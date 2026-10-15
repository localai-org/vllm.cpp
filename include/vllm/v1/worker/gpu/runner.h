// Ported from: vllm/v1/worker/gpu/model_runner.py @ e24d1b24
// (GPUModelRunner: initialize_kv_cache / execute_model / sample_tokens / sample /
// postprocess_sampled — the T0 slice) + the decode-first reorder from
// vllm/v1/attention/backends/utils.py::reorder_batch_to_split_decodes_and_prefills.
//
// Scope (M1.8 Task 4): the batched PAGED model runner — the concrete
// ModelRunnerBase (Task 1) the Executor pass-through drives. It owns the
// persistent InputBatch (M1.5), the KV caches (allocated from the M1.3
// KVCacheConfig), and the Sampler (M1.7), and threads a step through the MRV2
// forward/sample SPLIT:
//   * execute_model(scheduler_output): update_states (M1.5, admit new reqs incl.
//     prefill_token_ids + apply cached diffs) -> DECODE-FIRST REORDER -> the
//     paged step build (prepare_inputs M1.5 -> the per-KV-group attention
//     metadata M1.6) -> Qwen3_5Model::Forward (Task 3, over the KV caches) ->
//     STASH the [num_actual_tokens, vocab] logits + the StepInputs. Returns
//     std::nullopt (MRV2 "forward done"), or nullopt-with-empty-stash for a
//     0-token flush step (no forward run).
//   * sample_tokens(): gather the logits_indices rows -> [num_reqs, vocab] ->
//     make_sampling_metadata (M1.7, dense order — CLOSES the M1.7 wiring dep) ->
//     Sampler::forward -> ModelRunnerOutput -> WRITE the sampled tokens BACK into
//     the InputBatch so the next step's prepare_inputs sees them.
//
// ─── THE V1-ALGORITHM / MRV2-CONTRACT COMPOSITION (recorded) ────────────────
// We target the MRV2 runner's forward/sample SPLIT + scheduler-output contract,
// but IMPLEMENT it via the host-array V1 algorithm (persistent InputBatch holds
// the MultiGroupBlockTable; prepare_inputs emits dense-order StepInputs; the
// SamplingMetadata is built fresh each step in the SAME dense order). This
// composes with NO idx_mapping slot-indirection: the gathered logits
// (logits[logits_indices]), the SamplingMetadata rows, the attention metadata,
// and the sampled-token write-back all align on ONE dense order. The MRV2 staged
// per-slot GPU buffers (RequestState / InputBuffers / UvaBackedTensor /
// idx_mapping / apply_staged_writes) are axis-2 STORAGE, deferred to M2 (see
// .agents/vllm-v1-v2.md).
//
// ─── THE DECODE-FIRST REORDER (four-way ordering contract) ──────────────────
// Before building metadata the runner reorders the batch decode-first-then-
// prefill (reorder_batch_to_split_decodes_and_prefills, threshold 1) because the
// GDN split (M1.6 SplitDecodesAndPrefills) assumes decodes lead. ALL consumers
// {attention seq_lens/block_table, GDN segmentation, forward input_ids/positions,
// logits_indices, SamplingMetadata rows, write-back} are built AFTER the reorder,
// in that one order. Inert for a batch-of-1 / pure-decode / pure-prefill batch;
// it only bites a MIXED hybrid batch. Uses InputBatch::swap_states (M1.5).
//
// ─── DEFERRED (marked 1:1 stubs; slot in without reshaping the interface) ────
//   cudagraphs, spec-decode / rejection sampling, LoRA, multimodal / encoder,
//   pipeline parallelism, EPLB, kv-connector, prompt-logprobs,
//   async / AsyncOutput (ModelRunnerOutput returned synchronously), the MRV2
//   staged per-slot buffers + idx_mapping (M2), IntermediateTensors / dummy_run /
//   is_profile. compute_logits: our Qwen3_5Model::Forward already applies lm_head
//   and returns full [num_actual_tokens, vocab] logits, so sample_tokens only
//   GATHERS the logits_indices rows (no separate compute_logits pass at T0).
#ifndef VLLM_V1_WORKER_GPU_RUNNER_H_
#define VLLM_V1_WORKER_GPU_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vllm/config/speculative.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"  // SPEC-MTP I5d hidden tap + draft
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/model_executor/models/qwen3_dflash.h"  // SPEC-DFLASH D5 draft + aux taps
#include "vllm/model_executor/models/qwen3_dspark.h"  // SPEC-DSPARK W5 draft + Markov head
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/sample/device_scratch.h"  // HostBufferStaging (#1313)
#include "vllm/v1/sample/sampler.h"
#include "vllm/v1/worker/gpu/async_output.h"
#include "vllm/v1/worker/gpu/input_batch.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vllm/v1/worker/gpu/prepare_inputs.h"
#include "vllm/v1/worker/gpu/pool/pooling_runner.h"  // PoolingRunner (pooling arch)
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm::v1::kv_offload {
class KVConnector;  // KV-EXTERNAL-CACHE: worker-side store/load seam (fwd-decl).
}  // namespace vllm::v1::kv_offload

namespace vllm::v1 {
struct RejectionSamplerOutput;

// Decode-first reorder (utils.py::reorder_batch_to_split_decodes_and_prefills @
// e24d1b24, T0 subset). Reorders `input_batch`'s active [0, num_reqs) requests so
// all requests with num_scheduled_tokens <= decode_threshold at the front, using
// InputBatch::swap_states. The short_extend / long_extend chunked-prefill
// sub-regions collapse into the {decode, prefill} split at T0 (the gate models
// full-prefill in one chunk under the budget). Returns true iff the batch was
// modified. The builder's SplitDecodesAndPrefills (M1.6) then finds the boundary.
bool reorder_batch_to_split_decodes_and_prefills(
    InputBatch& input_batch, const SchedulerOutput& scheduler_output,
    int decode_threshold = 1);

// apply_grammar_bitmask (vllm/v1/structured_output/utils.py::apply_grammar_bitmask
// @ e24d1b24, M3.4 Task 3). Applies the per-step structured-output grammar mask
// to the gathered [num_logits, vocab] f32 `logits` (the exact tensor the runner
// feeds Sampler::forward), IN PLACE, BEFORE sampling.
//
// The `grammar_output` bitmask rows are COMPACTED — one row per structured
// request, ordered as `grammar_output.structured_output_request_ids`. `logits`
// rows are in the runner's DENSE batch order (`req_ids`, one per active request).
// This reorders each structured req's compacted bitmask row onto its dense logits
// row (utils.py:112-140), unpacks it (bit (t & 31) of word (t >> 5) SET => token t
// ALLOWED; CLEAR => FORBIDDEN), and sets every forbidden token's logit to -inf
// (reusing the M1.7 apply_allowed_token_ids -inf masking). Non-structured rows are
// untouched (all-allowed).
//
// `scheduled_spec_decode_tokens` supplies each req's spec-token count so the
// per-req logit offset (utils.py:117-118,134-138) matches upstream; spec-decode is
// deferred at T0, so the runner passes an EMPTY map (offset 0 => exactly one
// bitmask row per structured req, aligned to its dense logits row).
void apply_grammar_bitmask(
    const GrammarOutput& grammar_output,
    const std::vector<std::string>& req_ids,
    const std::map<std::string, std::vector<int32_t>>&
        scheduled_spec_decode_tokens,
    vt::Queue& queue, vt::Tensor& logits);

// PAGE-LOCKED host staging for a D2H copy the host does not wait for at the
// copy's issue point (SPEC-DFLASH2 A2-2, #2802).
//
// WHY IT IS NOT A `std::vector`. `cudaMemcpyAsync` of device-to-host into
// PAGEABLE memory is synchronous with respect to the host: the driver stages the
// bytes through its own pinned buffer and the call does not return until that
// staging is done. So a copy into a `std::vector` on a copy queue is not
// asynchronous at all, and every event around it is decoration. This tree
// already knows that — the async sampled-id route allocates its destination with
// `vt::Backend::AllocPinned` (`src/vllm/v1/worker/gpu/async_output.cpp`), which
// is what upstream gets for free by copying into torch CPU memory
// (`vllm/v1/worker/gpu/async_utils.py:124-125` @ pin 5559679229).
//
// WHY A GROW RETAINS THE OLD BLOCK INSTEAD OF FREEING IT. Freeing a staging
// block that a queued copy still writes is the same defect this wave's review
// found in the accept walk's argmax scratch: correct while the wait is in step,
// silently wrong the moment a later wave moves the wait. Retaining costs one
// block per DISTINCT LARGER STEP SHAPE — the ask is `rows * width + rows` — so a
// serving ramp that adds one request at a time retains one block per request
// added, bounded by the batch. It is not a handful and it is not a leak either:
// the sizes are strictly increasing and the shape is bounded, so the total is on
// the order of N^2/2 int32 for a batch of N. The precise statement is worth
// making because page-locked memory is the scarce kind. Everything is released
// together in the destructor.
//
// On a backend without page-locked memory `AllocPinned` forwards to `Alloc`
// (`vt::Backend::AllocPinned`), so this degrades to a plain host buffer and the
// CPU tier sees the same tokens either way.
class PinnedGrowStaging {
 public:
  PinnedGrowStaging() = default;
  ~PinnedGrowStaging();
  PinnedGrowStaging(const PinnedGrowStaging&) = delete;
  PinnedGrowStaging& operator=(const PinnedGrowStaging&) = delete;

  // A block of at least `elems` int32 values, page-locked. Stable until the
  // NEXT call that asks for more than the current block holds; the block it
  // replaces stays allocated, so a copy still in flight into the old block is
  // writing memory this object still owns.
  int32_t* Get(vt::Backend& backend, size_t elems);

  // How many blocks are alive, so the retain rule above is assertable and not
  // only readable. One after the first `Get`; one more per grow.
  size_t live_blocks() const { return blocks_.size(); }
  // The element count the current block holds.
  size_t elems() const { return elems_; }

 private:
  vt::Backend* backend_ = nullptr;
  std::vector<int32_t*> blocks_;
  size_t elems_ = 0;
};

// ─── SPEC-DFLASH2 A2-4 (#3004): THE COMMITTED-ID DOWNLOAD ───────────────────
//
// Brings ONE step's committed sampled ids to host memory the propose arms can
// read, from wherever the async decode arm left them, and returns a pointer to
// them. The shape is `AsyncOutput`'s (async_utils.py:29-44 @ pin
// 5559679229bc961848b121ccdeaa8fa5d79bec98) and the same one A2-2 gave the
// verify download: record a fork event on the MAIN queue so the copy is ordered
// after the sampler and after the write-back scatter, make the COPY queue wait
// it, issue the D2H there into PAGE-LOCKED memory, record a completion event on
// the copy queue, and block the host on that event alone. The main queue is
// never synchronized.
//
// WHY IT IS UNCONDITIONAL, and this is the point of the wave. `sample_tokens_
// async` has three write-back branches and #2920 had to ask WHICH one ran,
// because the two CUDA ones leave the step's committed ids unreadable on the
// host — the device-mirror branch leaves the host array's VALUES stale on
// purpose, the UMA branch writes it from a `LaunchScatterLastSampled` that
// nothing waits. That question was a `committed_ids_on_host` flag and a refusal
// by name, and on GB10's integrated default the UMA branch runs and hits it,
// which is what blocks A2-5. Calling this on EVERY branch deletes the question
// instead of answering it: there is no residence to read, and a FOURTH write-back
// branch cannot forget to set a flag that no longer exists. On the host branch
// the copy is a host-to-host memcpy of `num_reqs` int64s — paid only under
// `spec_on()`, and cheaper than the read-back it replaces being wrong.
//
// OWNERSHIP AND LIFETIME, stated because this row has conflated "owns
// everything" with "freed in the destructor" twice. The returned pointer is
// INTO `staging`, which owns it; the caller owns nothing and frees nothing. It
// is valid until `staging` is destroyed — a grow RETAINS the block it replaces
// (PinnedGrowStaging above), so even a later, larger call does not invalidate
// it. `staging` must outlive every copy issued into it; the runner's member is
// runner-lifetime and its blocks are released by `~PinnedGrowStaging`, which
// runs after `~GPUModelRunner`'s body and therefore after that body's drain.
// The host read below is well-defined because `SynchronizeEvent` has completed
// the copy before this returns; the block is not being written when it is read.
//
// `dev_ids` is `AsyncOutputSlot::device_sampled_ids`, the sampler's int64
// output. The page-locked block is measured in int32 elements, so an id costs
// two of them.
const int64_t* DownloadCommittedIds(vt::Backend& backend, vt::Queue& main_q,
                                    vt::Queue& copy_q, vt::Event& fork,
                                    vt::Event& ready, const void* dev_ids,
                                    int num_reqs, PinnedGrowStaging& staging);

// The batched paged model runner (upstream GPUModelRunner, T0 slice).
class GPUModelRunner final : public ModelRunnerBase {
 public:
  // Generic model-factory path. `model` owns/borrows its concrete weights and
  // must outlive the runner (LoadedEngine declares it before runner_).
  //
  // SPEC-MTP I5d-pre: the trailing draft parameters are the ENABLING seam for the
  // verify/propose loop (I5d). All three default to "no speculation", so every
  // existing construction site is byte-identical: `spec_config == nullopt` and a
  // null `draft_model` leave the runner on the current single-model decode path.
  // When set (I5d), `draft_model` is the MTP draft the runner owns for its
  // lifetime (built via LoadedModel::BuildMtpDraft) and `draft_kv` is its paged
  // KV layer (the `fa_draft` group). This increment only STORES them; nothing
  // reads them yet.
  GPUModelRunner(const HfConfig& config, LoadedModel& model,
                 const KVCacheConfig& kv_cache_config, vt::Queue queue,
                 int max_num_reqs, int max_model_len, int max_num_batched_tokens,
                 std::optional<vllm::SpeculativeConfig> spec_config =
                     std::nullopt,
                 std::unique_ptr<vllm::Qwen3_5MTPModel> draft_model = nullptr,
                 std::vector<PagedKvCache> draft_kv = {});

  // Construct the runner over a model (config + weights) and allocate the KV
  // caches from `kv_cache_config` (initialize_kv_cache). `queue` selects the
  // device (CPU at T0; CUDA dgx-pending). The InputBatch is sized from
  // max_num_reqs / max_model_len / max_num_batched_tokens and the per-KV-group
  // block sizes taken from the KVCacheConfig groups. `config` and `weights` must
  // outlive the runner (held by reference, mirroring upstream self.model).
  GPUModelRunner(const HfConfig& config, const Qwen3_5MoeWeights& weights,
                 const KVCacheConfig& kv_cache_config, vt::Queue queue,
                 int max_num_reqs, int max_model_len,
                 int max_num_batched_tokens);

  // DENSE-arch overload (27B, Qwen3_5ForConditionalGeneration / num_experts==0).
  // Identical to the MoE constructor except the model runs through the dense
  // weights + the paged dense forward (Qwen3_5DenseModel::Forward). The KV-cache
  // layout is config-driven (same GDN + full-attn hybrid backbone), so
  // initialize_kv_cache is unchanged. `config` and `weights` must outlive the
  // runner. The MoE-only fp4 decode-graph fast path stays inert on this arch.
  GPUModelRunner(const HfConfig& config, const Qwen3_5DenseWeights& weights,
                 const KVCacheConfig& kv_cache_config, vt::Queue queue,
                 int max_num_reqs, int max_model_len,
                 int max_num_batched_tokens);

  ~GPUModelRunner();

  // ModelRunnerBase (the MRV2 execute_model / sample_tokens split).
  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override;
  ModelRunnerOutput sample_tokens(
      const std::optional<GrammarOutput>& grammar_output) override;

  // sample_tokens_async (ENG-ASYNC-SCHED W3 sampler-OUTPUT half). The overlap
  // variant of sample_tokens: it produces the sampled ids DEVICE-RESIDENT (the
  // Sampler writes them into a fresh device buffer, no host download), does the
  // on-GPU post_update (last_sampled scatter), and returns an
  // AsyncGPUModelRunnerOutput that issued the single non-blocking sampled-id D2H
  // on the runner's COPY queue with a completion event — the MAIN queue is never
  // synchronized. The engine calls get_output() during output processing (off the
  // model's critical path), which blocks only on that copy event, so the copy
  // overlaps the next step's forward: this is the ~3.25 ms/step GPU-idle capture.
  // Token-exact with sample_tokens by construction (same argmax, same ids); on a
  // synchronous/unified backend the copy is a memcpy and events no-op. Only
  // meaningful when runner_supports_async() is true (async engaged). Mirrors
  // gpu_model_runner.py:4673-4693 (AsyncGPUModelRunnerOutput construction).
  std::unique_ptr<AsyncModelRunnerOutput> sample_tokens_async(
      const std::optional<GrammarOutput>& grammar_output) override;

  // take_draft_token_ids (SPEC-MTP I5d; gpu/model_runner.py:1483-1489). Returns
  // the out-of-band drafts the verify/propose loop produced this step, moving
  // them out (a second call returns nullopt). EngineCore::post_step pulls these
  // and installs them on the scheduler for the next verify step. nullopt when no
  // speculator is configured (the production default) — byte-identical.
  std::optional<DraftTokenIds> take_draft_token_ids() override {
    std::optional<DraftTokenIds> out = std::move(pending_drafts_);
    pending_drafts_.reset();
    return out;
  }

  // runner_supports_async (mirror of the vLLM compat gate feeding
  // SchedulerConfig::ResolveAsyncScheduling — vllm/config/vllm.py:990-1038).
  // TRUE iff the runner advertises async SCHEDULING capability: the env opt-in
  // (VT_ASYNC_RUNNER, default ON since the 2026-07-17 flip) on a backend whose
  // async-output primitives are wired (CUDA, or CPU's synchronous
  // degeneration for the CPU gate). With it true, ResolveAsyncScheduling
  // resolves the AsyncScheduler + mcb=2 by default, mirroring vLLM.
  // VT_ASYNC_RUNNER=0 rolls the runner back (false → sync Scheduler);
  // VT_ASYNC_SCHED=0 rolls only the scheduler back in the same binary.
  //
  // SPEC-DFLASH2 W7 (#1824): this is deliberately NOT async_input_combine_ —
  // that lever carries the spec veto (the device combine is not draft-aware,
  // I5e) and stays OFF under a speculator, while async SCHEDULING now stays ON
  // for the Eagle-type family exactly as upstream resolves it
  // (vllm/config/vllm.py:1064-1112): the spec engine keeps the synchronous
  // host input path and the host spec sampler, and the scheduler still
  // overlaps step N+1 with step N's output processing. The two were one flag
  // while spec forced sync scheduling; W7 is the commit that splits them.
  bool runner_supports_async() const override { return async_sched_supported_; }

  // SPEC-DFLASH2 W7 (#1824): whether the ENGINE resolved async scheduling ON.
  // Set once by LoadedEngine after ResolveAsyncEnabled (the runner cannot know
  // — the resolution also reads VT_ASYNC_SCHED and the pooling arm); read by
  // execute_model's draft-placeholder fill and computed-token correction.
  void set_async_scheduling(bool enabled) { use_async_scheduling_ = enabled; }
  bool use_async_scheduling() const { return use_async_scheduling_; }

  // ─── Accessors (for tests + the ordering identity gate) ────────────────────
  InputBatch& input_batch() { return input_batch_; }
  const InputBatch& input_batch() const { return input_batch_; }
  const std::vector<PagedKvCache>& attn_kv() const { return attn_kv_; }
  // The ENGINE-level attention backend selected PER full-attention KV GROUP
  // (resolved inside initialize_kv_cache's full-attn region via
  // vllm::v1::SelectAttentionBackendName — the same walk the registry test
  // covers), parallel to attn_kv(): one name per attention layer, in layer
  // order. Dense groups always resolve (a platform with no registered dense
  // backend throws loudly at init). An MLA group resolves TRITON_MLA where one
  // is registered (CUDA); on a device without a registered MLA backend (CPU,
  // ROCm today) the entry is EMPTY and that group's execution stays op-driven
  // (TritonMLAImpl on the fused cache — not registry-gated), which is why the
  // vector as a whole is empty only when no full-attention group exists at all
  // (a pure-GDN / pooling model caches no paged KV).
  const std::vector<std::string>& attn_backend_names() const {
    return attn_backend_names_;
  }
  const std::vector<GdnStateCache>& gdn_state() const { return gdn_state_; }
  // The compact GDN state-slot pool size (== max_num_reqs). Exposed for the
  // state-slot uniqueness regression tests.
  int64_t gdn_state_slots() const { return gdn_state_slots_; }
  // The step inputs / metadata built for the most recent execute_model (the four
  // consumers the ordering identity test aligns).
  const StepInputs& last_step() const { return exec_state_.step; }
  const CommonAttentionMetadata& last_attn_meta() const {
    return exec_state_.attn_meta;
  }
  const GDNAttentionMetadata& last_gdn_meta() const {
    return exec_state_.gdn_meta;
  }
  // Read-only output inspection. Copy the carrier to retain an owning output
  // across execute_model; the reference itself is step-local. These accessors
  // do not wait for GPU work or extend a non-owning logits view.
  const ForwardLogits& last_forward_logits() const { return exec_state_.logits; }
  const Qwen3_5MTPHiddenStates& last_spec_hidden() const {
    return exec_state_.spec_hidden;
  }
  // SPEC-MTP I5d acceptance telemetry accessors (the gate reads these).
  int64_t spec_drafts_proposed() const { return spec_drafts_proposed_; }
  int64_t spec_drafts_accepted() const { return spec_drafts_accepted_; }
  // SPEC-MTP-K-GT-1 (#81) PER-DEPTH acceptance. Index d counts the drafts
  // verified and accepted at draft depth d+1, so `[0]` is each request's first
  // drafted token and `[k-1]` its deepest. The aggregate counters above cannot
  // answer "how deep did acceptance actually reach", which is what #81's M1 asks
  // for and the only signal an acceptance-driven depth policy could use. The
  // vectors grow to the deepest draft the engine has verified.
  //
  // Their SIZE is NOT a witness that the propose loop ran k times, and an earlier
  // revision of this comment claimed it was. The size is grown from
  // `step.num_draft_tokens_per_req[i]`, which is the LENGTH of the emitted draft
  // list, a pure function of how `propose_drafts` slices whatever the proposer
  // returned. A propose that runs ONE forward and pads all k columns with its
  // step-0 draft satisfies it exactly, and the emitted tokens do not move either,
  // because greedy plus accept-iff-equal makes the sequence independent of k.
  // That mutation was applied and the whole depth suite stayed green on it. The
  // witness that DOES separate them is `spec_mtp_draft_decode_forwards()` below.
  const std::vector<int64_t>& spec_drafts_proposed_by_depth() const {
    return spec_drafts_proposed_by_depth_;
  }
  const std::vector<int64_t>& spec_drafts_accepted_by_depth() const {
    return spec_drafts_accepted_by_depth_;
  }
  // SPEC-MTP-K-GT-1 (#81): the WORK witness, counted where the work happens.
  //
  // `spec_mtp_propose_calls()` counts MTP propose invocations that reached the
  // proposer. `spec_mtp_draft_decode_forwards()` counts draft DECODE forwards
  // those calls actually executed, incremented after each forward returns.
  // Upstream runs `num_speculative_steps - 1` of them per propose
  // (`_multi_step_decode`, autoregressive/speculator.py:374-419 @ 555967922), so
  // the exact relation a caller asserts is
  //
  //     spec_mtp_draft_decode_forwards() == spec_mtp_propose_calls() * (k - 1)
  //
  // with both sides falling to 0 forwards at k=1. Nothing about the shape of the
  // emitted draft list can satisfy that equality, so it is the assertion a
  // propose that SHORT-CIRCUITS or CLAMPS the loop fails.
  //
  // It does NOT see padding, and an earlier revision of this comment said it
  // did. A propose that runs every forward, discards the results and writes the
  // step-0 draft into all k columns increments this counter exactly as the real
  // loop does. That mutation was applied and the whole depth suite stayed green
  // on it. `spec_mtp_proposals_with_varied_drafts()` below is the counter that
  // separates those two.
  int64_t spec_mtp_propose_calls() const { return spec_mtp_propose_calls_; }
  int64_t spec_mtp_draft_decode_forwards() const {
    return spec_mtp_draft_decode_forwards_;
  }
  // SPEC-MTP-K-GT-1 (#81): the RESULT witness, computed on the array the
  // proposer DELIVERED rather than inside the loop that filled it.
  //
  // It counts propose calls whose returned draft row held, for at least one
  // request, a column that differed from column 0. A propose that pads every
  // column with its step-0 draft leaves this at 0 at every k, because a padded
  // row is by construction a pure function of its own first column, and that is
  // the whole reason the counter exists. At k=1 it is 0 by definition, since a
  // one-column row has nothing to differ from.
  //
  // Its bound is stated here because it is easy to over-read. It says the
  // delivered array carries information the prefill draft alone does not
  // determine. It does NOT say that column j came from forward j, so an
  // off-by-one in the column index or a broken carry still satisfies it. It is
  // also a NECESSARY rather than a sufficient condition, and one the fixture
  // participates in: a drafter that resampled the same token on every step of
  // every call would leave it 0 while running the loop correctly. Measured on
  // the synthetic CPU gate model that does happen on individual calls (a
  // `2 2 2` row at k=3) and never on all of them, so the assertion the depth
  // suite makes is `> 0` over a run and never a per-call one. Per-column
  // provenance stays owed to the DGX gate on real weights, and NOT as a
  // non-zero acceptance count at depth, which a padded row also earns whenever
  // the target's greedy continuation repeats a token. It is owed as an
  // acceptance-RATE comparison against a padded control, specified in the
  // row spec under `## Owed`.
  int64_t spec_mtp_proposals_with_varied_drafts() const {
    return spec_mtp_proposals_with_varied_drafts_;
  }
  int full_attn_group_id() const { return full_attn_group_id_; }
  int gdn_group_id() const { return gdn_group_id_; }
  int64_t num_blocks() const { return num_blocks_; }
  // The per-block byte cost the attention-cache allocator ACTUALLY used, taken
  // from the KV SPEC (`spec->page_size_bytes()`, upstream
  // vllm/v1/kv_cache_interface.py:380-398) rather than reconstructed from the
  // HF config. 0 when the model has no full-attention group. This is the
  // positive signal that the spec-driven allocation path RAN — a compiled but
  // unexercised path leaves it 0, and a `page_size_padded` spec produces a
  // value the old HF-config arithmetic could not.
  int64_t fa_page_size_bytes() const { return fa_page_size_bytes_; }

  // FIX-KV-GROUP-LAYER-COUNT (#1963, #1966). What `initialize_kv_cache`
  // ALLOCATED, summed over every buffer it created, so a gate can compare the
  // sizing arithmetic against the allocation rather than against a second copy
  // of the same formula. Both are 0 before `initialize_kv_cache` runs.
  //
  //   ...paged_bytes() — the block-scaled half: one buffer per full-attention
  //     layer plus the speculative draft layer's. This is the half a
  //     `--kv-cache-memory` budget is supposed to bound, and
  //     `KVBytesPerBlock(cfg) * cfg.num_blocks` is supposed to equal it.
  //   ...allocated_bytes() — that plus the recurrent (GDN/Mamba) conv and SSM
  //     state, which is sized per sequence slot and not per block, and which
  //     `recurrent_state_bytes(cfg, max_num_reqs)` is supposed to equal.
  int64_t kv_cache_allocated_paged_bytes() const;
  int64_t kv_cache_allocated_bytes() const;

  // #810: the per-layer KV class `initialize_kv_cache` RESOLVED, index == model
  // layer index, one entry per hidden layer. `kNone` is a layer that no KV
  // cache group named and that therefore caches nothing — NemotronH's 23
  // MoE blocks, which upstream's module walk yields no kv_cache_spec entry for
  // (`gpu_model_runner.py:7785-7801`). Recorded because the routing decision is
  // otherwise observable only as buffer COUNTS, and a count cannot see a
  // routing inversion: 3 recurrent + 1 attention has the same counts whichever
  // three layers got which.
  //
  // KV-DSV4-MULTICACHE W3 (#2068) adds the fourth value. `kMultiCache` means
  // THIS LAYER'S CACHES ARE NOT DESCRIBED BY THE POSITIONAL `attn_kv[fa_idx]`
  // CONVENTION — read `layer_attn_kv_indices()[l]` instead. A DeepSeek-V4 C4A
  // layer has four entries there; layers 0 and 1 have exactly one (the SWA cache
  // alone, since `get_kv_cache_spec` returns None for `compress_ratio <= 1`,
  // `vllm/models/deepseek_v4/attention.py:626-630`) and are STILL kMultiCache,
  // because one cache reached by name is not the same thing as one cache reached
  // by position.
  enum class LayerKvClass : uint8_t {
    kNone = 0,
    kFullAttention = 1,
    kRecurrent = 2,
    kMultiCache = 3,
  };
  const std::vector<LayerKvClass>& layer_kv_class() const {
    return layer_kv_class_;
  }

  // KV-DSV4-MULTICACHE W3 (#2068): the published groups this runner carries.
  //
  // `attn_group_ids()` lists EVERY non-eagle group whose spec is an
  // `AttentionSpec`, in publication order; `recurrent_group_ids()` lists every
  // `kMamba` group. `full_attn_group_id()` and `gdn_group_id()` keep their old
  // meanings and their old values — the FIRST non-eagle full-attention/MLA group
  // and a Mamba group — because eleven call sites read them and none of them
  // means anything different.
  const std::vector<int>& attn_group_ids() const { return attn_group_ids_; }
  const std::vector<int>& recurrent_group_ids() const {
    return recurrent_group_ids_;
  }

  // KV-DSV4-MULTICACHE W3 (#2068): per-layer indices into `attn_kv()`, index ==
  // model layer index. EMPTY for every topology the positional convention can
  // express, which is the same empty-means-unchanged contract
  // `per_layer_attn_specs` states (`include/vllm/v1/kv_cache_interface.h`).
  const std::vector<std::vector<int32_t>>& layer_attn_kv_indices() const {
    return layer_attn_kv_indices_;
  }
  // The name each entry of `attn_kv()` was PUBLISHED under — upstream's
  // `static_forward_context` key. EMPTY on every uniform topology.
  const std::vector<std::string>& attn_kv_layer_names() const {
    return attn_kv_layer_names_;
  }
  // The third forward channel exactly as `ModelRegistry::Forward` receives it.
  // `size() == 0` on every uniform topology.
  const vllm::MultiKvCacheIndex& multi_kv_index() const {
    return multi_kv_index_;
  }

  // Async-scheduling device-input path (ENG-ASYNC-SCHED W3 runner leaf). When
  // ON, execute_model rebuilds each decode row's input token id from the
  // GPU-resident-analog InputBatch::last_sampled_tokens via
  // combine_sampled_and_draft_tokens INSTEAD of the host token_ids_cpu read —
  // the piece that lets step N+1 be prepared without waiting on step N's sampled
  // token to cross to the host (the ~3.25 ms/step idle). DEFAULT ON since the
  // 2026-07-17 flip (VT_ASYNC_RUNNER default ON): the production runner engages the
  // async device path so `runner_supports_async` is TRUE by default; the DGX
  // token-exactness gates + the 6ea7856 discriminator proved greedy tokens are
  // bit-identical to the sync path. VT_ASYNC_RUNNER=0 rolls it back to the
  // synchronous host path at construction; tests toggle it directly. Greedy tokens
  // are bit-identical in both modes (combine writes the same id sample_tokens wrote
  // back to token_ids_cpu).
  void set_async_input_combine(bool enabled) {
    async_input_combine_ = enabled;
  }
  bool async_input_combine() const { return async_input_combine_; }
  // ENG-ASYNC-SCHED depth-2 lifetime guard, exposed for the regression test.
  // TRUE between a sample_tokens_async that left main-queue work outstanding and
  // the next execute_model that drains it (see async_forward_in_flight_).
  bool async_forward_in_flight() const { return async_forward_in_flight_; }
  bool kv_cache_backend_resident() const {
    return kv_cache_backend_resident_;
  }
  vt::Device device() const { return queue_.device; }

  // KV-EXTERNAL-CACHE (LMCache) worker-side seam. A non-owning pointer to the
  // SAME connector instance the scheduler holds (single-process engine); null =
  // no connector = ZERO behaviour change (the load/store paths are skipped, the
  // KV cache is byte-identical to production). Set by LoadedEngine right after
  // both runner_ and scheduler_ are built. Only the concrete LMCacheConnector
  // carries a worker-side store/load (dynamic_cast in runner.cpp); a base
  // connector with no worker transport is a no-op here.
  void set_kv_connector(kv_offload::KVConnector* connector) {
    kv_connector_ = connector;
  }
  kv_offload::KVConnector* kv_connector() const { return kv_connector_; }

  // SPEC-DFLASH D5 (DF-ENGINE-INTEGRATION): wire the separately-loaded z-lab
  // DFlash draft into the verify/propose loop. `weights`/`config` are borrows
  // owned by LoadedEngine (they outlive the runner); `k` = num_speculative_tokens
  // (block-1's block, the (1+k) query). Sets use_dflash(): on the verify forward
  // the runner captures the D1 multi-tap (target_layer_ids) instead of the MTP
  // single hidden tap, and propose_drafts routes to the DFlash block propose.
  // Idempotent; null weights leaves the runner on the MTP/non-spec path.
  void set_dflash_draft(const vllm::Qwen3DFlashWeights* weights,
                        const vllm::HfConfig* config, int k);

  // SPEC-DSPARK W5: wire the separately-loaded DSpark draft into the SAME
  // verify/propose loop. DSpark inherits DFlash's context accumulation and block
  // forward unchanged (Qwen3DSparkModel(DFlashQwen3Model)); it differs only in
  // the query-block layout (N rows with the anchor itself predicting, or the
  // DFlash 1+N fill-in when `sample_from_anchor` is false) and in sampling
  // (sequential Markov instead of one parallel argmax). Wiring it therefore
  // routes through the same aux-tap capture and the same device KV store —
  // set_dflash_draft is called internally with `&weights->backbone`, so
  // use_dflash() stays the predicate for the shared machinery and use_dspark()
  // only switches the propose tail. Idempotent; null leaves the runner alone.
  void set_dspark_draft(const vllm::Qwen3DSparkWeights* weights,
                        const vllm::HfConfig* config, int k,
                        bool sample_from_anchor);

  // SAMPLE-PROMPT-LOGPROBS route-observation seam. Const, no behaviour, exposed
  // for the gate: the row count the LAST forward actually produced, the token
  // count that step ran on, and its request count. On EVERY step where no
  // request asked for prompt logprobs the forward gathers before lm_head, so
  // last_forward_rows() == step_num_logits(); a step that owes prompt logits
  // instead takes the full-logits route and returns num_actual_tokens rows.
  // test_llm_engine §9(g) asserts that DECISION directly, because an on-vs-off
  // comparison inside ONE build cannot see a change to the shared route — both
  // arms move together (review finding 2 on PR #235).
  int64_t last_forward_rows() const { return exec_state_.logits.rows; }
  int last_forward_num_actual_tokens() const {
    return exec_state_.num_actual_tokens;
  }
  int last_forward_num_reqs() const { return exec_state_.num_reqs; }
  // The expanded logit-row count for the stashed step (StepInputs::cu_num_logits
  // back, == exec_state_.num_reqs on the non-speculative default path). Public
  // so the gate above can name the expected value instead of re-deriving it.
  int step_num_logits() const;

 private:
  // Owns one persistent cache allocation. CUDA defaults to vt::Alloc-backed
  // device storage; CPU and VT_DEVICE_KV_CACHE=0 retain the host-vector
  // diagnostic fallback. Tensor/PagedKvCache views never own this memory.
  class CacheBuffer {
   public:
    CacheBuffer(vt::Device device, vt::Queue& queue, size_t bytes,
                bool backend_resident);
    ~CacheBuffer();
    CacheBuffer(const CacheBuffer&) = delete;
    CacheBuffer& operator=(const CacheBuffer&) = delete;

    void* data() {
      return backend_resident_ ? backend_data_ : host_data_.data();
    }

    // The byte size this buffer was constructed with — what the allocation
    // COST, not what a formula predicts it cost. `kv_cache_allocated_bytes()`
    // below sums these (FIX-KV-GROUP-LAYER-COUNT, #1963).
    size_t bytes() const { return bytes_; }

   private:
    vt::Device device_;
    bool backend_resident_ = false;
    size_t bytes_ = 0;
    void* backend_data_ = nullptr;
    std::vector<uint8_t> host_data_;
  };

  // Compatibility path for direct synthetic-weight runner tests. The wrapper
  // is type-erased but borrows the caller-owned concrete weights. Widened with
  // the same defaulted SPEC-MTP I5d-pre draft seam so the concrete-weight
  // constructors (which delegate here) stay byte-identical.
  GPUModelRunner(const HfConfig& config,
                 std::unique_ptr<LoadedModel> owned_model,
                 const KVCacheConfig& kv_cache_config, vt::Queue queue,
                 int max_num_reqs, int max_model_len, int max_num_batched_tokens,
                 std::optional<vllm::SpeculativeConfig> spec_config =
                     std::nullopt,
                 std::unique_ptr<vllm::Qwen3_5MTPModel> draft_model = nullptr,
                 std::vector<PagedKvCache> draft_kv = {});

  // ARCH-ONE-SURFACE ROW 6: the pooling counterpart of sample_tokens (mirror
  // of gpu/model_runner.py:1586-1607 + pool/pooling_runner.py:29-42). Consumes
  // the stashed forward result — for the pooling arch those are the
  // [rows, hidden] post-final-norm hidden states, NOT vocab logits — applies
  // the model's Pooler via pooling_runner_, and returns a ModelRunnerOutput
  // whose pooler_output carries one pooled vector per fully-prefilled request
  // (nullopt for rows still consuming prefill chunks — the same
  // seq_len == prompt_len validity predicate as is_valid, pooling_runner.py:
  // 40-41, which our discard mask already computes). sampled_token_ids rows
  // stay EMPTY: a pooling step samples nothing.
  ModelRunnerOutput pool_tokens();

  // Allocate the per-full-attn-layer paged KV buffers + the per-GDN-layer
  // persistent mamba ssm/conv buffers from the KVCacheConfig groups.
  void initialize_kv_cache(const KVCacheConfig& kv_cache_config);
  // ENG-RECURRENT-MULTISTATE (#2131): append ONE recurrent layer's complete
  // state set to `recurrent_state_buf_`, one buffer per published state, in
  // `MambaSpec::shapes` order. Both allocation sites in initialize_kv_cache (the
  // legacy path and the multi-cache path) call it, so neither can grow its own
  // idea of how many states a layer has.
  void alloc_recurrent_layer_states(
      vt::Device dev, const std::vector<vt::DType>& state_dtypes,
      const std::vector<int64_t>& state_row_elems);
  // Build the [num_reqs, num_cols] committed block-table slice for a KV group.
  std::vector<int32_t> gather_block_table(int group_id, int num_reqs,
                                          int* num_cols) const;
  // MODEL-MM-QWEN4-EXP W5c-2 (#2249 item 3): gather EVERY published group's
  // block table into `group_block_tables_`, in group order.
  //
  // The two named-id gathers above serve the TARGET attention group and the
  // recurrent group, and there is no third name to add — `qwen4_exp` publishes
  // a third group (the QSA indexer side cache, an `MLAAttentionSpec` at
  // compress_ratio 4) and DeepSeek-V4-Flash publishes seven. Upstream never had
  // a named-id shape: its per-group metadata loop runs over
  // `enumerate(kv_cache_groups)` and assigns each group its own table
  // (`vllm/v1/worker/gpu_model_runner.py:2551-2567` @ pin 5559679229,
  // `cm.block_table_tensor = _get_block_table(kv_cache_gid)`), where
  // `_get_block_table` (`:2318-2334`) is exactly `gather_block_table`.
  //
  // MULTI-CACHE ONLY. Every uniform topology leaves `group_block_tables_` empty
  // and `multi_kv_index_`'s two new pointers null, so no model shipping today
  // pays a copy or can observe the channel — the same empty-means-unchanged
  // contract the other three W3 vectors keep.
  void gather_group_block_tables(int num_reqs);
  // Rewrite the GDN group's block-table col 0 into a COMPACT per-sequence state
  // slot in [0, gdn_state_slots_). The GDN mamba state is one recurrent state
  // per SEQUENCE, kept in a compact cache sized by max_num_reqs — NOT the
  // attention num_blocks (which grows with concurrency×seq_len and made the f32
  // ssm_state the dominant memory consumer). The pool is keyed on the request's
  // stable IDENTITY (req_id), NOT the mamba pool block-id: under our MambaSpec
  // "none" config the group uses a sub-sequence block_size, so once a sequence
  // exceeds one block MambaManager::remove_skipped_blocks nulls every block but
  // the last and block-table col 0 collapses to the shared null block-id 0 —
  // block-id keying then maps every long sequence to ONE slot (a duplicate live
  // state index / cross-request state corruption). vLLM reaches the same
  // per-sequence state index via mamba_get_block_table_tensor (gathering the
  // CURRENT state block); because our compact per-sequence cache makes the
  // physical block-id irrelevant, the sequence identity is the correct key: each
  // live request owns exactly one slot for its whole lifetime, freed only when
  // it leaves the batch and reused only after. Only col 0 is read by the GDN
  // builder (state indices).
  void remap_gdn_state_slots(std::vector<int32_t>& gdn_bt, int gdn_cols,
                             int num_reqs,
                             const std::vector<std::optional<std::string>>&
                                 req_ids);

  const HfConfig& config_;
  // Production: model_ borrows LoadedEngine::model_. Direct runner tests use a
  // small owned adapter that in turn borrows their concrete weights.
  std::unique_ptr<LoadedModel> owned_model_;
  LoadedModel* model_ = nullptr;
  // SPEC-MTP I5d-pre draft seam (all inert unless a SpeculativeConfig is set).
  // spec_config_ carries the resolved method/k; draft_model_ is the MTP draft the
  // runner owns for its lifetime (built via LoadedModel::BuildMtpDraft);
  // draft_attn_kv_ is the draft's paged KV layer (the `fa_draft` group). Stored
  // here so the I5d verify/propose loop can reach them; UNREAD in this increment
  // (nullopt / null / empty on every existing construction site).
  std::optional<vllm::SpeculativeConfig> spec_config_;
  std::unique_ptr<vllm::Qwen3_5MTPModel> draft_model_;
  std::vector<PagedKvCache> draft_attn_kv_;
  vt::Queue queue_;
  // #1313 — the [rows, vocab] logits the ON-DEVICE sampler runs on are assembled
  // from a HOST buffer whenever the forward returned ForwardLogits.host
  // (nemotron_h, laguna, qwen3_vl). Those bytes must be reachable by the queue
  // device before any sampling kernel dereferences them: free on a unified
  // backend (GB10 wraps in place, unchanged), a grow-only H2D copy on a discrete
  // one, where the old host address was simply illegal.
  //
  // TWO buffers, not one: collect_prompt_logprobs stages the prompt rows while
  // the assembled sample-logits tensor is still live (sample_tokens calls it
  // between assemble_sample_logits and the sampler), so a shared buffer would
  // invalidate the tensor the sampler is about to read.
  HostBufferStaging sample_logits_staging_;
  HostBufferStaging prompt_logprobs_staging_;
  InputBatch input_batch_;
  Sampler sampler_;
  // ARCH-ONE-SURFACE ROW 6 (mirror of gpu/model_runner.py:368-369
  // `if self.is_pooling_model ...: self.pooling_runner = PoolingRunner(model)`):
  // non-null iff the loaded model's registration declares is_pooling_model and
  // the model owns a Pooler. sample_tokens then routes to pool_tokens() — the
  // POOLED DATA takes the place of sampled tokens (model_runner.py:1586-1607).
  // Null for every text arch: the sampler path below is byte-identical.
  std::unique_ptr<vllm::PoolingRunner> pooling_runner_;

  // ─── ENG-MM-INPUT-PIPELINE P2 (#2379): the multimodal step state ──────────
  //
  // `supports_mm_inputs()` mirrors `self.supports_mm_inputs`
  // (gpu_model_runner.py:530, `grep -c 'self.supports_mm_inputs ='` == 1),
  // resolved ONCE at construction from the registration's two required hooks.
  // FALSE for every text architecture, and then every member below stays empty
  // and every mm branch in execute_model is not entered.
  //
  // The PREDICATE the mm branch actually takes is this flag AND some request in
  // the batch carrying mm_features — one condition wider than upstream's, which
  // routes an mm model's text-only step through inputs_embeds too. The reason is
  // that our registered mm forwards are not all embeds-only: Muse Glimmer's has
  // both branches and Qwen3-VL's has only the mm one, so the runner cannot force
  // either shape for a step that carries no multimodal request at all. That is a
  // RUNTIME predicate, which is why §6.5 of the spec retires the "text steps are
  // byte-identical by construction" claim and proves it by mutation instead.
  // Derived at READ time from the registration rather than cached in a member,
  // because this runner has four constructors and a cached flag is four places
  // to forget. The lookup is one pointer dereference on a ModelFactory.
  bool supports_mm_inputs() const {
    return model_ != nullptr && ModelRegistry::SupportsMmInputs(*model_);
  }
  // `self.uses_mrope`: whether the registration supplies the OPTIONAL M-RoPE
  // hook. False is upstream's `uses_mrope == False`, and the model then reads
  // the ordinary 1-D `ModelForwardInput::positions`.
  bool uses_mrope() const {
    return model_ != nullptr && ModelRegistry::UsesMrope(*model_);
  }
  // `self.requests: dict[str, CachedRequestState]` — the per-REQUEST state that
  // lives beside the per-SLOT InputBatch. Kept only when supports_mm_inputs(),
  // because it holds a copy of every prompt and a text engine has no reader.
  std::unordered_map<std::string, CachedRequestState> req_states_;
  // `self.encoder_cache: dict[str, torch.Tensor]`, keyed by MM_HASH and never by
  // (req_id, input_id) — gpu_model_runner.py:2995
  // (`grep -c 'self.encoder_cache\[mm_hash\] = output'` == 1). Two requests that
  // send the same image share one entry, which is the whole point of the hash
  // key, and an entry leaves only when the SCHEDULER reports it evicted.
  std::unordered_map<std::string, MmEncoderOutput> encoder_cache_;

  // The row slices this step's placeholder positions take from the encoder
  // cache, plus the per-token mask that says which positions they are.
  struct MmGather {
    // Borrowed views into `encoder_cache_` entries, in the order their `true`
    // positions appear in `is_mm_embed`. Valid only while this step runs.
    std::vector<vt::Tensor> mm_embeds;
    std::vector<std::shared_ptr<MmEncoderLifetime>> mm_lifetimes;
    // [total_num_scheduled_tokens]; `char` because std::vector<bool> has no
    // contiguous buffer to hand a model.
    std::vector<char> is_mm_embed;
  };

  // `free_encoder_mm_hashes` -> drop those entries (the worker half of the
  // scheduler's eviction). No-op on a step whose list is empty, which is every
  // text step.
  void free_evicted_encoder_outputs(const SchedulerOutput& scheduler_output);
  // `_init_mrope_positions` (gpu_model_runner.py:1654, grep -c == 1): the 3-D
  // prompt positions, computed ONCE per request at admission because they need
  // the WHOLE prompt. Only for a request that carries mm_features.
  void init_mrope_positions(const SchedulerOutput& scheduler_output);
  // `_execute_mm_encoder` (gpu_model_runner.py:2998, grep -c == 1): run the
  // TOWER over exactly the items the scheduler named this step and store each
  // output under its mm_hash. This is the first `src/` caller of a vision tower.
  void execute_mm_encoder(const SchedulerOutput& scheduler_output);
  // `_gather_mm_embeddings` (gpu_model_runner.py:3220, grep -c == 1).
  MmGather gather_mm_embeddings(const SchedulerOutput& scheduler_output,
                                int total_num_scheduled_tokens,
                                int draft_lookahead = 0);
  // `_calc_mrope_positions` (gpu_model_runner.py:2748, grep -c == 1): [3, T]
  // row-major. Image prompt coordinates are SLICED from the per-request array;
  // text prompt rows in a mixed batch use ordinary token positions. The
  // completion part is SYNTHESISED as `context_len + i + delta` on all three
  // axes, which is why one int per request carries M-RoPE across every decode
  // step (gpu_model_runner.py:2786, grep -c == 1).
  std::vector<int32_t> calc_mrope_positions(
      const SchedulerOutput& scheduler_output, int total_num_scheduled_tokens);
  // Whether ANY request in the current batch carries multimodal items. The
  // second half of the mm predicate; see supports_mm_inputs() above.
  bool batch_carries_mm() const;

  // KV group layout (resolved from the KVCacheConfig).
  int full_attn_group_id_ = -1;
  int gdn_group_id_ = -1;
  // KV-DSV4-MULTICACHE W3 (#2068): the generalized group layout. See the
  // accessors above. `multi_cache_topology_` is TRUE only when the published
  // group set leaves groups over after the target attention group, the recurrent
  // group and the `fa_draft` slot — the same set the W2 refusal computes — so it
  // is FALSE for every model shipping today and the members below stay empty.
  std::vector<int> attn_group_ids_;
  std::vector<int> recurrent_group_ids_;
  bool multi_cache_topology_ = false;
  std::vector<std::vector<int32_t>> layer_attn_kv_indices_;
  std::vector<std::string> attn_kv_layer_names_;
  // ENG-MULTIKV-BYNAME: the BY-NAME index over EVERY published cache, paged and
  // recurrent alike, in published GROUP order and then the group's own
  // `layer_names` order — upstream's own insertion order for the single
  // `kv_caches` dict (`vllm/v1/worker/gpu_model_runner.py:7365-7372` @ pin
  // 5559679229). These five are parallel to EACH OTHER and back
  // `multi_kv_index_`'s five pointers.
  //
  // They are NOT parallel to `attn_kv_`, and that is the whole change: W3's
  // three vectors were, so a `MambaSpec` group — which contributes no `attn_kv_`
  // entry — could not be represented in them at all, and `Find()` answered -1
  // for all 34 of `qwen4_exp`'s recurrent states (#2343). `kv_index_payload_*`
  // carries the locator a position used to imply: entry `i` is
  // `attn_kv_[slot]` when its kind is `kPaged` and `gdn_state_[slot]` when it is
  // `kRecurrent`. Empty on every uniform topology.
  std::vector<std::string> kv_index_layer_names_;
  std::vector<int32_t> kv_index_group_ids_;
  std::vector<int32_t> kv_index_layer_indices_;
  std::vector<uint8_t> kv_index_payload_kinds_;
  std::vector<int32_t> kv_index_payload_slots_;
  // W5c-2 (#2249 item 3): one gathered block table per PUBLISHED GROUP, indexed
  // by group id (NOT parallel to `attn_kv_` — a table belongs to a group, and
  // every layer in the group shares it, which is upstream's own fan-out). Sized
  // once at `initialize_kv_cache` on the multi-cache path and refilled by
  // `gather_group_block_tables` on every step; empty on every uniform topology.
  std::vector<std::vector<int32_t>> group_block_tables_;
  std::vector<int32_t> group_block_table_cols_;
  vllm::MultiKvCacheIndex multi_kv_index_;
  int64_t num_blocks_ = 0;
  // Per-block attention-cache bytes as reported by the KV spec (see the
  // fa_page_size_bytes() accessor).
  int64_t fa_page_size_bytes_ = 0;
  // #810: per-layer KV class, index == model layer index (layer_kv_class()).
  std::vector<LayerKvClass> layer_kv_class_;
  // Persistent-batch capacity = max concurrent sequences. The GDN mamba-state
  // cache is sized by this (one recurrent state per sequence), decoupled from
  // the attention num_blocks. See remap_gdn_state_slots.
  int max_num_reqs_ = 0;
  // Flattened-token bound for one step; sizes the W4 device input_ids mirror.
  int max_num_batched_tokens_ = 0;
  int64_t gdn_state_slots_ = 0;
  std::shared_ptr<RecurrentPrefixSnapshotIndex> recurrent_prefix_snapshots_;
  int64_t prefix_snapshot_base_ = 0;
  void copy_recurrent_state_slot(int64_t source, int64_t destination);
  void publish_recurrent_prefixes(const StepInputs& step);
  // Compact GDN state-slot allocator: request identity (req_id) -> slot in
  // [0, gdn_state_slots_); free list of unused slots. Keyed on the sequence, not
  // the mamba pool block-id (see remap_gdn_state_slots for why block-id keying
  // collapsed long concurrent sequences onto one slot).
  std::unordered_map<std::string, int32_t> gdn_slot_of_req_;
  std::vector<int32_t> gdn_free_slots_;
  // Reused per-step scratch for the live-request set in remap_gdn_state_slots:
  // cleared and refilled each step (buckets stay allocated) so the compact
  // slot remap does no per-step set allocation.
  std::unordered_set<std::string> gdn_alive_scratch_;

  // Owned persistent cache storage plus the non-owning views used by forward.
  // CUDA uses backend allocations by default (VT_DEVICE_KV_CACHE=0 restores the
  // former host-vector storage for same-binary attribution); CPU stays host.
  // Full-attention KV is bf16 by default, or f32 under VT_KV_CACHE_F32.
  bool kv_cache_backend_resident_ = false;
  // Async-scheduling device-input opt-in (see set_async_input_combine). Default
  // from VT_ASYNC_RUNNER at construction; OFF keeps the sync host path.
  bool async_input_combine_ = false;
  // Async SCHEDULING capability (see runner_supports_async): the same
  // env/backend predicate WITHOUT the spec veto. W7 (#1824).
  bool async_sched_supported_ = false;
  // Whether the engine resolved async scheduling ON (set_async_scheduling;
  // LoadedEngine calls it once after ResolveAsyncEnabled). Gates the
  // draft-placeholder fill + computed-token correction in execute_model.
  bool use_async_scheduling_ = false;
  // ─── SPEC-DFLASH2 A2-4 (#3004): prev_num_draft_len AND ITS CORRECTION ──────
  //
  // `prev_num_draft_len` is upstream's name and upstream's value: the draft
  // count the PREVIOUS step scheduled for a request, written where the drafts
  // are spliced (`gpu_input_batch.py:517` @ pin
  // 5559679229bc961848b121ccdeaa8fa5d79bec98, `request.prev_num_draft_len =
  // num_spec_tokens`) and read at the next `_update_states` (`:1356`). W7
  // carried the same value under the name `prev_sched_draft_counts_`; the rename
  // is not cosmetic, because the three maps below are one mechanism and had to
  // be named after the mechanism upstream has.
  //
  // Keyed by req_id rather than by req_state slot, deliberately: these are
  // one-step facts about a REQUEST, so they neither move through `condense` nor
  // swap through `swap_states`. Upstream keeps them on `CachedRequestState` for
  // the same reason. All three are rebuilt every spec step and are empty
  // otherwise.
  std::map<std::string, int> prev_num_draft_len_;
  // req_id -> the `num_computed_tokens` THIS RUNNER wrote for it last step, so
  // the optimistic value below needs no accept result. Upstream's
  // `prev_positions` (`gpu_model_runner.py:2155`), which its GPU correction
  // kernel reads for exactly this.
  std::map<std::string, int> prev_num_computed_tokens_;
  // req_id -> `valid_sampled_token_count` for its last verify step: how many
  // tokens the accept walk emitted (accepted drafts plus the bonus). Upstream's
  // `_get_valid_sampled_token_count()` (`gpu_model_runner.py:1530`), which it
  // reads back from the device; ours is written on the host in
  // `sample_tokens_with_rejection` and that is the ONE remaining host-visibility
  // dependency of the correction — see `apply_deferred_spec_decode_corrections`.
  std::map<std::string, int> prev_valid_sampled_count_;
  // One entry per request whose `num_computed_tokens_cpu` this step set
  // OPTIMISTICALLY — assuming every one of last step's drafts was accepted —
  // and that therefore owes a correction. Upstream's
  // `deferred_spec_decode_corrections` list (`gpu_model_runner.py:1256,
  // 1379-1381`), carrying the same three fields.
  struct DeferredSpecDecodeCorrection {
    std::string req_id;
    int req_index = 0;
    int optimistic_num_accepted = 0;
  };
  std::vector<DeferredSpecDecodeCorrection> deferred_spec_corrections_;
  // ENG-ASYNC-SCHED depth-2 LIFETIME GUARD. sample_tokens_async DEFERS the main
  // queue's completion to the consuming step's get_output() (one step_with_batch_
  // queue call later), so when it returns the previous step's forward / sample /
  // scatter kernels are STILL IN FLIGHT and still reading exec_state_ (device
  // logits, the StepInputs host arrays) and writing input_batch_.last_sampled_
  // tokens. The next execute_model() would otherwise reset exec_state_ and mutate
  // input_batch_ (update_states condense/swap) WHILE those kernels run — a
  // use-after-free / host-heap corruption that only real GPU overlap exposes
  // (serialized runs — the CPU eager backend, compute-sanitizer — never see it).
  // Set true whenever sample_tokens_async leaves main-queue work outstanding;
  // execute_model() drains it before touching any shared state.
  bool async_forward_in_flight_ = false;
  // Dedicated COPY queue for the async sampled-id D2H (async_output_copy_stream,
  // gpu_model_runner.py:711-716,1137-1141). Created lazily on the first async
  // sample so the sync path allocates no extra stream; destroyed in the dtor.
  // id == 0 marks "not created". The copy runs OUTSIDE any CUDA-graph capture.
  vt::Queue async_copy_queue_{};
  // Lazily create + return the async-output copy queue on the runner's device.
  vt::Queue& get_or_create_async_copy_queue();
  // SPEC-DFLASH2 A2-2: the two persistent events of the verify D2H's
  // kCopyQueueEvent route (see ensure_verify_events). Null-handle no-ops on a
  // synchronous backend; destroyed in the dtor.
  vt::Event verify_fork_event_{};
  vt::Event verify_ready_event_{};
  bool verify_events_created_ = false;
  // The kCopyQueueEvent route's D2H destination: one page-locked block holding
  // the accept walk's `sampled` rows followed by its `num_sampled`. Pageable
  // memory would make the copy host-synchronous and the events above decorative
  // (see PinnedGrowStaging). Grown, never shrunk, and freed in the dtor.
  PinnedGrowStaging verify_download_staging_;
  // SPEC-DFLASH2 A2-4 (#3004): the DECODE arm's D2H destination — one
  // page-locked block holding this step's committed sampled ids (int64, so two
  // int32 elements each). Separate from the verify staging above because the two
  // downloads carry different payloads; the fork/ready EVENT pair is shared,
  // which is safe because `sample_tokens_async` takes exactly one of the two
  // routes per step (the verify arm returns before the decode arm can be
  // reached) and each route synchronizes its ready event before it returns.
  // Grown, never shrunk, and freed by `~PinnedGrowStaging` — after the
  // destructor body's drain, which is why that drain was moved to the top.
  PinnedGrowStaging committed_ids_staging_;
  // Persistent pool of the per-step overlap resources (device sampled-id buffer +
  // pinned host buffer + events), so sample_tokens_async does NO per-step
  // cudaMalloc/cudaHostAlloc/cudaEventCreate (each of which device-syncs and
  // would serialize the depth-2 overlap). Lazily created on the first async
  // sample; freed in the dtor. Mirrors torch's caching device/pinned allocators.
  std::unique_ptr<AsyncOutputPool> async_output_pool_;
  AsyncOutputPool& get_or_create_async_output_pool();

  // ─── ENG-ASYNC-SCHED W4: CUDA device-resident async inputs ──────────────────
  // last_sampled_tokens held in a persistent DEVICE buffer, the AUTHORITATIVE copy
  // (the scatter writes it on the main queue and the next step's combine reads it
  // there, so no sampled id crosses to the host on the critical path). Exactly
  // what upstream does on every platform — vllm/v1/worker/gpu/states.py:64 makes
  // last_sampled_tokens a GPU tensor unconditionally.
  //
  // Two reasons to engage it (async_device_mirror, gated VT_ASYNC_DEVICE_MIRROR):
  //  - DISCRETE GPU: REQUIRED. Its host arrays are not device-addressable, so the
  //    W3 host fallback has to synchronize the main stream to read the sampled ids,
  //    which makes the depth-2 async scheduler overlap nothing.
  //  - INTEGRATED GPU (GB10): the W3 kernels CAN operate on the runner's host
  //    arrays in place (pageable host memory is device-addressable), but then
  //    update_states' condense reorders that host array between the scatter that
  //    writes it and the combine that reads it — a read-after-write that pins the
  //    async drain to the TOP of execute_model. Moving last_sampled_tokens onto
  //    this device buffer removes that host RAW (condense records ops replayed
  //    on-queue instead), which lets the drain MOVE past the bulk host prep.
  //
  // Sized once from the batch bound; all zero-initialized. The per-step uploads
  // copy from the caller's pageable host buffers on purpose (see stage_upload).
  //
  // Sized once from the batch bound; all zero-initialized. The per-step uploads
  // copy from the caller's pageable host buffers on purpose (see stage_upload).
  struct AsyncDeviceInputs {
    int32_t* last_sampled = nullptr;     // [max_num_reqs], AUTHORITATIVE
    int32_t* prefill_len = nullptr;      // [max_num_reqs]
    int32_t* query_start_loc = nullptr;  // [max_num_reqs + 1]
    int32_t* seq_lens = nullptr;         // [max_num_reqs]
    int32_t* input_ids = nullptr;        // [max_num_batched_tokens]
    int32_t* ops = nullptr;              // [4 * max_num_reqs] structural replay
    // SPEC-DFLASH2 A2-3 (#2911): the device mirror of
    // `InputBatch::draft_tokens`, [max_num_reqs * draft_stride] ROW-MAJOR per
    // req_state slot. Null and zero-strided without a speculator.
    //
    // OWNERSHIP AND LIFETIME, stated together because they are two claims and
    // this row has been bitten by conflating them. `AsyncDeviceInputs` OWNS every
    // pointer in this struct, including this one. WHEN they may be freed is the
    // separate question: only in `~GPUModelRunner`, which is the sole `Free` site
    // and which DRAINS `queue_` before it frees. That drain is the guarantee,
    // and the sentence that stood here — "the runner and therefore its queue are
    // going away" — was not one: the runner does not own `queue_`, does not
    // destroy it, and a kernel queued against these buffers is still queued when
    // the runner's members are destroyed. Freeing under it is the defect
    // `20d225e54` repaired one level down in
    // `RejectionSamplerDeviceOutput::Release`, and this is the same tree's other
    // half of it. They are
    // allocated once on first use and are never per-step, never block-scoped, and
    // never released by a scope exit. That is deliberate. A2-4 exists to move the
    // verify wait past the propose, and a buffer whose free is tied to a scope
    // the wait can move out of is a use-after-free pre-installed for that wave:
    // the kernel is still queued against it, the read returns garbage drafts, and
    // because verify is lossless the emitted tokens stay correct and NOTHING
    // raises. Moving a wait cannot move a free here, because no wait frees this.
    int32_t* draft_tokens = nullptr;
    // SPEC-DFLASH2 A2-3 REPAIR (#2911): the device mirror of
    // `StepInputs::cu_num_logits`, [max_num_reqs + 1] exclusive prefix sum, the
    // twin of `query_start_loc` above and uploaded beside it every step.
    //
    // IT EXISTS BECAUSE A NULL IS NOT A NEUTRAL DEFAULT HERE. The kernel reads a
    // null `cu_num_logits` as arange(num_reqs + 1), so every row's `num_logits`
    // is 1 and its `num_draft_tokens` is 0. On a decode step that is the right
    // answer. On a VERIFY step it makes `logits_start` == `query_end - 1`, which
    // IS the last draft slot: the committed token is written over the last
    // draft and no drafts are scattered at all. Verify is lossless, so the
    // emitted tokens stay correct and only ACCEPTANCE falls — reason A's class,
    // which no token gate in this tree can see (#1366,
    // `.agents/specs/dflash2-async-spec-sampler.md` §"Reason A"). This arm
    // therefore mirrors the same per-request vector the host arm passes, rather
    // than passing a null that is only correct while the veto stands.
    int32_t* cu_num_logits = nullptr;    // [max_num_reqs + 1]
    int32_t draft_stride = 0;            // == InputBatch::num_speculative_steps
    int64_t input_ids_capacity = 0;      // elements in `input_ids`
    int32_t max_reqs = 0;
  };
  std::unique_ptr<AsyncDeviceInputs> async_device_inputs_;
  // Allocate on first use, or return nullptr when the mirror is not engaged
  // (non-CUDA backend, async not engaged, or VT_ASYNC_DEVICE_MIRROR=0). Caller
  // treats nullptr as "take the pre-W4 host-array path".
  AsyncDeviceInputs* get_or_create_async_device_inputs();
  // True when this runner mirrors the async inputs onto the device: CUDA, async
  // engaged, VT_ASYNC_DEVICE_MIRROR not "0", on a real GPU (integrated OR discrete;
  // not the CPU backend). Memoized. Default ON since the 2026-08-06 correctness
  // flip (ROW-SERVE-ASYNC-LLM P0); VT_ASYNC_DEVICE_MIRROR=0 is the rollback.
  bool async_device_mirror() const;
  mutable int async_device_mirror_cached_ = -1;  // -1 unknown, 0 no, 1 yes
  // VT_ASYNC_EXECUTOR (default OFF): the decode-graph slot double-buffer lever.
  // When on AND the previous step's stashed logits are a NON-owning decode-graph
  // slot view (non_owning_view), the depth-2 drain before the forward is skipped —
  // hazard-A (the eager owning-logits reset UAF) is absent for a slot view, and
  // hazard-C (the persistent decode-graph host inputs being overwritten while the
  // previous replay still reads them) is instead guarded inside the model by a
  // 2-slot parity ring + per-slot reuse event. OFF routes through today's single-
  // slot code with the drain intact (byte-identical). Memoized like the mirror.
  bool async_executor() const;
  mutable int async_executor_cached_ = -1;  // -1 unknown, 0 no, 1 yes
  // Push the recorded InputBatch structural edits (seed/move/swap) to the device
  // mirror in stream order, then clear the log. No-op without a mirror.
  void replay_last_sampled_ops(AsyncDeviceInputs& dev);
  // Upload `src` into `dst` through the pinned staging buffer on the main queue.
  void stage_upload(AsyncDeviceInputs& dev, int32_t* dst, const int32_t* src,
                    int64_t count);
  // Assemble the [num_reqs, vocab] logits the sampler runs on (the three-case
  // device/host gather from the stashed forward result) and apply the grammar
  // bitmask, IN the exact order the sync path uses. Shared by sample_tokens and
  // sample_tokens_async. `sampled_logits` is caller-owned scratch the returned
  // Tensor view may alias (host / VT_GPU_SAMPLE=0 paths); it must outlive the
  // sampler call. Requires exec_state_.num_reqs > 0.
  // #2534: the final-logit dump (VT_DUMP_LOGITS). Called from BOTH sampling
  // paths, because the production default takes the device-resident branch of
  // sample_tokens_async and not sample_tokens.
  void dump_step_logits(const vt::Tensor& logits,
                        const RejectionSamplerOutput* verification = nullptr);

  vt::Tensor assemble_sample_logits(
      const std::optional<GrammarOutput>& grammar_output,
      std::vector<float>& sampled_logits);
  // (step_num_logits is declared in the public section above — the
  // SAMPLE-PROMPT-LOGPROBS route gate names it.)
  // SAMPLE-PROMPT-LOGPROBS (gpu_model_runner.py:5612-5719, called at :3841).
  // Score the extra logit rows the forward produced for prompt positions, fold
  // each request's chunk into its accumulated tensor, and move out the tensors
  // whose prompt finished this step. Returns immediately — no branch taken, no
  // allocation — when the stashed step named no prompt rows, which is every
  // step unless a request asked for prompt logprobs.
  void collect_prompt_logprobs(
      std::map<std::string, LogprobsTensors>& prompt_logprobs_dict);
  // A request's prompt-logprob tensor height: num_prompt_tokens - 1.
  int prompt_logprob_positions(const std::string& req_id) const;
  // Forget in-progress prompt logprobs whose request left the batch (abort).
  void drop_stale_prompt_logprobs();
  // WHERE THE ACCEPT WALK'S OUTPUTS CROSS TO THE HOST (SPEC-DFLASH2 A2-2,
  // #2802). The walk itself is identical either way — same kernel, same queue,
  // same numbers; only the wait moves, and moving it is the point of the wave.
  enum class VerifyDownload {
    // Copy on the MAIN queue and `Synchronize` it. One full compute-stream
    // drain per verify step. This is what `RejectionSampler::forward` has
    // always done and what the synchronous sampler keeps.
    kMainQueueDrain,
    // Fork the COPY queue off the main queue with an event, copy there into
    // page-locked host memory, and block the HOST on that copy event alone.
    // The main queue is never SYNCHRONIZED, and the host still waits in step:
    // the copy waited the fork event, so by the time the host is released the
    // main queue has drained anyway. What this changes is the SHAPE — the copy
    // is no longer a main-stream operation and the wait is on a copy-queue
    // event a later wave can move past the propose. A2-4 and A2-5 are what
    // actually move it (`.agents/specs/dflash2-async-spec-sampler.md`).
    // Mirrors `AsyncOutput` (async_utils.py:29-44), which is how upstream gets
    // its spec-step sampled ids across without stalling the compute stream
    // (model_runner.py:1492-1499).
    kCopyQueueEvent,
  };

  // The SPEC-DECODE VERIFY half (SPEC-REJECTION I3): route the expanded
  // [Σ(1+k_i), vocab] logits through the greedy rejection sampler, write the
  // accepted tokens back, and record num_accepted_tokens. Called by BOTH
  // sample_tokens and sample_tokens_async, each IFF
  // StepRoutesToVerify(exec_state_.step.num_draft_tokens) — the one predicate,
  // which requires a configured SpeculativeConfig. Mirrors
  // gpu/model_runner.py:1129-1140.
  ModelRunnerOutput sample_tokens_with_rejection(
      vt::Tensor& logits,
      VerifyDownload download = VerifyDownload::kMainQueueDrain);

  // The PROPOSE that follows a verify (SPEC-MTP I5d): derive num_sampled /
  // num_rejected from the num_accepted_tokens the verify just wrote and call
  // propose_drafts. Shared by both sampling entry points so the two cannot
  // derive the same two vectors differently. Inert unless spec_on().
  void propose_after_verify(int num_reqs);

  // The PROPOSE that follows a PLAIN DECODE step (SPEC-MTP I5d), lifted out of
  // `sample_tokens` by SPEC-DFLASH2 A2-4 (#2920) for exactly the reason A2-2
  // lifted `propose_after_verify` out: `sample_tokens_async` needs the SAME
  // derivation, and a second hand-written copy of "every generating row sampled
  // one token, rejected none" is a rule with two expressions.
  //
  // Upstream has ONE propose tail for both routes — `sample()` picks the
  // sampler on `input_batch.num_draft_tokens == 0`
  // (gpu/model_runner.py:1129 @ pin 5559679229) and returns that sampler's own
  // `num_sampled` / `num_rejected`, which the single
  // `if self.speculator is not None:` tail at :1524-1547 then proposes from.
  // We have two entry points and two arms, so the pair of derivations is named
  // once each and APPLIED four times rather than written four times.
  //
  // Inert unless spec_on().
  void propose_after_decode(int num_reqs);

  // ─── SPEC-DFLASH2 A2-4 (#3004): THE DEFERRED HALF OF THE CORRECTION ────────
  //
  // Applies every correction `execute_model`'s optimistic pass queued, and is
  // upstream's `correct_spec_decode_token_counts` closure
  // (`gpu_model_runner.py:1529-1555` @ pin 5559679229) with its arithmetic
  // unchanged: `correction = optimistic_num_accepted - num_accepted`, then
  // `num_computed_tokens -= correction`.
  //
  // WHAT W7 DID INSTEAD, and why it had to be replaced. W7 wrote the EXACT value
  // at schedule time — `num_tokens_no_spec - 1`, the newest committed token's
  // position — which is only computable because the previous step's accept walk
  // ran on the host before this step was scheduled. That is precisely the
  // property A2-2 started removing and A2-5 finishes removing, so the rule was
  // correct and self-invalidating. The optimistic value below needs NO accept
  // result: it is `prev_num_computed_tokens + 1 + prev_num_draft_len`, all of it
  // this runner's own bookkeeping.
  //
  // WHERE IT IS CALLED, AND WHERE UPSTREAM CALLS IT. Upstream calls it AFTER the
  // model launch (`:4524-4525`, "now the batch has been launched we can wait for
  // corrections from the previous model forward without breaking async
  // scheduling") and keeps the in-step positions right with a DEVICE correction
  // inside `_prepare_inputs` (`update_num_computed_tokens_for_batch_change`,
  // `:2148-2166`). We compute positions on the host, so this must run before
  // `prepare_inputs` reads `num_computed_tokens_cpu` and it does. Moving it past
  // the launch needs that device kernel and is listed under the row spec's
  // `## Owed` for A2-5; nothing here claims the wait has moved.
  //
  // THE ONE REMAINING HOST DEPENDENCY, named so A2-5 knows what to replace:
  // `prev_valid_sampled_count_`, which `sample_tokens_with_rejection` writes on
  // the host. Upstream reads the same number off the device
  // (`valid_sampled_token_count_gpu`). What A2-4 buys is that the dependency is
  // now one input to one function instead of a value the schedule-time rewrite
  // could not run without.
  void apply_deferred_spec_decode_corrections();

  // Lazily create the two persistent events the kCopyQueueEvent route records
  // (fork: copy-queue-waits-main; ready: D2H completion). Created ONCE and
  // re-recorded each step — never a per-step CreateEvent, which on CUDA
  // device-syncs and would serialize the very overlap this route buys
  // (the AsyncOutputPool comment records that measurement).
  void ensure_verify_events();

  // ── SPEC-MTP I5d verify/propose loop helpers ────────────────────────────────
  // Whether a speculator is configured (nullopt on the production default path,
  // so every helper below is unreachable and the runner is byte-identical).
  bool spec_on() const { return spec_config_.has_value(); }
  // k = resolved num_speculative_tokens (0 when spec off).
  int num_spec() const {
    return spec_config_.has_value()
               ? spec_config_->ResolvedNumSpeculativeTokens()
               : 0;
  }
  // The DECODE-REORDER THRESHOLD this step's batch is split on — upstream's
  // resolved `self.reorder_batch_threshold`, which `_may_reorder_batch` passes
  // into `reorder_batch_to_split_decodes_and_prefills`
  // (gpu_model_runner.py:1126-1130 @ pin 5559679229). Upstream resolves it as
  // the min over the attention groups' builders once the builders exist
  // (`calculate_reorder_batch_threshold`, :7194-7212), and a builder that
  // supports spec-as-decode raises it to `1 + (2 if parallel_drafting else 1) *
  // k` (`_init_reorder_batch_threshold`, backend.py:657-687), which
  // gdn_attn.py:112 requests for every speculative configuration via
  // `supports_spec_as_decode=self.use_spec_decode`. We have one runner-level
  // reorder and no per-group builder registry to take a min over, so the
  // speculative raise IS the resolved value here.
  //
  // `parallel_drafting` is READ, never assumed: the resolvers set it for
  // `dflash` and `dspark` and for nothing else (speculative.py:963-964,
  // mirrored at include/vllm/config/speculative.h ResolveDflash/ResolveDspark),
  // so `mtp`, `ngram` and `draft_model` give `1 + k` and the block drafters
  // give `1 + 2k`.
  //
  // BYTE-IDENTICAL WITHOUT A SPECULATOR: `num_spec()` is 0 there and
  // `SpecAsDecodeReorderThreshold` returns 1, which is the value the reorder's
  // declaration already defaulted to.
  int reorder_batch_threshold() const {
    return static_cast<int>(SpecAsDecodeReorderThreshold(
        num_spec(),
        spec_config_.has_value() && spec_config_->parallel_drafting));
  }
  // Run the k=1 MTP propose after this step's sampling and stash the drafts for
  // take_draft_token_ids (gpu/model_runner.py:1455-1489). Uses the stashed target
  // hidden tap + verify attn metadata; `num_sampled`/`num_rejected` are the
  // per-req accept accounting (1/0 on a plain non-spec step). No-op unless spec_on.
  void propose_drafts(const std::vector<int32_t>& num_sampled,
                      const std::vector<int32_t>& num_rejected);
  // SPEC-DFLASH D5: the DFlash branch of propose_drafts. Combines this step's D1
  // multi-tap into per-token features (CombineAuxFeatures), ACCUMULATES the
  // accepted-prefix features into each request's growing combined-feature context
  // (rolling back the rejected drafts by appending only num_sampled features —
  // the num_rejected rollback, dflash/speculator.py:300-413), then runs the
  // non-autoregressive (1+k) block propose over that context (DflashProposeBlock)
  // and stashes the k drafts/request. Only reachable when use_dflash().
  void propose_drafts_dflash(const std::vector<int32_t>& num_sampled,
                             const std::vector<int32_t>& num_rejected);
  // The shared block-propose body of the DFlash and DSpark branches. Everything
  // through the context accumulation and the block forward is IDENTICAL for the
  // two (DSpark inherits it upstream); the two differ only in `num_query_per_req`
  // / `first_sample_offset` (the anchor-as-first-prediction layout) and in how the
  // resulting block logits become draft ids, which `sample` supplies. `anchors`
  // receives each proposing row's anchor token in the target vocab (DFlash
  // ignores it; DSpark seeds its sequential chain with it).
  void propose_drafts_block(
      const std::vector<int32_t>& num_rejected, const vllm::Qwen3DFlashWeights& backbone,
      const vllm::HfConfig& config, int num_query_per_req,
      const std::function<std::vector<std::vector<int32_t>>(
          const std::vector<float>& block_logits, int num_propose_rows,
          const std::vector<int32_t>& anchors)>& sample);
  // SPEC-DSPARK W5: the DSpark branch of propose_drafts — the shared body above
  // with the anchor-aware layout and the sequential Markov sampler
  // (SampleDsparkBlockDrafts). Only reachable when use_dspark().
  void propose_drafts_dspark(const std::vector<int32_t>& num_sampled,
                             const std::vector<int32_t>& num_rejected);
  // SPEC-NGRAM (ROAD-V1-D3): the draft-FREE branch of propose_drafts. Runs the
  // host-side n-gram matcher (v1/spec_decode/ngram_proposer) over each generating
  // request's own committed context (input_batch_.token_ids_cpu[i,
  // :num_tokens_no_spec[i]]) and stashes the variable-length (0..k) drafts. No
  // draft model, no hidden tap, no draft KV. Only reachable when use_ngram().
  void propose_drafts_ngram(const std::vector<int32_t>& num_sampled,
                            const std::vector<int32_t>& num_rejected);
  // method=="ngram": the draft-free proposer. Distinct from spec_on() (which is
  // any speculator) and use_dflash(); gates the propose routing + suppresses the
  // (unused) hidden-tap capture on the verify forward.
  bool use_ngram() const {
    return spec_config_.has_value() && spec_config_->use_ngram();
  }
  // The drafts produced this step, pending pull by EngineCore::post_step. Empty
  // (nullopt) on the default path.
  //
  // SPEC-DFLASH2 A2-3 (#2911): this is NO LONGER the drafts' residence, only the
  // out-of-band delivery to the scheduler. `take_draft_token_ids` MOVES it out,
  // so it cannot be what the next step's placeholder fill or the combine's
  // scatter read. Their residence is `input_batch_.draft_tokens`, which survives
  // the pull. Upstream has the same pair for the same reason: the device
  // `req_states.draft_tokens` for the workers, and the `DraftTokensHandler`'s
  // host copy for the scheduler (model_runner.py:1548 and :1553-1556).
  std::optional<DraftTokenIds> pending_drafts_;
  // The async placeholder fill's FRESHNESS half (SPEC-DFLASH2 A2-3 repair,
  // #2911), which A2-3 deleted with `pending_drafts_` and did not restate. The
  // rule, and why `InputBatch::draft_tokens` cannot carry it, are on
  // `ProposedDraftLedger` in `prepare_inputs.h`. Written by `set_draft_tokens`
  // and `clear_draft_tokens`, read and consumed by `FillDraftsForStep`, which
  // `execute_model` runs on EVERY async-scheduling step (the consume is a
  // property of the step, not of the fill; see prepare_inputs.h).
  ProposedDraftLedger proposed_drafts_;
  // SPEC-DFLASH2 A2-3 (#2911): the ONE producer seam for both residences.
  // Scatters `drafts` into `input_batch_.draft_tokens` by req_state slot and
  // stashes it in `pending_drafts_` for the out-of-band pull — upstream's
  // `req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens` followed by
  // `draft_tokens_handler.set_draft_tokens(...)`, in one place so a propose arm
  // cannot write one and forget the other. Every `propose_drafts*` arm ends here.
  void set_draft_tokens(DraftTokenIds&& drafts);
  // The no-drafts-this-step counterpart: clears the stash AND zeroes every slot's
  // valid count, so a stale count cannot make the next fill read a row this step
  // did not write. The row CONTENTS are deliberately left alone, exactly as
  // upstream leaves `req_states.draft_tokens` untouched and governs with the
  // valid count instead.
  void clear_draft_tokens();
  // SPEC-MTP I5d acceptance telemetry (spec §5 gate: measured nonzero acceptance).
  // spec_drafts_proposed_ counts draft tokens VERIFIED, spec_drafts_accepted_ the
  // subset the rejection sampler accepted. accepted/proposed is the acceptance
  // rate; total generated / total verify steps is the effective speedup proxy.
  int64_t spec_drafts_proposed_ = 0;
  int64_t spec_drafts_accepted_ = 0;
  // SPEC-MTP-K-GT-1 (#81): the same two counts, split by draft depth. Grown on
  // demand to the deepest draft verified, so they stay EMPTY on the default
  // no-speculation path.
  std::vector<int64_t> spec_drafts_proposed_by_depth_;
  std::vector<int64_t> spec_drafts_accepted_by_depth_;
  // SPEC-MTP-K-GT-1 (#81): the propose-side depth witnesses. All three stay 0
  // unless the MTP proposer runs. The first two carry the RATIO k-1, which the
  // draft list shape cannot produce. The third reads the delivered array, which
  // is the only place padding is visible. See the accessors above for why the
  // per-depth vectors serve neither purpose.
  int64_t spec_mtp_propose_calls_ = 0;
  int64_t spec_mtp_draft_decode_forwards_ = 0;
  int64_t spec_mtp_proposals_with_varied_drafts_ = 0;
  // ── SPEC-DFLASH D5 (DF-ENGINE-INTEGRATION) ──────────────────────────────────
  // The separately-loaded DFlash draft (borrows owned by LoadedEngine; null
  // unless method=="dflash"). use_dflash() gates the aux-tap capture + the DFlash
  // propose branch. dflash_tap_layer_ids_ = the draft's target_layer_ids (the D1
  // multi-tap capture-after indices), resolved once from the draft config.
  const vllm::Qwen3DFlashWeights* dflash_weights_ = nullptr;
  const vllm::HfConfig* dflash_config_ = nullptr;
  int dflash_k_ = 0;
  std::vector<int32_t> dflash_tap_layer_ids_;
  bool use_dflash() const { return dflash_weights_ != nullptr; }
  // ── SPEC-DSPARK W5 ──────────────────────────────────────────────────────────
  // The separately-loaded DSpark draft (borrow owned by LoadedEngine; null unless
  // method=="dspark"). When set, dflash_weights_ points at `&dspark_weights_->
  // backbone`, so every piece of shared machinery — the aux multi-tap capture,
  // the per-request device KV store, the context-aware block forward — runs
  // UNCHANGED, and use_dspark() only redirects the propose tail to the sequential
  // Markov sampler and the anchor-aware block layout.
  const vllm::Qwen3DSparkWeights* dspark_weights_ = nullptr;
  bool dspark_sample_from_anchor_ = true;
  bool use_dspark() const { return dspark_weights_ != nullptr; }
  // Per-request PERSISTENT context KV store (D9 persistent paged draft-KV — the
  // perf form of vLLM's incrementally-written draft KV cache). One entry holds
  // that request's per-layer bf16 context K/V (K normed+RoPE'd, V raw) for its
  // committed positions 0..ctx_len-1. Each verify step projects ONLY the
  // newly-accepted rows (AppendContextKVDeviceRows) and APPENDS them, instead of
  // re-projecting the whole growing context (the D5/D7 O(context^2) recompute).
  // Bit-identical to the recompute by per-row projection independence; rejected
  // drafts' rows are never appended (rollback = don't-append).
  //
  // D11 A-wire: the store is the DEVICE-RESIDENT append-only draft-KV store
  // (DflashDeviceKVStore, opaque). AppendContextKVDevice keeps the projected bf16
  // K/V on-device (no D<->H round trip) and ForwardBlockLogitsWithDeviceKV runs
  // the block forward straight off it — bit-identical to the D9 host path, and
  // the capture-ready substrate for Parts B/C. shared_ptr-to-incomplete is safe:
  // MakeDeviceKVStore constructs the control block (with its deleter) in
  // qwen3_dflash.cpp.
  //
  // KEYED BY REQUEST ID, NOT BY BATCH ROW (#2008). These three fields were three
  // arrays indexed by the runner's condensed-dense batch row, with a fourth
  // recording each row's occupant so a reused slot could reset. A row index is
  // not stable for a request's lifetime here: `InputBatch::condense` slides a
  // live request down into the hole a finished neighbour left, and `swap_states`
  // exchanges two live rows. Both permute every per-slot array they own —
  // including the block-table rows — and neither knows these exist, because they
  // live on the runner rather than in `InputBatch`. So the survivor of a
  // completed pair met the departed request's bookkeeping, the occupant test
  // read a changed id, the store was reset to EMPTY under a request still using
  // it, and `propose_drafts_block`'s position invariant then refused. #2008
  // measured what that costs: DFlash2 served c=1 at 24.70 out tok/s and VOIDed
  // at c=2 with ok=1, after which every later request on that server came back
  // `[request submitted to a stopped AsyncLLM]`.
  //
  // Upstream keys the same state to the request on both of its paths. Its V2
  // runner, where DFlash2 lives, has no `condense` at all — a finished request's
  // slot returns to a free list and stays that request's for its lifetime
  // (`vllm/v1/worker/gpu/states.py:29,100,132` @ `b389ac2946`) — and every
  // cross-step speculator tensor is indexed through
  // `req_state_idx = idx_mapping[req_idx]`
  // (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:536`). Its legacy V1
  // runner does condense, and there the draft's block-table row moves with the
  // request (`vllm/v1/worker/gpu_input_batch.py:786` ->
  // `vllm/v1/worker/block_table.py:367-373`). A request id is the key that
  // survives any reordering of OUR batch, so every permutation the batch can
  // perform is a no-op here — which is the property that makes upstream's
  // speculator indifferent to row order in the first place.
  //
  // Entries are pruned each propose against `InputBatch`'s own membership, so a
  // finished or preempted request releases its device store on the step after it
  // leaves the batch.
  struct DflashReqCtx {
    std::shared_ptr<vllm::DflashDeviceKVStore> store;
    // Committed context length L. Kept in lockstep with the store's own num_ctx,
    // and asserted against it every propose (SPEC-DFLASH2 W8, #1838).
    int32_t ctx_len = 0;
    // #1919: this request no longer fits the store and runs on the target alone.
    // STICKY for the request's lifetime, and that is forced rather than chosen:
    // `propose_drafts_block` keeps `ctx_len` in lockstep with the store's
    // `num_ctx` and asserts both against the target's committed positions, so a
    // step that declines to append breaks that lockstep and every later step for
    // the same request must decline too. Upstream's own skip is monotone in the
    // same way: its `num_tokens >= max_model_len` condition only ever becomes
    // true (`vllm/v1/spec_decode/ngram_proposer.py:156-159`). Being a property
    // of the REQUEST — which the row-indexed form could only approximate, and
    // its comment already claimed — it now simply ends with the request.
    bool disabled = false;
  };
  std::unordered_map<std::string, DflashReqCtx> dflash_ctx_;
  // #1919: the store's resolved capacity, taken ONCE at set_dflash_draft from
  // this engine's own max_model_len. The "no longer fits" flag it pairs with is
  // `DflashReqCtx::disabled` above.
  vllm::Qwen3DFlashModel::DflashCtxStoreSizing dflash_ctx_sizing_;
  // Draft KV cache (`fa_draft` group) backing storage, owned by the runner and
  // allocated in initialize_kv_cache when spec is on. draft_attn_kv_ (declared
  // above) views into these buffers. Empty on the default path.
  std::vector<std::unique_ptr<CacheBuffer>> draft_attn_buf_;
  std::vector<std::unique_ptr<CacheBuffer>> full_attn_buf_;
  // GDN convolution and recurrent caches have independent dtypes. This mirrors
  // MambaStateDtypeCalculator::_mamba_state_dtype: mamba_cache_dtype="auto"
  // makes conv state use model dtype (BF16 for the gate models), while Qwen3.5's
  // verify hook copies config.json mamba_ssm_dtype into the temporal/SSM cache
  // dtype (FP32 for both gate checkpoints). Kernels accumulate in FP32 and store
  // each cache through its own declared dtype. Allocation consumes the
  // MambaSpec directly on every backend; raw storage is sized independently and
  // all-zero bytes represent +0.0 in each supported floating type.
  vt::DType gdn_conv_cache_dtype_ = vt::DType::kF32;
  vt::DType gdn_ssm_cache_dtype_ = vt::DType::kF32;
  // ENG-RECURRENT-MULTISTATE (#2131): outer index is the recurrent LAYER, in
  // layer order; inner index is the STATE, in `MambaSpec::shapes` order. This
  // replaces the `ssm_buf_` / `conv_buf_` pair, which could not hold a third
  // state and so made the two-state assumption a property of the runner's
  // storage rather than of any model. Upstream's storage is one raw page per
  // layer that the layer itself slices into N states
  // (`vllm/model_executor/layers/mamba/abstract.py:29-43` @ pin 5559679229);
  // here the split is one buffer per state and the ORDER carries the
  // correspondence.
  std::vector<std::vector<std::unique_ptr<CacheBuffer>>> recurrent_state_buf_;
  std::vector<PagedKvCache> attn_kv_;
  std::vector<GdnStateCache> gdn_state_;
  // Per-layer attention backend names, parallel to attn_kv_ (see accessor).
  // A dense entry is never empty; an MLA entry may be (op-driven execution).
  std::vector<std::string> attn_backend_names_;

  // ── KV-EXTERNAL-CACHE (LMCache) worker-side store/load ──────────────────────
  // Non-owning; null (default) = inert. See set_kv_connector.
  kv_offload::KVConnector* kv_connector_ = nullptr;
  // Per-request count of prompt blocks already STORED to the external cache, so
  // a multi-step (chunked) prefill stores each full block exactly once and a
  // decode step re-stores nothing. Only touched when kv_connector_ != nullptr.
  std::unordered_map<std::string, int> connector_stored_blocks_;
  // Drain + apply the connector's recorded external-prefix loads into the KV
  // cache blocks BEFORE the forward reads them (load-before-compute). No-op
  // unless kv_connector_ is a worker-capable connector with pending loads.
  void ConnectorLoadExternalKv();
  // After the forward has written this step's KV, STORE every newly-complete
  // prompt block of each running request to the external cache (offload-prompt-
  // only). No-op unless kv_connector_ is a worker-capable connector.
  void ConnectorStorePromptKv(const SchedulerOutput& scheduler_output);

  // SAMPLE-PROMPT-LOGPROBS: the partially-filled prompt-logprob tensor of every
  // request whose prompt is still being consumed, keyed by req_id. A chunked
  // prefill fills it slice by slice and the final chunk moves it out; upstream
  // hangs the same tensor off the per-request state object as
  // `request.in_progress_prompt_logprobs_cpu` (gpu_model_runner.py:5645-5651,
  // cleared at :5712), which we have no equivalent of on the runner. Empty
  // unless a request asked for prompt logprobs.
  std::map<std::string, LogprobsTensors> in_progress_prompt_logprobs_;

  // Stashed forward result between execute_model and sample_tokens (upstream
  // ExecuteModelState — hidden_states + input_batch handoff, here the full
  // logits + the dense-order step). num_reqs == 0 marks a 0-token flush step.
  struct ExecuteModelState {
    // The forward result. DEFAULT: a DEVICE-resident [num_reqs, vocab] logits
    // buffer (ForwardLogits::device_*) fed straight to the sampler — NO per-step
    // full-logits D2H. On the VT_LOGITS_GATHER=0 opt-out it carries host logits
    // ([num_actual_tokens, vocab]) and sample_tokens re-gathers on host as before.
    ForwardLogits logits;
    int num_actual_tokens = 0;
    int num_reqs = 0;
    StepInputs step;
    CommonAttentionMetadata attn_meta;
    GDNAttentionMetadata gdn_meta;
    std::vector<std::string> req_ids;  // dense order (== input_batch order)
    // Per-request expanded verify row counts, retained until sampling. The
    // grammar mask uses sizes only; async draft values may be filled later.
    std::map<std::string, std::vector<int32_t>> scheduled_spec_decode_tokens;
    // SPEC-MTP I5d: the target's post-final-norm [T,H] hidden tap captured this
    // step (ModelForwardInput::hidden_tap output), consumed by propose_drafts to
    // run the MTP drafter. Empty (null storage) unless spec is on.
    Qwen3_5MTPHiddenStates spec_hidden;
    // Borrowed shifted visual slices retain their encoder parents until the
    // post-sampling draft merge/proposal has completed. Gather before request
    // progress advances, merge after shift/splice knows the sampled token.
    std::optional<MmGather> spec_mm_gather;
    std::vector<int32_t> spec_mm_positions;
    // SPEC-DFLASH D5: the target's D1 MULTI-tap captured this step
    // (ModelForwardInput::aux_tap output) — the residual stream at the draft's
    // target_layer_ids as [T, H×taps] bf16, consumed by propose_drafts_dflash.
    // Empty (null storage) unless use_dflash(); mutually exclusive with
    // spec_hidden (MTP single tap).
    Qwen3_5AuxTaps spec_aux;
    // discard_request_mask (gpu_model_runner.py:2048): per dense batch row, 1 iff
    // the request is still consuming its known prefill tokens this step
    // (optimistic seq_len < num_tokens) and so must NOT sample — its sampled
    // token is cleared to empty before returning.
    std::vector<uint8_t> discard;
  } exec_state_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_RUNNER_H_

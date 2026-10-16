// Ported from: vllm/v1/worker/gpu_input_batch.py @ e24d1b24
//
// Scope (M1.5 Task 2): the PERSISTENT per-slot input batch the model runner
// keeps alive across steps — `CachedRequestState` (the worker's cached copy of
// a scheduled request) and `InputBatch` (the num_reqs-major arrays mutated in
// place as requests are admitted / finished / condensed). Two ported types:
//   * CachedRequestState: req_id, prompt_token_ids, sampling_params, the
//     per-KV-cache-group block_ids, num_computed_tokens, output_token_ids, and
//     the num_prompt_tokens / num_tokens / get_token_id helpers.
//   * InputBatch: the per-slot arrays (req_ids indexed by slot, token_ids_cpu,
//     num_tokens_no_spec, num_prompt_tokens, num_computed_tokens_cpu, the
//     MultiGroupBlockTable, and the per-slot sampling metadata the M1.7 sampler
//     needs — temperature / top_p / top_k / {frequency,presence,repetition}
//     penalties) + the req_id_to_index map, add_request / remove_request /
//     condense (swap-remove densification), and num_reqs.
// Behavioral only: no CUDA, no model. The "tensors" are plain host arrays.
//
// ─── V1 ALGORITHM / MRV2 CONTRACT (recorded) ────────────────────────────────
// The source of truth is the V1 GPU worker file `gpu_input_batch.py` (NOT the
// MRV2 `gpu/input_batch.py`, which at e24d1b24 is a TRANSIENT per-step object
// that holds NO block table; its persistent state is split into `req_states` +
// staged `BlockTables` on the runner — deferred to M2 with the `vt` device).
// The persistent-InputBatch-holds-the-MultiGroupBlockTable design ported here
// is the identical bookkeeping minus the device wrapper. But it is DRIVEN by
// the MRV2 scheduler-output CONTRACT, not the MRV1 runner's admission path:
//   * new requests carry `prefill_token_ids` (= all_token_ids = prompt+output
//     at schedule time; NewRequestData::prefill_token_ids, M1.4). The token-id
//     seed of `token_ids_cpu` is prompt_token_ids then output_token_ids, which
//     by construction equals prefill_token_ids — see from_new_request.
//   * resumed-from-preemption requests are folded in AS NEW (they arrive in
//     scheduled_new_reqs), so there is a single admission path: add_request.
//   * cached diffs are `num_computed_tokens` + `new_block_ids` (Task 3 applies
//     them; add_request seeds the initial state).
// The V1 runner's MRV1-shape admission (`resumed_from_preemption` /
// `resumed_req_ids` / per-req `all_token_ids` reconstruction) is DEAD under our
// MRV2 scheduler and is NOT ported.
//
// The C++ file location follows the repo's `worker/gpu/` layout (as with Task
// 1's block_table); the `Ported from` ref above cites the ACTUAL upstream file.
//
// ─── HOST-ARRAY-FOR-DEVICE-TENSOR DEVIATION (recorded) ──────────────────────
// Upstream keeps paired cpu/device tensors (token_ids_cpu numpy view over a
// cpu tensor; temperature/top_p/... as device tensors with pinned cpu staging
// buffers copied via copy_slice in _make_sampling_metadata). At T0 there is no
// device: every array is a host std::vector. The per-slot "*_cpu" arrays are
// what add_request / condense write; the device-side sampling tensors + the
// copy_slice staging are the sampler's (M1.7) concern and are not materialized
// here. The boolean predicates (all_greedy / no_top_p / ...) the sampler keys
// on ARE ported so M1.7 can build SamplingMetadata from these arrays.
//
// ─── DEFERRED upstream slot state (marked; T0 never populates) ──────────────
//   - LoRA: request_lora_mapping / lora_id_to_request_ids / lora_id_to_lora_
//     request (+ make_lora_inputs).
//   - Spec decode: spec_token_ids (per-slot draft list), num_accepted_tokens
//     (per-slot, seeded to 1) and update_req_spec_token_ids are LANDED for
//     SPEC-MTP (I2 ABI freeze). They are inert on the default path: with no
//     speculator configured the runner never calls update_req_spec_token_ids, so
//     spec_token_ids stays empty and num_accepted_tokens stays 1 everywhere.
//   - Multimodal: mm_features (on CachedRequestState) and the M-RoPE pair are
//     LANDED for ENG-MM-INPUT-PIPELINE P2 (#2379); see the field notes. All
//     three are per-REQUEST state on CachedRequestState, not per-slot InputBatch
//     arrays, so nothing in the slot machinery (add/remove/condense/swap) moves.
//     STILL DEFERRED: req_prompt_embeds / prompt_is_token_ids / is_token_ids.
//   - Structured output / logits processors: batch_update_builder's added/moved
//     tracking, logitsprocs, allowed_token_ids_mask, bad_words_token_ids,
//     thinking-budget. Only the REMOVED-index tracking (which drives add-hole
//     fill + condense) is ported — as a minimal RemovedTracker.
//   - Logprobs: num_logprobs / logprob_token_ids.
//   - Pooling: pooling_params / pooling_states / is_pooling_model branch.
//   - Async scheduling: prev_sampled_token_ids / update_async_* .
//   - SamplingMetadata construction (_make_sampling_metadata / refresh_metadata)
//     is deferred to M1.7 — the SamplingMetadata type is not yet landed; the
//     per-slot arrays + boolean predicates it consumes ARE provided here.
#ifndef VLLM_V1_WORKER_GPU_INPUT_BATCH_H_
#define VLLM_V1_WORKER_GPU_INPUT_BATCH_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/multimodal/inputs.h"  // MultiModalFeatureSpec (P2, #2379)
#include "vllm/sampling_params.h"
#include "vllm/v1/core/sched/output.h"  // NewRequestData (from_new_request)
#include "vllm/v1/sample/metadata.h"    // SamplingMetadata (make_sampling_metadata)
#include "vllm/v1/worker/gpu/block_table.h"

namespace vllm::v1 {

// The worker's cached, persistent copy of a scheduled request.
// (Upstream: vllm/v1/worker/gpu_input_batch.py CachedRequestState — T0 field
// subset; generator / lora / prompt_embeds / pooling DEFERRED. mm_features and
// the M-RoPE pair are LANDED, P2 #2379.)
struct CachedRequestState {
  std::string req_id;
  // Upstream list[int] | None; always the token path at T0.
  std::vector<int32_t> prompt_token_ids;
  // Upstream SamplingParams | None; always present for generation (pooling
  // DEFERRED). Stored by value, already PostInit'd by the frontend.
  SamplingParams sampling_params;
  // Per-KV-cache-group block ids (upstream tuple[list[int], ...]).
  std::vector<std::vector<int>> block_ids;
  int num_computed_tokens = 0;
  std::vector<int32_t> output_token_ids;

  // num_prompt_tokens (upstream __post_init__ via
  // length_from_prompt_token_ids_or_embeds); no prompt_embeds at T0.
  int num_prompt_tokens = 0;

  // ── ENG-MM-INPUT-PIPELINE P2 (#2379) ──────────────────────────────────────
  // mm_features: the worker's copy of the request's multimodal items
  // (gpu_model_runner.py:1293 `mm_features=new_req_data.mm_features`, grep -c ==
  // 1). The encoder step runs the tower over the items the scheduler named, the
  // gather slices their outputs into this step's token window, and the M-RoPE
  // init reads their spans. EMPTY on every text request, and the runner's whole
  // mm arm is predicated on some request in the step having a non-empty one.
  std::vector<multimodal::MultiModalFeatureSpec> mm_features;

  // mrope_positions: [3 * num_prompt_tokens] row-major, computed ONCE at
  // admission (gpu_model_runner.py:1654 `_init_mrope_positions`, grep -c == 1)
  // because it needs the whole prompt. EMPTY unless the model declares an
  // M-RoPE hook AND this request carries mm items.
  std::vector<int32_t> mrope_positions;
  // mrope_position_delta: THE cross-step carrier (gpu_model_runner.py:2786,
  // grep -c == 1). A completion position is `context_len + i + delta` on all
  // three axes, so no completion position is ever stored — which is why a
  // request that decodes for a thousand steps still costs one int here.
  int64_t mrope_position_delta = 0;

  std::vector<std::string> prompt_block_hashes;

  // Build a CachedRequestState from the MRV2 NewRequestData contract. The seed
  // is prefill_token_ids (= all_token_ids): prompt_token_ids is the prompt,
  // output_token_ids is the tail of prefill_token_ids beyond the prompt. This
  // is the MRV2-contract adaptation of the V1 algorithm (the V1 runner instead
  // reconstructed all_token_ids per-req; that path is dead here).
  static CachedRequestState from_new_request(const NewRequestData& new_req);

  // num_tokens: num_prompt_tokens + len(output_token_ids).
  int num_tokens() const {
    return num_prompt_tokens + static_cast<int>(output_token_ids.size());
  }

  // get_token_id(idx): prompt then output; -1 past the end (upstream).
  int get_token_id(int idx) const;

  // Recompute num_prompt_tokens from prompt_token_ids (call after populating
  // fields when not using from_new_request). Mirrors __post_init__.
  void finalize() { num_prompt_tokens = static_cast<int>(prompt_token_ids.size()); }
};

// The persistent per-slot input batch. (Upstream: vllm/v1/worker/
// gpu_input_batch.py InputBatch — T0 subset.)
class InputBatch {
 public:
  // Upstream positional order: max_num_reqs, max_model_len,
  // max_num_batched_tokens, (device dropped), vocab_size, block_sizes,
  // kernel_block_sizes, (trailing knobs DEFERRED). block_sizes /
  // kernel_block_sizes are per KV cache group.
  // num_speculative_steps sizes the per-slot draft buffer's row (upstream's
  // `RequestStates.__init__(..., num_speculative_steps)`, states.py:34,71-77).
  // 0 — the default and every non-speculative runner — makes the buffer
  // zero-width, which is upstream's `torch.zeros(max_num_reqs, 0)` exactly.
  // max_num_blocks_per_req / slot_mapping_modes are upstream's InputBatch
  // arguments of the same names (gpu_model_runner.py:7375-7394 @ e126687a9a),
  // forwarded to MultiGroupBlockTable. nullopt keeps its derivation
  // (cdiv(max_model_len, block_size), every group kTokenToKvSlot). The runner
  // passes block_table_geometry's values (FIX-BLOCK-TABLE-ROW-WIDTH).
  InputBatch(int max_num_reqs, int max_model_len, int max_num_batched_tokens,
             int vocab_size, std::vector<int> block_sizes,
             std::vector<int> kernel_block_sizes,
             int num_speculative_steps = 0,
             std::optional<std::vector<int>> max_num_blocks_per_req = std::nullopt,
             std::optional<std::vector<SlotMappingMode>> slot_mapping_modes =
                 std::nullopt);

  // add_request: place `request` into a slot (a freed hole if one exists, else
  // append at num_reqs), fill the per-slot arrays from it, add the block-table
  // rows, and record the sampling metadata. Returns the assigned slot index.
  int add_request(const CachedRequestState& request);

  // remove_request: mark the slot for `req_id` empty and return its index, or
  // nullopt if unknown. MUST be followed by condense() before the next read of
  // the dense [0, num_reqs) range (upstream contract).
  std::optional<int> remove_request(const std::string& req_id);

  // condense: slide active requests down into the freed holes so [0, num_reqs)
  // is dense again, fixing req_id_to_index and swapping/moving every per-slot
  // array + the block-table rows. (Upstream InputBatch.condense.)
  void condense();

  // swap_states(i1, i2): swap the two slots' per-slot state in place (the token
  // prefix, num_* scalars, block-table rows, sampling params, and the per-req
  // seed), fixing req_id_to_index. Mirrors gpu_input_batch.py::swap_states
  // (@ e24d1b24) — the primitive the runner's decode-first reorder
  // (reorder_batch_to_split_decodes_and_prefills, M1.8 Task 4) drives. Only the
  // active token prefix (max active count of the two rows) is copied, exactly
  // as upstream (copying full max_model_len rows is unnecessary during
  // reordering). DEFERRED (T0 always empty/absent): is_token_ids /
  // req_prompt_embeds / request_lora_mapping / num_accepted_tokens /
  // bad_words_token_ids / allowed_token_ids_mask + the batch_update_builder.moved
  // logitsprocs tracking. The membership *_reqs sets are keyed by req_id (not
  // slot), so they need no swap.
  void swap_states(int i1, int i2);

  // num_reqs (property): len(req_id_to_index).
  int num_reqs() const { return static_cast<int>(req_id_to_index.size()); }

  // make_sampling_metadata: return the SamplingMetadata for the current dense
  // [0, num_reqs) prefix. Port of gpu_input_batch.py::_make_sampling_metadata
  // (build_sampling_metadata) + refresh_metadata's batch-change gate
  // (gpu_input_batch.py:812-830): upstream caches self.sampling_metadata and
  // rebuilds ONLY when the batch changes (add / remove / condense / swap). We
  // mirror that here — the result is cached and rebuilt only when
  // sampling_metadata_dirty_ is set by a batch mutation (rescan §6 item e). The
  // one deviation: because our port COPIES output_token_ids (upstream holds a
  // live reference), we also rebuild every step when penalties are active
  // (!no_penalties()), so the decode-advancing output tokens stay fresh; the
  // greedy / no-penalties gate workload gets the full caching win bit-identically
  // (the metadata then depends only on the request set + static sampling params).
  const SamplingMetadata& make_sampling_metadata() const;

  // Sampling predicates the M1.7 sampler keys on (upstream properties).
  bool all_greedy() const { return random_reqs.empty(); }
  bool all_random() const { return greedy_reqs.empty(); }
  bool no_top_p() const { return top_p_reqs.empty(); }
  bool no_top_k() const { return top_k_reqs.empty(); }
  bool no_penalties() const {
    return presence_penalties_reqs.empty() &&
           frequency_penalties_reqs.empty() &&
           repetition_penalties_reqs.empty();
  }
  // ROAD-V1-C7 predicates (gpu_input_batch.py no_allowed_token_ids / min-p).
  bool no_min_p() const { return min_p_reqs.empty(); }
  bool no_allowed_token_ids() const { return has_allowed_token_ids.empty(); }
  // max_num_logprobs (gpu_input_batch.py:1150-1151): max requested count, or
  // unset when no request asked for logprobs. Our -1 ("all") sentinel dominates.
  std::optional<int> max_num_logprobs() const;

  // Convenience token-id read (flat row-major token_ids_cpu; row stride is
  // max_model_len). Mirrors the numpy token_ids_cpu[req, col] access.
  int32_t token_id(int req_index, int col) const {
    return token_ids_cpu[static_cast<size_t>(req_index) * max_model_len + col];
  }

  // ─── Public state (mirrors upstream's accessible attributes) ──────────────
  int max_num_reqs;
  int max_model_len;
  int max_num_batched_tokens;
  int vocab_size;

  // req_ids indexed by slot; nullopt only transiently during remove/condense.
  std::vector<std::optional<std::string>> req_ids;
  std::unordered_map<std::string, int> req_id_to_index;

  // Per-slot token buffer, flat [max_num_reqs, max_model_len] row-major.
  std::vector<int32_t> token_ids_cpu;
  // Per-slot scalar arrays [max_num_reqs].
  std::vector<int32_t> num_tokens_no_spec;
  std::vector<int32_t> num_prompt_tokens;
  std::vector<int32_t> num_computed_tokens_cpu;

  // ─── Async-scheduling (ENG-ASYNC-SCHED W3) per-slot state [max_num_reqs] ────
  // Mirror of RequestState.last_sampled_tokens / prefill_len
  // (vllm/v1/worker/gpu/states.py:64,105-122). Populated only on the async runner
  // path; the synchronous path never reads them (production default).
  //
  // last_sampled_tokens[slot]: the last token the sampler produced for this
  // request, kept per req_state so the next step can build its decode input id
  // WITHOUT the sampled-id host round-trip (combine_sampled_and_draft_tokens).
  // On CUDA this becomes the GPU-resident RequestState.last_sampled_tokens; on
  // CPU it is a host array (the runner leaf is device-neutral). sample_tokens
  // writes it each step; add_request seeds it for a resumed/PD-disagg request
  // (0 < num_computed <= prefill_len) so its first decode reads the right id.
  std::vector<int32_t> last_sampled_tokens;
  // ENG-ASYNC-SCHED W4: the ordered log of STRUCTURAL edits made to
  // last_sampled_tokens since the runner last drained it — the seed at
  // add_request, the row move at condense, the row swap at swap_states.
  //
  // On a DISCRETE GPU the authoritative copy of last_sampled_tokens is a device
  // buffer (upstream keeps it device-resident on every platform,
  // states.py:64), so the host no longer holds the VALUES those edits move
  // around. It does know the INDICES, which is all a device replay needs. The
  // runner drains this each step and applies it in stream order before the
  // combine reads the buffer; see vt::cuda::LaunchApplyLastSampledOps.
  //
  // Upstream needs no equivalent because it never condenses: states.py:132
  // returns a finished request's slot to a free list and the slot index is
  // stable for the request's lifetime. This log is the price of our condensed
  // dense batch, not a deviation in what the state MEANS.
  //
  // Inert unless the async runner path is engaged on a device that mirrors the
  // array: the ops are recorded unconditionally (a few ints per admitted or
  // finished request, off the per-token path) and simply discarded otherwise.
  struct LastSampledOp {
    enum Kind : int32_t { kSeed = 0, kMove = 1, kSwap = 2 };
    int32_t kind = kSeed;
    int32_t a = 0;      // seed/move destination, or the first swapped slot
    int32_t b = 0;      // move source, or the second swapped slot
    int32_t value = 0;  // seed value; unused by move/swap
  };
  std::vector<LastSampledOp> last_sampled_ops;
  // prefill_len[slot]: the number of tokens KNOWN at admission (prompt + any
  // pre-existing output = num_tokens() at add_request), fixed for the request's
  // lifetime. combine gates on seq_len > prefill_len to tell a decode row (splice
  // the sampled token) from a prefill/chunked-prefill row (keep the prompt).
  std::vector<int32_t> prefill_len;

  // ─── SPEC-DFLASH2 A2-3 (#2911): the per-req_state DRAFT BUFFER ────────────
  // Mirror of `RequestStates.draft_tokens`
  // (vllm/v1/worker/gpu/states.py:71-77, zeroed at :113) @ pin
  // 5559679229bc961848b121ccdeaa8fa5d79bec98.
  //
  // `num_valid_draft_tokens` BESIDE IT HAS NO UPSTREAM COUNTERPART, and the
  // anchor this comment used to give for it was wrong. A2-3 cited
  // `gpu_model_runner.py:883-895`; at the pin those lines are the LEGACY
  // runner's n-gram-GPU async D2H buffer set (`_num_valid_draft_tokens`,
  // `_num_valid_draft_tokens_cpu`, its event and its copy stream), allocated
  // only when `speculative_config.use_ngram_gpu()`, and they are not a
  // per-req_state count paralleling `states.draft_tokens` at all. The NEW
  // runner keeps no such count: it takes each row's draft length from the
  // scheduler's own `scheduled_spec_decode_tokens`
  // (`vllm/v1/worker/gpu/model_runner.py:941-949`), which we cannot, because
  // under async scheduling that list is `-1` placeholders and the count of what
  // this runner actually proposed is exactly what the fill needs. It is our
  // construct, named after upstream's, and it is recorded as a divergence here
  // rather than dressed as a mirror.
  //
  // WHAT IT IS FOR. Two consumers need to agree on what this runner drafted:
  // the async placeholder fill in `execute_model`, and
  // `combine_sampled_and_draft_tokens`'s draft scatter (A2-1). Before A2-3 the
  // fill read `pending_drafts_` — host, ragged, req_id-keyed — and the scatter
  // had no buffer at all, so its call sites passed an empty one and refused any
  // step that scheduled drafts. This is the single residence both now read, with
  // `propose_drafts` as the one producer, which is upstream's shape.
  //
  // WHO OWNS IT AND FOR HOW LONG. The `InputBatch` owns it, and the InputBatch
  // is a by-value member of `GPUModelRunner`: it is allocated once at runner
  // construction and released only when the runner is destroyed. It is NEVER a
  // per-step allocation and never block-scoped. That distinction is deliberate
  // and it is the reason this wave does not repeat the hazard the row has now
  // been bitten by twice: a buffer a kernel still has queued against it must not
  // be freed by a scope exit, and a wave that MOVES A WAIT later (A2-4 moves the
  // verify wait past the propose) must not thereby move a free. Nothing here can
  // be freed by moving a wait, because nothing frees it until the runner dies.
  // Its device mirror (`GPUModelRunner::AsyncDeviceInputs::draft_tokens`) has the
  // same lifetime and the same reason, stated again at its own declaration.
  //
  // SIZED BY THE req_state POOL, NOT BY `num_reqs`, and that is a correctness
  // requirement rather than a convenience. The scatter indexes
  // `draft_tokens[req_state_idx * stride + b]`; the CUDA counterpart
  // (`src/vt/cuda/cuda_combine_tokens.cu`) has NOTHING bounding that read. A
  // buffer sized by this step's request count lets a high req_state slot index
  // past the allocation with every host check satisfied, and the quiet outcome —
  // garbage in the draft slots — costs acceptance and nothing else, because
  // speculative decoding is lossless. `.agents/specs/dflash2-async-spec-sampler.md`
  // names this and requires exactly this sizing.
  //
  // draft_tokens: [max_num_reqs * num_speculative_steps] ROW-MAJOR per req_state
  // slot, the flattened form of upstream's 2-D tensor. Empty when
  // num_speculative_steps is 0.
  std::vector<int32_t> draft_tokens;
  // num_valid_draft_tokens[slot]: how many of that row's entries this runner
  // actually proposed for the request in that slot. The row is padded to the
  // stride, so the count is what tells a 2-draft row in a k=3 buffer from a
  // 3-draft one. [max_num_reqs].
  std::vector<int32_t> num_valid_draft_tokens;
  // The draft buffer's row stride == upstream's `draft_tokens.stride(0)`
  // (input_batch.py:381,396). The speculator's MAX draft length, so a shorter
  // row is padded rather than short.
  int num_speculative_steps = 0;

  // The per-request KV-cache block table (one BlockTable per group).
  MultiGroupBlockTable block_table;

  // Per-slot sampling metadata [max_num_reqs] (M1.7 sampler input).
  std::vector<float> temperature_cpu;
  std::vector<float> top_p_cpu;
  std::vector<int32_t> top_k_cpu;
  std::vector<float> frequency_penalties_cpu;
  std::vector<float> presence_penalties_cpu;
  std::vector<float> repetition_penalties_cpu;

  // ─── ROAD-V1-C7 SAMPLE-CORE / SAMPLE-LOGIT-FILTERS per-slot controls ───────
  // min_p (MinPLogitsProcessor); [max_num_reqs], 0 disables the row. Moved with
  // the row on condense/swap like the penalty arrays.
  std::vector<float> min_p_cpu;
  // req_index -> MinTokensState (min_tokens floor + all_stop_token_ids to mask
  // while output_len < min_tokens; gpu_input_batch.py MinTokensLogitsProcessor).
  std::map<int, MinTokensState> min_tokens;
  // req_index -> (token_id -> additive bias) (LogitBiasLogitsProcessor).
  std::map<int, std::map<int32_t, float>> logit_bias;
  // req_index -> custom host logits-processor callback (ROAD-V1-C7
  // `custom_logit_processor`). Only present for a request that registered one;
  // make_sampling_metadata emits it as SamplingMetadata.logits_processors.
  std::map<int, LogitsProcessorCallback> logits_processors;
  // req_id -> requested sample-logprob count (gpu_input_batch.py:434-440). The
  // `-1` ("all logprobs") sentinel is WIDENED to vocab_size at add_request, as
  // upstream does, so every value here is a concrete count and one gathered
  // shape reaches every consumer. Preserving the sentinel instead was a
  // recorded deviation until it turned out to crash the engine
  // (specs/logprobs-all-sentinel.md).
  std::map<std::string, int> num_logprobs;
  // req_id -> the EXPLICIT vocab ids to score (gpu_input_batch.py:273,443-444).
  // Keyed by req_id like its `num_logprobs` sibling, so condense() and
  // swap_states() need do nothing: reindexing cannot invalidate an id key.
  // make_sampling_metadata re-derives the req_INDEX keys SamplingMetadata
  // carries (gpu_input_batch.py:934-951) over the live batch only.
  std::map<std::string, std::vector<int32_t>> logprob_token_ids;
  // req_id -> requested PROMPT-logprob count (gpu_model_runner.py:1305-1310,
  // where upstream keeps it on the runner rather than the input batch; it is
  // seeded and dropped at exactly the two sites num_logprobs above is, so it
  // lives beside it). The `-1` sentinel IS widened to vocab_size here — the
  // opposite of num_logprobs — because the prompt path feeds GatherLogprobs
  // directly, which needs a concrete column count, and never reaches the
  // Sampler that reads our sentinel. See specs/prompt-logprobs.md.
  std::map<std::string, int> num_prompt_logprobs;
  // Lazily-allocated [max_num_reqs][vocab_size] EXCLUDE mask (TRUE == mask this
  // token to -inf). Empty until the first request with allowed_token_ids; a row
  // is all-TRUE then the allowed ids are cleared to FALSE (gpu_input_batch.py
  // :446-467). apply_allowed_token_ids reads TRUE == exclude.
  std::vector<std::vector<uint8_t>> allowed_token_ids_mask;
  // req_index -> bad-words token-id n-grams (gpu_input_batch.py:469-471).
  std::map<int, std::vector<std::vector<int32_t>>> bad_words_token_ids;

  // Membership sets driving the sampling predicates (upstream *_reqs sets).
  std::unordered_map<std::string, char> greedy_reqs;
  std::unordered_map<std::string, char> random_reqs;
  std::unordered_map<std::string, char> top_p_reqs;
  std::unordered_map<std::string, char> top_k_reqs;
  std::unordered_map<std::string, char> frequency_penalties_reqs;
  std::unordered_map<std::string, char> presence_penalties_reqs;
  std::unordered_map<std::string, char> repetition_penalties_reqs;
  // ROAD-V1-C7 predicate sets (keyed by req_id, so reindexing is a no-op).
  std::unordered_map<std::string, char> min_p_reqs;
  std::unordered_map<std::string, char> has_allowed_token_ids;

  // Per-slot output token ids (nullopt on a freed slot). Consumed by the M1.7
  // sampler for penalties; mirrors upstream req_output_token_ids.
  std::vector<std::optional<std::vector<int32_t>>> req_output_token_ids;

  // Per-slot RNG seed (nullopt = no per-request seed). Sourced from
  // sampling_params.seed at add_request (upstream request.generator ==
  // sampling_params.seed; gpu_input_batch.py:413-414). make_sampling_metadata
  // emits it as SamplingMetadata.generators (req_index -> seed) so seeded random
  // sampling is deterministic; a request without a seed uses the sampler's
  // batch-default RNG (upstream NOTE at gpu_input_batch.py:251-252). This CLOSES
  // the M1.7 sampling-state seed carry. min_p / min_tokens / logit_bias /
  // allowed_token_ids / bad_words per-slot tracking stay DEFERRED (see
  // make_sampling_metadata).
  std::vector<std::optional<int64_t>> seeds;

  // Per-slot speculative token ids (gpu_input_batch.py:286). Empty on the
  // default path; set by update_req_spec_token_ids when a verify step schedules
  // drafts. Kept as a per-slot list so _get_active_token_count / condense swap
  // match upstream exactly.
  std::vector<std::vector<int32_t>> spec_token_ids;

  // Per-slot accepted-token count (gpu_input_batch.py:240-243). Seeded to 1 at
  // add_request (one token generated by default); the rejection sampler (I3)
  // overwrites it after verification and the GDN spec kernel (I5) reads it to
  // select the recurrent-state slot. Stays 1 everywhere on the default path.
  // Frozen SPEC-MTP metadata ABI (I2).
  std::vector<int32_t> num_accepted_tokens;

  // update_req_spec_token_ids (gpu_input_batch.py:484-509): install the drafts
  // scheduled for `req_id` this step into slot req_index — clear the prior list,
  // look up scheduled_spec_tokens[req_id] (absent -> no drafts, just cleared),
  // splice the ids into token_ids_cpu after num_tokens_no_spec (placeholders the
  // async runner overwrites in _prepare_input_ids), and record them in
  // spec_token_ids[req_index]. No-op when the request has no scheduled drafts.
  // Frozen SPEC-MTP metadata ABI (I2); the runner (I5) drives it.
  void update_req_spec_token_ids(
      int req_index, const std::string& req_id,
      const std::map<std::string, std::vector<int32_t>>& scheduled_spec_tokens);

 private:
  // Minimal port of BatchUpdateBuilder's REMOVED-index tracking — the only part
  // add_request (hole fill) + condense (densification) depend on. `removed`
  // returns the freed indices sorted DESCENDING; peek/pop return the LOWEST.
  // (The added/moved logitsprocs tracking is deferred.)
  class RemovedTracker {
   public:
    void removed_append(int index);
    bool has_removed() const { return !removed_.empty(); }
    // Lowest removed index (or nullopt); does not pop.
    std::optional<int> peek_removed();
    // Pop + return the lowest removed index (or nullopt).
    std::optional<int> pop_removed();
    // Freed indices sorted descending (upstream `removed` property).
    const std::vector<int>& removed();

   private:
    void ensure_sorted();
    std::vector<int> removed_;
    bool is_sorted_ = false;
  };

  // _register_add_request: pick the slot (freed hole, else num_reqs).
  int register_add_request();
  // _get_active_token_count: num_tokens_no_spec[i] + len(spec_token_ids[i]).
  int get_active_token_count(int req_index) const;

  // The uncached SamplingMetadata builder (upstream _make_sampling_metadata).
  SamplingMetadata build_sampling_metadata() const;

  RemovedTracker removed_tracker_;

  // Cached sampling metadata + its batch-change dirty flag (upstream
  // self.sampling_metadata + batch_update_builder). Mutable so the const
  // make_sampling_metadata() can refresh the cache; the dirty flag is set by
  // every batch mutation (add_request / remove_request / condense / swap_states).
  mutable SamplingMetadata sampling_metadata_cache_;
  mutable bool sampling_metadata_dirty_ = true;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_INPUT_BATCH_H_

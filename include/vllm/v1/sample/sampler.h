// Ported from: vllm/v1/sample/sampler.py @ e24d1b24 (the `Sampler` nn.Module).
//
// The V1 Sampler assembles the ordered sampling pipeline over the model's final
// logits [num_reqs, vocab] (row-major f32) and produces a SamplerOutput. The
// ORDER is the correctness core and mirrors Sampler.forward + Sampler.sample
// EXACTLY (sampler.py docstring steps 1-9):
//   1. If logprobs requested, snapshot raw_logprobs = compute_logprobs(logits)
//      BEFORE any mutation (logprobs_mode raw_logprobs; raw_logits is a stub).
//   2. logits -> float32 (already f32 here).
//   3. apply allowed_token_ids whitelist mask.
//   4. apply bad_words exclusion.
//   5. non-argmax-invariant procs, in upstream order: min_tokens then logit_bias.
//   6. penalties (repetition, frequency, presence).
//   7. sample(): greedy argmax snapshot (if !all_random); if all_greedy return
//      it; apply_temperature (temp<eps->1.0 guard when !all_random);
//      argmax-invariant procs (min_p); top_k_top_p; probs=softmax; random_sample;
//      where(temp<eps, greedy, random) per row for a mixed batch.
//   8. gather_logprobs(raw_logprobs, num_logprobs, sampled): top-k values+indices,
//      the sampled token's logprob, and ranks via batched_count_greater_than
//      (rank = #{vocab logprobs >= the sampled token's logprob}; 1-based — the
//      max-logprob token has rank 1). Concat [sampled_col | topk] -> [n, k+1].
//   9. Return SamplerOutput{ sampled_token_ids [n,1], logprobs_tensors }.
//
// The Sampler is device-agnostic: it composes the vt sampling ops (M1.7 Task 2/3)
// on whatever Queue's device the logits live on. Per-row host decisions (the
// where-merge, materializing the top_k/top_p/temperature/seed vectors) mirror the
// Task 2/3 host-array-then-op pattern. CPU is the correctness gate at T0; a
// CUDA-Queue run is dgx-pending.
//
// Step 8 has a SECOND form (`SAMPLE-LOGPROB-TOKEN-IDS`): when a request carries
// `logprob_token_ids` (generative scoring), gather_specific_token_logprobs
// returns the sampled token plus EXACTLY the requested vocab ids — padded to the
// batch-wide width with -inf — with the sampled token's rank still over the FULL
// vocab. It also drives whether the step-1 snapshot is taken at all
// (sampler.py:86), and it WINS over a logprobs count when a request supplies
// both (sampler.py:133-136).
//
// DEFERRED (marked 1:1 stubs, see sampler.cpp): spec-decode bonus-token
// (predict_bonus_token), thinking-budget, and the logprobs_mode variants beyond
// raw_logprobs.
#ifndef VLLM_V1_SAMPLE_SAMPLER_H_
#define VLLM_V1_SAMPLE_SAMPLER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/v1/engine/types.h"     // SamplerOutput
#include "vllm/v1/sample/metadata.h"  // SamplingMetadata
#include "vt/tensor.h"

namespace vllm::v1 {

// LogprobsMode (vllm/config/model.py::LogprobsMode). Only raw_logprobs is
// implemented at T0; the raw_logits / processed_* modes are marked stubs (the
// all-greedy processed-logprobs branch and the raw_logits clone are deferred).
// Which tensor the returned logprobs are read from (model.py:82,221 /
// sampler.py:87-93,255-302). The RAW pair is snapshotted before any logits
// processor runs, so it describes the MODEL's distribution; the PROCESSED pair
// is taken after temperature and top-k/top-p, so it describes the distribution
// actually SAMPLED from — a token top-k masked away reads -inf there and its
// true value in the raw modes. Both distinctions are user-visible.
enum class LogprobsMode {
  kRawLogprobs,       // default: compute_logprobs(logits) before any mutation
  kRawLogits,         // the raw logits themselves, before any mutation
  kProcessedLogprobs, // logprobs after temperature/top-k/top-p
  kProcessedLogits,   // logits after temperature/top-k/top-p
};

// Sampler (sampler.py::Sampler). Stateless apart from the logprobs_mode; the
// forward call carries all per-step state via the SamplingMetadata.
class Sampler {
 public:
  // Out-of-line (like the dtor) so the pimpl'd GreedyArgmaxScratch is complete
  // wherever a Sampler is constructed/destroyed.
  explicit Sampler(LogprobsMode logprobs_mode = LogprobsMode::kRawLogprobs);
  ~Sampler();
  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;

  // Sampler.forward. `logits` [num_reqs, vocab] f32 is mutated in place through
  // the pipeline (upstream's logits.to(float32) is a no-op copy when already f32,
  // so upstream mutates in place too). Returns the sampled tokens (+ logprobs).
  //
  // `sampled_ids_out` (ENG-ASYNC-SCHED W3, default nullptr = byte-identical sync
  // path): when non-null it must be a device-resident int64 [num_reqs] (or
  // [num_reqs,1]) tensor. The sampler writes the final sampled ids into it so the
  // async output D2H (AsyncGPUModelRunnerOutput) — not the sampler — owns the
  // single host copy. On the all-greedy, no-logprobs gate path this stays FULLY
  // device-resident (GreedyArgmax writes the tensor directly; no host download,
  // no main-queue synchronize) and the returned SamplerOutput.sampled_on_device
  // is true with an empty host `sampled_token_ids`. Any batch with random rows or
  // logprobs falls back to the host `sample()` path and then copies the host ids
  // into `sampled_ids_out` (correct, no zero-copy win — the greedy gate is the
  // overlap target; mirrors async_utils.py keeping ids GPU-side at :31).
  SamplerOutput forward(vt::Queue& q, vt::Tensor& logits,
                        const SamplingMetadata& sampling_metadata,
                        vt::Tensor* sampled_ids_out = nullptr) const;

  // compute_logprobs + gather_logprobs over PROMPT positions
  // (gpu_model_runner.py:5688-5697, SAMPLE-PROMPT-LOGPROBS). `logits` is a
  // read-only [n, vocab] f32 view of the rows the forward produced for prompt
  // positions; `target_token_ids` [n] names the token each row is scored
  // against — the token that FOLLOWS that position, since a prompt logprob asks
  // how likely the token that actually came next was.
  //
  // Prompt positions bypass every logits processor, which is why upstream notes
  // that the raw_* and processed_* logprobs_modes coincide here (:5691-5693);
  // the logits are consequently never mutated. Deliberately the same
  // log_softmax op and the same gather the sampled path runs, so a prompt
  // logprob and a sampled logprob over the same distribution agree bit-for-bit.
  LogprobsTensors compute_prompt_logprobs(
      vt::Queue& q, const vt::Tensor& logits, int num_logprobs,
      const std::vector<int64_t>& target_token_ids) const;

 private:
  // Sampler.sample. Runs steps 7a-7f; returns the [num_reqs] host token ids.
  //
  // `processed_out` (sampler.py:246-248,262-271,286-302): when non-null AND the
  // mode is one of the processed_* pair, sample() fills it with the [n, vocab]
  // host snapshot taken at the point upstream takes it — after the logits
  // processors on the all-greedy early return, and after temperature and
  // top-k/top-p otherwise. Left EMPTY under the raw_* modes, which snapshot in
  // forward() before any mutation.
  std::vector<int64_t> sample(vt::Queue& q, vt::Tensor& logits,
                              const SamplingMetadata& sampling_metadata,
                              std::vector<float>* processed_out = nullptr) const;

  // GreedyArgmax over `logits` -> [n] host int64 ids, reusing a PERSISTENT
  // device + pinned-host scratch (grow-only) so the greedy decode hot path does
  // NO per-step cudaMalloc/cudaFree/cudaHostAlloc (each device-syncs). Mirrors
  // vLLM's persistent sampled-id + pinned buffers (gpu_model_runner.py:873-878).
  std::vector<int64_t> greedy_argmax_host(vt::Queue& q, const vt::Tensor& logits,
                                          int64_t n) const;

  LogprobsMode logprobs_mode_;
  // Persistent greedy-argmax scratch (pimpl; defined in sampler.cpp). Lazily
  // created on the first greedy call; mutable because forward()/sample() are
  // const and this is pure allocation state, not sampling semantics.
  struct GreedyArgmaxScratch;
  mutable std::unique_ptr<GreedyArgmaxScratch> greedy_scratch_;
  uint64_t default_seed_ = 0;
  mutable uint64_t random_step_ = 0;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_SAMPLE_SAMPLER_H_

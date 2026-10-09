// Ported from: vllm/v1/worker/gpu/spec_decode/mtp/speculator.py (MTPSpeculator)
// + vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (propose :126-271,
//   _prefill :332-370) @ e24d1b24. `sample_draft` / `_greedy_sample_draft` are NOT
//   in that file: they are inherited from `DraftModelSpeculator` in
//   vllm/v1/worker/gpu/spec_decode/speculator.py (class :69, `_greedy_sample_draft`
//   :276-280 @ 555967922). The chain is
//   MTPSpeculator -> AutoRegressiveSpeculator -> DraftModelSpeculator ->
//   BaseSpeculator, and the ABC at :29 declares no such method.
//
// Scope (SPEC-MTP increment I5c, row SPEC-MTP): the k=1 greedy MTP propose — the
// paged draft-prefill forward + argmax draft-token pick — assembled from the
// bricks the earlier increments landed:
//   * prepare_prefill_inputs (I5b) shifts/splices the target verify batch into the
//     draft input_ids + last_token_indices (autoregressive/speculator.py:185-195);
//   * Qwen3_5MTPModel::ForwardPaged (I5c) runs the head + one paged full-attn
//     decoder layer over the DRAFT KV layer using the target's slot mapping
//     (qwen3_5_mtp.py:129-165 over the paged backend, speculator.py:346 _run_model);
//   * the shared lm_head + a per-request argmax over the last_token_indices rows
//     picks the drafted token (spec_decode/speculator.py:276-280), and k=1
//     EARLY-EXITS after this one forward (autoregressive/speculator.py:238-240)
//     — no multi-step decode.
//
// This is a CALLABLE, tested propose brick. It is NOT wired into the runner STEP
// loop (that is I5d): nothing on the production path constructs an MtpProposer or
// calls propose(), so with no SpeculativeConfig this TU's object code is
// unreachable and the engine is byte-identical. The draft KV layer it consumes is
// allocated only when speculative decoding is on (MakeQwen3_5KVCacheSpec num_spec).
#ifndef VLLM_V1_WORKER_GPU_SPEC_DECODE_MTP_SPECULATOR_H_
#define VLLM_V1_WORKER_GPU_SPEC_DECODE_MTP_SPECULATOR_H_

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"      // PagedKvCache
#include "vllm/model_executor/models/qwen3_5_mtp.h"   // Qwen3_5MTPModel
#include "vllm/v1/attention/backend.h"                // CommonAttentionMetadata
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm::v1 {

// The k=1 greedy MTP propose (autoregressive/speculator.py:126-271, k=1 branch).
// Runs exactly one paged draft-prefill forward and returns the drafted token id
// per request.
//
//   draft                the MTP draft model (shares the target embed/lm_head)
//   target_attn_meta     the just-completed VERIFY step's full-attn metadata
//                        (query_start_loc / seq_lens / block_table / slot_mapping).
//                        The draft REUSES it unchanged — identical batch shape and
//                        KV layout (speculator.py NOTE :162-167, :222-234).
//   draft_kv             the MTP head's OWN paged K/V layer (index num_hidden_layers)
//   target_hidden        [T,H] bf16 device — the target model's post-final-norm
//                        hidden tap (ForwardDeviceTap output; the drafter's
//                        `hidden_states` input, qwen3_5_mtp.py:129-140)
//   target_input_ids     [T] i32 — the verify step's flat input ids
//   target_positions     [T] i64 — the verify step's flat positions
//   idx_mapping          [num_reqs] i32 — batch_idx -> req_state slot
//   last_sampled         [>= max req_state] i32 — per req_state just-sampled id
//   next_prefill_tokens  [>= max req_state] i32 — per req_state next prefill id
//   num_sampled          [num_reqs] i32 — I3 RejectionSampler output
//   num_rejected         [num_reqs] i32 — I3 RejectionSampler output
//   max_num_reqs         CUDA-graph request-count padding bound (>= num_reqs)
//
// Optional prefill_inputs_embeds is the already shift-spliced [T,H] draft
// embedding batch. Its owner must survive the entire call. Later generated
// draft decode rows still use ordinary token lookup; draft RoPE stays 1-D.
//
// num_reqs is target_attn_meta.num_reqs; T is target_input_ids.size(). Returns
// draft_tokens [num_reqs] (draft_tokens[:num_reqs, :1] flattened for k=1).
std::vector<int32_t> MtpProposePrefill(
    const vllm::Qwen3_5MTPModel& draft,
    const CommonAttentionMetadata& target_attn_meta,
    vllm::PagedKvCache& draft_kv, const vt::Tensor& target_hidden,
    const std::vector<int32_t>& target_input_ids,
    const std::vector<int64_t>& target_positions,
    const std::vector<int32_t>& idx_mapping,
    const std::vector<int32_t>& last_sampled,
    const std::vector<int32_t>& next_prefill_tokens,
    const std::vector<int32_t>& num_sampled,
    const std::vector<int32_t>& num_rejected, int max_num_reqs,
    vt::Queue& queue, const vt::Tensor* prefill_inputs_embeds = nullptr);

// The FULL greedy MTP propose at any depth (SPEC-MTP-K-GT-1, issue #81) —
// upstream `AutoRegressiveSpeculator.propose` (:129-274 @ 555967922) end to end:
// the one paged draft prefill above, the `num_speculative_steps == 1` EARLY EXIT
// (:238-240), and otherwise `prepare_decode_inputs` (:242-251) followed by
// `_multi_step_decode` (:266-272), which runs k-1 single-token draft decode
// steps over the draft's OWN paged KV layer.
//
// The extra arguments over MtpProposePrefill are exactly what the decode half
// needs and the prefill half does not:
//
//   num_speculative_tokens  k (>= 1). k == 1 reproduces MtpProposePrefill
//                           EXACTLY, and runs no further forward.
//   max_model_len           the position / seq_len clamp bound
//                           (speculator.py:638,644,733,737)
//   block_size              the draft KV group's page size, for the per-step
//                           slot mapping (see draft_decode_slot_mapping)
//
// `num_speculative_tokens` is a PARAMETER, not a value read from a config inside
// the loop, so the depth a step drafts at is a decision the CALLER owns. That is
// the seam a scheduler-supplied depth would use (upstream decides it in
// `vllm/v1/core/sched/scheduler.py:1122-1126`). No depth policy is implemented
// here.
//
// `draft_tokens` is [num_reqs * k] ROW-MAJOR: request r's drafts are
// `draft_tokens[r * k .. r * k + k)` in draft order. This is the flattened form
// of upstream's `self.draft_tokens[:num_reqs]` (:274).
//
// `num_draft_decode_forwards` is NOT diagnostics. It is a value a caller can
// assert that a propose which did NOT run the loop cannot satisfy, and it exists
// because the obvious witnesses do not work.
//
// It is NOT the ONLY such value, and an earlier revision of this comment said it
// was. `GPUModelRunner::spec_mtp_proposals_with_varied_drafts()`, named at the end
// of this block, is a second one, and a mutation that short-circuits the loop reds
// BOTH. The "only" reading matters because it argues against adding another
// assertion, and that is the reasoning a padded propose already walked through
// once on this row. A further witness here is in scope, not redundant.
//
// The witnesses this field replaced, and why each one fails:
//
//   * The draft LIST LENGTH cannot serve. Whatever this function returns is
//     sliced into k-token lists by the runner, so a propose that ran ONE forward
//     and repeated its step-0 draft across all k columns yields k drafts per
//     request, a per-depth counter of SIZE k, and a byte-identical token stream,
//     because greedy plus accept-iff-equal makes the emission independent of k.
//     That exact mutation was applied to this function and the depth suite stayed
//     green on it, which is why this field exists.
//   * Non-zero acceptance at depth >= 2 cannot serve on CPU. Acceptance is
//     measured at ZERO at every depth on the synthetic gate model, so the
//     acceptance profile is identical between the real loop and the padded fake.
//     It does not serve on real weights either, and an earlier revision of this
//     comment called it the ONLY one of these that PROVES per-column provenance.
//     It proves no such thing. A padded row is `t0 t0 ...`, so its column 1 is
//     accepted exactly when the target's own greedy continuation repeats `t0`
//     (accept-iff-equal, `rejection_sampler.h`), which real text does routinely
//     on runs of whitespace, punctuation and indentation. A padded drafter can
//     therefore report `spec_drafts_accepted_by_depth()[1] > 0` on real weights,
//     and a broken carry only lowers the RATE rather than zeroing the count.
//     What closes provenance is an acceptance-RATE comparison against a PADDED
//     CONTROL on the identical workload, which the owed DGX gate specifies.
//   * PER-CALL distinctness of the k drafts cannot serve. A correct drafter may
//     repeat a token, and on a 24-entry vocabulary it does: a `2 2 2` row at k=3
//     is measured on the synthetic gate model. That property is the model's
//     rather than the loop's.
//
// It is incremented AFTER each draft decode forward RETURNS, so it counts work
// performed rather than intent, and it is exactly `k - 1` on every call.
//
// It is NOT a witness against padding, and it never was. A propose that runs
// every step and then overwrites `draft_tokens` with its step-0 draft reports
// `k - 1` here truthfully. What that mutation cannot survive is a check on the
// DELIVERED array, so the runner keeps one:
// `GPUModelRunner::spec_mtp_proposals_with_varied_drafts()` counts the calls
// whose returned row was not a pure function of its own first column. That is
// the AGGREGATE form of the per-call distinctness rejected above, it is measured
// non-zero at every k >= 2 on the gate model, and its own bound is recorded
// beside it in runner.h.
struct MtpDraftProposal {
  std::vector<int32_t> draft_tokens;      // [num_reqs * k] row-major
  int64_t num_draft_decode_forwards = 0;  // == k - 1, one per executed step
};

MtpDraftProposal MtpProposeDrafts(
    const vllm::Qwen3_5MTPModel& draft,
    const CommonAttentionMetadata& target_attn_meta,
    vllm::PagedKvCache& draft_kv, const vt::Tensor& target_hidden,
    const std::vector<int32_t>& target_input_ids,
    const std::vector<int64_t>& target_positions,
    const std::vector<int32_t>& idx_mapping,
    const std::vector<int32_t>& last_sampled,
    const std::vector<int32_t>& next_prefill_tokens,
    const std::vector<int32_t>& num_sampled,
    const std::vector<int32_t>& num_rejected, int max_num_reqs,
    int num_speculative_tokens, int max_model_len, int block_size,
    vt::Queue& queue, const vt::Tensor* prefill_inputs_embeds = nullptr);

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_SPEC_DECODE_MTP_SPECULATOR_H_

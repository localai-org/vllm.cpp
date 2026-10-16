// Ported from vllm/v1/worker/gpu/spec_decode/{mtp,autoregressive}/speculator.py
// — the greedy MTP propose. The k=1 prefill half landed as SPEC-MTP increment
// I5c against pin e24d1b24; the multi-step decode half is SPEC-MTP-K-GT-1
// (issue #81) against pin 555967922, where the same file carries the loop at
// :242-274. See the header for scope and the exact upstream anchors.
#include "vllm/v1/worker/gpu/spec_decode/mtp/speculator.h"

#include <cstddef>
#include <utility>

#include "vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_decode_inputs.h"
#include "vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_prefill_inputs.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm::v1 {
namespace {

// What `_prefill` (:335-371) leaves behind for the decode steps: the step-0
// draft token, the draft model's hidden state at each request's
// last_token_indices row (:367-371), and that row's POSITION (:346), which
// prepare_decode_inputs advances from.
struct PrefillOutcome {
  std::vector<int32_t> draft_tokens;    // [num_reqs]
  vllm::Qwen3_5MTPHiddenStates hidden;  // [T,H] device (the whole forward)
  std::vector<int64_t> sampled_rows;    // [num_reqs] last_token_indices
  std::vector<int32_t> positions;       // [num_reqs] at those rows
};

PrefillOutcome ProposePrefill(
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
    vt::Queue& queue, const vt::Tensor* prefill_inputs_embeds) {
  const int64_t num_reqs = target_attn_meta.num_reqs;
  const int64_t T = static_cast<int64_t>(target_input_ids.size());
  VT_CHECK(num_reqs > 0, "MtpProposePrefill: empty batch");
  VT_CHECK(static_cast<int64_t>(target_positions.size()) == T,
           "MtpProposePrefill: positions length must equal token count");
  VT_CHECK(target_attn_meta.num_actual_tokens == T,
           "MtpProposePrefill: attn metadata token count must equal T");

  // ── Draft input-prep (I5b): shift-splice the verify batch into the drafter's
  // input_ids + last_token_indices; the draft query_start_loc / seq_lens equal the
  // target's (rejected positions padded, not compacted — speculator.py:185-195). ──
  const SpecPrefillInputs spi = prepare_prefill_inputs(
      target_input_ids, target_positions, target_attn_meta.query_start_loc,
      target_attn_meta.seq_lens, idx_mapping, last_sampled, next_prefill_tokens,
      num_sampled, num_rejected, max_num_reqs);

  // The draft REUSES the target's attention metadata + slot mappings unchanged
  // (identical batch shape + KV layout, speculator.py:222-234). ForwardPaged takes
  // i32 positions (as the target/standalone forwards do); the drafter positions are
  // a copy of the target's, so narrow the I5b i64 positions.
  std::vector<int32_t> positions32(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    positions32[static_cast<size_t>(t)] =
        static_cast<int32_t>(spi.positions[static_cast<size_t>(t)]);

  // ── The one paged draft forward (I5c) + shared lm_head. ──────────────────────
  vllm::Qwen3_5MTPHiddenStates hidden = draft.ForwardPaged(
      spi.input_ids, positions32, target_hidden, target_attn_meta, draft_kv, queue,
      /*spec_step_idx=*/0, prefill_inputs_embeds);
  PrefillOutcome out;
  // ── Greedy draft pick over each request's last (sampled) row
  // (spec_decode/speculator.py:276-280). ──────────────────────────────────────
  out.sampled_rows.assign(
      spi.last_token_indices.begin(),
      spi.last_token_indices.begin() + static_cast<size_t>(num_reqs));
  // Keep all forward rows for draft KV/feedback, but apply the head only to the
  // sampled rows, as the pinned producer does before sample_draft.
  const auto selected = draft.GatherHiddenRows(hidden.tensor, out.sampled_rows, queue);
  const auto logits = draft.ComputeLogits(selected.tensor, queue);
  VT_CHECK(logits.on_device() && logits.rows == num_reqs,
           "MtpProposePrefill: unexpected selected draft logits shape");
  out.draft_tokens = draft.SelectDraftTokens(logits, queue);

  // :346 — the positions of those same rows. The decode half advances from them.
  // The k=1 caller drops them, and dropping a host vector costs nothing.
  out.positions.resize(static_cast<size_t>(num_reqs));
  for (int64_t r = 0; r < num_reqs; ++r) {
    out.positions[static_cast<size_t>(r)] =
        positions32[static_cast<size_t>(out.sampled_rows[
            static_cast<size_t>(r)])];
  }
  out.hidden = std::move(hidden);
  return out;
}

// The per-step attention metadata for a draft DECODE forward: one query token
// per request over the DRAFT KV layer. Upstream rebuilds this each step through
// `self.block_tables.compute_slot_mappings` + `_build_draft_attn_metadata`
// (:388-402). The block table is the TARGET's, unchanged, because the draft KV
// group has the same block geometry — the same reason the draft prefill reuses
// the target's slot mapping verbatim (:222-234).
CommonAttentionMetadata BuildDraftDecodeMeta(
    const CommonAttentionMetadata& target_attn_meta,
    const SpecDecodeInputs& inputs, int block_size) {
  const int num_reqs = inputs.num_reqs();
  CommonAttentionMetadata meta;
  meta.query_start_loc = inputs.query_start_loc;
  meta.query_start_loc_cpu = inputs.query_start_loc;
  meta.seq_lens = inputs.seq_lens;
  meta.seq_lens_cpu = inputs.seq_lens;
  meta.num_computed_tokens_cpu.resize(static_cast<size_t>(num_reqs));
  meta.num_reqs = num_reqs;
  meta.num_actual_tokens = num_reqs;  // exactly one query token per request
  meta.max_query_len = 1;
  meta.max_seq_len = 0;
  for (int r = 0; r < num_reqs; ++r) {
    const int32_t seq_len = inputs.seq_lens[static_cast<size_t>(r)];
    // This step's token is the request's last, so everything before it is
    // already computed. That is the decode-step form of query_len == 1.
    meta.num_computed_tokens_cpu[static_cast<size_t>(r)] = seq_len - 1;
    if (seq_len > meta.max_seq_len) meta.max_seq_len = seq_len;
  }
  meta.block_table_tensor = target_attn_meta.block_table_tensor;
  meta.block_table_num_cols = target_attn_meta.block_table_num_cols;
  meta.slot_mapping = draft_decode_slot_mapping(
      meta.block_table_tensor, meta.block_table_num_cols, inputs.positions,
      block_size);
  meta.causal = target_attn_meta.causal;
  return meta;
}

}  // namespace

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
    vt::Queue& queue, const vt::Tensor* prefill_inputs_embeds) {
  return ProposePrefill(draft, target_attn_meta, draft_kv, target_hidden,
                        target_input_ids, target_positions, idx_mapping,
                        last_sampled, next_prefill_tokens, num_sampled,
                        num_rejected, max_num_reqs, queue, prefill_inputs_embeds)
      .draft_tokens;
}

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
    vt::Queue& queue, const vt::Tensor* prefill_inputs_embeds) {
  VT_CHECK(num_speculative_tokens >= 1,
           "MtpProposeDrafts: num_speculative_tokens must be at least 1");
  const int num_reqs = target_attn_meta.num_reqs;
  const int k = num_speculative_tokens;

  MtpDraftProposal result;

  PrefillOutcome prefill = ProposePrefill(
      draft, target_attn_meta, draft_kv, target_hidden, target_input_ids,
      target_positions, idx_mapping, last_sampled, next_prefill_tokens,
      num_sampled, num_rejected, max_num_reqs, queue, prefill_inputs_embeds);

  // :238-240 — the k=1 EARLY EXIT. Byte-for-byte the pre-depth path: one
  // forward, one argmax, no decode state and no second gather. No decode
  // forward runs, so the reported count stays 0, which is what k-1 is at k=1.
  if (k == 1) {
    result.draft_tokens = std::move(prefill.draft_tokens);
    return result;
  }

  std::vector<int32_t> drafts(static_cast<size_t>(num_reqs) *
                              static_cast<size_t>(k));

  // :242-251 — the decode entry state, built from what the prefill left behind.
  SpecDecodeInputs inputs = prepare_decode_inputs(
      prefill.draft_tokens, target_attn_meta.seq_lens, num_rejected,
      prefill.positions, max_model_len);
  // Step 0's token is already drafted, so record it and advance to step 1. k >= 2
  // here, so this call never takes the final-step early return.
  const bool has_more = update_draft_inputs(prefill.draft_tokens, /*step=*/0, k,
                                            drafts, inputs, max_model_len);
  VT_CHECK(has_more, "MtpProposeDrafts: k >= 2 must leave a decode step to run");

  // :367-371 — the drafter's own hidden state at the SAMPLED rows becomes the
  // next forward's `hidden_states` input. From step 2 on, each decode forward
  // already produces exactly one row per request, so the carry is the identity
  // (:714-726) and no further gather happens.
  vllm::Qwen3_5MTPHiddenStates carry =
      draft.GatherHiddenRows(prefill.hidden.tensor, prefill.sampled_rows, queue);
  // `prefill.hidden` is deliberately kept ALIVE for the rest of this function.
  // The gather ENQUEUES its row copies, and on an asynchronous backend they have
  // not necessarily run when this line is reached; releasing the source buffer
  // returns it to the device pool, from which the very next allocation inside
  // ForwardPaged could take those bytes and overwrite them before the copies
  // read them. It is one verify-sized [T,H] draft activation held across k-1
  // draft steps, which is small beside the draft KV it is protecting.

  // :266-272 `_multi_step_decode` — `for step in range(1, num_speculative_steps)`.
  for (int step = 1; step < k; ++step) {
    const CommonAttentionMetadata decode_meta =
        BuildDraftDecodeMeta(target_attn_meta, inputs, block_size);
    // :434-441 `_generate_draft` — one draft forward over num_reqs tokens. The
    // MTP head is layer_type="full_attention", so this reads and writes only the
    // draft's own paged KV layer, at the slots decode_meta names.
    vllm::Qwen3_5MTPHiddenStates hidden =
        draft.ForwardPaged(inputs.input_ids, inputs.positions, carry.tensor,
                           decode_meta, draft_kv, queue);
    // Counted AFTER the forward returns, so this reports work PERFORMED rather
    // than intent. It is the witness a propose that short-circuits or clamps the
    // loop cannot forge. It is not a witness against PADDING, because a loop
    // that runs every step and then discards what it sampled increments it just
    // the same. The caller reads the delivered array for that. See the
    // MtpDraftProposal comment for what each witness sees and what none of them
    // do.
    ++result.num_draft_decode_forwards;
    vllm::ForwardLogits logits = draft.ComputeLogits(hidden.tensor, queue);
    VT_CHECK(logits.on_device() && logits.rows == num_reqs,
             "MtpProposeDrafts: unexpected draft decode logits shape");
    const std::vector<int32_t> drafted = draft.SelectDraftTokens(logits, queue);

    // :460-471 `update_draft_inputs` — record the step and feed it forward.
    const bool more =
        update_draft_inputs(drafted, step, k, drafts, inputs, max_model_len);
    // Releasing the PREVIOUS carry is safe here, unlike the prefill handoff
    // above: SelectDraftTokens synchronised the queue two lines up, so the
    // forward that read it has completed and its storage cannot be handed to a
    // later allocation while still in use.
    carry = std::move(hidden);
    if (!more) break;
  }
  result.draft_tokens = std::move(drafts);
  return result;
}

}  // namespace vllm::v1

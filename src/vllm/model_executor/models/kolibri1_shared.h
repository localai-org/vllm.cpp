// Kolibri-1 — the forward's device-generic helpers, shared VERBATIM by the
// CPU row (kolibri1_forward.cpp, MODEL-TEXT-kolibri-1) and the Tenstorrent
// B2b-i dense-resident device forward (kolibri1_tt_forward.cpp,
// MODEL-TEXT-kolibri-1-tenstorrent).
//
// This is a PURE RELOCATION in the dense_attn_block.h extraction idiom: the
// sigmoid-logit-add router and the per-step attention-metadata upload moved
// out of kolibri1_forward.cpp's anonymous namespace UNCHANGED (renamed only
// to carry the Kolibri1 prefix at namespace scope), so the CPU row's op
// sequence is byte-identical — the same functions, the same order, the same
// bytes — and its gates (test_kolibri1, test_kolibri1_w2,
// test_kolibri1_moe_glue, test_kolibri1_w3) pin that. The B2b-i device
// forward consumes the SAME routing math (the addendum inherits the CPU
// row's f32 sigmoid/sigmoid compute path) instead of re-deriving a parallel
// copy that could drift.
//
// Both helpers are device-generic: they run over any vt backend's queue
// through the production vt ops. Nothing here is CPU-specific or
// Tenstorrent-specific.
#pragma once

#include <cmath>  // std::exp (SigmoidLogitAddRouting)
#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // dense_attn::StepInputs
#include "vllm/v1/attention/backend.h"                    // CommonAttentionMetadata
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {

// ── Router: sigmoid-logit-add (kolibri1.py:126-142) ─────────────────────────
// Selection on logits + bias, weights = sigmoid of the UNBIASED logits, NO
// renormalisation (norm_topk_prob=false). Router logits are f32 (the gate
// linear's out_dtype=torch.float32 upstream, :171-177). Ties break to the
// LOWER expert index, matching torch.topk's stable CPU order for the exact
// ties a test constructs deliberately.
struct Kolibri1HostRouting {
  std::vector<int32_t> ids;    // [T, top_k]
  std::vector<float> weights;  // [T, top_k]
};

inline Kolibri1HostRouting SigmoidLogitAddRouting(
    const std::vector<float>& logits, const std::vector<float>& bias,
    int64_t t, int64_t num_experts, int64_t top_k) {
  Kolibri1HostRouting r;
  r.ids.resize(static_cast<size_t>(t * top_k));
  r.weights.resize(static_cast<size_t>(t * top_k));
  for (int64_t i = 0; i < t; ++i) {
    const float* row = &logits[static_cast<size_t>(i * num_experts)];
    // Partial selection: top_k passes of argmax over the biased scores.
    std::vector<char> taken(static_cast<size_t>(num_experts), 0);
    for (int64_t kk = 0; kk < top_k; ++kk) {
      int64_t best = -1;
      float best_score = 0.0f;
      for (int64_t e = 0; e < num_experts; ++e) {
        if (taken[static_cast<size_t>(e)]) continue;
        const float score = row[e] + bias[static_cast<size_t>(e)];
        if (best < 0 || score > best_score) {
          best = e;
          best_score = score;
        }
      }
      taken[static_cast<size_t>(best)] = 1;
      // The WEIGHT reads the unbiased logit (docstring kolibri1.py:134-136).
      r.weights[static_cast<size_t>(i * top_k + kk)] =
          1.0f / (1.0f + std::exp(-row[best]));
      r.ids[static_cast<size_t>(i * top_k + kk)] = static_cast<int32_t>(best);
    }
    // norm_topk_prob=false: the renormalisation branch (:140-141) is skipped.
  }
  return r;
}

// ── Per-step device inputs (the Kolibri-1 variant) ──────────────────────────
// positions / slot_mapping / block_table / seq_lens / query_start_loc
// uploaded once per forward step. The CPU row calls this per layer (the
// values are layer-invariant); the B2b-i device forward calls it once per
// step — the same bytes, the same order, one upload instead of N.
inline dense_attn::StepInputs Kolibri1BuildStepInputs(
    dense_attn::Dev d, const v1::CommonAttentionMetadata& meta,
    const std::vector<int32_t>& positions) {
  const int64_t t = static_cast<int64_t>(positions.size());
  dense_attn::DBuf d_positions(d, vt::DType::kI32, {t}, positions.data());
  dense_attn::DBuf d_slot_mapping(
      d, vt::DType::kI64,
      {static_cast<int64_t>(meta.slot_mapping.size())},
      meta.slot_mapping.data());
  dense_attn::DBuf d_block_table(
      d, vt::DType::kI32, {meta.num_reqs, meta.block_table_num_cols},
      const_cast<int32_t*>(meta.block_table_tensor.data()));
  dense_attn::DBuf d_seq_lens(d, vt::DType::kI32, {meta.num_reqs},
                             const_cast<int32_t*>(meta.seq_lens.data()));
  dense_attn::DBuf d_query_start_loc(
      d, vt::DType::kI32, {meta.num_reqs + 1},
      const_cast<int32_t*>(meta.query_start_loc.data()));
  dense_attn::StepInputs si;
  si.positions = std::move(d_positions);
  si.slot_mapping = std::move(d_slot_mapping);
  si.block_table = std::move(d_block_table);
  si.seq_lens = std::move(d_seq_lens);
  si.query_start_loc = std::move(d_query_start_loc);
  return si;
}

}  // namespace vllm

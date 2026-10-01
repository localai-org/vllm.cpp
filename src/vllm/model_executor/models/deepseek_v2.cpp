// DeepSeek-V2 (`DeepseekV2ForCausalLM`) forward — MLA campaign W7. This is the
// first forward in this tree whose attention is MLA: it composes W6's
// `mla::ForwardMlaAttentionBlock` (which itself composes W3's cache write, W4's
// absorbed MQA decode and W5's materialized-MHA chunked prefill) with the
// DeepSeek MoE block and the dense `first_k_dense_replace` layers.
//
// Grounding (@ pin e24d1b24), file:line on both sides:
//   ForwardBody / ForwardLayers  <- vllm/model_executor/models/deepseek_v2.py
//                                   :1347-1520 `DeepseekV2Model.forward`
//   RunLayer                     <- deepseek_v2.py:1262-1345
//                                   `DeepseekV2DecoderLayer.forward`
//                                   (input_layernorm(h, residual) -> self_attn ->
//                                    post_attention_layernorm(h, residual) -> mlp)
//   DenseMlp                     <- deepseek_v2.py:270-274 `DeepseekV2MLP.forward`
//   MoeBlock                     <- deepseek_v2.py:395-424 `DeepseekV2MoE.forward`
//                                   + vllm/model_executor/layers/fused_moe/router/
//                                     grouped_topk_router.py:80-161
//                                   + fused_moe/runner/moe_runner.py:392-407
//                                     (routed_scaling_factor then + shared_output)
//   BuildMlaBatchSplit           <- mla_attention.py:1640-1649 + :1806-1810
//   BuildMlaStep                 <- mla_attention.py:1652-1830
//                                   (`MLACommonMetadataBuilder.build`, non-DCP)
//
// ─── THREE THINGS THAT ARE NEW FOR THIS MODEL FAMILY ────────────────────────
// 1. SHARED EXPERTS. Qwen3-Coder had none, and the only other shared-expert path
//    in this tree (Qwen3.6's) carries a SIGMOID GATE. DeepSeek's does not:
//    `DeepseekV2MoE.shared_experts` is a plain `DeepseekV2MLP` (:344-357) whose
//    output is ADDED to the routed sum. So this block does NOT reuse
//    `RunMoeBlock` / `SharedExpert` (which would apply Qwen's gate and Qwen's
//    ungrouped softmax router); it calls the same underlying vt:: ops directly.
//    That also means ZERO edit to the 27B/35B/Coder MoE paths.
// 2. THE GROUPED (`noaux_tc`-shaped) ROUTER. `use_grouped_topk=True` is
//    UNCONDITIONAL for this family (:370-378) — even V2-Lite, whose `n_group ==
//    topk_group == 1` makes the grouping a no-op. We pass the grouped args
//    through to W3's `vt::MoeRouterTopK` extension rather than special-casing.
// 3. THE BATCH ORDER. MLA's metadata is built from a batch that upstream's
//    scheduler already reordered (decodes first; with-context prefills leading
//    the prefill tail). `BuildMlaBatchSplit` VALIDATES that rather than assuming
//    it — W6's gate measured 0.86 relative error from exactly that violation.
//
// ─── RECORDED DEVIATIONS ────────────────────────────────────────────────────
// (a) `routed_scaling_factor` is applied to the ROUTING WEIGHTS
//     (`MoeRouterTopKArgs::routed_scaling_factor`, i.e.
//     grouped_topk_router.py:159-160) rather than to the combined routed OUTPUT
//     (moe_runner.py:402-406, which vLLM's CUDA path selects via
//     `apply_routed_scale_to_output=True`). The routed combine is linear in the
//     weights, so the two are the same function; they differ only in float
//     rounding order, and the shared-expert term — which must NOT be scaled — is
//     added after either way. DeepSeek-V2-Lite has `routed_scaling_factor: 1.0`,
//     so on the W8 gate vehicle the two are bit-identical.
// (b) The chunked-prefill workspace size mirrors
//     `determine_chunked_prefill_workspace_size` (mla_attention.py:1422-1451)
//     with `max_model_len := config.max_position_embeddings` and
//     `max_num_seqs := this step's num_reqs`, because the model forward has no
//     VllmConfig. With the default (unclamped) `--max-model-len` those are the
//     same numbers upstream uses.
// (c) W9 landed the decode CUDA-graph sibling (`DeepseekV2DecodeGraph`, at the
//     bottom of this TU). The per-row-slice A-projection issue recorded at W6 was
//     re-measured at W9 and is discussed there.
#include "vllm/model_executor/models/deepseek_v2.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/attention/mla_chunked_context.h"
#include "vllm/model_executor/models/decode_graph_sizes.h"  // DecodeGraphSizes/PadToCaptureSize
#include "vllm/model_executor/moe_placement_seam.h"
#include "vllm/model_executor/layers/linear.h"             // UnquantizedMlpGateUpMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight glue
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/device_pool.h"
#include "vllm/model_executor/models/mla_attention.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::DeviceTokenIds
#include "vllm/model_executor/models/step_token_ids.h"   // #1305: the slot's device ids
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/breakable_graph.h"  // ENG-CUDAGRAPH-BREAK W3: the shared capture seam
#include "vt/ops.h"

namespace vllm {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

using namespace dense_attn;  // Dev / DBuf / MakeTensor / Reshape / ResidentWeight

namespace {

// `MlaStep` — the per-step device inputs the MLA block needs — moved to
// `deepseek_v2.h` at #699 W4a, so a second MLA model can drive the same build
// instead of re-deriving it. The comment that documented it moved with it.

// ─── W8 diagnostic counters (deepseek_v2.h `MlaBatchSplitStats`) ─────────────
// Written once per forward from the runner's single forward thread; read only by
// the paged-engine gate, which uses them to PROVE the engine actually produced
// mixed / with-context MLA batches rather than a stream of trivial ones.
MlaBatchSplitStats& MutableMlaBatchSplitStats() {
  static MlaBatchSplitStats stats;
  return stats;
}

void RecordMlaBatchSplit(const MlaBatchSplit& sp, int num_reqs) {
  MlaBatchSplitStats& st = MutableMlaBatchSplitStats();
  ++st.steps;
  if (sp.num_prefills == 0 && sp.num_decodes > 0) ++st.decode_only_steps;
  if (sp.num_decodes == 0 && sp.num_prefills > 0) ++st.prefill_only_steps;
  if (sp.num_decodes > 0 && sp.num_prefills > 0) ++st.mixed_steps;
  if (sp.num_prefills_with_context > 0) ++st.with_context_prefill_steps;
  st.max_num_decodes = std::max<int64_t>(st.max_num_decodes, sp.num_decodes);
  st.max_num_prefills = std::max<int64_t>(st.max_num_prefills, sp.num_prefills);
  st.max_num_reqs = std::max<int64_t>(st.max_num_reqs, num_reqs);
  st.total_decode_tokens += sp.num_decode_tokens;
  st.total_prefill_tokens += sp.num_prefill_tokens;
}

template <typename T>
Tensor UploadInto(Dev d, std::vector<DBuf>& owned, DType dt,
                  const std::vector<int64_t>& shape, const std::vector<T>& host) {
  owned.emplace_back(d, dt, shape, host.data());
  return owned.back().t();
}

// Upload a CONTIGUOUS RANGE of a caller-owned vector without materializing a
// temporary copy of it.
//
// ─── WHY THIS EXISTS: THE CUDA-GRAPH DANGLING-SOURCE BUG (MLA campaign W9) ───
// The obvious spelling of a sub-range upload is
//     std::vector<int32_t> slice(v.begin() + a, v.begin() + b);
//     UploadInto(..., slice);
// and under EAGER execution it is correct: `Copy` is a cudaMemcpyAsync issued
// before `slice` dies, and on GB10 the host pointer is device-addressable so the
// copy reads it in stream order.
//
// Under CUDA-GRAPH CAPTURE it is a USE-AFTER-FREE. The capture turns that
// cudaMemcpyAsync into a graph node that BAKES the source address, and `slice`
// is destroyed the moment BuildMlaStep returns. Every subsequent REPLAY then
// copies whatever now occupies that freed heap block into the decode
// `block_table` / `seq_lens`. The observed signature was exactly this: the first
// replay (issued immediately after capture, before the block was reused) is
// CORRECT, replay #2 onwards diverges by ~19 logits, and once the garbage
// block-table entry goes out of range the KV gather faults with "an illegal
// memory access was encountered". compute-sanitizer serializes enough to hide
// the fault and shows only the wrong numbers.
//
// So the whole captured region must upload ONLY from storage whose address the
// graph driver keeps alive and refreshes IN PLACE across replays — i.e. the
// caller's own metadata vectors (the driver's persistent per-size SizeSlot). This
// helper is how the decode half does that.
template <typename T>
Tensor UploadRange(Dev d, std::vector<DBuf>& owned, DType dt,
                   const std::vector<int64_t>& shape, const std::vector<T>& host,
                   size_t offset) {
  owned.emplace_back(d, dt, shape, host.data() + offset);
  return owned.back().t();
}

}  // namespace

// `MLACommonMetadataBuilder.build` (mla_attention.py:1652-1830), non-DCP branch.
//
// EXPORTED (declared in deepseek_v2.h) rather than file-local since #699 W4a:
// dots3-note's full-attention layer is `Dots3NoteFullAttention(
// DeepseekV2MLAAttention)` upstream and drives the SAME `mla::
// ForwardMlaAttentionBlock` over the same paged MLA cache, so it needs the same
// per-step metadata. Re-deriving it in a second TU would be the hand-rolled
// parallel path AGENTS.md forbids. Nothing about the body changed in the move.
MlaStep BuildMlaStep(Dev d, const std::vector<int32_t>& positions,
                     const CommonAttentionMetadata& am, int64_t block_size,
                     int64_t max_model_len) {
  MlaStep s;
  s.split = BuildMlaBatchSplit(am);
  const MlaBatchSplit& sp = s.split;
  RecordMlaBatchSplit(sp, am.num_reqs);
  const int64_t T = static_cast<int64_t>(positions.size());

  s.positions = UploadInto(d, s.owned, DType::kI32, {T}, positions);
  s.slot_mapping = UploadInto(d, s.owned, DType::kI64, {T}, am.slot_mapping);

  s.meta.num_decode_tokens = sp.num_decode_tokens;
  const int64_t cols = am.block_table_num_cols;

  // --- decode half (triton_mla.py:214-216, 245-246) ---
  if (sp.num_decodes > 0) {
    // The decode half is a PREFIX of the batch (the scheduler reorders decodes
    // first — mla_attention.py:1420 `reorder_batch_threshold = 1`), so both
    // sources are a prefix of the caller's own persistent vectors and need NO
    // temporary. This is load-bearing for CUDA-graph capture: see UploadRange.
    s.meta.decode.block_table = UploadRange(d, s.owned, DType::kI32,
                                            {sp.num_decodes, cols},
                                            am.block_table_tensor, /*offset=*/0);
    s.meta.decode.seq_lens = UploadRange(d, s.owned, DType::kI32, {sp.num_decodes},
                                         am.seq_lens, /*offset=*/0);
    s.meta.decode.max_seq_len = sp.decode_max_seq_len;
  }

  // --- prefill half (mla_attention.py:1652-1682) ---
  // The prefill block table is the SUFFIX of the batch's block table, so it too
  // uploads straight out of the caller's vector (no temporary). `prefill_cu_seqlens_q`
  // is DERIVED data owned by `s.split`, which lives inside this MlaStep for the
  // whole forward — safe eagerly, and never reached under capture because the
  // graph driver only ever captures PURE-DECODE steps (num_prefills == 0).
  if (sp.num_prefills > 0) {
    s.meta.prefill_cu_seqlens_q = UploadInto(d, s.owned, DType::kI32,
                                             {sp.num_prefills + 1},
                                             sp.prefill_cu_seqlens_q);
    s.meta.prefill_block_table =
        UploadRange(d, s.owned, DType::kI32, {sp.num_prefills, cols},
                    am.block_table_tensor,
                    static_cast<size_t>(sp.num_decodes * cols));
    s.meta.max_query_len = sp.prefill_max_query_len;

    if (sp.num_prefills_with_context > 0) {
      // DEVIATION (b) — see the TU header.
      const int64_t workspace = mla::DetermineChunkedPrefillWorkspaceSize(
          max_model_len, am.num_reqs, block_size);
      const mla::MlaChunkedContextMetadata cm = mla::BuildMlaChunkedContext(
          sp.prefill_context_lens, sp.prefill_cu_seqlens_q, workspace, block_size);
      s.meta.prefill_tokens_with_context = cm.prefill_tokens_with_context;
      s.meta.chunk_workspace_tokens = workspace;
      const int64_t np = cm.num_prefills;
      const int32_t row = std::max<int32_t>(cm.max_token_num_over_chunk, 1);
      for (int32_t i = 0; i < cm.num_chunks; ++i) {
        const std::vector<int32_t> cu(
            cm.cu_seq_lens.begin() + static_cast<size_t>(i) * (np + 1),
            cm.cu_seq_lens.begin() + static_cast<size_t>(i + 1) * (np + 1));
        const std::vector<int32_t> starts(
            cm.starts.begin() + static_cast<size_t>(i) * np,
            cm.starts.begin() + static_cast<size_t>(i + 1) * np);
        const std::vector<int32_t> t2s(
            cm.token_to_seq.begin() + static_cast<size_t>(i) * row,
            cm.token_to_seq.begin() + static_cast<size_t>(i + 1) * row);
        mla::MlaChunkDeviceMetadata cd;
        cd.cu_seq_lens = UploadInto(d, s.owned, DType::kI32, {np + 1}, cu);
        cd.starts = UploadInto(d, s.owned, DType::kI32, {np}, starts);
        cd.token_to_seq = UploadInto(d, s.owned, DType::kI32, {row}, t2s);
        cd.total_tokens = cm.chunk_total_token[static_cast<size_t>(i)];
        cd.max_seq_len = cm.max_seq_lens[static_cast<size_t>(i)];
        s.meta.chunks.push_back(cd);
      }
    }
  }
  return s;
}

namespace {

// The MLA block's DEVICE-resident weight views for one layer. ResidentWeight
// uploads once on first touch and memoizes on the OwnedTensor, so this is a
// pointer shuffle after the first step.
mla::MlaBlockWeights ResidentMla(Dev d, const DeepseekV2MlaWeights& w,
                                 const DeepseekV2Params& p,
                                 const Tensor& rope_cache) {
  const mla::MlaBlockDims& dm = p.mla;
  mla::MlaBlockWeights m;
  if (dm.has_q_lora()) {
    m.fused_qkv_a_proj = ResidentWeight(d, w.fused_qkv_a_proj);
    m.q_a_layernorm = ResidentWeight(d, w.q_a_layernorm, {dm.q_lora_rank});
    m.q_b_proj = ResidentWeight(d, w.q_b_proj);
  } else {
    m.kv_a_proj_with_mqa = ResidentWeight(d, w.kv_a_proj_with_mqa);
    m.q_proj = ResidentWeight(d, w.q_proj);
  }
  m.kv_a_layernorm = ResidentWeight(d, w.kv_a_layernorm, {dm.kv_lora_rank});
  m.kv_b_proj = ResidentWeight(d, w.kv_b_proj);
  m.w_uk_t = ResidentWeight(d, w.w_uk_t);
  m.w_uv = ResidentWeight(d, w.w_uv);
  m.o_proj = ResidentWeight(d, w.o_proj);
  m.rope_cos_sin_cache = rope_cache;
  return m;
}

// `DeepseekV2MLP.forward` (deepseek_v2.py:270-274): merged gate_up GEMM ->
// SiluAndMul -> down GEMM. Identical op shape to the Qwen3-dense MlpBlock.
DBuf DenseMlp(Dev d, const DeepseekV2DenseMlp& w, const Tensor& dh, int64_t T,
              int64_t H, int64_t I) {
  // gate_up MatmulBT -> SiluAndMul via the SHARED bf16 gate-up MLP seam
  // (layers::UnquantizedMlpGateUpMethod). Byte-for-byte the same op sequence the
  // inline path ran; this single fold covers BOTH the dense-layer MLP and the
  // shared-expert epilogue (the two DenseMlp call sites), and inherits the nvfp4
  // GateUpFusedMarlinD arm for free on a quantized checkpoint. (Tier-A1 fold,
  // arch-fusion-fold-plan-2026-07-30.)
  DBuf act = layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I).Apply(d, dh);
  Tensor wdn = ResidentWeight(d, w.down_proj);  // [H, I] raw-NK
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wdn);
  return out;
}

// Per-layer RESIDENT expert device-pointer arrays for the grouped bf16 MoE GEMM
// (the DeepSeek analog of qwen3_5.cpp's MoeBf16Resident). Uploaded ONCE at first
// touch, keyed on the layer's weight-struct address, and never freed for the
// process lifetime — the same retire-don't-free contract the existing grouped
// MoE paths rely on so nothing can dangle.
struct MoePtrs {
  void* gate = nullptr;
  void* up = nullptr;
  void* down = nullptr;
  std::map<int64_t, void*> tok_map;  // T -> [T*top_k] i32 pair->token row map
  bool ready = false;
};
MoePtrs& MoePtrsFor(const DeepseekV2MoeWeights* key) {
  static std::map<const DeepseekV2MoeWeights*, MoePtrs> table;
  return table[key];
}

bool GroupedMoeEligible(Dev d, const DeepseekV2MoeWeights& w) {
  // S7: this gates availability of the bf16 grouped-GEMM MoE kernel, which is
  // registered ONLY for CUDA (cuda_matmul_nvfp4.cu:2799). `vt::OpRegistered` is
  // therefore byte-identical to the old `is_cuda()` (true iff the op is realized
  // for this device — kCUDA today, on every build) and drops the device token —
  // exactly the S4 class-A conversion (audit §9.2), safe because the op is
  // CUDA-only (not the dual-registered fp4/fp8 ops S6 had to leave).
  if (!vt::OpRegistered(vt::OpId::kMoeGroupedGemmBf16, d.q.device.type))
    return false;
  if (w.expert_gate.empty()) return false;
  // vt::MoeGroupedGemmBf16 reads the [K,N] Matmul-B orientation only.
  for (const OwnedTensor* t : {&w.expert_gate[0], &w.expert_up[0], &w.expert_down[0]})
    if (t->nk) return false;
  return true;
}

// `DeepseekV2MoE.forward` (deepseek_v2.py:395-424) + the FusedMoE runner's
// routed/shared composition (moe_runner.py:392-407).
DBuf MoeBlock(Dev d, const DeepseekV2MoeWeights& w, const DeepseekV2Params& p,
              const Tensor& dh, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.n_routed_experts;
  const int64_t top_k = p.num_experts_per_tok;
  const int64_t I = p.moe_intermediate_size;
  const int64_t P = T * top_k;

  // --- router: logits = gate(hidden) then grouped top-k ---------------------
  // `router_logits, _ = self.gate(hidden_states)` (:417). The gate is a
  // ReplicatedLinear whose OUTPUT dtype is `_get_moe_router_dtype(config)`
  // (`:309-314`, `GateLinear(..., out_dtype=self.router_dtype)`), which is f32
  // for `model_type == "glm_moe_dsa"` (`:127`) and for any config declaring
  // `moe_router_dtype: "float32"` (`:131`), and None — i.e. the model dtype —
  // otherwise. DeepSeek-V2 and V2-Lite declare neither and stay bf16.
  //
  // THE ANNOTATED f32, and the one buffer on this path that is deliberately
  // wider than the checkpoint's dtype. `vt::MoeRouterTopK` already accepts any
  // float logits dtype and already scores in f32, so widening the STORE is the
  // whole delta: it is the rounding between the GEMM and the top-k that
  // upstream removes, not the arithmetic inside either.
  //
  // The three anchors in the previous version of this comment were all stale —
  // `:413` for the gate call (now `:417`) and `:120-130` for
  // `_get_moe_router_dtype` (now `:123-133`) — and they were correct at the
  // PRIOR pin. Re-read at `5559679229` on 2026-08-30.
  Tensor drg = ResidentWeight(d, w.router_gate);  // [H,E] Matmul-B
  DBuf dlog(d, p.router_dtype_is_f32 ? DType::kF32 : DType::kBF16, {T, E});
  vt::Matmul(d.q, dlog.t(), dh, drg);

  vt::MoeRouterTopKArgs args{};
  args.top_k = static_cast<int>(top_k);
  args.renormalize = p.norm_topk_prob;
  args.scoring_func = p.scoring_func;
  args.num_expert_group = static_cast<int>(p.n_group);
  args.topk_group = static_cast<int>(p.topk_group);
  args.routed_scaling_factor = p.routed_scaling_factor;  // DEVIATION (a)
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  std::unique_ptr<Tensor> bias;
  if (!w.e_score_correction_bias.Empty()) {
    bias = std::make_unique<Tensor>(ResidentWeight(d, w.e_score_correction_bias, {E}));
  }
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(), args, bias.get());

  // --- routed experts -------------------------------------------------------
  DBuf expert_out(d, DType::kBF16, {T, top_k, H});
  if (GroupedMoeEligible(d, w)) {
    // ~3 grouped bf16 GEMM launches, fully on device (the same op sequence the
    // gated Qwen3-Coder bf16 MoE runs: vt::MoeGroupedGemmBf16 x2 -> MoeSiluMul
    // -> vt::MoeGroupedGemmBf16). The top-k ids [T,top_k] ARE the per-pair
    // expert ids viewed [P].
    MoePtrs& mr = MoePtrsFor(&w);
    if (!mr.ready) {
      std::vector<int64_t> gp(static_cast<size_t>(E)), up(static_cast<size_t>(E)),
          dp(static_cast<size_t>(E));
      for (int64_t e = 0; e < E; ++e) {
        const size_t se = static_cast<size_t>(e);
        gp[se] = reinterpret_cast<int64_t>(ResidentWeight(d, w.expert_gate[se]).data);
        up[se] = reinterpret_cast<int64_t>(ResidentWeight(d, w.expert_up[se]).data);
        dp[se] = reinterpret_cast<int64_t>(ResidentWeight(d, w.expert_down[se]).data);
      }
      const size_t eb = static_cast<size_t>(E) * sizeof(int64_t);
      auto upload = [&](const std::vector<int64_t>& h) {
        void* q = d.b.Alloc(eb);
        d.b.Copy(d.q, q, h.data(), eb);
        return q;
      };
      mr.gate = upload(gp);
      mr.up = upload(up);
      mr.down = upload(dp);
      mr.ready = true;
    }
    auto tok_it = mr.tok_map.find(T);
    if (tok_it == mr.tok_map.end()) {
      std::vector<int32_t> tok_map(static_cast<size_t>(P));
      for (int64_t i = 0; i < P; ++i)
        tok_map[static_cast<size_t>(i)] = static_cast<int32_t>(i / top_k);
      const size_t tb = static_cast<size_t>(P) * sizeof(int32_t);
      void* q = d.b.Alloc(tb);
      d.b.Copy(d.q, q, tok_map.data(), tb);
      tok_it = mr.tok_map.emplace(T, q).first;
    }
    Tensor eids = Reshape(dtid.t(), {P});
    Tensor gate_ptrs = MakeTensor(mr.gate, DType::kI64, d.q.device, {E});
    Tensor up_ptrs = MakeTensor(mr.up, DType::kI64, d.q.device, {E});
    Tensor down_ptrs = MakeTensor(mr.down, DType::kI64, d.q.device, {E});
    Tensor dtok = MakeTensor(tok_it->second, DType::kI32, d.q.device, {P});
    // Tier-A4 fold: fused gate+up GEMM + SwiGLU in ONE vt op (the same shared op
    // Qwen3-Coder routes through), BIT-IDENTICAL to the old {gate GEMM; up GEMM;
    // MoeSiluMul} — decode fuses the two f32 [P,I] round-trips away, prefill reuses
    // the tuned grouped GEMM twice + the identical silu-mul. Then the down GEMM.
    DBuf dact(d, DType::kBF16, {P, I});
    vt::MoeGroupedGemmBf16GateUpSilu(d.q, dact.t(), dh, eids, &dtok, gate_ptrs, up_ptrs);
    Tensor eo = Reshape(expert_out.t(), {P, H});
    vt::MoeGroupedGemmBf16(d.q, eo, dact.t(), eids, nullptr, down_ptrs);
  } else {
    // REFERENCE path (CPU, and any non-Matmul-B expert layout): download the
    // routing decision, gather each activated expert's token rows, run its
    // SwiGLU MLP, scatter back. Same numerics per (token, slot) as the grouped
    // path modulo GEMM accumulation order.
    std::vector<int32_t> ids(static_cast<size_t>(P));
    dtid.Download(d, ids.data());
    std::vector<std::vector<std::pair<int64_t, int64_t>>> lists(
        static_cast<size_t>(E));
    for (int64_t t = 0; t < T; ++t)
      for (int64_t j = 0; j < top_k; ++j)
        lists[static_cast<size_t>(ids[static_cast<size_t>(t * top_k + j)])]
            .push_back({t, j});
    expert_out.Zero(d);
    const size_t row_bytes = static_cast<size_t>(H) * vt::SizeOf(DType::kBF16);
    for (int64_t e = 0; e < E; ++e) {
      const auto& list = lists[static_cast<size_t>(e)];
      if (list.empty()) continue;
      const int64_t n = static_cast<int64_t>(list.size());
      DBuf xg(d, DType::kBF16, {n, H});
      for (int64_t r = 0; r < n; ++r) {
        d.b.Copy(d.q, static_cast<char*>(xg.ptr()) + static_cast<size_t>(r) * row_bytes,
                 static_cast<const char*>(dh.data) +
                     static_cast<size_t>(list[static_cast<size_t>(r)].first) * row_bytes,
                 row_bytes);
      }
      DBuf g(d, DType::kBF16, {n, I}), u(d, DType::kBF16, {n, I});
      vt::Matmul(d.q, g.t(), xg.t(), ResidentWeight(d, w.expert_gate[static_cast<size_t>(e)]));
      vt::Matmul(d.q, u.t(), xg.t(), ResidentWeight(d, w.expert_up[static_cast<size_t>(e)]));
      DBuf a(d, DType::kBF16, {n, I});
      vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());
      DBuf o(d, DType::kBF16, {n, H});
      vt::Matmul(d.q, o.t(), a.t(), ResidentWeight(d, w.expert_down[static_cast<size_t>(e)]));
      for (int64_t r = 0; r < n; ++r) {
        const auto& tj = list[static_cast<size_t>(r)];
        d.b.Copy(d.q,
                 static_cast<char*>(expert_out.ptr()) +
                     static_cast<size_t>(tj.first * top_k + tj.second) * row_bytes,
                 static_cast<const char*>(o.ptr()) + static_cast<size_t>(r) * row_bytes,
                 row_bytes);
      }
    }
  }

  // --- shared experts + weighted combine ------------------------------------
  // DeepSeek's shared expert is a PLAIN MLP whose output is added to the routed
  // sum (deepseek_v2.py:344-357; moe_runner.py:407 `shared_output + fused_output`)
  // — NO sigmoid gate, unlike Qwen3.6's. `vt::MoeCombine`'s optional `shared`
  // term is exactly that add.
  const bool has_shared = p.n_shared_experts > 0 && !w.shared.Empty();
  DBuf out(d, DType::kBF16, {T, H});
  if (has_shared) {
    DBuf shared = DenseMlp(d, w.shared, dh, T, H, p.shared_intermediate_size());
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), &shared.t());
  } else {
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), nullptr);
  }
  return out;
}

// One `DeepseekV2DecoderLayer` (deepseek_v2.py:1262-1345). The bf16 residual
// stream is threaded through the two fused add+RMSNorm producers exactly as in
// every other model in this tree; the FP16-overflow rescale at :1298-1308 /
// :1338-1343 is guarded on `hidden_states.dtype == torch.float16` and is
// therefore INACTIVE for a bf16 model (we are bf16), so it is not ported.
void RunLayer(Dev d, const DeepseekV2LayerWeights& layer, const DeepseekV2Params& p,
              Tensor& hidden, std::shared_ptr<void>& hidden_hold, DBuf& res,
              const MlaStep& step, Tensor& kv_cache, v1::TritonMLAImpl& impl,
              int64_t T, int64_t layer_index) {
  const int64_t H = p.hidden_size;
  const float eps = p.rms_norm_eps;

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dhn.t(), hidden, w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dhn.t(), hidden, w_in, vt::RmsNormArgs{eps, false}, &res.t());
  }

  DBuf attn(d, DType::kBF16, {T, H});
  Tensor attn_t = attn.t();
  const mla::MlaBlockWeights mw = ResidentMla(d, layer.attn, p, *step.rope_cache);
  mla::ForwardMlaAttentionBlock(d, p.mla, mw, dhn.t(), step.positions, kv_cache,
                                step.slot_mapping, step.meta, impl, attn_t);

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
  }

  // `self.mlp(hidden_states)` — MoE or dense, per first_k_dense_replace (:1214).
  // ENG-HYBRID-PLACEMENT W3d: the MoE arm goes through the shared seam, inert by
  // construction when this layer is not placed. The DENSE arm is untouched —
  // placement moves ROUTED EXPERTS, and a dense MLP has none.
  DBuf mlp = layer.is_moe
                 ? vllm::RunMoePlaced(d, layer_index, dh2.t(), T, H,
                                      [&](Dev pd, const Tensor& h) {
                                        return MoeBlock(pd, layer.moe, p, h, T);
                                      })
                 : DenseMlp(d, layer.dense, dh2.t(), T, H, p.intermediate_size);
  // The MLP output becomes the new residual-stream delta; keep its storage alive
  // until the next producer replaces it.
  auto* held = new DBuf(std::move(mlp));
  hidden = held->t();
  hidden_hold = std::shared_ptr<void>(held, [](void* q) { delete static_cast<DBuf*>(q); });
}

void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// The EMBED step, hoisted out of the layer region so it can stay OUTSIDE a CUDA
// graph capture (the embedding path takes a device flag through a cudaMalloc +
// stream sync — the same reason qwen3_moe.cpp:200 keeps `EmbedInto` outside).
// #1305 — THE SCOPED DEVICE-ID OVERRIDE is consumed here, once per forward.
// Both registrations that reach this TU (`DeepseekV2ForCausalLM` and
// `Glm4MoeLiteForCausalLM`) publish `ModelForwardInput::device_token_ids` through
// `detail::DeviceTokenIdsScope`; it is null on every path except the asynchronous
// CUDA runner, where the combine has already spliced each decode row's sampled
// token into the DEVICE identifiers and left the host vector deliberately stale.
// Both readers below — `detail::TakeDeviceTokenIds` on the graph path and
// `detail::ApplyDeviceTokenIds` on the eager one — CLEAR it, so the first embed
// in a forward is the one that gets it. Both live in `qwen3_5_internal.h` beside
// the scope that publishes them, because four models consume it and each used to
// spell the same two bodies out for itself.
// EMBED FROM AN ALREADY-RESIDENT ID TENSOR. The decode-graph driver holds its
// identifiers in a `vllm::StepTokenIds` whose device address is stable for the
// life of the slot, so this arm takes the tensor instead of re-uploading a host
// vector into a fresh per-step allocation.
void EmbedInto(Dev d, DBuf& hidden, const Tensor& ids,
               const DeepseekV2Weights& weights) {
  const DeepseekV2Params& p = weights.params;
  Tensor dtab = ResidentWeight(d, weights.embed_tokens, {p.vocab_size, p.hidden_size});
  Tensor h = hidden.t();
  vt::Embedding(d.q, h, dtab, ids);
}

void EmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& token_ids,
               const DeepseekV2Weights& weights) {
  const DeepseekV2Params& p = weights.params;
  EmbedGather(d, hidden, token_ids, weights.embed_tokens, p.vocab_size,
              p.hidden_size, "deepseek v2 embed");
}

// The CAPTURABLE region: everything after the embedding — the MLA step metadata
// upload, the 27 decoder layers, the final norm and the lm_head GEMM. Takes the
// already-embedded hidden states as a Tensor so the graph driver can hand it a
// PERSISTENT buffer whose address the captured kernels bake in.
DBuf ForwardLayers(Dev d, const Tensor& hidden_in,
                   const std::vector<int32_t>& positions,
                   const CommonAttentionMetadata& am,
                   const std::vector<PagedKvCache>& attn_kv,
                   const DeepseekV2Weights& weights,
                   const std::vector<int32_t>& logits_indices) {
  const DeepseekV2Params& p = weights.params;
  const int64_t T = hidden_in.shape[0];
  const int64_t H = p.hidden_size;
  const int64_t vocab = p.vocab_size;
  const float eps = p.rms_norm_eps;
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "deepseek-v2: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(p.num_hidden_layers),
           "deepseek-v2: one MLA PagedKvCache per layer required");
  VT_CHECK(T > 0, "deepseek-v2: empty batch");

  Tensor hidden = hidden_in;
  std::shared_ptr<void> hidden_hold;

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  const int64_t block_size = attn_kv[0].block_size;
  MlaStep step = BuildMlaStep(d, positions, am, block_size, p.max_position_embeddings);
  // The rope cache is per-MODEL, not per-step; hand the block a device view.
  const Tensor rope = ResidentWeight(d, weights.rope_cos_sin_cache);
  step.rope_cache = &rope;

  v1::TritonMLAImpl impl;
  const int64_t head_size = p.mla.head_size();
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const PagedKvCache& kv = attn_kv[static_cast<size_t>(l)];
    VT_CHECK(kv.num_kv_heads == 1 && kv.head_size == head_size,
             "deepseek-v2: the MLA cache must be 1-head, "
             "kv_lora_rank + qk_rope_head_dim wide (MLAAttentionSpec)");
    Tensor kv_cache = MakeTensor(kv.data, kv.dtype, d.q.device,
                                 {kv.num_blocks, kv.block_size, head_size});
    RunLayer(d, weights.layers[static_cast<size_t>(l)], p, hidden, hidden_hold, res,
             step, kv_cache, impl, T, /*layer_index=*/l);
  }

  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dnorm.t(), hidden, w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dnorm.t(), hidden, w_fn, vt::RmsNormArgs{eps, false}, &res.t());
  }

  const bool tied = p.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);
  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kBF16,
               do_gather ? std::vector<int64_t>{
                               static_cast<int64_t>(logits_indices.size()), H}
                         : std::vector<int64_t>{1, 1});
  if (do_gather) {
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];
  DBuf logits(d, DType::kF32, {n_out, vocab});
  if (tied)
    vt::MatmulBT(d.q, logits.t(), src, lm);
  else
    vt::Matmul(d.q, logits.t(), src, lm);
  return logits;
}

// Full eager forward: embed then the capturable layer region. Used by
// DeepseekV2Model::Forward/ForwardDevice and by the graph driver's eager
// fallback + cold-size pre-warm step.
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& am,
                 const std::vector<PagedKvCache>& attn_kv,
                 const DeepseekV2Weights& weights,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  DBuf hidden(d, DType::kBF16, {T, weights.params.hidden_size});
  EmbedInto(d, hidden, token_ids, weights);
  return ForwardLayers(d, hidden.t(), positions, am, attn_kv, weights,
                       logits_indices);
}

ForwardLogits WrapDeviceLogits(DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  fl.device_storage = dlogits.ReleaseShared();
  return fl;
}

// ─── W9 decode-CUDA-graph support helpers ───────────────────────────────────
// Byte-for-byte the qwen3_moe.cpp:232-303 helpers (ViewDeviceLogits /
// CopyInPlace / BuildPaddedDecodeAttn), re-expressed here because they are file
// -local to that TU. The padding contract is IDENTICAL and it is identical for a
// reason: it is the contract of `CommonAttentionMetadata` itself, not of any
// particular attention backend.

// NON-OWNING [rows, vocab] f32 view over a buffer the graph slot keeps alive.
// Stream ordering guarantees the sampler's later reads see the replay's writes;
// the next same-size replay overwrites the buffer, so in-place sampler mutation
// is safe.
ForwardLogits ViewDeviceLogits(void* base, vt::Device device, int64_t rows,
                               int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = MakeTensor(base, DType::kF32, device, {rows, vocab});
  fl.device_storage = std::shared_ptr<void>(base, [](void*) {});
  return fl;
}

// Overwrite dst's CONTENTS without moving dst.data() when the sizes already
// match — the captured host->device copies bake the source address in.
template <typename T>
void CopyInPlace(std::vector<T>& dst, const std::vector<T>& src) {
  if (dst.size() != src.size()) {
    dst = src;
  } else {
    std::copy(src.begin(), src.end(), dst.begin());
  }
}

// Build the S-padded PURE-DECODE inputs from the real B-request step (B <= S).
//
// WHY THE PADDING IS INERT FOR **MLA** SPECIFICALLY (this is the part that does
// NOT come for free from the Qwen3-Coder argument, and it was re-derived rather
// than assumed):
//   * `BuildMlaBatchSplit` sees query_len == 1 for EVERY row (real and padded),
//     so the split is `num_decodes == S, num_prefills == 0` — the padded rows
//     land in the DECODE half, the only half a pure-decode step has, and the
//     with-context prefill ordering invariant is vacuous.
//   * The absorbed MQA decode kernel (`vt::MlaDecodeAttention`) is per-request:
//     each query row reduces only over ITS OWN `seq_lens[r]` tokens taken through
//     ITS OWN `block_table[r]` row. `seq_lens = 1` + block-table row 0 is an
//     in-bounds single-token read whose output row is discarded.
//   * `slot_mapping = -1` makes the MLA cache write (`vt::MlaCacheWrite`, the W3
//     op) SKIP the row exactly as `ReshapeAndCache` does, so no real KV page is
//     perturbed — this is the property that makes padding safe to REPLAY, not
//     merely safe to compute.
//   * Everything else in the step (both RMSNorms, the A/B projections, the two
//     batched absorption GEMMs, the router, the grouped MoE GEMM, the shared
//     expert, the lm_head) is per-token with no cross-row reduction.
// At S == B this is a bit-identical rebuild of the eager inputs.
void BuildPaddedDecodeAttn(int64_t S, const std::vector<int32_t>& tok,
                           const std::vector<int32_t>& pos,
                           const CommonAttentionMetadata& am,
                           std::vector<int32_t>& tok_out,
                           std::vector<int32_t>& pos_out,
                           CommonAttentionMetadata& am_out) {
  const int64_t cols = am.block_table_num_cols;

  tok_out.assign(static_cast<size_t>(S), 0);
  pos_out.assign(static_cast<size_t>(S), 0);
  std::copy(tok.begin(), tok.end(), tok_out.begin());
  std::copy(pos.begin(), pos.end(), pos_out.begin());

  am_out = am;  // carries causal + block_table_num_cols + max_seq_len
  am_out.num_reqs = static_cast<int>(S);
  am_out.num_actual_tokens = static_cast<int>(S);
  am_out.max_query_len = 1;  // pure decode
  // W10 (#1857): a pure-decode rewrite is never spec-classified. Belt on the
  // vt shape guard's braces (S == q*S only at q == 1).
  am_out.uniform_spec_query_len = 0;
  am_out.slot_mapping.assign(static_cast<size_t>(S), -1);
  std::copy(am.slot_mapping.begin(), am.slot_mapping.end(),
            am_out.slot_mapping.begin());
  am_out.seq_lens.assign(static_cast<size_t>(S), 1);
  std::copy(am.seq_lens.begin(), am.seq_lens.end(), am_out.seq_lens.begin());
  am_out.block_table_tensor.assign(static_cast<size_t>(S * cols), 0);
  std::copy(am.block_table_tensor.begin(), am.block_table_tensor.end(),
            am_out.block_table_tensor.begin());
  am_out.query_start_loc.resize(static_cast<size_t>(S + 1));
  for (int64_t i = 0; i <= S; ++i)
    am_out.query_start_loc[static_cast<size_t>(i)] = static_cast<int32_t>(i);
}

}  // namespace

// ─── W8: the split-shape counters (deepseek_v2.h) ───────────────────────────
const MlaBatchSplitStats& GetMlaBatchSplitStats() {
  return MutableMlaBatchSplitStats();
}
void ResetMlaBatchSplitStats() { MutableMlaBatchSplitStats() = MlaBatchSplitStats{}; }

// ─── the batch split + its ORDERING invariant ───────────────────────────────
MlaBatchSplit BuildMlaBatchSplit(const CommonAttentionMetadata& am) {
  MlaBatchSplit s;
  const int num_reqs = am.num_reqs;
  VT_CHECK(num_reqs >= 0, "deepseek-v2: negative num_reqs");
  VT_CHECK(static_cast<int>(am.query_start_loc.size()) == num_reqs + 1,
           "deepseek-v2: query_start_loc must have num_reqs + 1 entries");
  VT_CHECK(static_cast<int>(am.seq_lens.size()) >= num_reqs,
           "deepseek-v2: seq_lens must have at least num_reqs entries");

  // `split_decodes_and_prefills(..., decode_threshold=reorder_batch_threshold)`
  // with `reorder_batch_threshold == 1` (mla_attention.py:1420): a request is a
  // DECODE iff its query length is <= 1, and decodes must form a PREFIX.
  bool seen_prefill = false;
  for (int r = 0; r < num_reqs; ++r) {
    const int32_t q_len = am.query_start_loc[static_cast<size_t>(r) + 1] -
                          am.query_start_loc[static_cast<size_t>(r)];
    VT_CHECK(q_len >= 0, "deepseek-v2: non-monotonic query_start_loc");
    const bool is_decode = q_len <= 1;
    if (is_decode) {
      VT_CHECK(!seen_prefill,
               "deepseek-v2 MLA batch ordering: request " + std::to_string(r) +
                   " is a DECODE (query_len <= 1) but follows a PREFILL. MLA "
                   "requires decodes to be packed FIRST "
                   "(reorder_batch_to_split_decodes_and_prefills; the block "
                   "slices q[:num_mqa_tokens] / q[num_mqa_tokens:], "
                   "mla_attention.py:700-737).");
      ++s.num_decodes;
      s.num_decode_tokens += q_len;
      s.decode_max_seq_len =
          std::max(s.decode_max_seq_len, am.seq_lens[static_cast<size_t>(r)]);
    } else {
      seen_prefill = true;
      ++s.num_prefills;
      s.num_prefill_tokens += q_len;
      s.prefill_max_query_len = std::max(s.prefill_max_query_len, q_len);
    }
  }

  // The prefill tail, RELATIVE to its own start (mla_attention.py:1670-1675).
  s.prefill_cu_seqlens_q.reserve(static_cast<size_t>(s.num_prefills) + 1);
  s.prefill_cu_seqlens_q.push_back(0);
  bool seen_without_context = false;
  for (int r = s.num_decodes; r < num_reqs; ++r) {
    const int32_t q_len = am.query_start_loc[static_cast<size_t>(r) + 1] -
                          am.query_start_loc[static_cast<size_t>(r)];
    s.prefill_cu_seqlens_q.push_back(s.prefill_cu_seqlens_q.back() + q_len);
    const int32_t ctx = am.seq_lens[static_cast<size_t>(r)] - q_len;
    VT_CHECK(ctx >= 0,
             "deepseek-v2: request " + std::to_string(r) +
                 " has seq_len < query_len (negative context)");
    s.prefill_context_lens.push_back(ctx);
    // THE W6 INVARIANT: `prefill_tokens_with_context =
    // prefill_query_start_loc_cpu[num_prefills_with_context]`
    // (mla_attention.py:1806-1810) is a PREFIX LENGTH, so every with-context
    // prefill must precede every context-free one. A batch that violates this
    // measured 0.86 relative error in W6's gate — fail loudly instead.
    if (ctx > 0) {
      VT_CHECK(!seen_without_context,
               "deepseek-v2 MLA batch ordering: prefill request " +
                   std::to_string(r) +
                   " HAS context but follows a context-free prefill. MLA "
                   "requires with-context prefills to LEAD the prefill tail "
                   "(prefill_tokens_with_context is a prefix length, "
                   "mla_attention.py:1806-1810).");
      ++s.num_prefills_with_context;
    } else {
      seen_without_context = true;
    }
  }
  VT_CHECK(s.num_decode_tokens + s.num_prefill_tokens == am.num_actual_tokens,
           "deepseek-v2: query_start_loc does not sum to num_actual_tokens");
  return s;
}

std::vector<float> DeepseekV2Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV2Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * weights.params.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits DeepseekV2Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV2Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits =
      ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(std::move(dlogits), n_out, weights.params.vocab_size);
}

// ─── DeepseekV2DecodeGraph (MLA decode CUDA-graph driver) — MLA campaign W9 ───
//
// The MLA sibling of Qwen3MoeDecodeGraph (qwen3_moe.cpp:374), Qwen3_5DecodeGraph
// (qwen3_5.cpp:5902) and Qwen3_5DenseDecodeGraph (qwen3_5.cpp:6104): the SAME
// cold -> warm -> capture -> replay state machine, the SAME padded-batch capture
// set (decode_graph_sizes.h, mirroring vLLM `_set_cudagraph_sizes` reduced to the
// full-decode-cudagraph regime + `compilation/cuda_graph.py`'s pad-to-nearest
// dispatch), the SAME persistent fixed-address host inputs and persistent
// embed/logits buffers — driving the MLA + DeepSeek-MoE decode forward.
//
// Ported from: vllm/v1/worker/gpu_model_runner.py::GPUModelRunner @ e24d1b24
//   (`_dummy_run` warm-up then capture, then graph dispatch per decode step) +
//   vllm/compilation/cuda_graph.py (`CUDAGraphWrapper.__call__`: pad the batch to
//   a captured size, replay, else run eager).
//
// GRAPH-SAFETY AUDIT of the MLA decode path (capture requires stable pointers, no
// host sync and no stream-ordered alloc inside the region). What is NEW versus
// the three existing drivers is the MLA half; the rest is shared machinery:
//   * Embedding (device flag cudaMalloc + stream sync) stays OUTSIDE (EmbedInto).
//   * `BuildMlaStep`'s metadata uploads all come from the slot's PERSISTENT host
//     vectors, so the captured H2D copies read fixed addresses. A block-table
//     column-count change reallocates one of them -> the slot's graph is dropped
//     and re-captured (the `cols_changed` guard below), exactly as the FA-2
//     drivers do.
//   * On a PURE-DECODE step the split is num_prefills == 0, so the ENTIRE chunked
//     -context branch of BuildMlaStep (the workspace sizing, the per-chunk
//     descriptor uploads, `ComputeMlaPrefillContext`) is not reached. The
//     capturable region is therefore only the decode half — no data-dependent
//     chunk count, no host-visible loop bound.
//   * `vt::MlaDecodeAttention`'s split-KV partial workspace comes from the shared
//     DevicePool, whose blocks are recycled and never returned to the driver; the
//     cold pre-warm step at this exact padded size populates every size class the
//     capture then reuses, so the capture itself performs no cudaMalloc. Its
//     split count is derived from the HOST `max_seq_len`, which only sizes the
//     grid — each split's KV range is recomputed in-kernel from the DEVICE
//     `seq_lens`, so a captured graph stays CORRECT as sequences grow. This is
//     the identical contract the already-gated FA-2 decode graphs rely on.
//   * The MoE grouped-GEMM index scratch and split-K partials are graph-safe by
//     design (retire-don't-free), and the per-layer expert device-pointer arrays
//     + pair->token row map (`MoePtrs`) are uploaded once at first touch during
//     the pre-warm step.
//   * `ResidentWeight` uploads every weight once, on first touch (pre-warm) —
//     including the load-time-absorbed `W_UK`/`W_UV` and the YaRN rope cache.
//   * cuBLASLt's workspace is a one-time per-context cudaMalloc.
//
// NUMERICS: a replay is the same kernels in the same order over the same buffers,
// so the real rows are BIT-IDENTICAL to the eager forward. The W8 SACRED gate is
// re-run with this default-ON and must stay 8/8; it does.
struct DeepseekV2DecodeGraph::Impl {
  Impl(const DeepseekV2Weights& w, vt::Queue q, int64_t max_reqs)
      : weights(w), queue(q), max_num_reqs(max_reqs) {
    // ENG-CUDAGRAPH-BREAK W3 (#1291): the FRAMEWORK-WIDE switch is the SEAM's,
    // not this driver's. `vt::GraphCaptureEnabled()` reads `VLLM_CPP_CUDAGRAPH`
    // once per process into a function-local static; six drivers each read that
    // variable for themselves before this row, so there was no one switch that
    // turned capture off. The DeepSeek-LOCAL rollback stays here, because it is a
    // per-model A/B lever for exactly this driver (W9) and not a copy of the
    // shared one.
    const char* local = std::getenv("VT_DEEPSEEK_CUDAGRAPH");
    const bool local_on = (local == nullptr) || local[0] != '0';
    Backend& b = vt::GetBackend(queue.device.type);
    // The TT captured arm of this family is not gated (#2812): explicit opt-in only.
    enabled = vt::GraphCaptureEnabled() && local_on &&
              platforms::GetPlatform(queue.device.type).support_static_graph_mode() &&
              !platforms::GetPlatform(queue.device.type).static_graph_requires_opt_in() &&
              b.SupportsGraphCapture();
  }
  ~Impl() {
    // No DestroyGraph loop: every segment handle belongs to the slot's
    // `vt::BreakableGraph`, whose destructor releases it through
    // `Backend::DestroyGraph`. That routing is what lets ENG-CUDAGRAPH-DEDUP
    // (#1162) interpose at the backend later without editing this driver.
  }

  struct SizeSlot {
    std::vector<int32_t> token_ids;  // [S]
    std::vector<int32_t> positions;  // [S]
    CommonAttentionMetadata attn_meta;
    std::unique_ptr<DBuf> hidden;  // [S,H] bf16 persistent embed target
    std::unique_ptr<DBuf> logits;  // [S,vocab] f32 held graph output
    // #1305: this slot's per-step input IDENTIFIERS, in a device buffer whose
    // address does not move, refreshed through `vt::PersistentStepInput` from the
    // padded host vector and then from the runner's device mirror when it is
    // live. Before it, every arm below re-uploaded the HOST vector into a fresh
    // allocation and the mirror was never read at all.
    StepTokenIds ids;
    // ENG-CUDAGRAPH-BREAK W3 (#1291): the instantiated graph, its handle
    // ownership, its release and its `captured()` state now live in the shared
    // seam instead of in a raw `void*` plus a `bool` this driver maintained by
    // hand. `vt::BreakableGraph` is non-copyable and is constructed in place by
    // `slots[S]`, so the map still owns one per padded size.
    vt::BreakableGraph graph;
    int bt_cols = -1;  // captured block-table column count
    bool warm = false;
    int64_t replays = 0;

    void Refresh(const std::vector<int32_t>& tok, const std::vector<int32_t>& pos,
                 const CommonAttentionMetadata& am) {
      CopyInPlace(token_ids, tok);
      CopyInPlace(positions, pos);
      CopyInPlace(attn_meta.slot_mapping, am.slot_mapping);
      CopyInPlace(attn_meta.block_table_tensor, am.block_table_tensor);
      CopyInPlace(attn_meta.seq_lens, am.seq_lens);
      CopyInPlace(attn_meta.query_start_loc, am.query_start_loc);
      attn_meta.num_reqs = am.num_reqs;
      attn_meta.num_actual_tokens = am.num_actual_tokens;
      attn_meta.max_query_len = am.max_query_len;
      attn_meta.max_seq_len = am.max_seq_len;
      attn_meta.block_table_num_cols = am.block_table_num_cols;
      attn_meta.causal = am.causal;
    }
  };

  const DeepseekV2Weights& weights;
  vt::Queue queue;
  int64_t max_num_reqs = 0;  // == max_num_seqs; padded decode batch cap
  bool enabled = false;

  std::map<int64_t, SizeSlot> slots;  // padded size S -> slot
  int64_t replays = 0;
  bool any_captured = false;
};

DeepseekV2DecodeGraph::DeepseekV2DecodeGraph(const DeepseekV2Weights& weights,
                                             vt::Queue queue, int64_t max_num_reqs)
    : impl_(std::make_unique<Impl>(weights, queue, max_num_reqs)) {}

DeepseekV2DecodeGraph::~DeepseekV2DecodeGraph() = default;

bool DeepseekV2DecodeGraph::captured() const { return impl_->any_captured; }
int64_t DeepseekV2DecodeGraph::replay_count() const { return impl_->replays; }

ForwardLogits DeepseekV2DecodeGraph::Step(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv) {
  const int64_t B = static_cast<int64_t>(token_ids.size());
  Backend& b = vt::GetBackend(impl_->queue.device.type);
  Dev d{b, impl_->queue};
  const int64_t vocab = impl_->weights.params.vocab_size;
  const int64_t H = impl_->weights.params.hidden_size;

  // Pure decode passes identity logits_indices (gather is a no-op), so the
  // capturable region returns the full [S,vocab].
  const std::vector<int32_t> kNoGather;
  const int64_t S = PadToCaptureSize(B, impl_->max_num_reqs);
  if (!impl_->enabled || S < 0) {
    DBuf lg = ForwardBody(d, token_ids, positions, attn_meta, attn_kv,
                          impl_->weights, kNoGather);
    return WrapDeviceLogits(std::move(lg), B, vocab);
  }

  Impl::SizeSlot& s = impl_->slots[S];
  const int cols = attn_meta.block_table_num_cols;
  std::vector<int32_t> ptok, ppos;
  CommonAttentionMetadata pam;
  BuildPaddedDecodeAttn(S, token_ids, positions, attn_meta, ptok, ppos, pam);

  const bool cols_changed = (s.bt_cols != -1 && s.bt_cols != cols);
  s.Refresh(ptok, ppos, pam);
  s.bt_cols = cols;
  // #1305 — THE IDENTIFIERS, ON THE SEAM. Bind this padded size's device
  // destination once, then refresh it for this step: the padded host vector
  // first (authoritative for the inert padding rows and for every row when no
  // mirror is live), and then the runner's DEVICE identifiers over the real
  // prefix when the asynchronous combine has patched them there. Both copies are
  // enqueued on the main queue, so the second is ordered after the combine
  // instead of racing it, and all three arms below embed from the SAME stable
  // address.
  const detail::DeviceTokenIds ov = detail::TakeDeviceTokenIds();
  s.ids.Ensure(d, S);
  s.ids.Refresh(d, s.token_ids, ov.ids, ov.count);
  if (cols_changed && s.graph.captured()) {
    // Reset() releases every segment through Backend::DestroyGraph and returns
    // the container to its as-constructed state, which is also what lets the
    // next capture open a scope on it (the scope refuses a container that
    // already holds one).
    s.graph.Reset();
    s.warm = false;
  }

  // Fast path: replay. `BuildMlaStep` (and with it `RecordMlaBatchSplit`) does
  // NOT run on a replay, so the W8 split-shape counters are recorded here from
  // the SAME padded metadata the captured region was built from — the diagnostic
  // stays exactly as informative as it is on the eager path, which is what the
  // paged-engine gate asserts on.
  if (s.graph.captured()) {
    EmbedInto(d, *s.hidden, s.ids.t(), impl_->weights);
    RecordMlaBatchSplit(BuildMlaBatchSplit(s.attn_meta), s.attn_meta.num_reqs);
    // Through the seam's container, never `Backend::ReplayGraph` directly: the
    // container replays its segments in order (one, here, because a decode
    // capture is kFull) and owns the G3 replay counter the reachability gate
    // reads.
    s.graph.Replay(impl_->queue);
    ++s.replays;
    ++impl_->replays;
    // DIAGNOSTIC (VT_DEEPSEEK_GRAPH_VERIFY=1, off by default, dev-only): run the
    // EAGER forward on the SAME padded inputs and report the divergence. This is
    // the bisect harness for graph-safety work — it answers "is replay N wrong,
    // and by how much" without a second process.
    if (std::getenv("VT_DEEPSEEK_GRAPH_VERIFY") != nullptr) {
      DBuf ref = ForwardBody(d, s.token_ids, s.positions, s.attn_meta, attn_kv,
                             impl_->weights, kNoGather);
      std::vector<float> a(static_cast<size_t>(S) * vocab), e(a.size());
      d.b.Copy(d.q, a.data(), s.logits->ptr(), a.size() * sizeof(float));
      ref.Download(d, e.data());
      double worst = 0.0;
      int amax_g = 0, amax_e = 0;
      for (int64_t j = 0; j < vocab; ++j) {
        const size_t idx = static_cast<size_t>(j);
        worst = std::max(worst, static_cast<double>(std::abs(a[idx] - e[idx])));
        if (a[idx] > a[static_cast<size_t>(amax_g)]) amax_g = static_cast<int>(j);
        if (e[idx] > e[static_cast<size_t>(amax_e)]) amax_e = static_cast<int>(j);
      }
      std::fprintf(stderr,
                   "[DeepseekV2DecodeGraph:VERIFY] S=%lld replay#%lld max|graph-eager|"
                   "=%.6g argmax graph=%d eager=%d %s\n",
                   static_cast<long long>(S), static_cast<long long>(s.replays), worst,
                   amax_g, amax_e, amax_g == amax_e ? "" : "  <<< ARGMAX DIVERGED");
    }
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Warm: the pool, weight residency and per-shape kernel scratch were warmed for
  // this size by the previous (eager) step. CAPTURE the layer region once.
  if (s.warm) {
    EmbedInto(d, *s.hidden, s.ids.t(), impl_->weights);
    // ENG-CUDAGRAPH-BREAK W3 (#1291): the capture is the SHARED SEAM's, not this
    // driver's hand-rolled `BeginCapture`/`EndCaptureGraph` pair. The scope owns
    // the segment, the handle, its release, the drain a mid-capture throw needs
    // and the G3 counters.
    //
    // kFULL, and the mode is the whole argument. vLLM's v1 default
    // `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:63`) is documented at
    // `:630-632` as a FULL graph for DECODE batches and a piecewise one for
    // prefill and mixed batches, and `decode_mode()` (`:65-66`) returns the full
    // half. This is a decode driver, so its capture is ONE segment with the MLA
    // decode attention INSIDE it — byte-identical in shape to the region this
    // replaces. Opening it kPiecewise would turn every layer's attention into an
    // eager call between two graph replays, which is not vLLM's decode behaviour
    // and which nothing in this row's record supports.
    std::optional<DBuf> lg;
    {
      vt::GraphCaptureScope scope(b, impl_->queue, s.graph, vt::GraphCaptureMode::kFull);
      lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, attn_kv,
                         impl_->weights, kNoGather);
    }  // ~GraphCaptureScope closes the segment and files it on s.graph
    // NOT CAPTURED covers TWO states, and returning `*lg` is correct for exactly
    // one of them. `~GraphCaptureScope` must swallow a throwing `EndCaptureGraph`
    // — a destructor that propagates terminates — so a FAILED capture leaves the
    // container reporting what an INERT scope reports.
    //
    //   * INERT (`capture_failed() == false`): the backend cannot capture, or
    //     `VLLM_CPP_CUDAGRAPH=0`. The scope made no backend call, the layer
    //     region above ran EAGERLY, and `*lg` is a real result.
    //   * FAILED (`capture_failed() == true`): `Backend::EndCaptureGraph` threw
    //     (`src/vt/cuda/cuda_backend.cu:229`, `Check()` at `:50`). Under stream
    //     capture NOTHING between `BeginCapture` and the throw executed: every
    //     kernel was RECORDED, so `*lg` is pool-recycled memory, and returning it
    //     would hand this step uncomputed device memory as its logits — silently
    //     wrong tokens, no fault, and a token gate cannot see it.
    //
    // So the failure PROPAGATES, carrying the runtime's own exception, which is
    // exactly what the pre-W3 driver did (its `s.graph = b.EndCaptureGraph(...)`
    // was unguarded). Gated at
    // `tests/vllm/models/test_deepseek_v2_decode_graph_seam.cpp`.
    if (!s.graph.captured()) {
      s.warm = false;  // either way this slot goes back to cold
      if (s.graph.capture_failed()) {
        const std::exception_ptr err = s.graph.capture_error();
        s.graph.Reset();  // clear the failure with the graph it described
        // The runtime's OWN diagnosis where the seam holds it. It is empty only
        // on the arm where an exception was already propagating THROUGH the
        // scope, which cannot reach this line; the refusal below is what makes
        // that unreachability an assertion rather than a claim.
        if (err) std::rethrow_exception(err);
        VT_CHECK(false,
                 "DeepseekV2DecodeGraph: the decode capture was ABANDONED and its logits "
                 "were never computed; refusing to return uncaptured device memory");
      }
      ForwardLogits drained = WrapDeviceLogits(std::move(*lg), B, vocab);
      drained.device_tensor =
          MakeTensor(drained.device_storage.get(), DType::kF32, d.q.device, {B, vocab});
      return drained;
    }
    s.logits = std::make_unique<DBuf>(std::move(*lg));
    impl_->any_captured = true;
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
      std::fprintf(stderr,
                   "[DeepseekV2DecodeGraph] captured MLA decode graph for padded "
                   "size S=%lld (real B=%lld)\n",
                   static_cast<long long>(S), static_cast<long long>(B));
    s.graph.Replay(impl_->queue);
    s.replays = 1;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Cold size: run one EAGER step (pre-warms the DevicePool size classes, the
  // resident weights / MoE expert pointer arrays + token map, and the MLA decode
  // split workspace for this size) and defer capture to the next same-size step.
  // This is a real decode step — nothing is wasted.
  s.hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{S, H});
  EmbedInto(d, *s.hidden, s.ids.t(), impl_->weights);
  DBuf lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, attn_kv,
                          impl_->weights, kNoGather);
  s.warm = true;
  ForwardLogits fl = WrapDeviceLogits(std::move(lg), B, vocab);
  fl.device_tensor =
      MakeTensor(fl.device_storage.get(), DType::kF32, d.q.device, {B, vocab});
  return fl;
}

}  // namespace vllm

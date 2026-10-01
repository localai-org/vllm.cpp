// GLM-5.3 (`GlmMoeDsaForCausalLM`) — THE FORWARD. W9 of
// `.agents/specs/glm-dsa-latest-deepseek.md` §3.7, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214). vLLM parity pin
// `5559679229bc961848b121ccdeaa8fa5d79bec98`.
//
// ─── WHAT THIS FILE IS, AND WHAT IT DELIBERATELY IS NOT ──────────────────────
// Upstream at the pin is `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
// pass` (`deepseek_v2.py:1930`), so this is a MIRROR job and every op sequence
// below has a `deepseek_v2.py` line beside it. It is not a design.
//
// It composes ONLY shared seams, and that is the whole of its content:
//
//   embed                     `vt::Embedding` over a BLOCK-QUANTIZED table
//   residual add + RMSNorm    `vt::FusedChain` (kFusedAddRmsNormStd)
//   attention                 `mla::ForwardMlaAttentionBlock`, per-layer dims
//                             from `GlmMoeDsaMlaSchedule`, over ONE
//                             `mla::MlaSharedSelection`
//   dense MLP / shared expert `layers::MlpGateUpMethodBase`
//   router                    `vt::MoeRouterTopK`
//   routed experts            `expert_stream::ExpertSlice` + `vt::MatmulBT`
//   combine                   `vt::MoeCombine`
//   lm_head                   `vt::MatmulBT`
//
// ─── THE ONE THING THIS FORWARD DOES THAT NOTHING ELSE IN THIS TREE DOES ─────
// It passes `mla::ForwardMlaAttentionBlock`'s ELEVENTH argument. That parameter
// has existed since W4 and had no production caller: `git grep` over `src`
// returns four call sites (`deepseek_v2.cpp`, `dots3_note_device.cpp`,
// `kimi_linear_device.cpp`, `minicpm3.cpp`) and every one of them passes ten
// arguments. Upstream allocates ONE `topk_indices_buffer` per model
// (`deepseek_v2.py:1372-1377`) and hands the SAME tensor to every layer
// (`:1395`); a `kFull` layer overwrites it and a `kShared` layer simply does
// not, so the bytes a shared layer attends through are the ones its owning full
// layer wrote EARLIER IN THIS SAME FORWARD PASS (`mla.py:180`). That is why the
// buffer is allocated once here, per forward, and why the layer loop runs in
// order.
//
// ─── AND THE ONE THING IT STILL REFUSES ──────────────────────────────────────
// A step in which any request RESUMES while the selection PRUNES. The indexer's
// `k` for a token is produced from that token's own hidden state
// (`deepseek_v2.py:808-810`), so a step that computes every token of every
// sequence has every index key in hand and a resumed step does not — it would
// need the indexer's own 128-wide side cache, which is a SECOND kv-cache group
// owned by `KV-DSV4-MULTICACHE` (spec O4, #1925, #2323). A FIRST token on a
// fresh prompt is reachable; a SECOND is not, and it says so by name rather
// than serving dense attention on a sparse model in silence.
#include "vllm/model_executor/models/glm_moe_dsa.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/expert_stream_seam.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::ApplyDeviceTokenIds
#include "vllm/model_executor/moe_placement_seam.h"
#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/deepseek_v2.h"  // MlaStep / BuildMlaStep
#include "vllm/model_executor/models/mla_attention.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using dense_attn::FusedChainAdoptEnabled;
using dense_attn::Reshape;
using dense_attn::ResidentWeight;
using v1::CommonAttentionMetadata;
using vt::DType;
using vt::Tensor;

// `FusedChainAdoptEnabled` is `deepseek_v2.h`'s; the residual add + RMSNorm goes
// through the SHARED `vt::FusedChain` catalog with the standalone call as the
// byte-exact rollback, exactly as `deepseek_v2.cpp` and `dots3_note_device.cpp`
// spell it. Upstream's decoder layer for this model IS `DeepseekV2DecoderLayer`.
void AddRmsNorm(Dev d, Tensor& out, const Tensor& in, const Tensor& w,
                Tensor& residual, float eps) {
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, out, in, w, &residual, vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, out, in, w, vt::RmsNormArgs{eps, false}, &residual);
  }
}

// ─── the expert-slice seam, and this model's resident fallback ───────────────
//
// WHY THIS FUNCTION IS NOT `dense_attn::ResidentWeight` WITH AN OFFSET, which is
// what the merged-tower path would otherwise do. The header's `ResidentWeight`
// STAGES a weight to the device when the platform needs staging, and the thing
// it would stage here is a whole `[E*out, K]` routed-expert tower: 187.312 GiB
// across this model's 228 of them. `qwen3_5.cpp` has a private `ResidentWeight`
// that refuses exactly that (`w.expert_streamed`), and spec O13 records that the
// two definitions are not interchangeable and that reconciling them has no owner
// yet. So this file does not call either one on a tower: it serves the slice
// itself, out of the mmap the loader borrowed, and refuses a platform that
// cannot follow a host pointer.
//
// WHEN IT IS EVEN REACHED. `expert_stream::ExpertSlice` handles the two live
// cases itself — a served slot, and a cache miss on a host-addressable device
// (which it satisfies in place rather than falling through, because prefill
// misses by construction). This fallback is what remains: streaming was never
// requested, or the device is DISCRETE. The first is the ordinary CPU gate and
// aliases; the second cannot be served at all and says so.
Tensor GlmResidentExpertSlice(Dev d, const OwnedTensor& w, int64_t N, int64_t K,
                              int64_t row_off) {
  const platforms::Platform& p = platforms::GetPlatform(d.q.device.type);
  VT_CHECK(p.is_cpu() || p.host_memory_is_device_addressable(),
           "glm-dsa forward: a routed-expert tower reached the resident-slice "
           "fallback on a DISCRETE device. This model's 228 towers are "
           "187.312 GiB and the expert-stream lane serves them out of HOST slot "
           "storage, which a discrete device's kernels cannot read "
           "(ENG-EXPERT-STREAM-DEVICE W2). Staging the tower is what "
           "CheckDeviceWeightFit refuses at load, and doing it here would be the "
           "same allocation one frame later");
  VT_CHECK(!w.bytes.empty(),
           "glm-dsa forward: a routed-expert tower has no host bytes to slice; "
           "its mapping was released or it was never loaded");
  // A repacked or elementwise-repacked tower cannot be sliced by row at all, and
  // the loader declines both transforms for this arm (`kGlmMoeDsaQuantRepack`).
  // Stating it here is what keeps that a checked fact rather than an inherited
  // one: a repack preserves the dtype AND the byte count, so nothing downstream
  // could notice.
  VT_CHECK(!w.repacked && !w.elem_kn_repacked,
           "glm-dsa forward: a REPACKED routed-expert tower reached the "
           "resident-slice fallback. This arm declines both load-time repacks "
           "(the sibling MODEL-MM-GLM53-FLASH row paid for one: 346 Q8_0 "
           "tensors decoded to garbage and the model emitted token id 0 for "
           "every position, #2241), so reaching this means a policy changed "
           "under the loader");
  const size_t row_bytes = vt::RowSizeBytes(w.dtype, K);
  Tensor t = MakeTensor(
      static_cast<void*>(const_cast<uint8_t*>(w.bytes.data()) +
                         static_cast<size_t>(row_off) * row_bytes),
      w.dtype, d.q.device, {N, K});
  return t;
}

Tensor GlmExpertSlice(Dev d, const OwnedTensor& w, int64_t N, int64_t K,
                      int64_t row_off, int64_t expert) {
  VT_CHECK(expert >= 0, "glm-dsa forward: a routed-expert index must be >= 0");
  return expert_stream::ExpertSlice(d, w, N, K, row_off, expert,
                                    &GlmResidentExpertSlice);
}

// ─── the MLA block's per-layer weights ──────────────────────────────────────
// The SPLIT A-projection arm (`q_a_proj` + `kv_a_proj_with_mqa`) rather than the
// fused one: on this artifact `attn_q_a` is Q5_K and `attn_kv_a_mqa` is Q8_0,
// two different block encodings, so no fused `[q_lora + kv_lora + rope, hidden]`
// weight exists to build. See `mla_attention.h`'s `q_a_proj`.
mla::MlaBlockWeights GlmResidentMla(Dev d, const GlmMoeDsaMlaWeights& w,
                                    const mla::MlaBlockDims& dm,
                                    const Tensor& rope_cache) {
  mla::MlaBlockWeights m;
  m.q_a_proj = ResidentWeight(d, w.q_a_proj);
  m.q_a_layernorm = ResidentWeight(d, w.q_a_layernorm, {dm.q_lora_rank});
  m.q_b_proj = ResidentWeight(d, w.q_b_proj);
  m.kv_a_proj_with_mqa = ResidentWeight(d, w.kv_a_proj_with_mqa);
  m.kv_a_layernorm = ResidentWeight(d, w.kv_a_layernorm, {dm.kv_lora_rank});
  // The absorbed trio, produced at load by `AbsorbMla` through the SHARED
  // `mla::AbsorbKvBProjBf16`. `kv_b_proj` serves the prefill (materialized-MHA)
  // arm and the chunked-context callback; `w_uk_t` / `w_uv` serve decode.
  m.kv_b_proj = ResidentWeight(d, w.kv_b_proj);
  m.w_uk_t = ResidentWeight(d, w.w_uk_t);
  m.w_uv = ResidentWeight(d, w.w_uv);
  m.o_proj = ResidentWeight(d, w.o_proj);
  m.rope_cos_sin_cache = rope_cache;
  if (dm.has_indexer()) {
    // A `kShared` layer has NO indexer at all (`deepseek_v2.py:1134-1135`), so
    // its five tensors stay empty and the block issues no indexer GEMM. The
    // loader already dropped the file's broadcast copies (spec D3), so there is
    // nothing here to leave behind by accident.
    m.indexer_wq_b = ResidentWeight(d, w.indexer.wq_b);
    m.indexer_wk = ResidentWeight(d, w.indexer.wk);
    m.indexer_weights_proj = ResidentWeight(d, w.indexer.weights_proj);
    m.indexer_k_norm_weight =
        ResidentWeight(d, w.indexer.k_norm_weight, {dm.index_head_dim});
    m.indexer_k_norm_bias =
        ResidentWeight(d, w.indexer.k_norm_bias, {dm.index_head_dim});
  }
  return m;
}

// ─── the SPARSE per-token MQA step ──────────────────────────────────────────
// The GLM counterpart of `BuildDots3NoteSparseStep`. It is a second BUILDER and
// not a second PREDICATE: the eligibility question both models ask is
// `mla::SparseStepEligibilityOf`, lifted into the seam by this wave so the two
// cannot drift.
//
// The builder itself differs from dots3-note's in one way that matters, and it
// is a property of the ARCHITECTURE rather than a choice: EVERY GLM-5.3 layer is
// sparse (22 carry an indexer, 57 reuse), where dots3-note has 33 sliding layers
// that carry no selection at all. So there is no per-layer routing here — a
// sparse step routes the whole model through per-token MQA, and a dense step
// routes all of it through the ordinary split.
struct GlmSparseStep {
  bool active = false;
  std::vector<DBuf> owned;
  mla::MlaBlockMetadata meta;
};

Tensor UploadInto(Dev d, std::vector<DBuf>& owned, DType dt,
                  const std::vector<int64_t>& shape, const void* host) {
  owned.emplace_back(d, dt, shape, host);
  return owned.back().t();
}

GlmSparseStep BuildGlmSparseStep(Dev d, const GlmMoeDsaParams& p,
                                 const CommonAttentionMetadata& am, int64_t T) {
  GlmSparseStep s;
  if (!mla::SparseStepEligibilityOf(p.index_topk, am).Active()) return s;

  const int num_reqs = am.num_reqs;
  const int64_t cols = am.block_table_num_cols;
  std::vector<int32_t> cu(static_cast<size_t>(num_reqs) + 1, 0);
  std::vector<int32_t> tok_seq_lens(static_cast<size_t>(T), 0);
  std::vector<int32_t> tok_block_table(static_cast<size_t>(T * cols), 0);
  for (int r = 0; r <= num_reqs; ++r) {
    cu[static_cast<size_t>(r)] = am.query_start_loc[static_cast<size_t>(r)];
  }
  int64_t max_seq = 1;
  for (int r = 0; r < num_reqs; ++r) {
    const int64_t o = cu[static_cast<size_t>(r)];
    const int64_t len = cu[static_cast<size_t>(r + 1)] - o;
    for (int64_t i = 0; i < len; ++i) {
      // Position `i` within its OWN request, because eligibility already
      // established that nothing was computed before this step. `seq_lens[t]` is
      // therefore the number of causal keys token `t` has, which is what the MQA
      // kernel bounds its walk by.
      tok_seq_lens[static_cast<size_t>(o + i)] = static_cast<int32_t>(i + 1);
      for (int64_t c = 0; c < cols; ++c) {
        tok_block_table[static_cast<size_t>((o + i) * cols + c)] =
            am.block_table_tensor[static_cast<size_t>(r * cols + c)];
      }
    }
    max_seq = std::max(max_seq, len);
  }

  s.active = true;
  s.meta.num_decode_tokens = T;
  s.meta.decode.block_table =
      UploadInto(d, s.owned, DType::kI32, {T, cols}, tok_block_table.data());
  s.meta.decode.seq_lens =
      UploadInto(d, s.owned, DType::kI32, {T}, tok_seq_lens.data());
  s.meta.decode.max_seq_len = static_cast<int>(max_seq);
  // NON-EMPTY is what declares the step sparse to the MLA block; `BuildMlaStep`
  // never sets this field and only a sparse builder does.
  s.meta.indexer_cu_seqlens_q = cu;
  return s;
}

// ─── the MLP arms ───────────────────────────────────────────────────────────
// The SPLIT gate/up method rather than the merged one, for the same reason the
// A-projection takes its split arm: llama.cpp writes `ffn_gate` and `ffn_up` as
// two tensors with independently chosen block encodings. Routed through
// `layers::MlpGateUpMethodBase` so the fused-kernel scheme choice keeps one
// home.
DBuf DenseMlp(Dev d, const GlmMoeDsaMlpWeights& w, const Tensor& dh, int64_t T,
              int64_t H, int64_t I) {
  DBuf act = layers::UnquantizedMlpGateUpSplitMethod(&w.gate_proj, &w.up_proj, I)
                 .Apply(d, dh);
  Tensor wdn = ResidentWeight(d, w.down_proj);  // [H, I] raw-NK
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wdn);
  return out;
}

// One routed expert's SwiGLU MLP over the STACKED towers, taken a slice at a
// time through the expert-streaming seam. Byte-identical to reading the tower
// in place — `ExpertSlice` returns a view of the same bytes, in a slot or in the
// mapping — which is what makes the streamed and unstreamed arms comparable
// inside one process (spec §3.6 G3).
DBuf ExpertMlp(Dev d, const GlmMoeDsaMoeWeights& w, const Tensor& x, int64_t e,
               int64_t n, int64_t H, int64_t I) {
  // Declare the LARGEST of the three slices BEFORE taking any, so the slot store
  // is sized once and correctly. gate/up and down are not the same size on a
  // dynamic (UD) quant that keeps `down_proj` at a higher precision, and this
  // artifact is exactly that: `blk.6.ffn_gate_exps` is IQ2_XXS while
  // `blk.6.ffn_down_exps` is IQ3_XXS. Sizing from whichever slice arrived first
  // would trip the store's own check mid-decode.
  expert_stream::ExpertStreamLane::Reserve(
      std::max({static_cast<size_t>(I) * vt::RowSizeBytes(w.gate_exps.dtype, H),
                static_cast<size_t>(I) * vt::RowSizeBytes(w.up_exps.dtype, H),
                static_cast<size_t>(H) * vt::RowSizeBytes(w.down_exps.dtype, I)}));
  DBuf g(d, DType::kBF16, {n, I});
  DBuf u(d, DType::kBF16, {n, I});
  vt::MatmulBT(d.q, g.t(), x, GlmExpertSlice(d, w.gate_exps, I, H, e * I, e));
  vt::MatmulBT(d.q, u.t(), x, GlmExpertSlice(d, w.up_exps, I, H, e * I, e));
  DBuf act(d, DType::kBF16, {n, I});
  vt::MoeSiluMul(d.q, act.t(), g.t(), u.t());
  DBuf out(d, DType::kBF16, {n, H});
  vt::MatmulBT(d.q, out.t(), act.t(),
               GlmExpertSlice(d, w.down_exps, H, I, e * H, e));
  return out;
}

// `DeepseekV2MoE.forward` (`deepseek_v2.py:395-424`) + the FusedMoE runner's
// routed/shared composition (`moe_runner.py:392-407`).
DBuf MoeBlock(Dev d, const GlmMoeDsaMoeWeights& w, const GlmMoeDsaParams& p,
              const Tensor& dh, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.n_routed_experts;
  const int64_t top_k = p.num_experts_per_tok;
  const int64_t I = p.moe_intermediate_size;
  const int64_t P = T * top_k;

  // --- router: `router_logits, _ = self.gate(hidden_states)` (`:417`) --------
  // THE ONE DELIBERATELY WIDER OP ON THIS MODEL PATH, and the polarity rule says
  // to annotate it rather than let it pass. Upstream's `GateLinear` holds the
  // gate at the MODEL dtype and takes the cuBLAS bf16 x bf16 -> f32 tier
  // (`fused_moe/router/gate_linear.py`, tier 5); the OUTPUT dtype is f32 because
  // `_get_moe_router_dtype` forces it for `model_type == "glm_moe_dsa"`
  // (`deepseek_v2.py:123-133`, `:127`). This artifact stores `ffn_gate_inp` at
  // F32 — llama.cpp always does — and the forward keeps it there and widens the
  // ACTIVATION to match, so the GEMM is f32 x f32 -> f32.
  //
  // Three reasons, and the third is the one that decides it. It is the smallest
  // GEMM in the model (`[T,6144] x [256,6144]`, ~1.6 MFLOP per token against
  // ~11 GiB of expert bytes). vLLM's own `force_fp32_compute` arm stores this
  // exact weight in fp32 "so the fallback linear path computes in fp32" when no
  // specialized kernel is available, which is this box. And the output feeds a
  // DISCRETE top-k, where the error is bimodal and narrowing the file's own f32
  // to bf16 would be us discarding precision the artifact carries, on the one
  // decision no tolerance bounds.
  //
  // The cost is one `vt::CastF32` of `[T, hidden]` per MoE layer.
  Tensor drg = ResidentWeight(d, w.router);  // f32 [E, H] raw-NK
  DBuf dh_f32(d, DType::kF32, {T, H});
  vt::CastF32(d.q, dh_f32.t(), dh);
  DBuf dlog(d, p.router_dtype_is_f32 ? DType::kF32 : DType::kBF16, {T, E});
  vt::MatmulBT(d.q, dlog.t(), dh_f32.t(), drg);

  vt::MoeRouterTopKArgs args{};
  args.top_k = static_cast<int>(top_k);
  args.renormalize = p.norm_topk_prob;
  // SIGMOID unconditionally, and it is a CHECKED fact rather than a default:
  // `ParseGlmMoeDsaParams` refuses any `scoring_func` but `sigmoid` and any
  // `topk_method` but `noaux_tc` (`deepseek_v2.py:313-318`), so no config that
  // reaches this line describes anything else.
  args.scoring_func = vt::MoeScoringFunc::kSigmoid;
  args.num_expert_group = static_cast<int>(p.n_group);
  args.topk_group = static_cast<int>(p.topk_group);
  args.routed_scaling_factor = static_cast<float>(p.routed_scaling_factor);
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  std::unique_ptr<Tensor> bias;
  if (!w.e_score_correction_bias.Empty()) {
    // f32, as upstream builds it (`deepseek_v2.py:317-318`,
    // `dtype=torch.float32`). It is ADDED to the sigmoid scores before the
    // top-k.
    bias = std::make_unique<Tensor>(
        ResidentWeight(d, w.e_score_correction_bias, {E}));
  }
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(), args, bias.get());

  // --- routed experts -------------------------------------------------------
  // The per-expert gather/scatter reference form, which is `deepseek_v2.cpp`'s
  // non-grouped branch. There is no grouped arm here ON PURPOSE: the grouped
  // keep-quant op consumes a WHOLE tower, so it bypasses the slot lane entirely
  // — the same conflict `qwen3_5.cpp` resolves by turning grouping off when
  // streaming is on. This row has no speed axis (spec O10), and a grouped MoE
  // that silently un-streams a 187 GiB expert set is the invisible-fallback
  // shape this campaign keeps finding. Recorded as owed rather than taken.
  std::vector<int32_t> ids(static_cast<size_t>(P));
  dtid.Download(d, ids.data());
  std::vector<std::vector<std::pair<int64_t, int64_t>>> lists(
      static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < top_k; ++j) {
      const int32_t e = ids[static_cast<size_t>(t * top_k + j)];
      VT_CHECK(e >= 0 && e < E,
               "glm-dsa forward: the router returned expert id " +
                   std::to_string(e) + " outside [0, " + std::to_string(E) + ")");
      lists[static_cast<size_t>(e)].push_back({t, j});
    }
  }
  DBuf expert_out(d, DType::kBF16, {T, top_k, H});
  expert_out.Zero(d);
  const size_t row_bytes = static_cast<size_t>(H) * vt::SizeOf(DType::kBF16);
  for (int64_t e = 0; e < E; ++e) {
    const auto& list = lists[static_cast<size_t>(e)];
    if (list.empty()) continue;
    const int64_t n = static_cast<int64_t>(list.size());
    DBuf xg(d, DType::kBF16, {n, H});
    for (int64_t r = 0; r < n; ++r) {
      d.b.Copy(d.q,
               static_cast<char*>(xg.ptr()) + static_cast<size_t>(r) * row_bytes,
               static_cast<const char*>(dh.data) +
                   static_cast<size_t>(list[static_cast<size_t>(r)].first) *
                       row_bytes,
               row_bytes);
    }
    DBuf o = ExpertMlp(d, w, xg.t(), e, n, H, I);
    for (int64_t r = 0; r < n; ++r) {
      const auto& tj = list[static_cast<size_t>(r)];
      d.b.Copy(d.q,
               static_cast<char*>(expert_out.ptr()) +
                   static_cast<size_t>(tj.first * top_k + tj.second) * row_bytes,
               static_cast<const char*>(o.ptr()) +
                   static_cast<size_t>(r) * row_bytes,
               row_bytes);
    }
  }

  // --- shared expert + weighted combine -------------------------------------
  // DeepSeek's shared expert is a PLAIN MLP whose output is ADDED to the routed
  // sum (`deepseek_v2.py:344-357`; `moe_runner.py:407`
  // `shared_output + fused_output`) — no sigmoid gate. Its width is
  // `moe_intermediate_size * n_shared_experts`, NOT `intermediate_size`.
  const bool has_shared = p.n_shared_experts > 0 && !w.shared.Empty();
  DBuf out(d, DType::kBF16, {T, H});
  if (has_shared) {
    DBuf shared =
        DenseMlp(d, w.shared, dh, T, H, p.moe_intermediate_size * p.n_shared_experts);
    Tensor shared_t = shared.t();
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), &shared_t);
  } else {
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), nullptr);
  }
  return out;
}

// One `DeepseekV2DecoderLayer` (`deepseek_v2.py:1262-1345`). The fp16-overflow
// rescale at `:1298-1308` / `:1338-1343` is guarded on
// `hidden_states.dtype == torch.float16` and is INACTIVE for a bf16 model, so it
// is not ported — the same reading `deepseek_v2.cpp` records.
void RunLayer(Dev d, const GlmMoeDsaLayerWeights& lw, const GlmMoeDsaParams& p,
              const mla::MlaBlockDims& dims, Tensor& hidden,
              std::shared_ptr<void>& hidden_hold, DBuf& res, const MlaStep& step,
              const mla::MlaBlockMetadata& lmeta, Tensor& kv_cache,
              v1::TritonMLAImpl& impl, mla::MlaSharedSelection* shared,
              int64_t T, int64_t layer_index) {
  const int64_t H = p.hidden_size;
  const float eps = static_cast<float>(p.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, lw.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  Tensor dhn_t = dhn.t(), res_t = res.t();
  AddRmsNorm(d, dhn_t, hidden, w_in, res_t, eps);

  DBuf attn(d, DType::kBF16, {T, H});
  Tensor attn_t = attn.t();
  const mla::MlaBlockWeights mw = GlmResidentMla(d, lw.attn, dims, *step.rope_cache);
  // THE ELEVENTH ARGUMENT. `shared` is upstream's per-model
  // `topk_indices_buffer` (`deepseek_v2.py:1372-1377`, handed to every layer at
  // `:1395`). A `kFull` layer WRITES its selection into it; a `kShared` layer
  // READS whatever is there, which is the preceding full layer's selection
  // because nothing overwrote it (`mla.py:180`).
  mla::ForwardMlaAttentionBlock(d, dims, mw, dhn.t(), step.positions, kv_cache,
                                step.slot_mapping, lmeta, impl, attn_t,
                                // `attn_pre_o_proj` is DeepSeek-V4's early return
                                // (#2323); this model applies `o_proj` in the block.
                                /*attn_pre_o_proj=*/nullptr, shared);

  Tensor w_post = ResidentWeight(d, lw.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  Tensor dh2_t = dh2.t(), attn_ro = attn.t();
  AddRmsNorm(d, dh2_t, attn_ro, w_post, res_t, eps);

  // `self.mlp` is `DeepseekV2MoE` on a MoE layer and `DeepseekV2MLP` otherwise
  // (`deepseek_v2.py:1214-1218`). The MoE arm goes through `vllm::RunMoePlaced`,
  // the one seam every architecture's routed-expert compute routes through —
  // inert by construction when no placement plan is configured, which is every
  // load that configured none.
  DBuf mlp = lw.is_moe
                 ? vllm::RunMoePlaced(d, layer_index, dh2.t(), T, H,
                                      [&](Dev pd, const Tensor& h) {
                                        return MoeBlock(pd, lw.moe, p, h, T);
                                      })
                 : DenseMlp(d, lw.dense, dh2.t(), T, H, p.intermediate_size);
  auto* held = new DBuf(std::move(mlp));
  hidden = held->t();
  hidden_hold = std::shared_ptr<void>(held, [](void* q) { delete static_cast<DBuf*>(q); });
}

void GatherRows(Dev d, void* dst, const Tensor& src,
                const std::vector<int32_t>& idx, int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s) {
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
  }
}

// The refusal a step that cannot be selected for raises. Built from the SAME
// eligibility object the route is taken from, so the two cannot come apart —
// the failure `dots3_note_device.cpp` records at length and this file inherits
// rather than rediscovers.
void RefuseUnservableStep(const GlmMoeDsaParams& p,
                          const mla::SparseStepEligibility& elig) {
  VT_CHECK(
      !elig.prunes || elig.Active(),
      "GlmMoeDsaForCausalLM forward: request " +
          std::to_string(elig.prunes_req) + " needs " +
          std::to_string(elig.prunes_len) + " keys against `index_topk` " +
          std::to_string(p.index_topk) +
          ", so this STEP's DSA selection PRUNES (deepseek_v2.py:1068) — and " +
          (elig.resumes
               ? ("request " + std::to_string(elig.resumes_req) +
                  " in the SAME step resumes from " +
                  std::to_string(elig.resumes_from) +
                  " already-computed tokens, whose index keys are not in hand")
               : std::string("this step's attention metadata is not shaped the "
                             "way the sparse route reads it")) +
          ". " + GlmMoeDsaForwardRefusal());
}

// The capturable region: the MLA step metadata, the 78 decoder layers, the final
// norm and the lm_head GEMM.
DBuf ForwardLayers(Dev d, const Tensor& hidden_in,
                   const std::vector<int32_t>& positions,
                   const CommonAttentionMetadata& am,
                   const std::vector<PagedKvCache>& attn_kv,
                   const GlmMoeDsaWeights& weights,
                   const std::vector<int32_t>& logits_indices) {
  const GlmMoeDsaParams& p = weights.params;
  const int64_t T = hidden_in.shape[0];
  const int64_t H = p.hidden_size;
  const int64_t vocab = p.vocab_size;
  const float eps = static_cast<float>(p.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "glm-dsa forward: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(p.num_hidden_layers),
           "glm-dsa forward: one MLA PagedKvCache per backbone layer is "
           "required (" + std::to_string(attn_kv.size()) + " given for " +
           std::to_string(p.num_hidden_layers) + " layers)");

  const std::vector<mla::MlaBlockDims> schedule = GlmMoeDsaMlaSchedule(p);
  VT_CHECK(static_cast<int64_t>(schedule.size()) == p.num_hidden_layers,
           "glm-dsa forward: the MLA schedule must have one entry per layer");

  Tensor hidden = hidden_in;
  std::shared_ptr<void> hidden_hold;
  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  const int64_t block_size = attn_kv[0].block_size;
  MlaStep step =
      BuildMlaStep(d, positions, am, block_size, p.max_position_embeddings);
  const Tensor rope = ResidentWeight(d, weights.rope_cos_sin_cache);
  step.rope_cache = &rope;

  // The route and the refusal, from ONE object.
  const mla::SparseStepEligibility elig =
      mla::SparseStepEligibilityOf(p.index_topk, am);
  RefuseUnservableStep(p, elig);
  const GlmSparseStep sparse = BuildGlmSparseStep(d, p, am, T);

  // ONE selection buffer for the whole model, allocated per forward and handed
  // to every layer in order — upstream's `topk_indices_buffer`
  // (`deepseek_v2.py:1372-1377`, `:1395`). Allocated ONLY on a sparse step:
  // on a dense step no layer writes or reads it, and `nullptr` is the state the
  // block's own `reuse_selection` branch never reaches.
  DBuf sel_idx, sel_cnt;
  mla::MlaSharedSelection shared{};
  mla::MlaSharedSelection* shared_ptr = nullptr;
  if (sparse.active) {
    sel_idx = DBuf(d, DType::kI32, {T, p.index_topk});
    sel_cnt = DBuf(d, DType::kI32, {T});
    // Zeroed so a defect that fails to write the buffer produces a legible
    // selection of position 0 rather than whatever the pool last held. The
    // block refuses a `skip_topk` layer with no buffer outright, so this is not
    // a fallback — it is a deterministic starting state for a mutation to move.
    sel_idx.Zero(d);
    sel_cnt.Zero(d);
    shared.topk_indices = sel_idx.t();
    shared.valid_counts = sel_cnt.t();
    shared_ptr = &shared;
  }

  v1::TritonMLAImpl impl;
  const int64_t head_size = p.mla_kv_head_size();
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const PagedKvCache& kv = attn_kv[static_cast<size_t>(l)];
    VT_CHECK(kv.num_kv_heads == 1 && kv.head_size == head_size,
             "glm-dsa forward: the MLA cache must be 1-head and "
             "kv_lora_rank + qk_rope_head_dim wide (MLAAttentionSpec)");
    Tensor kv_cache = MakeTensor(kv.data, kv.dtype, d.q.device,
                                 {kv.num_blocks, kv.block_size, head_size});
    // EVERY layer of this model is sparse — 22 carry an indexer and 57 reuse —
    // so the metadata choice is per STEP and not per layer, unlike dots3-note
    // where the sliding layers keep the ordinary split.
    const mla::MlaBlockMetadata& lmeta =
        sparse.active ? sparse.meta : step.meta;
    RunLayer(d, weights.layers[static_cast<size_t>(l)], p,
             schedule[static_cast<size_t>(l)], hidden, hidden_hold, res, step,
             lmeta, kv_cache, impl, shared_ptr, T, /*layer_index=*/l);
  }

  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  Tensor dnorm_t = dnorm.t(), res_t = res.t();
  AddRmsNorm(d, dnorm_t, hidden, w_fn, res_t, eps);

  // The tie is a property of the FILE: the loader leaves `lm_head` empty exactly
  // when the GGUF shipped no `output.weight`.
  const bool tied = weights.lm_head.Empty();
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
  // BOTH arms are `MatmulBT`: the GGUF stores `token_embd` and `output` alike as
  // `[vocab, hidden]` raw-NK, so a tied model needs no second orientation. This
  // is where `deepseek_v2.cpp` differs, and it differs because ITS safetensors
  // `lm_head` is packed the other way.
  vt::MatmulBT(d.q, logits.t(), src, lm);
  return logits;
}

DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& am,
                 const std::vector<PagedKvCache>& attn_kv,
                 const GlmMoeDsaWeights& weights,
                 const std::vector<int32_t>& logits_indices) {
  const GlmMoeDsaParams& p = weights.params;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  VT_CHECK(T > 0, "glm-dsa forward: empty batch");
  VT_CHECK(weights.absorbed,
           "GlmMoeDsaForCausalLM forward: this model did not go through the "
           "post-load absorption, so `kv_b_proj`, `w_uk_t` and `w_uv` are empty "
           "and the MLA block has nothing to attend with. Only "
           "LoadGlmMoeDsaFromGguf produces a usable GlmMoeDsaWeights; a "
           "hand-constructed one is refused here rather than one frame deeper "
           "with a message about a single missing tensor "
           "(.agents/specs/glm-dsa-latest-deepseek.md §3.7 W9 F4)");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == p.num_hidden_layers,
           "glm-dsa forward: the weights carry " +
               std::to_string(weights.layers.size()) + " layers and the config "
               "declares " + std::to_string(p.num_hidden_layers));

  // ONE FORWARD IS ONE STEP. The guard clears the slot cache's per-step eviction
  // protection and advances the hotness clock; without it the lane switches
  // itself off partway through the third token and says nothing. It refuses to
  // nest, which is why it is taken HERE — the single body both public entry
  // points call — and nowhere below.
  expert_stream::ExpertStreamStepGuard step_guard;

  DBuf hidden(d, DType::kBF16, {T, p.hidden_size});
  // The embedding table is BLOCK-QUANTIZED (Q4_K on this artifact) and stays
  // that way: `vt::Embedding` dequantizes ONE ROW per gathered id, mirroring
  // `ggml_compute_forward_get_rows_q`. Expanding a `[154880, 6144]` table to
  // bf16 would be 1.77 GiB for the sake of a gather.
  dense_attn::EmbedGather(d, hidden, token_ids, weights.embed_tokens,
                          p.vocab_size, p.hidden_size, "glm-dsa embed");
  return ForwardLayers(d, hidden.t(), positions, am, attn_kv, weights,
                       logits_indices);
}

}  // namespace

std::vector<float> GlmMoeDsaModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const GlmMoeDsaWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv,
                             weights, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) *
                            static_cast<size_t>(weights.params.vocab_size));
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits GlmMoeDsaModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const GlmMoeDsaWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv,
                             weights, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  ForwardLogits fl;
  fl.rows = n_out;
  fl.vocab = weights.params.vocab_size;
  fl.device_tensor = dlogits.t();
  fl.device_storage = dlogits.ReleaseShared();
  return fl;
}

}  // namespace vllm

// MiMoV2 (`MiMoV2ForCausalLM`) device forward pass (MODEL-TEXT-mimo-v2, W3).
//
// Implements the full text-only LLM arm of MiMoV2:
//   - Hybrid full/sliding-window attention with per-layer KV geometry split
//   - v_head_dim (128) != head_dim (192) — the asymmetric V pattern
//   - partial RoPE (rotary_dim=64 of head_dim=192)
//   - attention_value_scale on V before KV cache write
//   - attention_sink_bias on SWA layers (gpt-oss mechanism)
//   - Layer 0 dense MLP, layers 1..47 MoE (256 experts, sigmoid + noaux_tc)
//
// The forward follows the Dots3Note ForwardDevice pattern:
//   embed → per-layer (RMSNorm+residual → attention → post-norm+residual → MLP/MoE)
//   → final norm → lm_head → logits
//
// Reference: .agents/specs/mimov2.md, dots3_note_device.cpp, deepseek_v4_registry.cpp.

#include "vllm/model_executor/models/mimo_v2.h"
#include "vllm/model_executor/models/mimo_v2_weights.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/host_token_ids.h"
#include "vllm/model_executor/models/kv_cache_route.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_common.h"

#include "vt/ops.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace vllm {

using vt::DType;
using vt::Tensor;

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using dense_attn::Reshape;
using dense_attn::ResidentWeight;

namespace {

// ---------------------------------------------------------------------------
// GatherRows — copies selected rows from a device tensor (dots3_note pattern).
// ---------------------------------------------------------------------------
void GatherRows(Dev d, void* dst, const Tensor& src,
                const std::vector<int32_t>& idx, int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// ---------------------------------------------------------------------------
// WrapDeviceLogits — wraps a device buffer into ForwardLogits (dots3_note pattern).
// ---------------------------------------------------------------------------
ForwardLogits WrapDeviceLogits(DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  fl.device_storage = dlogits.ReleaseShared();
  return fl;
}

// ---------------------------------------------------------------------------
// Linear — dispatches to bf16 Matmul or EXL3 GEMM based on weight type.
// MiMoV2Projection bf16 weights are loaded via LoadBf16Transposed → [K,N]
// (Matmul-B layout), so vt::Matmul: out[M,N] = a[M,K] @ b[K,N].
// ---------------------------------------------------------------------------
DBuf Linear(Dev d, const Tensor& x, const MiMoV2Projection& w, DType out_dtype) {
  if (w.IsExl3()) {
    return dense_attn::Exl3MatmulD(d, x, w.exl3, out_dtype);
  }
  Tensor wt = ResidentWeight(d, w.bf16);  // [K, N]
  const int64_t M = x.shape[0];
  const int64_t N = wt.shape[1];
  DBuf out(d, out_dtype, {M, N});
  vt::Matmul(d.q, out.t(), x, wt);
  return out;
}

// ---------------------------------------------------------------------------
// DenseMlp — layer 0: separate gate/up/down projections with SwiGLU.
// gate_proj/up_proj: [intermediate, H] → MatmulBT
// down_proj: [H, intermediate] → MatmulBT
// ---------------------------------------------------------------------------
DBuf DenseMlp(Dev d, const MiMoV2DenseMlpWeights& w, const Tensor& dh,
              int64_t T, int64_t /*H*/, int64_t I) {
  DBuf g = Linear(d, dh, w.gate_proj, DType::kBF16);  // [T, I]
  DBuf u = Linear(d, dh, w.up_proj, DType::kBF16);    // [T, I]
  DBuf a(d, DType::kBF16, {T, I});
  vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());           // silu(g) * u
  return Linear(d, a.t(), w.down_proj, DType::kBF16);  // [T, H]
}

// ---------------------------------------------------------------------------
// MoeBlock — layers 1..47: 256 experts, sigmoid routing, no shared expert.
// CPU reference path: download routing, per-expert gather/scatter.
// ---------------------------------------------------------------------------
DBuf MoeBlock(Dev d, const MiMoV2MoeWeights& w, const MiMoV2Params& p,
              const Tensor& dh, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.num_experts;
  const int64_t top_k = p.num_experts_per_tok;
  const int64_t I = p.moe_intermediate_size;
  const int64_t P = T * top_k;

  // --- router ---
  // router_gate loaded via LoadBf16Direct → on-disk [E, H] = [N, K] (not transposed).
  // MatmulBT: out[T, E] = dh[T, H] @ gate^T[H, E].
  Tensor drg = ResidentWeight(d, w.router_gate);
  DBuf dlog(d, DType::kBF16, {T, E});
  vt::MatmulBT(d.q, dlog.t(), dh, drg);  // [T, E]

  vt::MoeRouterTopKArgs args{};
  args.top_k = static_cast<int>(top_k);
  args.renormalize = p.norm_topk_prob;
  args.scoring_func = vt::MoeScoringFunc::kSigmoid;
  args.num_expert_group = static_cast<int>(p.n_group);
  args.topk_group = static_cast<int>(p.topk_group);
  args.routed_scaling_factor = static_cast<float>(p.routed_scaling_factor);

  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  Tensor bias = ResidentWeight(d, w.e_score_correction_bias, {E});
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(), args, &bias);

  // --- routed experts (CPU reference path) ---
  DBuf expert_out(d, DType::kBF16, {T, top_k, H});
  expert_out.Zero(d);

  // Download routing to host for the per-expert gather/scatter loop.
  std::vector<int32_t> hid(static_cast<size_t>(P));
  std::vector<float> htw(static_cast<size_t>(P));
  dtid.Download(d, hid.data());
  dtw.Download(d, htw.data());

  for (int64_t e = 0; e < E; ++e) {
    // Find tokens routed to this expert.
    std::vector<int32_t> token_rows;
    std::vector<int32_t> slot_rows;
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t k = 0; k < top_k; ++k) {
        const int64_t idx = t * top_k + k;
        if (hid[static_cast<size_t>(idx)] == static_cast<int32_t>(e)) {
          token_rows.push_back(static_cast<int32_t>(t));
          slot_rows.push_back(static_cast<int32_t>(idx));
          break;
        }
      }
    }
    if (token_rows.empty()) continue;

    const int64_t ne = static_cast<int64_t>(token_rows.size());
    // Gather hidden rows for this expert.
    DBuf gathered(d, DType::kBF16, {ne, H});
    GatherRows(d, gathered.ptr(), dh, token_rows, H);

    // Expert MLP: gate → up → SwiGLU → down
    DBuf g = Linear(d, gathered.t(), w.experts[static_cast<size_t>(e)].gate_proj,
                    DType::kBF16);  // [ne, I]
    DBuf u = Linear(d, gathered.t(), w.experts[static_cast<size_t>(e)].up_proj,
                    DType::kBF16);  // [ne, I]
    DBuf a(d, DType::kBF16, {ne, I});
    vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());

    DBuf o = Linear(d, a.t(), w.experts[static_cast<size_t>(e)].down_proj,
                    DType::kBF16);  // [ne, H]

    // Scatter into expert_out at slot_rows.
    for (int64_t i = 0; i < ne; ++i) {
      d.b.Copy(d.q,
               static_cast<char*>(expert_out.ptr()) +
                   static_cast<size_t>(slot_rows[static_cast<size_t>(i)]) *
                       static_cast<size_t>(H) * vt::SizeOf(DType::kBF16),
               static_cast<const char*>(o.ptr()) +
                   static_cast<size_t>(i) * static_cast<size_t>(H) *
                       vt::SizeOf(DType::kBF16),
               static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));
    }
  }

  // --- weighted combine (no shared expert) ---
  DBuf out(d, DType::kBF16, {T, H});
  vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(),
                 /*shared=*/nullptr, static_cast<float>(p.routed_scaling_factor));
  return out;
}

// ---------------------------------------------------------------------------
// AttentionBlock — custom MiMoV2 attention.
// Cannot reuse dense_attn::AttnBlock because MiMoV2 has:
//   - Separate q_proj/k_proj/v_proj (not fused QKV)
//   - v_head_dim (128) != head_dim (192)
//   - Per-layer KV head count (4 full / 8 SWA)
//   - Per-layer RoPE theta (10M full / 10K SWA)
//   - attention_value_scale on V before cache write
//   - attention_sink_bias on SWA layers
// ---------------------------------------------------------------------------
DBuf AttentionBlock(Dev d, const MiMoV2AttnWeights& w,
                    const MiMoV2Params& p, int64_t layer_idx,
                    const Tensor& dhn, const Tensor& positions,
                    const dense_attn::StepInputs& si,
                    const PagedKvCache& kv, int64_t T) {
  const int64_t Hq = p.num_attention_heads;     // 64
  const int64_t Dh = p.head_dim;                 // 192 (Q/K)
  const int64_t Dv = p.v_head_dim;                // 128 (V)
  const int64_t Hkv = kv.num_kv_heads;            // 4 (full) or 8 (SWA)
  const int64_t rot = p.rotary_dim;               // 64
  const bool is_full = p.hybrid_layer_pattern[static_cast<size_t>(layer_idx)] == 0;
  const double rope_theta = is_full ? p.rope_theta_full : p.rope_theta_swa;
  const float attn_scale = 1.0F / std::sqrt(static_cast<float>(Dh));

  // --- Q/K/V projections ---
  DBuf q = Linear(d, dhn, w.q_proj, DType::kBF16);  // [T, Hq*Dh]
  DBuf k = Linear(d, dhn, w.k_proj, DType::kBF16);  // [T, Hkv*Dh]
  DBuf v = Linear(d, dhn, w.v_proj, DType::kBF16);  // [T, Hkv*Dv]

  // Reshape to 3D for RoPE and cache.
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  Tensor v3 = Reshape(v.t(), {T, Hkv, Dv});

  // --- RoPE (in-place partial NeoX, rotary_dim=64) ---
  vt::RopeArgs ra{};
  ra.base = static_cast<float>(rope_theta);
  ra.rotary_dim = static_cast<int>(rot);
  ra.is_neox_style = true;
  vt::RopeNeox(d.q, q3, k3, positions, ra);

  // --- Scale V by attention_value_scale ---
  // `vs` OWNS the storage v3 is re-pointed at, and the KV-cache write below
  // reads it, so it must live in THIS scope, not the `if`: a DBuf destroyed at
  // the closing brace returns its block to the DevicePool — freed outright
  // under VT_POOL_BYPASS=1 — while WriteKvCache is still memcpy-ing out of it
  // (ASan heap-use-after-free in ReshapeAndCacheKernel,
  // ISSUE-LOCAL-01M4FK2H2Y3D2TXADS8JPRN492).
  DBuf vs;
  if (p.attention_value_scale != 0.0 && p.attention_value_scale != 1.0) {
    vs = DBuf(d, DType::kBF16, {T, Hkv, Dv});
    vt::MulScalar(d.q, vs.t(), v3, p.attention_value_scale);
    v3 = vs.t();
  }

  // --- KV cache write ---
  Tensor k_cache = dense_attn::KvSlice(kv, d.q.device, 0);
  Tensor v_cache = dense_attn::KvSlice(kv, d.q.device, 1);
  dense_attn::WriteKvCache(d.q, kv, k3, v3, k_cache, v_cache,
                           si.slot_mapping.t());

  // --- Paged attention ---
  // Output shape is [T, Hq, Dv] — matching v_head_dim, NOT head_dim.
  DBuf attn(d, DType::kBF16, {T, Hq, Dv});
  vt::PagedAttentionArgs pa{};
  pa.scale = attn_scale;
  pa.causal = true;
  pa.query_start_loc_host = nullptr;  // TODO: thread from meta if available
  pa.max_seq_len = 0;                  // let the kernel read from device

  // `sink` must outlive the vt::PagedAttention call below: pa.attn_sink POINTS
  // at it and the kernel dereferences it (args.attn_sink->Ptr<float>(),
  // src/vt/cpu/cpu_paged_attn.cpp:138). Declaring it inside the
  // `if (w.has_sink_bias)` block leaves a dangling stack pointer (the same
  // escape class as `vs` above, ISSUE-LOCAL-01M4FK2H2Y3D2TXADS8JPRN492). The
  // house pattern is deepseek_v4_dsa.cpp:265-271: t_sink lives in the
  // enclosing scope for exactly this reason.
  Tensor sink;
  if (is_full) {
    pa.window_size = std::nullopt;
    pa.attn_sink = nullptr;
  } else {
    // SWA: sliding_window=128 → window of 128 tokens.
    // AttentionWindow{left, right} where (W-1, 0) = causal decoder window of W tokens.
    pa.window_size = vt::AttentionWindow{static_cast<int32_t>(p.sliding_window - 1), 0};
    if (w.has_sink_bias) {
      sink = ResidentWeight(d, w.sink_bias, {Hq});
      pa.attn_sink = &sink;
    } else {
      pa.attn_sink = nullptr;
    }
  }

  dense_attn::ApplyKvCacheQuant(pa, kv);
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache,
                     si.block_table.t(), si.seq_lens.t(),
                     si.query_start_loc.t(), pa);

  // --- o_proj ---
  // o_proj input dim = Hq * Dv = 64 * 128 = 8192 (NOT Hq * Dh).
  Tensor o_in = Reshape(attn.t(), {T, Hq * Dv});
  return Linear(d, o_in, w.o_proj, DType::kBF16);  // [T, H]
}

}  // namespace

// ---------------------------------------------------------------------------
// ForwardMiMoV2Device — the device forward pass.
// Defined in the vllm namespace (not anonymous) so the registry TU can call it.
// ---------------------------------------------------------------------------
ForwardLogits ForwardMiMoV2Device(
    const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const MiMoV2Weights& weights,
    const MultiKvCacheIndex* multi_kv,
    vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  const MiMoV2Params& p = weights.params;
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t vocab = p.vocab_size;

  VT_CHECK(T > 0, "MiMoV2: empty token batch");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "MiMoV2: positions size mismatch");

  // --- Embedding ---
  DBuf hidden_buf(d, DType::kBF16, {T, H});
  {
    DBuf ids(d, DType::kI32, {T}, token_ids.data());
    Tensor tab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
    vt::Embedding(d.q, hidden_buf.t(), tab, ids.t());
  }
  Tensor hidden = hidden_buf.t();
  std::shared_ptr<void> hidden_hold;

  // --- Residual stream ---
  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  const float eps = static_cast<float>(p.rms_norm_eps);

  // --- Per-layer loop ---
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const MiMoV2LayerWeights& lw = weights.layers[static_cast<size_t>(l)];

    // Resolve this layer's KV cache via multi_kv.
    const PagedKvCache* kv_ptr = nullptr;
    if (multi_kv != nullptr) {
      const std::string name =
          "model.layers." + std::to_string(l) + ".self_attn";
      const int64_t idx = multi_kv->Find(name);
      VT_CHECK(idx >= 0 && idx < static_cast<int64_t>(attn_kv.size()),
               "MiMoV2: KV cache not found for layer " + std::to_string(l));
      kv_ptr = &attn_kv[static_cast<size_t>(idx)];
    } else {
      kv_ptr = &attn_kv[static_cast<size_t>(l)];
    }
    const PagedKvCache& kv = *kv_ptr;

    // Build step inputs (positions, slot_mapping, block_table, seq_lens, query_start_loc).
    // We use a lightweight HfConfig shim to interface with BuildStepInputs.
    // Actually, we build the step inputs manually since MiMoV2 has per-layer RoPE.
    const int64_t T_l = T;
    DBuf d_positions(d, DType::kI32, {T_l}, positions.data());
    DBuf d_slot_mapping(d, DType::kI64, {T_l}, attn_meta.slot_mapping.data());
    DBuf d_block_table(d, DType::kI32,
                        {attn_meta.num_reqs, attn_meta.block_table_num_cols},
                        attn_meta.block_table_tensor.data());
    DBuf d_seq_lens(d, DType::kI32, {attn_meta.num_reqs},
                     attn_meta.seq_lens.data());
    DBuf d_query_start_loc(d, DType::kI32, {attn_meta.num_reqs + 1},
                            attn_meta.query_start_loc.data());

    dense_attn::StepInputs si;
    si.positions = std::move(d_positions);
    si.slot_mapping = std::move(d_slot_mapping);
    si.block_table = std::move(d_block_table);
    si.seq_lens = std::move(d_seq_lens);
    si.query_start_loc = std::move(d_query_start_loc);
    // cos_sin / cos_sin_bf16 / rope_row_idx are not needed — we use vt::RopeNeox
    // (in-place, computes its own frequencies from positions + RopeArgs).

    // --- input_layernorm + residual add (fused or standalone) ---
    DBuf dhn(d, DType::kBF16, {T, H});
    Tensor w_in = ResidentWeight(d, lw.input_layernorm, {H});
    Tensor dhn_t = dhn.t();
    Tensor res_t = res.t();
    if (dense_attn::FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, dhn_t, hidden, w_in, &res_t,
                     vt::kFusedAddRmsNormStd, eps);
    } else {
      vt::RmsNorm(d.q, dhn_t, hidden, w_in, vt::RmsNormArgs{eps, false}, &res_t);
    }

    // --- attention ---
    DBuf attn = AttentionBlock(d, lw.attn, p, l, dhn.t(), si.positions.t(),
                                si, kv, T);

    // --- post_attention_layernorm + residual add ---
    DBuf dh2(d, DType::kBF16, {T, H});
    Tensor w_post = ResidentWeight(d, lw.post_attention_layernorm, {H});
    Tensor dh2_t = dh2.t();
    Tensor attn_t = attn.t();
    if (dense_attn::FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, dh2_t, attn_t, w_post, &res_t,
                     vt::kFusedAddRmsNormStd, eps);
    } else {
      vt::RmsNorm(d.q, dh2_t, attn_t, w_post, vt::RmsNormArgs{eps, false}, &res_t);
    }

    // --- MLP / MoE ---
    DBuf mlp;
    if (lw.is_moe) {
      mlp = MoeBlock(d, lw.moe, p, dh2.t(), T);
    } else {
      mlp = DenseMlp(d, lw.dense_mlp, dh2.t(), T, H, p.intermediate_size);
    }

    // Update hidden for next layer.
    auto* held = new DBuf(std::move(mlp));
    hidden = held->t();
    hidden_hold = std::shared_ptr<void>(held, [](void* q) {
      delete static_cast<DBuf*>(q);
    });
  }

  // --- final norm + lm_head ---
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  Tensor dnorm_t = dnorm.t();
  Tensor res_t = res.t();
  if (dense_attn::FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dnorm_t, hidden, w_fn, &res_t,
                   vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dnorm_t, hidden, w_fn, vt::RmsNormArgs{eps, false}, &res_t);
  }

  const bool tied = p.tie_word_embeddings;
  Tensor lm;
  if (tied) {
    lm = ResidentWeight(d, weights.embed_tokens, {vocab, H});
  } else {
    if (weights.lm_head.IsExl3()) {
      // EXL3 lm_head path
    } else {
      lm = ResidentWeight(d, weights.lm_head.bf16);
    }
  }

  // --- logits_indices gather ---
  const bool do_gather =
      !logits_indices.empty() &&
      static_cast<int64_t>(logits_indices.size()) < T;
  const int64_t n_idx = static_cast<int64_t>(logits_indices.size());
  DBuf dgather(d, DType::kBF16, {do_gather ? n_idx : int64_t{0}, H});
  Tensor src = dnorm.t();
  if (do_gather) {
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    src = dgather.t();
  }
  const int64_t n_out = do_gather ? n_idx : T;

  // --- lm_head GEMM ---
  // Tied: embed_tokens loaded via LoadBf16Direct → [vocab, H] = [N, K] (raw).
  //   → MatmulBT: out[T, vocab] = src[T, H] @ embed^T[vocab, H].
  // Untied bf16: lm_head loaded via LoadBf16Transposed → [H, vocab] = [K, N].
  //   → Matmul: out[T, vocab] = src[T, H] @ lm[H, vocab].
  // Untied EXL3: Exl3MatmulD handles it.
  if (tied) {
    DBuf logits(d, DType::kF32, {n_out, vocab});
    vt::MatmulBT(d.q, logits.t(), src, lm);
    return WrapDeviceLogits(std::move(logits), n_out, vocab);
  }
  if (!weights.lm_head.IsExl3()) {
    // lm is ResidentWeight(d, weights.lm_head.bf16) → [H, vocab] = [K, N]
    DBuf logits(d, DType::kF32, {n_out, vocab});
    vt::Matmul(d.q, logits.t(), src, lm);
    return WrapDeviceLogits(std::move(logits), n_out, vocab);
  }
  // EXL3 lm_head
  DBuf logits = Linear(d, src, weights.lm_head, DType::kF32);
  return WrapDeviceLogits(std::move(logits), n_out, vocab);
}

}  // namespace vllm

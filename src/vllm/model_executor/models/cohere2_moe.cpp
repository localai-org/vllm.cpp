// Cohere2MoeForCausalLM forward: a parallel-block decoder with an interleaved
// per-layer sliding window, RoPE on the sliding and forced-RoPE prefix-dense
// layers only (NoPE elsewhere), a dense MLP prefix, and a sigmoid (or softmax)
// top-k MoE with optional shared experts. Tied logits scaled by logit_scale.
//
// Grounding: vllm/model_executor/models/cohere2_moe.py @ a7c23ac96d
//   Cohere2MoeDecoderLayer.forward (:369-384): residual = h; n = norm(h);
//     h = residual + attn(n) + mlp(n). ONE norm feeds both branches.
//   Cohere2MoeAttention (:205-246): qkv (no bias) -> split -> GPT-J RoPE when
//     `sliding_window or force_rope` -> attn(scale head_dim**-0.5,
//     per_layer_sliding_window) -> o_proj.
//   Cohere2Moe.forward (:319-328): gate -> FusedMoE(custom routing) with the
//     shared expert added -> halved when the strategy is "average".
//   token_choice_with_bias (:56-72): sigmoid(logits.float()) -> top-k ->
//     renormalize only when norm_topk_prob.
//   Cohere2MoeForCausalLM.compute_logits (:537-541): embed_tokens @ h, scaled.
//
// Shared seams: every linear goes through layers::UnquantizedLinearMethod, every
// SwiGLU MLP (dense prefix, shared expert, each routed expert) through
// layers::UnquantizedMlpGateUpMethod, the router through vt::MoeRouterTopK and
// the combine through vt::MoeCombine; the window through ResolveAttentionWindow
// (ENG-ATTENTION-WINDOW). dense_attn::AttnBlock is the Qwen3 block (qk-norm,
// NeoX RoPE on every layer) and cannot express a NoPE layer or GPT-J pairing, so
// the attention block composes the same vt ops the Command-R block does.
#include "vllm/model_executor/models/cohere2_moe.h"

#include <cmath>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/attention/attention.h"  // ResolveAttentionWindow
#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::DType;
using vt::Tensor;
using v1::CommonAttentionMetadata;

using namespace dense_attn;

// input_layernorm / final norm (select_norm_impl, cohere2_moe.py:97-103). Both
// arms accumulate in f32, scale by the f32 weight and round once to the input
// dtype (rms_norm_func :75-82, commandr.py layer_norm_func).
void NormInto(Dev d, DBuf& out, const Tensor& x, const OwnedTensor& weight,
              const Cohere2MoeParams& p) {
  Tensor wt = ResidentWeight(d, weight, {p.hidden_size});
  if (p.use_rms_norm) {
    vt::RmsNorm(d.q, out.t(), x, wt, vt::RmsNormArgs{p.norm_eps, false});
  } else {
    vt::LayerNorm(d.q, out.t(), x, &wt, /*bias=*/nullptr, vt::LayerNormArgs{p.norm_eps});
  }
}

DBuf Linear(Dev d, const OwnedTensor& w, const Tensor& x, DType out) {
  return layers::UnquantizedLinearMethod(&w).Apply(d, x, out);
}

// Cohere2MoeMLP.forward (:142-146): gate_up -> SiluAndMul -> down.
DBuf Mlp(Dev d, const Cohere2MoeMlpWeights& w, const Tensor& x, DType adt) {
  const int64_t I = w.down_proj.shape[1];  // raw-NK [H, I]
  DBuf act = layers::UnquantizedMlpGateUpMethod(&w.gate_up_proj, I, adt).Apply(d, x);
  return Linear(d, w.down_proj, act.t(), adt);
}

DBuf AttnBlock(Dev d, const Cohere2MoeWeights& mw, const Cohere2MoeAttnWeights& w,
               int64_t layer_idx, const Tensor& x, const StepInputs& si,
               const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const Cohere2MoeParams& p = mw.params;
  const DType adt = mw.compute_dtype;
  const int64_t Hq = p.num_heads, Hkv = p.num_kv_heads, Dh = p.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "cohere2_moe: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "cohere2_moe: KV cache head dims mismatch config");

  DBuf qkv = Linear(d, w.qkv_proj, x, adt);  // QKVParallelLinear(bias=False)
  DBuf q(d, adt, {T, qdim});
  DBuf k(d, adt, {T, kdim});
  DBuf v(d, adt, {T, kdim});
  vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());

  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  const size_t li = static_cast<size_t>(layer_idx);
  // `if self.sliding_window or self.force_rope` (:242): a full-attention MoE
  // layer is NoPE; the full-attention prefix-dense layer keeps RoPE when
  // prefix_dense_sliding_window_pattern == 1.
  if (p.rope[li]) {
    Tensor cache = ResidentWeight(d, mw.rope_cos_sin);
    vt::RopeArgs ra;
    ra.rotary_dim = static_cast<int>(Dh);
    ra.is_neox_style = false;  // get_rope(..., is_neox_style=False) (:202)
    Tensor kr = k3;
    vt::RopeFromCache(d.q, q3, &kr, si.positions.t(), cache, ra);
  }

  Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});
  Tensor kw = k3;
  Tensor vw = v3;
  DBuf kcast(d, kv.dtype, {T, Hkv, Dh});
  DBuf vcast(d, kv.dtype, {T, Hkv, Dh});
  if (kv.dtype != adt) {
    if (kv.dtype == DType::kBF16) {
      vt::CastBf16(d.q, kcast.t(), k3);
      vt::CastBf16(d.q, vcast.t(), v3);
    } else {
      vt::CastF32(d.q, kcast.t(), k3);
      vt::CastF32(d.q, vcast.t(), v3);
    }
    kw = kcast.t();
    vw = vcast.t();
  }
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, si.slot_mapping.t());

  DBuf attn(d, adt, {T, Hq, Dh});
  // scaling = head_dim**-0.5 (:177).
  vt::PagedAttentionArgs pa{1.0F / std::sqrt(static_cast<float>(Dh)), meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  // per_layer_sliding_window = sliding_window + 1 (:211, :231). The resolver maps
  // W to FlashAttention's (W - 1, 0), so a query sees itself and the previous
  // `sliding_window` keys (flash_attn.py:1076-1080 @ a7c23ac96d).
  if (p.window[li].has_value())
    pa.window_size = ResolveAttentionWindow(p.window[li], std::nullopt,
                                            v1::AttentionType::kDecoder,
                                            DisableSlidingWindowActive());
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);
  return Linear(d, w.o_proj, Reshape(attn.t(), {T, qdim}), adt);  // o_proj, no bias
}

// Cohere2Moe.forward (:319-328).
DBuf MoeBlock(Dev d, const Cohere2MoeWeights& mw, const Cohere2MoeSparseWeights& w,
              const Tensor& x, int64_t T) {
  const Cohere2MoeParams& p = mw.params;
  const DType adt = mw.compute_dtype;
  const int64_t H = p.hidden_size, E = p.num_experts, K = p.top_k;

  // router_logits = gate(x): a ReplicatedLinear in the model dtype (:278-285).
  DBuf logits = Linear(d, w.router, x, adt);
  vt::MoeRouterTopKArgs args;
  args.top_k = static_cast<int>(K);
  args.renormalize = p.norm_topk_prob;  // renormalize=norm_topk_prob (:311)
  if (p.sigmoid_router) {
    // token_choice_with_bias (:56-72): sigmoid over ALL experts, top-k, no bias,
    // no grouping. vt::MoeRouterTopK defines sigmoid scoring on its grouped path
    // only; ONE group that survives is exactly the ungrouped top-k (the group
    // mask keeps every expert), the form kimi_linear.cpp already routes through.
    args.scoring_func = vt::MoeScoringFunc::kSigmoid;
    args.num_expert_group = 1;
    args.topk_group = 1;
  }
  DBuf topw(d, DType::kF32, {T, K});
  DBuf topi(d, DType::kI32, {T, K});
  vt::MoeRouterTopK(d.q, topw.t(), topi.t(), logits.t(), args);

  // Routed experts: gather each expert's tokens, run its SwiGLU MLP, scatter the
  // [n, H] result into the (token, slot) rows of expert_out [T, K, H].
  std::vector<int32_t> ids(static_cast<size_t>(T * K));
  topi.Download(d, ids.data());
  std::vector<std::vector<int64_t>> slots(static_cast<size_t>(E));
  for (int64_t s = 0; s < T * K; ++s) slots[static_cast<size_t>(ids[static_cast<size_t>(s)])].push_back(s);
  DBuf expert_out(d, adt, {T, K, H});
  const size_t row = static_cast<size_t>(H) * vt::SizeOf(adt);
  for (int64_t e = 0; e < E; ++e) {
    const std::vector<int64_t>& list = slots[static_cast<size_t>(e)];
    if (list.empty()) continue;
    const int64_t n = static_cast<int64_t>(list.size());
    DBuf xg(d, adt, {n, H});
    for (int64_t r = 0; r < n; ++r)
      d.b.Copy(d.q, static_cast<char*>(xg.ptr()) + static_cast<size_t>(r) * row,
               static_cast<const char*>(x.data) + static_cast<size_t>(list[static_cast<size_t>(r)] / K) * row,
               row);
    DBuf o = Mlp(d, w.experts[static_cast<size_t>(e)], xg.t(), adt);
    for (int64_t r = 0; r < n; ++r)
      d.b.Copy(d.q, static_cast<char*>(expert_out.ptr()) + static_cast<size_t>(list[static_cast<size_t>(r)]) * row,
               static_cast<const char*>(o.ptr()) + static_cast<size_t>(r) * row, row);
  }
  DBuf out(d, adt, {T, H});
  vt::MoeCombine(d.q, out.t(), expert_out.t(), topw.t(), /*shared=*/nullptr);
  if (w.has_shared) {
    // FusedMoE returns shared_output + routed_output (:323-325): each rounded to
    // the model dtype, then added, which is why this is not MoeCombine's
    // single-store shared operand.
    DBuf sh = Mlp(d, w.shared, x, adt);
    vt::Add(d.q, out.t(), out.t(), sh.t());
    // "average" halves the WHOLE output, routed + shared (:326-327).
    if (p.shared_average) vt::MulScalar(d.q, out.t(), out.t(), 0.5);
  }
  return out;
}

// Cohere2MoeDecoderLayer.forward (:369-384). `hidden` is updated in place.
void RunLayer(Dev d, const Cohere2MoeWeights& mw, const Cohere2MoeLayerWeights& lw,
              int64_t layer_idx, DBuf& hidden, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const DType adt = mw.compute_dtype;
  DBuf normed(d, adt, {T, mw.params.hidden_size});
  NormInto(d, normed, hidden.t(), lw.input_layernorm, mw.params);
  DBuf attn = AttnBlock(d, mw, lw.attn, layer_idx, normed.t(), si, meta, kv, T);
  DBuf mlp = lw.dense ? Mlp(d, lw.mlp, normed.t(), adt)
                      : MoeBlock(d, mw, lw.moe, normed.t(), T);
  // hidden = residual + attn + mlp, left to right (:383).
  vt::Add(d.q, hidden.t(), hidden.t(), attn.t());
  vt::Add(d.q, hidden.t(), hidden.t(), mlp.t());
}

void CheckWindowSwitch(const Cohere2MoeParams& p) {
  if (!DisableSlidingWindowActive()) return;
  for (const auto& w : p.window)
    VT_CHECK(!w.has_value(),
             "cohere2_moe: disable_sliding_window is not supported: the pinned "
             "cohere2_moe.py computes config.sliding_window + 1 for every "
             "sliding_attention layer and cannot run with the window disabled");
}

// The step inputs without the per-step NeoX cos|sin cache: this model ropes from
// its own resident GPT-J cache, indexed by the real positions.
StepInputs StepInputsNoRope(Dev d, const std::vector<int32_t>& positions,
                            const CommonAttentionMetadata& meta) {
  HfConfig no_rope;
  no_rope.rotary_dim = 0;
  return BuildStepInputs(d, positions, meta, no_rope);
}

DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& meta,
                 const std::vector<PagedKvCache>& attn_kv, const Cohere2MoeWeights& w,
                 const std::vector<int32_t>& logits_indices) {
  const Cohere2MoeParams& p = w.params;
  const DType adt = w.compute_dtype;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size, V = p.vocab_size;
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "cohere2_moe: positions length must match token_ids");
  VT_CHECK(static_cast<int64_t>(attn_kv.size()) == p.num_layers &&
               static_cast<int64_t>(w.layers.size()) == p.num_layers,
           "cohere2_moe: one layer weight set and one PagedKvCache per layer required");
  CheckWindowSwitch(p);

  DBuf hidden(d, adt, {T, H});
  {
    Tensor table = ResidentWeight(d, w.embed_tokens, {V, H});
    DBuf ids(d, DType::kI32, {T}, token_ids.data());
    vt::Embedding(d.q, hidden.t(), table, ids.t());
  }
  StepInputs si = StepInputsNoRope(d, positions, meta);
  for (int64_t l = 0; l < p.num_layers; ++l)
    RunLayer(d, w, w.layers[static_cast<size_t>(l)], l, hidden, si, meta,
             attn_kv[static_cast<size_t>(l)], T);

  DBuf normed(d, adt, {T, H});
  NormInto(d, normed, hidden.t(), w.final_norm, p);

  Tensor src = normed.t();
  const bool gather =
      !logits_indices.empty() && static_cast<int64_t>(logits_indices.size()) < T;
  DBuf rows(d, adt, gather ? std::vector<int64_t>{static_cast<int64_t>(logits_indices.size()), H}
                           : std::vector<int64_t>{1, 1});
  if (gather) {
    const size_t rb = static_cast<size_t>(H) * vt::SizeOf(adt);
    for (size_t s = 0; s < logits_indices.size(); ++s)
      d.b.Copy(d.q, static_cast<char*>(rows.ptr()) + s * rb,
               static_cast<const char*>(normed.ptr()) + static_cast<size_t>(logits_indices[s]) * rb,
               rb);
    src = rows.t();
  }
  // Tied logits: embed_tokens @ h (compute_logits :537-541). f32 because the
  // ForwardLogits carrier and the sampler read f32 in this tree (the same
  // store every dense registration here makes).
  DBuf logits = Linear(d, w.embed_tokens, src, DType::kF32);
  // LogitsProcessor(scale=logit_scale) (:510-513).
  if (p.logit_scale != 1.0) vt::MulScalar(d.q, logits.t(), logits.t(), p.logit_scale);
  return logits;
}

// A scratch single-request prefill over T tokens, for the per-layer entry.
struct ScratchPrefill {
  std::vector<float> kv_bytes;
  PagedKvCache kv;
  CommonAttentionMetadata meta;
  ScratchPrefill(const Cohere2MoeParams& p, int64_t T) {
    const int64_t bs = 16, blocks = (T + bs - 1) / bs;
    kv_bytes.assign(static_cast<size_t>(blocks * 2 * bs * p.num_kv_heads * p.head_dim), 0.0F);
    kv.data = kv_bytes.data();
    kv.dtype = DType::kF32;
    kv.num_blocks = blocks;
    kv.block_size = bs;
    kv.num_kv_heads = p.num_kv_heads;
    kv.head_size = p.head_dim;
    meta.num_reqs = 1;
    meta.num_actual_tokens = static_cast<int>(T);
    meta.query_start_loc = {0, static_cast<int32_t>(T)};
    meta.query_start_loc_cpu = meta.query_start_loc;
    meta.seq_lens = {static_cast<int32_t>(T)};
    meta.seq_lens_cpu = meta.seq_lens;
    meta.max_query_len = static_cast<int>(T);
    meta.max_seq_len = static_cast<int>(T);
    meta.block_table_num_cols = static_cast<int>(blocks);
    for (int64_t b = 0; b < blocks; ++b) meta.block_table_tensor.push_back(static_cast<int32_t>(b));
    for (int64_t t = 0; t < T; ++t) meta.slot_mapping.push_back(t);
    meta.causal = true;
  }
};

DBuf Upload(Dev d, DType dt, const std::vector<float>& host, int64_t T, int64_t H) {
  DBuf f(d, DType::kF32, {T, H}, host.data());
  if (dt == DType::kF32) return f;
  DBuf out(d, dt, {T, H});
  vt::CastBf16(d.q, out.t(), f.t());
  return out;
}

std::vector<float> DownloadF32(Dev d, DBuf& x, int64_t T, int64_t H) {
  std::vector<float> out(static_cast<size_t>(T * H));
  if (x.t().dtype == DType::kF32) {
    x.Download(d, out.data());
    return out;
  }
  DBuf f(d, DType::kF32, {T, H});
  vt::CastF32(d.q, f.t(), x.t());
  f.Download(d, out.data());
  return out;
}

}  // namespace

std::vector<float> Cohere2MoeModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Cohere2MoeWeights& weights, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf logits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                            logits_indices);
  std::vector<float> out(static_cast<size_t>(logits.t().Numel()));
  logits.Download(d, out.data());
  return out;
}

ForwardLogits Cohere2MoeModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Cohere2MoeWeights& weights, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf logits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                            logits_indices);
  ForwardLogits fl;
  fl.rows = logits.t().shape[0];
  fl.vocab = weights.params.vocab_size;
  fl.device_tensor = logits.t();
  fl.device_storage = logits.ReleaseShared();
  return fl;
}

std::vector<float> Cohere2MoeModel::DecoderLayer(const Cohere2MoeWeights& weights,
                                                 const Cohere2MoeLayerWeights& layer,
                                                 int64_t layer_idx,
                                                 const std::vector<float>& hidden_in,
                                                 const std::vector<int32_t>& positions,
                                                 vt::Queue& queue) {
  const Cohere2MoeParams& p = weights.params;
  VT_CHECK(layer_idx >= 0 && layer_idx < p.num_layers, "cohere2_moe: layer out of range");
  const int64_t T = static_cast<int64_t>(positions.size());
  VT_CHECK(static_cast<int64_t>(hidden_in.size()) == T * p.hidden_size,
           "cohere2_moe: hidden_in must be [T, hidden_size]");
  CheckWindowSwitch(p);
  Dev d{vt::GetBackend(queue.device.type), queue};
  ScratchPrefill scratch(p, T);
  DBuf hidden = Upload(d, weights.compute_dtype, hidden_in, T, p.hidden_size);
  StepInputs si = StepInputsNoRope(d, positions, scratch.meta);
  RunLayer(d, weights, layer, layer_idx, hidden, si, scratch.meta, scratch.kv, T);
  return DownloadF32(d, hidden, T, p.hidden_size);
}

std::vector<float> Cohere2MoeModel::FinalNorm(const Cohere2MoeWeights& weights,
                                              const std::vector<float>& hidden_in,
                                              vt::Queue& queue) {
  const int64_t H = weights.params.hidden_size;
  const int64_t T = static_cast<int64_t>(hidden_in.size()) / H;
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf x = Upload(d, weights.compute_dtype, hidden_in, T, H);
  DBuf out(d, weights.compute_dtype, {T, H});
  NormInto(d, out, x.t(), weights.final_norm, weights.params);
  return DownloadF32(d, out, T, H);
}

}  // namespace vllm

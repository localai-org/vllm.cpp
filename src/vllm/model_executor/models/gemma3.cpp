// Gemma-3 text (`Gemma3ForCausalLM`) dense forward — sweep W2, the first
// Gemma-family token-exact gate vehicle. Composed from the public vt:: ops and
// the shared dense-attention device glue (dense_attn_block.h: Dev/DBuf/
// ResidentWeight/KvSlice/StepInputs), with a Gemma-specific attention block and
// sandwich-norm decoder layer.
//
// Grounding: vllm/model_executor/models/gemma3.py @ e24d1b24. The Gemma
// vocabulary vs the Qwen3-dense path (qwen3.cpp):
//   - GemmaRMSNorm (1+w, fp32) at EVERY norm — RmsNormArgs{eps, gemma=true};
//   - the Gemma-2/3 SANDWICH layout (GLM-4 pattern, glm4.cpp): input +
//     pre_feedforward are fused add+RMSNorm; post_attention + post_feedforward
//     are standalone norms on the sublayer output before it re-enters residual;
//   - per-head Gemma q/k RMSNorm (gemma=true) before RoPE;
//   - GeGLU MLP (vt::GeluAndMul, gelu_pytorch_tanh) instead of SwiGLU;
//   - embedding scaled by sqrt(hidden) cast to bf16 (vt::MulScalar);
//   - attention scale = query_pre_attn_scalar**-0.5 (not head_dim**-0.5);
//   - DUAL per-layer RoPE theta: rope_theta on full-attn layers,
//     rope_local_base_freq on sliding layers, routed by sliding_window_pattern;
//   - per-layer sliding window (masked at the FA kernel; inert for contexts
//     < sliding_window, e.g. the short gate battery);
//   - NO attention output gate, NO attention logit soft-cap, NO qkv bias.
//
// On ROCm, linear-RoPE models use the executing e126687a9a compiled Gemma
// boundaries: scaled embedding variance, Q/K norm plus RoPE, erf GeGLU, and
// sandwich residual expressions. All model storage remains BF16. Other paths
// retain the existing materialized per-operation boundaries.
#include "vllm/model_executor/models/gemma3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/attention/attention.h"
#include "vllm/model_executor/layers/linear.h"            // UnquantizedMlpGateUpGeluMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/glue
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/device_pool.h"       // Pool
#include "vllm/model_executor/models/gemma3_decode_graph.h"
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/compiled_gemma.h"
#include "vt/ops.h"
#include "vt/persistent_step_input.h"
#include "vt/recipes.h"  // kFusedAddRmsNorm (Tier-B2)

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;
using namespace dense_attn;  // Dev/DBuf/ResidentWeight/KvSlice/StepInputs/Reshape

// Raw config.json scalar readers (Gemma-specific fields not yet typed on
// HfConfig; flat doc for the standalone Gemma3ForCausalLM checkpoint).
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number()) return fallback;
  return it->get<double>();
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number_integer()) return fallback;
  return it->get<int64_t>();
}

// The measured compiler partition is the linear-RoPE Gemma3 path. Keep the
// existing non-linear models and materialized backends on their prior route.
bool CompiledGemma(Dev d, const HfConfig& cfg) {
  return cfg.rope_parameters.rope_type == "linear" &&
         d.b.GetResidualNormPolicy() == vt::ResidualNormPolicy::kCompiledExpression;
}

// Per-layer Gemma-3 routing derived from the config.
struct Gemma3Layout {
  int64_t sliding_window_pattern;  // 1-in-N layers are full attention
  double rope_theta_global;        // full-attn layers
  double rope_theta_local;         // sliding layers (rope_local_base_freq)
  float attn_scale;                // query_pre_attn_scalar**-0.5
  int64_t sliding_window;          // window length (tokens)
  bool IsSliding(int64_t l) const {
    return ((l + 1) % sliding_window_pattern) != 0;
  }
};

Gemma3Layout MakeLayout(const HfConfig& cfg) {
  Gemma3Layout g;
  g.sliding_window_pattern = RawInt(cfg.raw, "sliding_window_pattern", 6);
  if (g.sliding_window_pattern <= 0) g.sliding_window_pattern = 1;
  g.rope_theta_global = cfg.rope_theta;
  g.rope_theta_local = RawDouble(cfg.raw, "rope_local_base_freq", 10000.0);
  const double qpas =
      RawDouble(cfg.raw, "query_pre_attn_scalar", static_cast<double>(cfg.head_dim));
  g.attn_scale = static_cast<float>(1.0 / std::sqrt(qpas));
  g.sliding_window = cfg.sliding_window.value_or(RawInt(cfg.raw, "sliding_window", 0));
  return g;
}

// One Gemma-3 self-attention block (gemma3.py::Gemma3Attention.forward). `dhn` is
// the input-normed hidden [T,H] bf16; returns the o_proj output [T,H] bf16.
// `rope_base` is this layer's RoPE theta (dual: global vs local). `attn_scale` is
// query_pre_attn_scalar**-0.5. `sliding_window` (>0) masks to the last W keys.
DBuf Gemma3AttnBlock(Dev d, const Gemma3AttnWeights& w, const HfConfig& cfg,
                     const Tensor& dhn, const StepInputs& si,
                     const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                     int64_t T, double rope_base, const OwnedTensor& rope_cache, float attn_scale,
                     std::optional<int64_t> sliding_window) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const DType adt = DType::kBF16;  // vLLM runs Gemma-3 bf16 (per-op stores)
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "gemma3: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "gemma3: KV cache head dims mismatch config");

  // Merged QKVParallelLinear (no bias): one raw-NK owner [qdim+2kdim, H]. D1 folds
  // the shared-input q/k/v GEMMs to ONE MatmulBT over the merged owner + a
  // contiguous QkvSplit (MergedQkvEnabled(), VT_QWEN3_QKV_MERGE default ON; =0
  // restores the byte-identical 3-shard, whose tiny GQA k/v GEMMs mirror vLLM's
  // single qkv GEMM + split numerically).
  const bool compiled = CompiledGemma(d, cfg);
  DBuf q, k, v;
  std::optional<DBuf> packed_view;
  Tensor q_tensor, k_tensor, v_tensor;
  Tensor wqkv = ResidentWeight(d, w.qkv_proj);
  if (compiled && T == 1 && MergedQkvEnabled()) {
    // A one-token projection has three contiguous slices. Keep its owner alive
    // through cache writes and attention, including captured graph execution.
    packed_view.emplace(d, adt, std::vector<int64_t>{T, qdim + 2 * kdim});
    vt::MatmulBT(d.q, packed_view->t(), dhn, wqkv);
    q_tensor = Reshape(packed_view->t().Slice(1, 0, qdim), {T, qdim});
    k_tensor = Reshape(packed_view->t().Slice(1, qdim, qdim + kdim), {T, kdim});
    v_tensor = Reshape(packed_view->t().Slice(1, qdim + kdim, qdim + 2 * kdim), {T, kdim});
  } else {
    q = DBuf(d, adt, {T, qdim});
    k = DBuf(d, adt, {T, kdim});
    v = DBuf(d, adt, {T, kdim});
    {
      if (MergedQkvEnabled()) {
        DBuf qkv(d, adt, {T, qdim + 2 * kdim});
        vt::MatmulBT(d.q, qkv.t(), dhn, wqkv);
        vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());
      } else {
        Tensor wq = wqkv.Slice(0, 0, qdim);
        Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
        Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
        vt::MatmulBT(d.q, q.t(), dhn, wq);
        vt::MatmulBT(d.q, k.t(), dhn, wk);
        vt::MatmulBT(d.q, v.t(), dhn, wv);
      }
    }

    q_tensor = q.t();
    k_tensor = k.t();
    v_tensor = v.t();
  }

  // Per-head Gemma q/k RMSNorm (GemmaRMSNorm(head_dim), 1+w) BEFORE RoPE, then
  // full-dim NeoX RoPE with this layer's theta.
  Tensor q2 = Reshape(q_tensor, {T * Hq, Dh});
  Tensor k2 = Reshape(k_tensor, {T * Hkv, Dh});
  Tensor q3 = Reshape(q_tensor, {T, Hq, Dh});
  Tensor k3 = Reshape(k_tensor, {T, Hkv, Dh});
  Tensor wqn = ResidentWeight(d, w.q_norm, {Dh});
  Tensor wkn = ResidentWeight(d, w.k_norm, {Dh});
  if (!compiled || cfg.rope_parameters.rope_type != "linear") {
    vt::RmsNorm(d.q, q2, q2, wqn, vt::RmsNormArgs{eps, true});
    vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, true});
  }
  vt::RopeArgs ra;
  ra.base = static_cast<float>(rope_base);
  ra.rotary_dim = static_cast<int>(Dh);  // full rotary_dim = head_dim
  if (cfg.rope_parameters.rope_type == "linear") {
    VT_CHECK(!rope_cache.Empty(), "gemma3: linear RoPE requires its loaded cache");
    Tensor cache =
        compiled ? ResidentWeight(
                       d, rope_cache, {},
                       [&](Tensor& target) {
                         // base.py:89-112 computes the cache on-device before its BF16 cast.
                         // CPU libm differs even after BF16 rounding at long positions.
                         const int64_t rows = target.shape[0];
                         std::vector<int64_t> positions(static_cast<size_t>(rows));
                         std::iota(positions.begin(), positions.end(), int64_t{0});
                         DBuf indices(d, DType::kI64, {rows}, positions.data());
                         // FP32 cache construction mirrors the primary; the resident cache is BF16.
                         DBuf generated(d, DType::kF32, {rows, Dh});
                         auto args = ra;
                         args.linear_scaling_factor =
                             sliding_window.has_value()
                                 ? 1.f
                                 : static_cast<float>(cfg.rope_parameters.factor.value_or(1.));
                         vt::RopeCosSinCache(d.q, generated.t(), indices.t(), args);
                         vt::CastBf16(d.q, target, generated.t());
                         d.b.Synchronize(d.q);  // The temporary host positions back the upload.
                       })
                 : ResidentWeight(d, rope_cache);
    if (compiled) {
      vt::FusedBinding binding{};
      binding.n = 8;
      binding.op[0] = &q2;
      binding.op[1] = &wqn;
      binding.op[2] = &k2;
      binding.op[3] = &wkn;
      binding.op[4] = &q3;
      binding.op[5] = &k3;
      binding.op[6] = &cache;
      binding.op[7] = const_cast<Tensor*>(&si.positions.t());
      vt::FusedParams params{};
      params.eps = eps;
      params.rope = ra;
      vt::FusedChain(d.q, vt::kAttnQkNormRopeGemma, binding, params);
    } else {
      vt::RopeFromCache(d.q, q3, &k3, si.positions.t(), cache, ra);
    }
  } else {
    vt::RopeNeox(d.q, q3, k3, si.positions.t(), ra);
  }

  // Write rope'd K + V into the paged cache. On the bf16 default (== cache dtype)
  // no cast; an f32 cache (CPU-synthetic A/B) down/up-casts K/V to match.
  Tensor v3 = Reshape(v_tensor, {T, Hkv, Dh});
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

  // Paged GQA attention: scale = query_pre_attn_scalar**-0.5; sliding layers
  // mask to the last `sliding_window` keys (FA window convention (W-1, 0)).
  DBuf attn(d, adt, {T, Hq, Dh});
  vt::PagedAttentionArgs pa{attn_scale, meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  // ENG-ATTENTION-WINDOW (#2388): see gemma2.cpp -- same shape, same reason.
  if (sliding_window.has_value() && *sliding_window > 0)
    pa.window_size = ResolveAttentionWindow(
        /*per_layer=*/std::nullopt, sliding_window,
        v1::AttentionType::kDecoder,
        /*disable_model_sliding_window=*/DisableSlidingWindowActive());
  // ROCM_ATTN passes W-1 into prefix_prefill's strict distance < window mask
  // (rocm_attn.py:309,475; prefix_prefill.py:435-437 at e126687a9a).
  // AttentionWindow itself remains inclusive for all shared-op callers.
  if (compiled && pa.window_size && pa.window_size->left > 0) --pa.window_size->left;
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  // o_proj (RowParallelLinear, no bias): [T, Hq*Dh] -> [T,H] bf16.
  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  return o;
}

// Gemma-3 GeGLU MLP (gemma3.py::Gemma3MLP): merged gate_up -> GeluAndMul(tanh) ->
// down. `dh2` is the pre-FF-normed hidden [T,H] bf16.
DBuf Gemma3MlpBlock(Dev d, const Gemma3MlpWeights& w, const HfConfig& cfg,
                    const Tensor& dh2, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  // gate_up MatmulBT -> GeluAndMul(tanh) via the SHARED bf16 GeGLU gate-up MLP seam
  // (layers::UnquantizedMlpGateUpGeluMethod). Byte-for-byte the inline sequence —
  // folds Gemma-3 onto the shared MlpGateUpMethodBase descriptor. (Tier-C1,
  // arch-fusion-fold-plan-2026-07-30.)
  DBuf act = layers::UnquantizedMlpGateUpGeluMethod(&w.gate_up_proj, I, CompiledGemma(d, cfg))
                 .Apply(d, dh2);
  Tensor wd = ResidentWeight(d, w.down_proj);
  DBuf down(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, down.t(), act.t(), wd);
  return down;
}

// One Gemma-3 decoder layer (gemma3.py::Gemma3DecoderLayer.forward), the sandwich
// pattern (GLM-4, glm4.cpp:158-188) with GemmaRMSNorm (1+w) throughout:
//   res += hidden;   dhn = gemmaNorm(res)      # input_layernorm  (fused)
//   attn = Attn(dhn)
//   attn = gemmaNorm(attn)                      # post_attention   (standalone)
//   res += attn;     dh2 = gemmaNorm(res)       # pre_feedforward  (fused)
//   mlp  = Mlp(dh2)
//   hidden = gemmaNorm(mlp)                      # post_feedforward (standalone)
void RunLayer(Dev d, const Gemma3LayerWeights& layer, const HfConfig& cfg, const Gemma3Layout& g,
              const OwnedTensor& rope_cache, const OwnedTensor& next_norm, int64_t l, DBuf& hidden,
              DBuf& res, const StepInputs& si, const CommonAttentionMetadata& meta,
              const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const vt::RmsNormArgs gemma{eps, true};

  // input_layernorm (fused add+GemmaRMSNorm): res += hidden; dhn = norm(res).
  // Routed through the shared fusion catalog (vt::kFusedAddRmsNorm); the Tier-0
  // composite is byte-identical to the standalone `vt::RmsNorm(..., &res)`. (Tier-B2,
  // arch-fusion-fold-plan-2026-07-30.)
  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  const bool compiled = CompiledGemma(d, cfg);
  DBuf dhn(d, DType::kBF16, {T, H});
  if (compiled)
    dhn = std::move(hidden);  // Already normalized at the preceding partition boundary.
  else if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNorm, eps);
  else
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, gemma, &res.t());

  // attention (dual-theta rope + per-layer sliding window)
  const bool sliding = g.IsSliding(l);
  const double rope_base = sliding ? g.rope_theta_local : g.rope_theta_global;
  std::optional<int64_t> window;
  if (sliding) window = g.sliding_window;
  DBuf attn = Gemma3AttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T,
                              rope_base, rope_cache, g.attn_scale, window);

  if (compiled) {
    Tensor post_attn = ResidentWeight(d, layer.post_attention_layernorm, {H});
    Tensor pre_ff = ResidentWeight(d, layer.pre_feedforward_layernorm, {H});
    DBuf dh2(d, DType::kBF16, {T, H});
    vt::FusedChain(d.q, dh2.t(), vt::SandwichNormInputs{attn.t(), res.t(), post_attn, pre_ff}, eps);
    DBuf mlp = Gemma3MlpBlock(d, layer.mlp, cfg, dh2.t(), T);
    Tensor post_ff = ResidentWeight(d, layer.post_feedforward_layernorm, {H});
    Tensor next = ResidentWeight(d, next_norm, {H});
    hidden = DBuf(d, DType::kBF16, {T, H});
    vt::FusedChain(d.q, hidden.t(),
                   vt::SandwichNormInputs{attn.t(), res.t(), post_attn, next, &mlp.t(), &post_ff},
                   eps, l + 1 < cfg.num_hidden_layers ? &res.t() : nullptr);
    return;
  }

  // post_attention_layernorm (STANDALONE GemmaRMSNorm, sandwich): attn = norm(attn).
  // NOT-FUSABLE onto kFusedAddRmsNorm — a sublayer-output post-norm with NO residual
  // add. Left standalone by design (fold-plan §B2 hazard). (Tier-B2)
  Tensor w_pa = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf attn_n(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, attn_n.t(), attn.t(), w_pa, gemma);

  // pre_feedforward_layernorm (fused add+GemmaRMSNorm): res += attn_n; dh2=norm(res).
  Tensor w_pf = ResidentWeight(d, layer.pre_feedforward_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dh2.t(), attn_n.t(), w_pf, &res.t(), vt::kFusedAddRmsNorm, eps);
  else
    vt::RmsNorm(d.q, dh2.t(), attn_n.t(), w_pf, gemma, &res.t());

  // GeGLU MLP
  DBuf mlp = Gemma3MlpBlock(d, layer.mlp, cfg, dh2.t(), T);

  // post_feedforward_layernorm (STANDALONE GemmaRMSNorm, sandwich): hidden=norm(mlp).
  // NOT-FUSABLE (no residual add). STANDALONE. (Tier-B2)
  Tensor w_pff = ResidentWeight(d, layer.post_feedforward_layernorm, {H});
  hidden = DBuf(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, hidden.t(), mlp.t(), w_pff, gemma);
}

// GatherRows: gather the idx-indexed rows of `src` [.,H] into contiguous `dst`.
void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

DBuf ForwardLayers(Dev d, DBuf hidden, const StepInputs& si,
                   const CommonAttentionMetadata& attn_meta,
                   const std::vector<PagedKvCache>& attn_kv, const Gemma3Weights& weights,
                   const HfConfig& config, const std::vector<int32_t>& logits_indices) {
  const int64_t T = hidden.t().shape[0], H = config.hidden_size, vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  const Gemma3Layout g = MakeLayout(config);
  const float nsqrt = std::sqrt(static_cast<float>(H));
  const double normalizer = static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(nsqrt)));
  const bool compiled = CompiledGemma(d, config);
  DBuf res(d, DType::kBF16, {T, H});
  if (compiled) {
    DBuf normed(d, DType::kBF16, {T, H});
    Tensor first_norm = ResidentWeight(d, weights.layers.front().input_layernorm, {H});
    vt::FusedChain(d.q, normed.t(), hidden.t(), first_norm,
                   vt::ScaledRmsNormArgs{static_cast<float>(normalizer), eps}, res.t());
    hidden = std::move(normed);
  } else {
    vt::MulScalar(d.q, hidden.t(), hidden.t(), normalizer);
    res.Zero(d);
  }

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, g,
             g.IsSliding(l) ? weights.rope_local : weights.rope_global,
             l + 1 < config.num_hidden_layers ? weights.layers[l + 1].input_layernorm
                                              : weights.final_norm,
             l, hidden, res, si, attn_meta, attn_kv[static_cast<size_t>(l)], T);

  // Final GemmaRMSNorm over the fused stream (res += hidden; gemma norm) via catalog.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (compiled)
    dnorm = std::move(hidden);  // Final sandwich partition emitted the normalized state.
  else if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNorm, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, true}, &res.t());

  // lm_head. Tied (gemma-3-1b): logits = hidden @ embed_tokens^T (MatmulBT over
  // the [vocab,H] embed table). Note the embed table used here is the UNSCALED
  // one — the sqrt(H) normalizer applies only to the input embedding lookup.
  const bool tied = weights.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);

  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kBF16, do_gather ? std::vector<int64_t>{
                                                static_cast<int64_t>(logits_indices.size()), H}
                                          : std::vector<int64_t>{1, 1});
  if (do_gather) {
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];
  // vLLM LogitsProcessor._apply_head preserves the BF16 model dtype.
  DBuf projected(d, DType::kBF16, {n_out, vocab});
  if (tied)
    vt::MatmulBT(d.q, projected.t(), src, lm);
  else
    vt::Matmul(d.q, projected.t(), src, lm);
  // FP32 is the existing runner/sampler boundary, after the model-dtype round.
  DBuf logits(d, DType::kF32, {n_out, vocab});
  vt::CastF32(d.q, logits.t(), projected.t());
  return logits;
}
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions, const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv, const Gemma3Weights& weights,
                 const HfConfig& config, const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "gemma3: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "gemma3: one PagedKvCache per layer required");

  // Embed then scale by sqrt(hidden) cast to bf16 (gemma3.py:328-341). The
  // normalizer is bf16(sqrt(H)); the multiply is f32 then rounded to bf16,
  // matching torch's bf16-scalar multiply.
  DBuf hidden(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, hidden, token_ids, weights.embed_tokens, vocab, H,
                "gemma3 embed");
  }
  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);
  return ForwardLayers(d, std::move(hidden), si, attn_meta, attn_kv, weights, config,
                       logits_indices);
}

ForwardLogits WrapDeviceLogits(Dev d, DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  fl.device_storage = dlogits.ReleaseShared();
  (void)d;
  return fl;
}

}  // namespace

// vLLM e126687a9a uses a full decode graph. Keep inputs outside capture and
// retain the existing Gemma layer body inside the shared graph scope.
struct Gemma3DecodeGraph::Impl {
  Impl(const Gemma3Weights& w, const HfConfig& c, Queue q)
      : weights(w),
        config(c),
        queue(q),
        backend(vt::GetBackend(q.device)),
        pool(ActivePool(backend)) {}
  ~Impl() { Reset(); }
  void Reset() {
    backend.Synchronize(queue);
    graph.Reset();
    pool.UnpinForGraph(backend, pinned);
    pinned.clear();
    logits = {};
    for (auto& cell : input_cells) cell.Unbind();
    token_ids.reset();
    inputs.reset();
    embedded.reset();
    warm = false;
  }
  bool SameCache(const ModelForwardInput& in) const {
    if (!inputs || in.attn_meta.block_table_num_cols > columns || in.attn_kv.size() != cache.size())
      return false;
    for (size_t i = 0; i < cache.size(); ++i) {
      const auto& a = cache[i];
      const auto& b = in.attn_kv[i];
      if (a.data != b.data || a.dtype != b.dtype || a.num_blocks != b.num_blocks ||
          a.block_size != b.block_size || a.num_kv_heads != b.num_kv_heads ||
          a.head_size != b.head_size)
        return false;
    }
    return true;
  }
  const Gemma3Weights& weights;
  const HfConfig& config;
  Queue queue;
  Backend& backend;
  DevicePool& pool;
  CommonAttentionMetadata metadata;
  std::vector<PagedKvCache> cache;
  int columns = 0;
  std::optional<DBuf> embedded;
  std::optional<DBuf> token_ids;
  std::optional<StepInputs> inputs;
  std::array<vt::PersistentStepInput, 6> input_cells;
  ForwardLogits logits;
  vt::BreakableGraph graph;
  DevicePool::StepDemand demand;
  std::vector<std::pair<size_t, void*>> pinned;
  bool warm = false;
};

Gemma3DecodeGraph::Gemma3DecodeGraph(const Gemma3Weights& weights, const HfConfig& config,
                                     Queue queue)
    : impl_(std::make_unique<Impl>(weights, config, queue)) {}
Gemma3DecodeGraph::~Gemma3DecodeGraph() = default;
bool Gemma3DecodeGraph::UsesQueue(const Queue& queue) const {
  return impl_->queue.device == queue.device && impl_->queue.handle == queue.handle &&
         impl_->queue.id == queue.id;
}

ForwardLogits Gemma3DecodeGraph::Step(const ModelForwardInput& in) {
  auto& s = *impl_;
  VT_CHECK(in.positions.size() == 1 && in.attn_meta.num_reqs == 1 &&
               in.attn_meta.num_actual_tokens == 1 && in.attn_meta.slot_mapping.size() == 1 &&
               in.attn_meta.seq_lens.size() == 1 && in.attn_meta.seq_lens[0] > 0 &&
               in.attn_meta.query_start_loc == std::vector<int32_t>({0, 1}) &&
               in.attn_meta.block_table_num_cols > 0 &&
               in.attn_meta.block_table_tensor.size() ==
                   static_cast<size_t>(in.attn_meta.block_table_num_cols),
           "gemma3 decode graph: invalid single-query metadata");
  Dev d{s.backend, s.queue};
  ActivePoolScope pool_scope(&s.pool);
  const bool new_inputs = !s.SameCache(in);
  bool table_changed = new_inputs;
  if (!new_inputs) {
    const auto& incoming = in.attn_meta.block_table_tensor;
    const auto& previous = s.metadata.block_table_tensor;
    table_changed = !std::equal(incoming.begin(), incoming.end(), previous.begin()) ||
                    std::any_of(previous.begin() + incoming.size(), previous.end(),
                                [](int32_t block) { return block != 0; });
  }
  if (new_inputs) {
    s.Reset();
    s.columns = static_cast<int>(std::max<int64_t>(in.attn_meta.block_table_num_cols, in.attn_kv.front().num_blocks));
    s.cache = in.attn_kv;
    s.metadata = in.attn_meta;
    s.metadata.block_table_num_cols = s.columns;
    s.metadata.block_table_tensor.resize(static_cast<size_t>(s.columns), 0);
    s.inputs.emplace(BuildStepInputs(d, in.positions, s.metadata, s.config));
    s.embedded.emplace(d, DType::kBF16, std::vector<int64_t>{1, s.config.hidden_size});
    s.token_ids.emplace(d, DType::kI32, std::vector<int64_t>{1});
    std::array<DBuf*, 6> buffers{&s.inputs->positions,       &s.inputs->slot_mapping,
                                 &s.inputs->block_table,     &s.inputs->seq_lens,
                                 &s.inputs->query_start_loc, &*s.token_ids};
    for (size_t i = 0; i < buffers.size(); ++i)
      s.input_cells[i].Bind(s.backend, buffers[i]->ptr(), buffers[i]->t().Bytes());
  }
  // Reserve the full table once. A new request or a new logical block changes
  // its contents, without changing a captured device address or row stride.
  s.metadata = in.attn_meta;
  s.metadata.block_table_num_cols = s.columns;
  s.metadata.block_table_tensor.resize(static_cast<size_t>(s.columns), 0);
  auto upload = [&](size_t index, const auto& source) {
    auto& cell = s.input_cells[index];
    VT_CHECK(cell.capacity() == source.size() * sizeof(source[0]),
             "gemma3 decode graph: input shape changed");
    cell.RefreshFromHost(d.q, source.data(), cell.capacity());
  };
  upload(0, in.positions);
  upload(1, in.attn_meta.slot_mapping);
  if (table_changed) upload(2, s.metadata.block_table_tensor);
  upload(3, in.attn_meta.seq_lens);
  if (new_inputs) upload(4, in.attn_meta.query_start_loc);
  upload(5, in.token_ids);
  // Embedding validates IDs with a synchronized bounds check. It drains these
  // uploads before the next step can overwrite the persistent pinned staging.
  Tensor table =
      ResidentWeight(d, s.weights.embed_tokens, {s.config.vocab_size, s.config.hidden_size});
  vt::Embedding(d.q, s.embedded->t(), table, s.token_ids->t());
  if (s.graph.captured()) {
    s.graph.Replay(s.queue);
    return s.logits;
  }
  auto forward = [&] {
    DBuf hidden(d, DType::kBF16, {1, s.config.hidden_size});
    d.b.Copy(d.q, hidden.ptr(), s.embedded->ptr(), hidden.t().Bytes());
    return ForwardLayers(d, std::move(hidden), *s.inputs, s.metadata, s.cache, s.weights, s.config,
                         {});
  };
  if (!s.warm) {
    s.pool.MarkStepBoundary();
    auto result = forward();
    s.demand = s.pool.StepDemandProfile();
    s.warm = true;
    return WrapDeviceLogits(d, std::move(result), 1, s.config.vocab_size);
  }
  s.pool.PreGrowForCapture(s.backend, s.demand);
  std::optional<DBuf> result;
  {
    vt::GraphCaptureScope scope(s.backend, s.queue, s.graph, vt::GraphCaptureMode::kFull);
    result.emplace(forward());
  }
  if (s.graph.capture_failed()) {
    const auto error = s.graph.capture_error();
    s.graph.Reset();
    if (error) std::rethrow_exception(error);
    throw std::runtime_error("gemma3 decode graph capture failed before executing its output");
  }
  if (!s.graph.captured()) {
    s.warm = false;
    return WrapDeviceLogits(d, std::move(*result), 1, s.config.vocab_size);
  }
  s.logits = WrapDeviceLogits(d, std::move(*result), 1, s.config.vocab_size);
  s.pinned = s.pool.PinForGraph(s.backend, s.demand);
  s.graph.Replay(s.queue);
  return s.logits;
}

std::optional<ForwardLogits> Gemma3DecodeGraphForward(std::unique_ptr<Gemma3DecodeGraph>& graph,
                                                      const Gemma3Weights& weights,
                                                      const ModelForwardInput& input) {
  if (!input.pure_decode || !input.gather_logits || input.device_token_ids != nullptr ||
      input.token_ids.size() != 1 || input.num_reqs != 1 || !input.attn_meta.causal ||
      input.config.rope_parameters.rope_type != "linear" || input.config.head_dim != 256 ||
      input.config.num_attention_heads != 2 * input.config.num_key_value_heads ||
      (!input.logits_indices.empty() && input.logits_indices != std::vector<int32_t>{0}) ||
      input.attn_kv.empty() ||
      input.attn_kv.size() != static_cast<size_t>(input.config.num_hidden_layers) ||
      !CompiledGemma(Dev{vt::GetBackend(input.queue.device), input.queue}, input.config) ||
      !vt::GraphCaptureEnabled() || !DevicePool::SupportsCapture())
    return std::nullopt;
  for (const auto& kv : input.attn_kv)
    if (kv.dtype != DType::kBF16 || (kv.block_size != 16 && kv.block_size != 32))
      return std::nullopt;
  if (!graph || !graph->UsesQueue(input.queue)) {
    const auto& platform = platforms::GetPlatform(input.queue.device.type);
    if (!platform.support_static_graph_for_model(input.config.architectures,
                                                 input.queue.device.index))
      return std::nullopt;
    graph = std::make_unique<Gemma3DecodeGraph>(weights, input.config, input.queue);
  }
  return graph->Step(input);
}

std::vector<float> Gemma3Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma3Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Gemma3Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma3Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm

// Gemma-2 text (`Gemma2ForCausalLM`) dense forward — sweep W4. The Gemma-2 gate
// vehicle, which PROVES the two soft-cap primitives (W3): an ATTENTION logit
// soft-cap threaded through PagedAttentionArgs into the paged-attention score,
// and a FINAL logit soft-cap on the output logits. Composed from the public vt::
// ops and the shared dense-attention device glue (dense_attn_block.h), with a
// Gemma-specific sandwich-norm decoder layer.
//
// Grounding: vllm/model_executor/models/gemma2.py @ e24d1b24. The Gemma-2
// vocabulary vs the Gemma-3 path (gemma3.cpp): the sandwich-norm layout, GeGLU
// MLP, sqrt(hidden) embed-scale, tied lm_head, qpas scaling and interleaved
// sliding window are IDENTICAL; the delta is
//   - NO per-head q/k RMSNorm (Gemma-3 added it; Gemma-2 does not have it);
//   - a SINGLE per-model RoPE theta (no dual local/global cache);
//   - an ATTENTION logit soft-cap (attn_logit_softcapping, 50.0) applied on the
//     scaled pre-softmax score inside paged attention — the token-changing
//     primitive this gate proves;
//   - a FINAL logit soft-cap (final_logit_softcapping, 30.0): logits =
//     cap*tanh(logits/cap). Monotone -> greedy argmax invariant, applied for
//     faithfulness and unit-gated bit-exact (W3).
//
// Numeric contract: bf16 per-op, matching vLLM's stores (dense_attn_block.h).
#include "vllm/model_executor/layers/attention/attention.h"
#include "vllm/model_executor/models/gemma2.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/linear.h"             // UnquantizedMlpGateUpGeluMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/glue
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/device_pool.h"       // Pool
#include "vllm/model_executor/models/qwen3_5_common.h"     // HostLogits
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/recipes.h"  // kFusedAddRmsNorm (Tier-B2)

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;
using namespace dense_attn;  // Dev/DBuf/ResidentWeight/KvSlice/StepInputs/Reshape

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

// Per-layer Gemma-2 routing derived from the config.
struct Gemma2Layout {
  double rope_theta;               // single per-model theta
  float attn_scale;                // query_pre_attn_scalar**-0.5
  float attn_logit_softcap;        // 0 => no attention soft-cap
  float final_logit_softcap;       // 0 => no final soft-cap
  int64_t sliding_window;          // window length (tokens)
  std::vector<bool> is_sliding;    // per-layer (from config.layer_types)
  bool IsSliding(int64_t l) const {
    return l >= 0 && static_cast<size_t>(l) < is_sliding.size() &&
           is_sliding[static_cast<size_t>(l)];
  }
};

Gemma2Layout MakeLayout(const HfConfig& cfg) {
  Gemma2Layout g;
  g.rope_theta = cfg.rope_theta;
  const double qpas =
      RawDouble(cfg.raw, "query_pre_attn_scalar", static_cast<double>(cfg.head_dim));
  g.attn_scale = static_cast<float>(1.0 / std::sqrt(qpas));
  g.attn_logit_softcap =
      static_cast<float>(RawDouble(cfg.raw, "attn_logit_softcapping", 0.0));
  // VT_GEMMA2_ATTN_SOFTCAP=0 forces the attention soft-cap off (same-binary A/B to
  // attribute a token delta to the cap vs the base attention numerics).
  if (const char* e = std::getenv("VT_GEMMA2_ATTN_SOFTCAP"); e != nullptr && e[0] == '0')
    g.attn_logit_softcap = 0.0f;
  g.final_logit_softcap =
      static_cast<float>(RawDouble(cfg.raw, "final_logit_softcapping", 0.0));
  g.sliding_window = cfg.sliding_window.value_or(RawInt(cfg.raw, "sliding_window", 0));

  // Interleaved local/global routing (gemma2.py:154). Prefer config.layer_types;
  // fall back to HF's alternating default (even-index layers sliding). Inert for
  // the short gate context, so the exact assignment does not affect correctness.
  const int64_t L = cfg.num_hidden_layers;
  g.is_sliding.assign(static_cast<size_t>(L), false);
  const auto it = cfg.raw.find("layer_types");
  if (it != cfg.raw.end() && it->is_array()) {
    for (int64_t l = 0; l < L && static_cast<size_t>(l) < it->size(); ++l)
      g.is_sliding[static_cast<size_t>(l)] =
          it->at(static_cast<size_t>(l)).is_string() &&
          it->at(static_cast<size_t>(l)).get<std::string>() == "sliding_attention";
  } else {
    for (int64_t l = 0; l < L; ++l) g.is_sliding[static_cast<size_t>(l)] = (l % 2 == 0);
  }
  return g;
}

// One Gemma-2 self-attention block (gemma2.py::Gemma2Attention.forward). `dhn` is
// the input-normed hidden [T,H] bf16; returns the o_proj output [T,H] bf16.
DBuf Gemma2AttnBlock(Dev d, const Gemma2AttnWeights& w, const HfConfig& cfg,
                     const Tensor& dhn, const StepInputs& si,
                     const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                     int64_t T, double rope_base, float attn_scale,
                     float attn_logit_softcap, std::optional<int64_t> sliding_window) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const DType adt = DType::kBF16;
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "gemma2: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "gemma2: KV cache head dims mismatch config");

  // Merged QKVParallelLinear (no bias): [qdim+2kdim, H]. D1 folds the shared-input
  // q/k/v GEMMs to ONE MatmulBT over the merged owner + a contiguous QkvSplit
  // (MergedQkvEnabled(), VT_QWEN3_QKV_MERGE default ON; =0 = byte-identical 3-shard).
  DBuf q(d, adt, {T, qdim});
  DBuf k(d, adt, {T, kdim});
  DBuf v(d, adt, {T, kdim});
  {
    Tensor wqkv = ResidentWeight(d, w.qkv_proj);
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

  // NeoX RoPE (single theta, full rotary_dim = head_dim). NO q/k norm (Gemma-2).
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  vt::RopeArgs ra;
  ra.base = static_cast<float>(rope_base);
  ra.rotary_dim = static_cast<int>(Dh);
  vt::RopeNeox(d.q, q3, k3, si.positions.t(), ra);

  // Write rope'd K + V into the paged cache (cast to cache dtype when different).
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

  // Paged GQA attention: scale = query_pre_attn_scalar**-0.5; ATTENTION logit
  // soft-cap (cap*tanh(score/cap)); sliding layers mask to the last W keys.
  DBuf attn(d, adt, {T, Hq, Dh});
  vt::PagedAttentionArgs pa{attn_scale, meta.causal};
  pa.logits_soft_cap = attn_logit_softcap;
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  // ENG-ATTENTION-WINDOW (#2388): the window DERIVATION is the shared
  // resolver's; the predicate stays here because it is model shape, not the
  // window rule. W3 replaced the local `VT_GEMMA2_SLIDING` reader with the ONE
  // model-agnostic switch, so Gemma-4, OLMo-2 and Muse-Glimmer now have the same
  // control that only Gemma-2 and Gemma-3 carried.
  if (sliding_window.has_value() && *sliding_window > 0)
    pa.window_size = ResolveAttentionWindow(
        /*per_layer=*/std::nullopt, sliding_window,
        v1::AttentionType::kDecoder,
        /*disable_model_sliding_window=*/DisableSlidingWindowActive());
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  // o_proj (RowParallelLinear, no bias): [T, Hq*Dh] -> [T,H] bf16.
  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  return o;
}

// Gemma-2 GeGLU MLP (gemma2.py::Gemma2MLP): merged gate_up -> GeluAndMul(tanh) ->
// down. `dh2` is the pre-FF-normed hidden [T,H] bf16.
DBuf Gemma2MlpBlock(Dev d, const Gemma2MlpWeights& w, const HfConfig& cfg,
                    const Tensor& dh2, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  // gate_up MatmulBT -> GeluAndMul(tanh) via the SHARED bf16 GeGLU gate-up MLP seam
  // (layers::UnquantizedMlpGateUpGeluMethod). Byte-for-byte the inline sequence —
  // folds Gemma-2 onto the shared MlpGateUpMethodBase descriptor. (Tier-C1,
  // arch-fusion-fold-plan-2026-07-30.)
  DBuf act = layers::UnquantizedMlpGateUpGeluMethod(&w.gate_up_proj, I).Apply(d, dh2);
  Tensor wd = ResidentWeight(d, w.down_proj);
  DBuf down(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, down.t(), act.t(), wd);
  return down;
}

// One Gemma-2 decoder layer (gemma2.py::Gemma2DecoderLayer.forward), the sandwich
// pattern with GemmaRMSNorm (1+w) throughout (identical to Gemma-3):
//   res += hidden;   dhn = gemmaNorm(res)      # input_layernorm  (fused)
//   attn = Attn(dhn)
//   attn = gemmaNorm(attn)                      # post_attention   (standalone)
//   res += attn;     dh2 = gemmaNorm(res)       # pre_feedforward  (fused)
//   mlp  = Mlp(dh2)
//   hidden = gemmaNorm(mlp)                      # post_feedforward (standalone)
void RunLayer(Dev d, const Gemma2LayerWeights& layer, const HfConfig& cfg,
              const Gemma2Layout& g, int64_t l, DBuf& hidden, DBuf& res,
              const StepInputs& si, const CommonAttentionMetadata& meta,
              const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const vt::RmsNormArgs gemma{eps, true};

  // input_layernorm: fused residual-add + GemmaRMSNorm (1+w) via the shared catalog
  // (vt::kFusedAddRmsNorm). Byte-identical to the standalone `vt::RmsNorm(..., &res)`.
  // (Tier-B2, arch-fusion-fold-plan-2026-07-30.)
  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNorm, eps);
  else
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, gemma, &res.t());

  const bool sliding = g.IsSliding(l);
  std::optional<int64_t> window;
  if (sliding) window = g.sliding_window;
  DBuf attn = Gemma2AttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T,
                              g.rope_theta, g.attn_scale, g.attn_logit_softcap, window);

  // post_attention_layernorm: SANDWICH sublayer-output post-norm — norm(attn) with
  // NO residual add. NOT-FUSABLE onto kFusedAddRmsNorm (that recipe's step0 is a
  // residual add this site does not have); folding it would be an INCORRECT fold.
  // Left STANDALONE by design (fold-plan §B2 hazard 4). (Tier-B2)
  Tensor w_pa = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf attn_n(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, attn_n.t(), attn.t(), w_pa, gemma);

  // pre_feedforward_layernorm: fused residual-add + GemmaRMSNorm (residual-carrying).
  Tensor w_pf = ResidentWeight(d, layer.pre_feedforward_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dh2.t(), attn_n.t(), w_pf, &res.t(), vt::kFusedAddRmsNorm, eps);
  else
    vt::RmsNorm(d.q, dh2.t(), attn_n.t(), w_pf, gemma, &res.t());

  DBuf mlp = Gemma2MlpBlock(d, layer.mlp, cfg, dh2.t(), T);

  // post_feedforward_layernorm: SANDWICH sublayer-output post-norm — norm(mlp) with
  // NO residual add. NOT-FUSABLE (same reason as post_attention). STANDALONE. (Tier-B2)
  Tensor w_pff = ResidentWeight(d, layer.post_feedforward_layernorm, {H});
  hidden = DBuf(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, hidden.t(), mlp.t(), w_pff, gemma);
}

void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const Gemma2Weights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "gemma2: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "gemma2: one PagedKvCache per layer required");

  const Gemma2Layout g = MakeLayout(config);

  // Embed then scale by sqrt(hidden) cast to bf16 (gemma2.py:276-283).
  DBuf hidden(d, DType::kBF16, {T, H});
  {
    EmbedGather(d, hidden, token_ids, weights.embed_tokens, vocab, H,
                "gemma2 embed");
  }
  const float nsqrt = std::sqrt(static_cast<float>(H));
  const double normalizer = static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(nsqrt)));
  vt::MulScalar(d.q, hidden.t(), hidden.t(), normalizer);

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, g, l, hidden, res, si,
             attn_meta, attn_kv[static_cast<size_t>(l)], T);

  // final norm: fused residual-add + GemmaRMSNorm (residual-carrying) via the catalog.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled())
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNorm, eps);
  else
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, true}, &res.t());

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
  DBuf logits(d, DType::kF32, {n_out, vocab});
  if (tied)
    vt::MatmulBT(d.q, logits.t(), src, lm);
  else
    vt::Matmul(d.q, logits.t(), src, lm);

  // FINAL logit soft-cap (gemma2.py:344-345). Monotone -> greedy argmax
  // invariant, but applied for faithfulness (and unit-gated bit-exact, W3).
  if (g.final_logit_softcap > 0.0f)
    vt::SoftCap(d.q, logits.t(), logits.t(), g.final_logit_softcap);
  return logits;
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

std::vector<float> Gemma2Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma2Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Gemma2Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma2Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm

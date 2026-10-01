// Gemma-4 text backbone forward (`Gemma4ForConditionalGeneration` language_model
// stack, unsloth/gemma-4-E4B-it) — MODEL-GEMMA4 G1. Composed from the public vt::
// ops + the shared dense-attention device glue (dense_attn_block.h). Grounding:
// vllm/model_executor/models/gemma4.py (Gemma4MLP :224-254, Gemma4Attention
// :374-552, Gemma4DecoderLayer :555-767, Gemma4Model embed/PLE :845-928 + forward
// :1289-1370), gemma4_rope.py (proportional RoPE), layernorm.py::RMSNorm (PLAIN,
// x*w — NOT the Gemma (1+w) offset). See .agents/specs/gemma4-multimodal.md.
//
// Primitive-by-primitive vs the Gemma-2/3 path (gemma3.cpp):
//   - PLAIN RMSNorm everywhere (vt::RmsNormArgs{eps,false}); V-norm is
//     weight-less (a ones[Dh] weight = identity scale).
//   - standalone-norm + explicit-add residual (NOT the gemma2/3 fused sandwich).
//   - Per-Layer Embeddings: a second embed table + projection combined once
//     (:845-898), then a per-layer gate/proj/norm fused into the residual
//     after the MLP (:753-761). The gate `gelu(gate_lin(h)) * ple_l` reuses
//     vt::GeluAndMul on [gate_lin | ple_l] (no elementwise-Mul op needed).
//   - heterogeneous per-layer head_dim (256 sliding / 512 full), scaling=1.0.
//   - YOCO KV-sharing: shared layers read the target layer's K/V cache and
//     compute NO K/V of their own.
//   - dual RoPE: full layers use the "proportional" cos/sin cache (head_dim
//     denominator + zero-padded non-rotary pairs); sliding layers standard neox.
//   - GeGLU MLP, sqrt(hidden) embed normalizer, per-layer scalar, final logit
//     soft-cap 30.0, tied lm_head.
//
// G1b STATUS: the runner now allocates a PER-LAYER KV head_dim
// (KVCacheConfig::per_layer_attn_specs, consumed in runner.cpp initialize_kv_cache),
// so the per-layer VT_CHECK(kv.head_size == Dh) below is satisfied on both the
// 256 (sliding) and 512 (global) layers and the strict e2e gate RUNS. YOCO
// KV-sharing is handled in the forward (kv_idx = target for shared layers); the
// shared layers' own — unused — buffers are still allocated (memory-only G1c
// residual, not a correctness gap).
#include "vllm/model_executor/layers/attention/attention.h"
#include "vllm/model_executor/models/gemma4.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/linear.h"             // UnquantizedMlpGateUpGeluMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/glue
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/device_pool.h"       // Pool
#include "vllm/model_executor/models/gemma4_moe.h"
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/fused_ops.h"

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

// The engine keeps HfConfig::raw as the FULL config.json (hf_config.cpp:414),
// while the mm-wrapper nests the language-model scalars under `text_config`.
// Gemma-4's per-arch fields (global_head_dim, hidden_size_per_layer_input,
// layer_types, rope_parameters, sliding_window, *_softcapping) therefore live in
// raw["text_config"], NOT at the top level — read them through this view.
const nlohmann::json& TextCfg(const nlohmann::json& raw) {
  const auto it = raw.find("text_config");
  if (it != raw.end() && it->is_object()) return *it;
  return raw;
}

// Read a per-layer-type rope field (rope_parameters[type][key]) with a fallback.
double RopeField(const nlohmann::json& doc, const char* type, const char* key,
                 double fallback) {
  const auto rp = doc.find("rope_parameters");
  if (rp == doc.end() || !rp->is_object()) return fallback;
  const auto t = rp->find(type);
  if (t == rp->end() || !t->is_object()) return fallback;
  const auto k = t->find(key);
  if (k == t->end() || k->is_null() || !k->is_number()) return fallback;
  return k->get<double>();
}

// Read a bf16 scalar [1] weight to host float (layer_scalar).
float ReadBf16Scalar(const OwnedTensor& w) {
  VT_CHECK(w.HasHostBytes(), "gemma4: layer_scalar host bytes required");
  const auto* p = static_cast<const uint16_t*>(w.View().data);
  return vt::BF16ToF32(*p);
}

// Config-derived layout.
struct Gemma4Layout {
  int64_t hidden = 0;
  int64_t num_layers = 0;
  int64_t num_q_heads = 0;
  int64_t num_kv_heads = 0;
  int64_t head_dim_sliding = 0;   // 256
  int64_t head_dim_full = 0;      // 512
  int64_t ple_dim = 0;            // hidden_size_per_layer_input (256)
  int64_t sliding_window = 0;
  double rope_theta_full = 1000000.0;
  double rope_partial_full = 0.25;  // partial_rotary_factor for full layers
  double rope_theta_sliding = 10000.0;
  float final_logit_softcap = 0.0f;
  float attn_logit_softcap = 0.0f;
  float embed_scale_ple = 0.0f;     // sqrt(ple_dim)
  double proj_scale = 0.0;          // hidden^-0.5
  double input_scale = 0.0;         // rsqrt(2)
};

Gemma4Layout MakeLayout(const HfConfig& cfg) {
  Gemma4Layout g;
  const nlohmann::json& raw = TextCfg(cfg.raw);
  g.hidden = cfg.hidden_size;
  g.num_layers = cfg.num_hidden_layers;
  g.num_q_heads = cfg.num_attention_heads;
  g.num_kv_heads = cfg.num_key_value_heads;
  g.head_dim_sliding = cfg.head_dim;
  g.head_dim_full = RawInt(raw, "global_head_dim", cfg.head_dim);
  g.ple_dim = RawInt(raw, "hidden_size_per_layer_input", 0);
  g.sliding_window = cfg.sliding_window.value_or(RawInt(raw, "sliding_window", 0));
  g.rope_theta_full = RopeField(raw, "full_attention", "rope_theta", 1000000.0);
  g.rope_partial_full =
      RopeField(raw, "full_attention", "partial_rotary_factor", 1.0);
  g.rope_theta_sliding =
      RopeField(raw, "sliding_attention", "rope_theta", 10000.0);
  g.final_logit_softcap =
      static_cast<float>(RawDouble(raw, "final_logit_softcapping", 0.0));
  g.attn_logit_softcap =
      static_cast<float>(RawDouble(raw, "attn_logit_softcapping", 0.0));
  g.embed_scale_ple = static_cast<float>(std::sqrt(static_cast<double>(g.ple_dim)));
  g.proj_scale = std::pow(static_cast<double>(g.hidden), -0.5);
  g.input_scale = 1.0 / std::sqrt(2.0);
  return g;
}

// Build the Gemma-4 "proportional" cos|sin cache for full-attention layers on
// host (gemma4_rope.py::Gemma4RotaryEmbedding). head_dim = Dh (512), rope_angles
// = rotary_dim/2 = int(Dh*partial)/2 non-zero pairs, the rest zero-padded to
// identity; angle denominator is Dh (not rotary_dim). Layout matches
// RopeCosSinCache: cos_sin[t, i]=cos, cos_sin[t, Dh/2 + i]=sin, over Dh pairs...
// stored as [P, Dh] with first Dh/2 = cos, second Dh/2 = sin. Double angle -> f32.
std::vector<float> ProportionalRopeCosSinHost(const Gemma4Layout& g, int64_t Dh,
                                              int64_t max_pos) {
  const int64_t pairs = Dh / 2;
  const int64_t rope_angles =
      static_cast<int64_t>(g.rope_partial_full * static_cast<double>(Dh)) / 2;
  const double base = g.rope_theta_full;
  std::vector<double> inv_freq(static_cast<size_t>(pairs), 0.0);
  for (int64_t j = 0; j < rope_angles && j < pairs; ++j) {
    const double exponent = static_cast<double>(2 * j) / static_cast<double>(Dh);
    inv_freq[static_cast<size_t>(j)] = 1.0 / std::pow(base, exponent);
  }
  const int64_t P = max_pos + 1;
  std::vector<float> host(static_cast<size_t>(P) * static_cast<size_t>(Dh), 0.0f);
  for (int64_t t = 0; t < P; ++t) {
    for (int64_t i = 0; i < pairs; ++i) {
      const double angle = static_cast<double>(t) * inv_freq[static_cast<size_t>(i)];
      host[static_cast<size_t>(t) * static_cast<size_t>(Dh) + static_cast<size_t>(i)] =
          static_cast<float>(std::cos(angle));
      host[static_cast<size_t>(t) * static_cast<size_t>(Dh) +
           static_cast<size_t>(pairs + i)] = static_cast<float>(std::sin(angle));
    }
  }
  return host;
}

DBuf BuildProportionalRopeCache(Dev d, const Gemma4Layout& g, int64_t Dh,
                                int64_t max_pos) {
  const std::vector<float> f32 = ProportionalRopeCosSinHost(g, Dh, max_pos);
  // NOTE: cache dtype bf16 to match the bf16 q/k it rotates (RopeFromCache wants
  // q/k/cache same dtype). vLLM keeps f32 cos/sin — a named bf16-rounding nuance.
  std::vector<uint16_t> host(f32.size());
  for (size_t i = 0; i < f32.size(); ++i) host[i] = vt::F32ToBF16(f32[i]);
  DBuf cache(d, DType::kBF16, {max_pos + 1, Dh}, host.data());
  return cache;
}

// Copy a strided source-row block into a contiguous destination sub-block. Used
// to assemble the [T, 2*ple] GeluAndMul input from the fresh gate GEMM (dense)
// and the per-layer slice of the [T, L, ple] PLE-input tensor (strided).
void CopyRow(Dev d, void* dst, const void* src, size_t bytes) {
  d.b.Copy(d.q, dst, src, bytes);
}

// One Gemma-4 self-attention block. `Dh` is this layer's head_dim (256/512).
// `kv` is the cache to attend (this layer's for non-shared, the target layer's
// for YOCO-shared). `rope_full` selects the proportional cache path.
DBuf Gemma4AttnBlock(Dev d, const Gemma4LayerWeights& w, const Gemma4Layout& g,
                     const Tensor& dhn, const StepInputs& si,
                     const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                     int64_t T, int64_t Dh, bool rope_full,
                     const Tensor& ones_dh, const Tensor* prop_cache,
                     std::optional<int64_t> sliding_window,
                     double rope_theta_sliding) {
  const int64_t H = g.hidden;
  const int64_t Hq = g.num_q_heads;
  const int64_t Hkv = w.num_kv_heads > 0 ? w.num_kv_heads : g.num_kv_heads;
  const float eps = 1e-6f;  // rms_norm_eps (E4B)
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  const DType adt = DType::kBF16;
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "gemma4: KV cache head dims mismatch this layer (heterogeneous KV — "
           "runner must allocate per-layer head_dim; see gemma4.h G1 note)");

  // TLS temps across layers (decode T=1 thrash).
  struct AttnTls {
    int dev = -1;
    int64_t T = 0, qdim = 0, kdim = 0, Hq = 0, Dh = 0;
    std::optional<DBuf> q, k, v, qkv, attn;
  };
  static thread_local AttnTls tls;
  if (tls.dev != d.q.device.index || tls.T != T || tls.qdim != qdim || tls.kdim != kdim ||
      tls.Hq != Hq || tls.Dh != Dh) {
    tls.q.emplace(d, adt, std::vector<int64_t>{T, qdim});
    tls.k.emplace(d, adt, std::vector<int64_t>{T, kdim});
    tls.v.emplace(d, adt, std::vector<int64_t>{T, kdim});
    tls.qkv.emplace(d, adt, std::vector<int64_t>{T, qdim + 2 * kdim});
    tls.attn.emplace(d, adt, std::vector<int64_t>{T, Hq, Dh});
    tls.dev = d.q.device.index;
    tls.T = T;
    tls.qdim = qdim;
    tls.kdim = kdim;
    tls.Hq = Hq;
    tls.Dh = Dh;
  }
  DBuf& q = *tls.q;
  DBuf& k = *tls.k;
  DBuf& v = *tls.v;
  DBuf& qkv = *tls.qkv;
  DBuf& attn = *tls.attn;

  // Merged QKVParallelLinear: one MatmulBT + QkvSplit (default).
  {
    Tensor wqkv = ResidentWeight(d, w.attn.qkv_proj);
    if (MergedQkvEnabled()) {
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

  // Q norm (PLAIN per-head RMSNorm over Dh). Always applied.
  Tensor q2 = Reshape(q.t(), {T * Hq, Dh});
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  Tensor wqn = ResidentWeight(d, w.attn.q_norm, {Dh});
  vt::RmsNorm(d.q, q2, q2, wqn, vt::RmsNormArgs{eps, false});

  if (!w.is_kv_shared) {
    // K norm (plain), then RoPE(q,k), then weight-less V norm (ones weight).
    Tensor k2 = Reshape(k.t(), {T * Hkv, Dh});
    Tensor wkn = ResidentWeight(d, w.attn.k_norm, {Dh});
    vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
  }

  // RoPE (both q,k roped; k discarded on shared layers, matching vLLM's
  // rotary_emb(pos,q,k)[0] which still rotates k).
  if (rope_full) {
    VT_CHECK(prop_cache != nullptr, "gemma4: full layer needs proportional cache");
    vt::RopeArgs ra;
    ra.rotary_dim = static_cast<int>(Dh);
    ra.is_neox_style = true;
    Tensor kk = k3;
    vt::RopeFromCache(d.q, q3, w.is_kv_shared ? nullptr : &kk, si.positions.t(),
                      *prop_cache, ra);
  } else {
    vt::RopeArgs ra;
    ra.base = static_cast<float>(rope_theta_sliding);
    ra.rotary_dim = static_cast<int>(Dh);
    vt::RopeNeox(d.q, q3, k3, si.positions.t(), ra);
  }

  if (!w.is_kv_shared) {
    // Weight-less V norm (ones weight = identity scale), then cache K/V.
    Tensor v2 = Reshape(v.t(), {T * Hkv, Dh});
    vt::RmsNorm(d.q, v2, v2, ones_dh, vt::RmsNormArgs{eps, false});

    Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});
    Tensor kw = k3;
    Tensor vw = v3;
    // Cast buffers only when dtype differs (rare for bf16 KV).
    if (kv.dtype != adt) {
      DBuf kcast(d, kv.dtype, {T, Hkv, Dh});
      DBuf vcast(d, kv.dtype, {T, Hkv, Dh});
      if (kv.dtype == DType::kBF16) {
        vt::CastBf16(d.q, kcast.t(), k3);
        vt::CastBf16(d.q, vcast.t(), v3);
      } else {
        vt::CastF32(d.q, kcast.t(), k3);
        vt::CastF32(d.q, vcast.t(), v3);
      }
      kw = kcast.t();
      vw = vcast.t();
      Tensor k_cache = KvSlice(kv, d.q.device, 0);
      Tensor v_cache = KvSlice(kv, d.q.device, 1);
      vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, si.slot_mapping.t());
    } else {
      Tensor k_cache = KvSlice(kv, d.q.device, 0);
      Tensor v_cache = KvSlice(kv, d.q.device, 1);
      vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, si.slot_mapping.t());
    }
  }

  // Paged GQA attention: scale = 1.0 (Q/K norms carry the scale).
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::PagedAttentionArgs pa{1.0f, meta.causal};
  if (g.attn_logit_softcap > 0.0f) pa.logits_soft_cap = g.attn_logit_softcap;
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  // ENG-ATTENTION-WINDOW (#2388). NOTE: this model has no kill switch, unlike
  // gemma2/gemma3; the asymmetry is real and owed as W3, not invented here.
  if (sliding_window.has_value() && *sliding_window > 0)
    pa.window_size = ResolveAttentionWindow(
        /*per_layer=*/std::nullopt, sliding_window,
        v1::AttentionType::kDecoder,
        /*disable_model_sliding_window=*/DisableSlidingWindowActive());
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  Tensor wo = ResidentWeight(d, w.attn.o_proj);
  DBuf o(d, DType::kBF16, {T, H});  // returned — not TLS
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  return o;
}

// GeGLU MLP (gemma4.py::Gemma4MLP).
DBuf Gemma4MlpBlock(Dev d, const Gemma4MlpWeights& w, int64_t H, int64_t I,
                    const Tensor& dh2, int64_t T) {
  // Dense GeGLU: TLS reuse of large gate_up [T,2I] + act [T,I] across layers.
  Tensor wgu = ResidentWeight(d, w.gate_up_proj);
  struct MlpTls {
    int dev = -1;
    int64_t T = 0, I = 0;
    std::optional<DBuf> gu, act;
  };
  static thread_local MlpTls tls;
  if (tls.dev != d.q.device.index || tls.T != T || tls.I != I) {
    tls.gu.emplace(d, DType::kBF16, std::vector<int64_t>{T, 2 * I});
    tls.act.emplace(d, DType::kBF16, std::vector<int64_t>{T, I});
    tls.dev = d.q.device.index;
    tls.T = T;
    tls.I = I;
  }
  DBuf& gu = *tls.gu;
  DBuf& act = *tls.act;
  vt::MatmulBT(d.q, gu.t(), dh2, wgu);
  vt::GeluAndMul(d.q, act.t(), gu.t());
  Tensor wd = ResidentWeight(d, w.down_proj);
  DBuf down(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, down.t(), act.t(), wd);
  return down;
}

void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// CLAIM-GEMMA4-MM-E2E: `inputs_embeds_override` (DEVICE bf16 [T, H]) and
// `ple_token_ids` (DEVICE i32 [T]) are the multimodal seam. ENG-MM-INPUT-PIPELINE
// P1 (#1358, #2300) moved both off host vectors and onto borrowed device views —
// see MultiModalForwardInput for why. When BOTH are null the
// function is byte-identical to the text path (embed from token_ids, PLE from
// token_ids) — the two text call sites pass null so the SACRED text 32/32 gate is
// unchanged by construction. When set (the registered mm forward), the hidden
// stream STARTS from the already-merged + already-sqrt(H)-scaled inputs_embeds
// (text rows scaled, vision soft-token rows the embed_vision projector output,
// gemma4_mm.py:1319/embed_input_ids masked-scatter), and the PLE embed_tokens_
// per_layer lookup uses the mm-masked ids (gemma4_mm.py:1962-1973). ple_proj still
// projects the merged inputs_embeds (project_per_layer_inputs, gemma4.py:908-912).
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const Gemma4Weights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices,
                 const Tensor* inputs_embeds_override = nullptr,
                 const Tensor* ple_token_ids = nullptr,
                 std::vector<std::vector<float>>* hidden_states_out = nullptr) {
  const int64_t T = inputs_embeds_override != nullptr
                        ? static_cast<int64_t>(positions.size())
                        : static_cast<int64_t>(token_ids.size());
  const Gemma4Layout g = MakeLayout(config);
  const int64_t H = g.hidden;
  const int64_t L = g.num_layers;
  const int64_t ple = g.ple_dim;
  const int64_t I = config.intermediate_size;
  const int64_t vocab = config.vocab_size;
  const float eps = 1e-6f;
  const vt::RmsNormArgs plain{eps, false};
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "gemma4: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(L),
           "gemma4: one PagedKvCache per layer required");
  // mm seam: the merged-embeds override and the PLE ids must both match T.
  if (inputs_embeds_override != nullptr)
    VT_CHECK(inputs_embeds_override->data != nullptr &&
                 inputs_embeds_override->Numel() == T * H,
             "gemma4 mm: inputs_embeds_override size != T*H");
  // The PLE ids are the mm-masked DEVICE ids when the seam supplies them, and the
  // step's own `token_ids` otherwise — the same either/or the host contract had.
  VT_CHECK(ple_token_ids != nullptr
               ? (ple_token_ids->data != nullptr && ple_token_ids->Numel() == T)
               : static_cast<int64_t>(token_ids.size()) == T,
           "gemma4 mm: ple_token_ids length != T");

  // Ones weights for the weight-less V-norm (identity scale), per head_dim.
  auto make_ones = [&](int64_t Dh) {
    std::vector<uint16_t> h(static_cast<size_t>(Dh), vt::F32ToBF16(1.0f));
    return DBuf(d, DType::kBF16, {Dh}, h.data());
  };
  DBuf ones_sliding = make_ones(g.head_dim_sliding);
  DBuf ones_full = make_ones(g.head_dim_full);

  // Proportional RoPE cache for full-attention layers (shared across them).
  int64_t max_pos = 0;
  for (int32_t p : positions) max_pos = std::max<int64_t>(max_pos, p);
  DBuf prop_cache = BuildProportionalRopeCache(d, g, g.head_dim_full, max_pos);

  // --- Token embedding * sqrt(hidden) (bf16). Also the PLE-projection input. ---
  // mm seam: when inputs_embeds_override is set the hidden stream STARTS from the
  // already-merged, already-sqrt(H)-scaled embeddings (text rows scaled at merge
  // time, vision soft-token rows the raw embed_vision output). No embed lookup /
  // no MulScalar — mirrors gemma4.py forward inputs_embeds branch (:908-909).
  DBuf tok(d, DType::kBF16, {T, H});
  if (inputs_embeds_override != nullptr) {
    // D2D from the caller's BORROWED view: `tok` seeds the residual stream and is
    // read again as the PLE-projection input, so the forward owns its own copy and
    // never writes through the seam's handle (MultiModalForwardInput contract).
    // `tok` is not written on this branch, so the copy is elidable; it is kept
    // because eliding it is unverifiable until the Gemma-4 checkpoint gate can run,
    // and the copy COUNT is unchanged against the host contract it replaces (that
    // one did H2D-into-`tok` then D2D-into-`hidden`). Recorded under `## Owed`.
    d.b.Copy(d.q, tok.ptr(), inputs_embeds_override->data, tok.bytes());
  } else {
    EmbedGather(d, tok, token_ids, weights.embed_tokens, vocab, H,
                "gemma4 embed");
    const float nsqrt = std::sqrt(static_cast<float>(H));
    const double normalizer =
        static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(nsqrt)));
    vt::MulScalar(d.q, tok.t(), tok.t(), normalizer);
  }

  // --- Per-Layer Embeddings precompute (gemma4.py:845-898). ---
  // Skipped when hidden_size_per_layer_input==0 (google/gemma-4-12B-it dense).
  DBuf ple_input(d, DType::kBF16, ple > 0 ? std::vector<int64_t>{T, L, ple}
                                          : std::vector<int64_t>{1, 1, 1});
  if (ple > 0) {
    // ple_emb = embed_tokens_per_layer(ids) * sqrt(ple)         -> [T, L*ple]
    // ple_proj = (per_layer_model_projection @ tok) * hidden^-0.5 -> [T, L*ple]
    //            reshape [T,L,ple]; RMSNorm(plain) over ple.
    // ple_input = (ple_proj + ple_emb) * rsqrt(2)               -> [T, L, ple]
    const int64_t LP = L * ple;
    DBuf ple_emb(d, DType::kBF16, {T, LP});
    {
      Tensor dtab = ResidentWeight(d, weights.embed_tokens_per_layer, {vocab, LP});
      // mm seam: the PLE ids arrive ALREADY on device; the text path uploads
      // `token_ids` exactly as it always did.
      DBuf owned_ids;
      Tensor dids = ple_token_ids != nullptr ? *ple_token_ids : Tensor{};
      if (ple_token_ids == nullptr) {
        owned_ids = DBuf(d, DType::kI32, {T}, token_ids.data());
        dids = owned_ids.t();
      }
      vt::Embedding(d.q, ple_emb.t(), dtab, dids);
    }
    vt::MulScalar(d.q, ple_emb.t(), ple_emb.t(),
                  static_cast<double>(g.embed_scale_ple));

    DBuf ple_proj(d, DType::kBF16, {T, LP});
    Tensor wproj = ResidentWeight(d, weights.per_layer_model_projection, {LP, H});
    vt::MatmulBT(d.q, ple_proj.t(), tok.t(), wproj);
    vt::MulScalar(d.q, ple_proj.t(), ple_proj.t(), g.proj_scale);

    // RMSNorm(plain) over the last dim ple.
    Tensor proj2 = Reshape(ple_proj.t(), {T * L, ple});
    Tensor wpn = ResidentWeight(d, weights.per_layer_projection_norm, {ple});
    vt::RmsNorm(d.q, proj2, proj2, wpn, plain);

    // (ple_proj + ple_emb) * rsqrt(2).
    vt::Add(d.q, ple_input.t(), Reshape(ple_proj.t(), {T, L, ple}),
            Reshape(ple_emb.t(), {T, L, ple}));
    vt::MulScalar(d.q, ple_input.t(), ple_input.t(), g.input_scale);
  }

  // Layout [L,T,ple] so each layer's slice is one contiguous D2D (not T row gathers).
  DBuf ple_by_layer(d, DType::kBF16,
                    ple > 0 ? std::vector<int64_t>{L, T, ple} : std::vector<int64_t>{1, 1, 1});
  if (ple > 0) {
    const size_t row = static_cast<size_t>(ple) * sizeof(uint16_t);
    const auto* src = static_cast<const char*>(ple_input.ptr());
    auto* dst = static_cast<char*>(ple_by_layer.ptr());
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t li = 0; li < L; ++li) {
        const size_t s_off =
            (static_cast<size_t>(t) * static_cast<size_t>(L) + static_cast<size_t>(li)) * row;
        const size_t d_off =
            (static_cast<size_t>(li) * static_cast<size_t>(T) + static_cast<size_t>(t)) * row;
        CopyRow(d, dst + d_off, src + s_off, row);
      }
    }
  }

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  // hidden state stream (each layer fully materializes h; no separate residual).
  DBuf hidden(d, DType::kBF16, {T, H});
  d.b.Copy(d.q, hidden.ptr(), tok.ptr(),
           static_cast<size_t>(T) * static_cast<size_t>(H) * sizeof(uint16_t));

  const size_t ple_row_bytes = static_cast<size_t>(ple) * sizeof(uint16_t);

  // Decode/prefill layer scratch reused across L (pool thrash was real on 30L MoE).
  struct LayerTls {
    int dev = -1;
    int64_t T = 0, H = 0, I = 0, ple = 0;
    std::optional<DBuf> dhn, attn_n, h1, dh2, h2, moe_in, n1, n2, sum, n3, mlp_n;
    std::optional<DBuf> gate_lin, ple_l, gated, contrib;
  };
  static thread_local LayerTls lt;
  if (lt.dev != d.q.device.index || lt.T != T || lt.H != H || lt.I != I || lt.ple != ple) {
    auto mk = [&](int64_t a, int64_t b) {
      return DBuf(d, DType::kBF16, std::vector<int64_t>{a, b});
    };
    lt.dhn.emplace(mk(T, H));
    lt.attn_n.emplace(mk(T, H));
    lt.h1.emplace(mk(T, H));
    lt.dh2.emplace(mk(T, H));
    lt.h2.emplace(mk(T, H));
    lt.moe_in.emplace(mk(T, H));
    lt.n1.emplace(mk(T, H));
    lt.n2.emplace(mk(T, H));
    lt.sum.emplace(mk(T, H));
    lt.n3.emplace(mk(T, H));
    lt.mlp_n.emplace(mk(T, H));
    lt.contrib.emplace(mk(T, H));
    if (ple > 0) {
      lt.gate_lin.emplace(mk(T, ple));
      lt.ple_l.emplace(mk(T, ple));
      lt.gated.emplace(mk(T, ple));
    } else {
      lt.gate_lin.reset();
      lt.ple_l.reset();
      lt.gated.reset();
    }
    lt.dev = d.q.device.index;
    lt.T = T;
    lt.H = H;
    lt.I = I;
    lt.ple = ple;
  }
  DBuf& dhn = *lt.dhn;
  DBuf& h1 = *lt.h1;
  DBuf& dh2 = *lt.dh2;
  DBuf& h2 = *lt.h2;
  DBuf& moe_in = *lt.moe_in;
  const size_t th_bytes = static_cast<size_t>(T) * static_cast<size_t>(H) * sizeof(uint16_t);
  DBuf& contrib = *lt.contrib;

  // MODEL-DIFFUSION-LTX25 L3: the per-layer hidden-state capture. Mirrors
  // transformers' `output_hidden_states=True` EXACTLY, because that is what
  // LTX-2.5's text encoder consumes (base_encoder.py:68-71) and it is the ONE
  // thing about the tuple that is easy to get wrong: transformers appends the
  // state BEFORE each decoder layer and then appends the FINAL-NORMED one after
  // the loop, so the tuple is
  //   [0]   the embeddings (already sqrt(hidden)-scaled, or the merged mm embeds)
  //   [i]   the output of decoder layer i-1, for i in 1..L-1
  //   [L]   model.norm(output of decoder layer L-1)
  // and the RAW output of the last decoder layer never appears. Count = L + 1.
  // Off (nullptr) on every existing call site, so no shipped path changes.
  std::vector<uint16_t> capture_row;
  const auto capture = [&](DBuf& buf) {
    if (hidden_states_out == nullptr) return;
    capture_row.resize(static_cast<size_t>(T) * static_cast<size_t>(H));
    buf.Download(d, capture_row.data());
    std::vector<float> host(capture_row.size());
    for (size_t i = 0; i < capture_row.size(); ++i) host[i] = vt::BF16ToF32(capture_row[i]);
    hidden_states_out->push_back(std::move(host));
  };
  if (hidden_states_out != nullptr) hidden_states_out->clear();

  for (int64_t l = 0; l < L; ++l) {
    capture(hidden);  // the state BEFORE layer `l`, transformers' append order
    const Gemma4LayerWeights& w = weights.layers[static_cast<size_t>(l)];
    const int64_t Dh = w.head_dim;
    const bool full = w.is_full_attention;
    const Tensor ones_dh = full ? ones_full.t() : ones_sliding.t();
    std::optional<int64_t> window;
    if (!full) window = g.sliding_window;

    const int64_t kv_idx = w.is_kv_shared ? w.kv_target_layer : l;
    VT_CHECK(kv_idx >= 0 && kv_idx < L, "gemma4: bad kv target layer");
    const PagedKvCache& kv = attn_kv[static_cast<size_t>(kv_idx)];

    Tensor w_in = ResidentWeight(d, w.input_layernorm, {H});
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, plain);

    static const bool layer_prof = [] {
      const char* e = std::getenv("VT_GEMMA4_PROFILE");
      return e && e[0] == '1';
    }();
    using clock = std::chrono::steady_clock;
    const auto t0 = layer_prof ? clock::now() : clock::time_point{};

    DBuf attn = Gemma4AttnBlock(d, w, g, dhn.t(), si, attn_meta, kv, T, Dh, full,
                                ones_dh, full ? &prop_cache.t() : nullptr, window,
                                g.rope_theta_sliding);

    Tensor w_pa = ResidentWeight(d, w.post_attention_layernorm, {H});
    // h1 = rmsnorm(attn) + hidden  (one kernel)
    vt::RmsNormPlusAdd(d.q, h1.t(), attn.t(), w_pa, hidden.t(), plain);

    if (layer_prof) d.b.Synchronize(d.q);
    const auto t1 = layer_prof ? clock::now() : clock::time_point{};

    Tensor w_pf = ResidentWeight(d, w.pre_feedforward_layernorm, {H});
    vt::RmsNorm(d.q, dh2.t(), h1.t(), w_pf, plain);
    DBuf mlp = Gemma4MlpBlock(d, w.mlp, H, I, dh2.t(), T);

    if (layer_prof) d.b.Synchronize(d.q);
    const auto t2 = layer_prof ? clock::now() : clock::time_point{};

    if (w.moe.enabled) {
      Tensor w_pf2 = ResidentWeight(d, w.moe.pre_feedforward_layernorm_2, {H});
      vt::RmsNorm(d.q, moe_in.t(), h1.t(), w_pf2, plain);
      Gemma4MoeScratch moe_out =
          RunGemma4Moe(d.q, w.moe, /*router_in=*/h1.t(), /*expert_in=*/moe_in.t(), T, H, eps);
      Tensor w_p1 = ResidentWeight(d, w.moe.post_feedforward_layernorm_1, {H});
      Tensor w_p2 = ResidentWeight(d, w.moe.post_feedforward_layernorm_2, {H});
      Tensor w_pff = ResidentWeight(d, w.post_feedforward_layernorm, {H});
      // h2 = rmsnorm(rmsnorm(mlp)+rmsnorm(moe), w_pff) + h1  — one fused kernel
      vt::DualRmsNormPlusRes(d.q, h2.t(), mlp.t(), w_p1, moe_out.tensor, w_p2, w_pff,
                                       h1.t(), plain);
    } else {
      Tensor w_pff = ResidentWeight(d, w.post_feedforward_layernorm, {H});
      vt::RmsNormPlusAdd(d.q, h2.t(), mlp.t(), w_pff, h1.t(), plain);
    }

    if (layer_prof) {
      d.b.Synchronize(d.q);
      const auto t3 = clock::now();
      static std::atomic<uint64_t> n{0}, us_attn{0}, us_mlp{0}, us_moe{0};
      auto us = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
      };
      us_attn.fetch_add(static_cast<uint64_t>(us(t0, t1)), std::memory_order_relaxed);
      us_mlp.fetch_add(static_cast<uint64_t>(us(t1, t2)), std::memory_order_relaxed);
      us_moe.fetch_add(static_cast<uint64_t>(us(t2, t3)), std::memory_order_relaxed);
      const uint64_t c = n.fetch_add(1, std::memory_order_relaxed) + 1;
      if (c == 32 || c % 128 == 0) {
        std::fprintf(stderr,
                     "gemma4 layer profile: calls=%llu attn_us=%.1f mlp_us=%.1f moe_us=%.1f "
                     "(attn%%=%.0f mlp%%=%.0f moe%%=%.0f)\n",
                     static_cast<unsigned long long>(c),
                     static_cast<double>(us_attn.load()) / c,
                     static_cast<double>(us_mlp.load()) / c,
                     static_cast<double>(us_moe.load()) / c,
                     100.0 * us_attn.load() / (us_attn.load() + us_mlp.load() + us_moe.load() + 1),
                     100.0 * us_mlp.load() / (us_attn.load() + us_mlp.load() + us_moe.load() + 1),
                     100.0 * us_moe.load() / (us_attn.load() + us_mlp.load() + us_moe.load() + 1));
      }
    }

    if (ple > 0) {
      Tensor wg = ResidentWeight(d, w.per_layer_input_gate, {ple, H});
      DBuf& gate_lin = *lt.gate_lin;
      DBuf& ple_l = *lt.ple_l;
      DBuf& gated = *lt.gated;
      vt::MatmulBT(d.q, gate_lin.t(), h2.t(), wg);
      // Contiguous [T,ple] slice for this layer from [L,T,ple] layout.
      const size_t layer_bytes = static_cast<size_t>(T) * ple_row_bytes;
      const char* src = static_cast<const char*>(ple_by_layer.ptr()) +
                        static_cast<size_t>(l) * layer_bytes;
      d.b.Copy(d.q, ple_l.ptr(), src, layer_bytes);
      vt::GeluMulSeparate(d.q, gated.ptr(), gate_lin.ptr(), ple_l.ptr(), T * ple,
                                    DType::kBF16);

      Tensor wp = ResidentWeight(d, w.per_layer_projection, {H, ple});
      vt::MatmulBT(d.q, contrib.t(), gated.t(), wp);
      Tensor w_pln = ResidentWeight(d, w.post_per_layer_input_norm, {H});
      vt::RmsNorm(d.q, contrib.t(), contrib.t(), w_pln, plain);
      vt::Add(d.q, h2.t(), h2.t(), contrib.t());
    }

    if (!w.layer_scalar.Empty()) {
      const double scalar = static_cast<double>(ReadBf16Scalar(w.layer_scalar));
      vt::MulScalar(d.q, h2.t(), h2.t(), scalar);
    }

    d.b.Copy(d.q, hidden.ptr(), h2.ptr(), th_bytes);
  }

  // Final norm (plain RMSNorm, standalone — residual is None in vLLM).
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, plain);
  // The LAST entry of transformers' hidden_states tuple is the FINAL-NORMED one,
  // not the raw output of the last decoder layer.
  capture(dnorm);

  // Tied lm_head + final logit soft-cap.
  const bool tied = weights.tie_word_embeddings || weights.lm_head.Empty();
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

std::vector<float> Gemma4Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma4Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

Gemma4HiddenStatesResult Gemma4Model::ForwardHiddenStates(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma4Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  Gemma4HiddenStatesResult out;
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices, nullptr, nullptr, &out.hidden_states);
  const int64_t n_out = dlogits.t().shape[0];
  out.logits.resize(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, out.logits.data());
  VT_CHECK(static_cast<int64_t>(out.hidden_states.size()) == config.num_hidden_layers + 1,
           "gemma4: hidden_states must be num_hidden_layers + 1 entries");
  return out;
}

ForwardLogits Gemma4Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma4Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

std::vector<float> Gemma4Model::ForwardMm(
    const vt::Tensor& inputs_embeds, const vt::Tensor& ple_token_ids,
    const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Gemma4Weights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  // token_ids is unused when inputs_embeds_override is set (T comes from
  // positions); pass an empty vector so the mm seam never dereferences it.
  const std::vector<int32_t> no_tokens;
  DBuf dlogits = ForwardBody(d, no_tokens, positions, attn_meta, attn_kv, weights,
                             config, logits_indices, &inputs_embeds,
                             &ple_token_ids);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

std::vector<float> Gemma4ProportionalRopeCosSin(const HfConfig& config, int64_t head_dim,
                                                int64_t max_pos) {
  VT_CHECK(head_dim > 0 && head_dim % 2 == 0 && max_pos >= 0,
           "gemma4: proportional rope table wants an even positive head_dim and a "
           "non-negative max_pos");
  return ProportionalRopeCosSinHost(MakeLayout(config), head_dim, max_pos);
}

}  // namespace vllm

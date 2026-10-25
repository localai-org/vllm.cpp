// RecursiveSI Flint (`FlintForCausalLM`) — config parser, CPU reference
// forward, and the model-registry registration (MODEL-TEXT-FLINT W1, spec
// .agents/specs/flint-cpu-gguf.md).
//
// A NEW translation unit with ONE REGISTER_VLLM_MODEL line and ZERO edit to
// any shared array (the qwen3_moe_registry.cpp seam). The GGUF arm is refused
// BY NAME per the row's scope (it is a separate wave: the `GgufArchArm`
// entry), and the reference forward refuses multi-sequence input by name —
// the engine PAGED KV-cache integration is a later wave, recorded in the
// spec's ## Owed.
//
// Reference numerics (pinned oracle llama-cpp-flint, PR #30928 head
// `1b44e30775dcb2f93952116b05f89e5bd797fb00`, src/models/flint.cpp +
// conversion/flint.py; RecursiveSI config.json field names):
//   - Standard pre-norm residual: h = attn(ln1(x)) + x;
//     out = mlp(ln2(h)) + h. NOT Kolibri-1's sandwich.
//   - Per-head q/k RMSNorm over the head dim BEFORE RoPE.
//   - Hybrid FFSSSF sliding/full attention; RoPE on the SLIDING layers only
//     (config.no_rope_layer_indices names the full layers).
//   - Attention scores in f32 (spec: "QK-normed logits reach tens"); the
//     per-head sink is folded into the softmax DENOMINATOR:
//     out = Σ exp(S_j)·V_j / (Σ exp(S_j) + exp(sink)).
//   - MoE router: f32 logits -> sigmoid -> top-6 -> renormalise
//     (expert_weights_norm). One UNGATED shared expert always added.
//   - Layers 0-1 dense (mlp_only_layers = [0, 1]).
//
// Linear outputs and norms round to bf16 (the model dtype every layer
// inherits, per AGENTS.md "Inherit vLLM defaults"); the score/router/softmax
// f32 wideness is the architecture's own, named in the spec.

#include "vllm/model_executor/models/flint.h"
#include "vllm/model_executor/models/flint_tt.h"
#include "vllm/model_executor/models/flint_gguf_weights.h"  // the GGUF arm

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits

namespace vllm {

namespace {

// ---- raw-json readers (the kolibri1_registry.cpp shape) ----

int64_t GetInt(const nlohmann::json& j, const char* key, int64_t def = 0) {
  if (const auto it = j.find(key); it != j.end() && it->is_number_integer())
    return it->get<int64_t>();
  if (const auto it = j.find(key); it != j.end() && it->is_number_float())
    return static_cast<int64_t>(it->get<double>());
  return def;
}

double GetDouble(const nlohmann::json& j, const char* key, double def = 0.0) {
  if (const auto it = j.find(key); it != j.end() && it->is_number())
    return it->get<double>();
  return def;
}

std::string GetString(const nlohmann::json& j, const char* key,
                      const char* def = "") {
  if (const auto it = j.find(key); it != j.end() && it->is_string())
    return it->get<std::string>();
  return def;
}

std::vector<int64_t> GetIntArray(const nlohmann::json& j, const char* key) {
  std::vector<int64_t> out;
  if (const auto it = j.find(key); it != j.end() && it->is_array())
    for (const auto& el : *it)
      if (el.is_number_integer()) out.push_back(el.get<int64_t>());
  return out;
}

std::vector<std::string> GetStringArray(const nlohmann::json& j,
                                        const char* key) {
  std::vector<std::string> out;
  if (const auto it = j.find(key); it != j.end() && it->is_array())
    for (const auto& el : *it)
      if (el.is_string()) out.push_back(el.get<std::string>());
  return out;
}

#define FLINT_CHECK(cond, msg)                                            \
  do {                                                                    \
    if (!(cond))                                                          \
      throw std::runtime_error(std::string("flint: ") + (msg));           \
  } while (0)

}  // namespace

bool FlintParams::LayerHasRope(int64_t i) const {
  return std::find(no_rope_layer_indices.begin(), no_rope_layer_indices.end(),
                   i) == no_rope_layer_indices.end();
}

bool FlintParams::IsDenseLayer(int64_t i) const {
  return std::find(mlp_only_layers.begin(), mlp_only_layers.end(), i) !=
         mlp_only_layers.end();
}

FlintParams ParseFlintParams(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  FlintParams p;

  p.hidden_size = config.hidden_size;
  p.num_hidden_layers = config.num_hidden_layers;
  p.vocab_size = config.vocab_size;
  p.num_attention_heads = config.num_attention_heads;
  p.num_key_value_heads = config.num_key_value_heads;
  p.head_dim = GetInt(raw, "head_dim", 0);
  p.sliding_window = GetInt(raw, "sliding_window", 0);
  p.rope_theta = GetDouble(raw, "rope_theta", 0.0);
  p.rms_norm_eps = GetDouble(raw, "rms_norm_eps", 1e-6);
  p.hidden_act = GetString(raw, "hidden_act", "silu");
  p.layer_types = GetStringArray(raw, "layer_types");
  p.no_rope_layer_indices = GetIntArray(raw, "no_rope_layer_indices");
  p.mlp_only_layers = GetIntArray(raw, "mlp_only_layers");
  p.num_experts = GetInt(raw, "num_experts", 0);
  p.num_experts_per_tok = GetInt(raw, "num_experts_per_tok", 0);
  p.moe_intermediate_size = GetInt(raw, "moe_intermediate_size", 0);
  p.shared_expert_intermediate_size =
      GetInt(raw, "shared_expert_intermediate_size", 0);
  p.tie_word_embeddings = false;
  if (const auto it = raw.find("tie_word_embeddings");
      it != raw.end() && it->is_boolean())
    p.tie_word_embeddings = it->get<bool>();

  // ---- validation: everything unrepresentable throws ----
  FLINT_CHECK(p.hidden_size > 0, "hidden_size must be positive");
  FLINT_CHECK(p.num_hidden_layers > 0, "num_hidden_layers must be positive");
  FLINT_CHECK(p.vocab_size > 0, "vocab_size must be positive");
  FLINT_CHECK(p.num_attention_heads > 0,
              "num_attention_heads must be positive");
  FLINT_CHECK(p.num_key_value_heads > 0,
              "num_key_value_heads must be positive");
  FLINT_CHECK(p.num_attention_heads % p.num_key_value_heads == 0,
              "num_attention_heads must be a multiple of num_key_value_heads "
              "(GQA)");
  FLINT_CHECK(p.head_dim > 0, "head_dim must be positive");
  FLINT_CHECK(p.layer_types.size() ==
                  static_cast<size_t>(p.num_hidden_layers),
              "layer_types length must equal num_hidden_layers — the hybrid "
              "pattern cannot be derived");
  for (const std::string& kind : p.layer_types)
    FLINT_CHECK(kind == "sliding_attention" || kind == "full_attention",
                "layer_types entry '" + kind +
                    "' is neither sliding_attention nor full_attention");
  FLINT_CHECK(p.rope_theta > 0.0, "rope_theta must be positive");
  FLINT_CHECK(p.rms_norm_eps > 0.0, "rms_norm_eps must be positive");
  FLINT_CHECK(p.hidden_act == "silu",
              "hidden_act '" + p.hidden_act +
                  "' is not the silu the architecture defines");
  FLINT_CHECK(!p.tie_word_embeddings,
              "tied word embeddings are not implemented — the checkpoint "
              "ships an untied lm_head");
  // The sliding pattern needs a window.
  const bool any_sliding =
      std::any_of(p.layer_types.begin(), p.layer_types.end(),
                  [](const std::string& s) { return s == "sliding_attention"; });
  FLINT_CHECK(!any_sliding || p.sliding_window > 0,
              "sliding layers require a positive sliding_window");
  // The no-RoPE set must be exactly the full-attention layers (the
  // architecture's own invariant: RoPE lives on the sliding layers only).
  for (int64_t i = 0; i < p.num_hidden_layers; ++i)
    FLINT_CHECK(p.LayerHasRope(i) == p.IsSlidingLayer(i),
                "no_rope_layer_indices must name exactly the "
                "full_attention layers");
  // Dense layers must precede the MoE layers (the converter refuses
  // otherwise: conversion/flint.py "dense layers must precede the MoE").
  for (size_t k = 0; k < p.mlp_only_layers.size(); ++k)
    FLINT_CHECK(p.mlp_only_layers[k] == static_cast<int64_t>(k),
                "mlp_only_layers must be the leading dense layers {0..n-1}");
  FLINT_CHECK(p.num_experts > 0, "num_experts must be positive");
  FLINT_CHECK(p.num_experts_per_tok > 0 &&
                  p.num_experts_per_tok <= p.num_experts,
              "num_experts_per_tok must be in (0, num_experts]");
  FLINT_CHECK(p.moe_intermediate_size > 0,
              "moe_intermediate_size must be positive");
  FLINT_CHECK(p.shared_expert_intermediate_size > 0,
              "shared_expert_intermediate_size must be positive — the "
              "architecture always carries the ungated shared expert");
  return p;
}

void ParseFlintConfig(const HfConfig& config) { (void)ParseFlintParams(config); }

v1::KVCacheConfig MakeFlintKVCache(const HfConfig& config, int block_size,
                                   int num_blocks) {
  const FlintParams p = ParseFlintParams(config);
  const vt::DType dtype = v1::ResolveKvCacheDType();

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  std::vector<std::string> full_layer_names;
  std::vector<std::string> swa_layer_names;
  for (int64_t i = 0; i < static_cast<int64_t>(p.layer_types.size()); ++i) {
    const std::string name = "model.layers." + std::to_string(i) + ".self_attn";
    if (p.IsSlidingLayer(i))
      swa_layer_names.push_back(name);
    else
      full_layer_names.push_back(name);
  }
  if (!full_layer_names.empty()) {
    kv.kv_cache_groups.emplace_back(
        std::move(full_layer_names),
        std::make_shared<v1::FullAttentionSpec>(
            block_size, static_cast<int>(p.num_key_value_heads),
            static_cast<int>(p.head_dim), dtype));
  }
  if (!swa_layer_names.empty()) {
    kv.kv_cache_groups.emplace_back(
        std::move(swa_layer_names),
        std::make_shared<v1::SlidingWindowSpec>(
            block_size, static_cast<int>(p.num_key_value_heads),
            static_cast<int>(p.head_dim), dtype,
            static_cast<int>(p.sliding_window)));
  }
  return kv;
}

// ---- the CPU reference forward ----

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

float Bf16(uint16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

uint16_t ToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  // Round-to-nearest-even on the truncated 16 low bits.
  uint32_t lsb = (u >> 16) & 1;
  u += 0x7fff + lsb;
  return static_cast<uint16_t>(u >> 16);
}

const uint16_t* Bf16Data(const OwnedTensor& t) {
  return reinterpret_cast<const uint16_t*>(t.bytes.data());
}

// y[T,N] = x[T,K] @ w[K,N], f32 accumulate over bf16 operands; y rounded to
// bf16 (the model dtype every layer inherits).
void MatmulBf16(const float* x, const OwnedTensor& w, int64_t t, float* y) {
  const int64_t k_dim = w.shape[0];
  const int64_t n_dim = w.shape[1];
  const uint16_t* wd = Bf16Data(w);
  for (int64_t i = 0; i < t; ++i) {
    for (int64_t n = 0; n < n_dim; ++n) y[i * n_dim + n] = 0.0f;
    for (int64_t k = 0; k < k_dim; ++k) {
      const float xv = x[i * k_dim + k];
      if (xv == 0.0f) continue;
      const uint16_t* wrow = wd + k * n_dim;
      for (int64_t n = 0; n < n_dim; ++n)
        y[i * n_dim + n] += xv * Bf16(wrow[n]);
    }
  }
  for (int64_t i = 0; i < t * n_dim; ++i) y[i] = Bf16(ToBf16(y[i]));
}

// y = bf16(x) * rms_w — a standard (non-gemma) RMSNorm over the last dim,
// variance in f32, output rounded to bf16.
void RmsNormBf16(const float* x, const OwnedTensor& w, int64_t rows,
                 int64_t dim, double eps, float* y) {
  const uint16_t* wd = Bf16Data(w);
  for (int64_t i = 0; i < rows; ++i) {
    double ss = 0.0;
    for (int64_t d = 0; d < dim; ++d) ss += double(x[i * dim + d]) * x[i * dim + d];
    const float inv = static_cast<float>(
        1.0 / std::sqrt(ss / double(dim) + eps));
    for (int64_t d = 0; d < dim; ++d)
      y[i * dim + d] = Bf16(ToBf16(x[i * dim + d] * inv * Bf16(wd[d])));
  }
}

// NeoX-style (rotate_half) RoPE over full head_dim, in place on q (rows x
// hq x dh) and k (rows x hkv x dh). The architecture is full rotary; the
// mistral.rs note pins RopePairing::HalfSplit == HF rotate_half.
void RopeNeoxHalfSplit(float* t, int64_t rows, int64_t heads, int64_t dh,
                       const std::vector<int32_t>& positions,
                       double theta) {
  const int64_t half = dh / 2;
  for (int64_t r = 0; r < rows; ++r) {
    const double pos = static_cast<double>(positions[static_cast<size_t>(r)]);
    for (int64_t h = 0; h < heads; ++h) {
      float* head = t + (r * heads + h) * dh;
      for (int64_t d = 0; d < half; ++d) {
        const double ang =
            pos / std::pow(theta, static_cast<double>(2 * d) / double(dh));
        const double c = std::cos(ang), s = std::sin(ang);
        const float x1 = head[d], x2 = head[d + half];
        head[d] = static_cast<float>(x1 * c - x2 * s);
        head[d + half] = static_cast<float>(x1 * s + x2 * c);
      }
    }
  }
}

}  // namespace

std::vector<float> FlintModel::Forward(const std::vector<int32_t>& token_ids,
                                       const std::vector<int32_t>& positions,
                                       const FlintWeights& w,
                                       const HfConfig& config) {
  (void)config;
  const FlintParams& p = w.params;
  const int64_t t = static_cast<int64_t>(token_ids.size());
  FLINT_CHECK(t > 0, "forward: empty token_ids");
  FLINT_CHECK(positions.size() == token_ids.size(),
              "forward: positions length must equal token_ids length");
  // Single-sequence reference arm: absolute positions must be CONTIGUOUS and
  // increasing (one flat sequence, no slots skipped). A paged engine
  // integration is a later wave of the row (spec ## Owed) — refused BY NAME
  // here, never silently.
  for (int64_t i = 1; i < t; ++i)
    FLINT_CHECK(positions[static_cast<size_t>(i)] ==
                    positions[static_cast<size_t>(i - 1)] + 1,
                "forward: the reference arm serves ONE contiguous sequence "
                "(got positions[" +
                    std::to_string(i) + "] == " +
                    std::to_string(positions[static_cast<size_t>(i)]) +
                    "); the paged engine arm is a later wave of "
                    "MODEL-TEXT-FLINT");

  const int64_t h = p.hidden_size;
  const int64_t hq = p.num_attention_heads;
  const int64_t hkv = p.num_key_value_heads;
  const int64_t dh = p.head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
  // The W0 device-vs-CPU dump (spec .agents/specs/flint-tenstorrent.md): the
  // same three stage sums the TT forward prints behind FLINT_TT_W0_DUMP,
  // tagged [FLINT-CPU], so the like-for-like distance is readable per stage.
  const bool cpu_dump = std::getenv("FLINT_CPU_DUMP") != nullptr;

  std::vector<float> x(static_cast<size_t>(t) * h);  // hidden stream
  {
    // Embedding lookup: embed_tokens [vocab, H].
    const uint16_t* emb = Bf16Data(w.embed_tokens);
    for (int64_t i = 0; i < t; ++i) {
      const int64_t tok = token_ids[static_cast<size_t>(i)];
      FLINT_CHECK(tok >= 0 && tok < p.vocab_size, "forward: token id out of range");
      // CONVERT, do not memcpy: the source is bf16 bits and the destination
      // is f32. A raw copy drops each bf16 into the low half of a zero-high
      // float — a ~1e-42 denormal (ISSUE-LOCAL-01M5QBGBM3Z7NT3D9TA0K1HT7T).
      for (int64_t d = 0; d < h; ++d)
        x[static_cast<size_t>(i) * h + d] = Bf16(emb[tok * h + d]);
    }
  }
  if (cpu_dump) {
    double s = 0.0;
    for (float v : x) s += v;
    std::fprintf(stderr, "[FLINT-CPU] embed sum=%.6f\n", s);
  }

  std::vector<float> normed(static_cast<size_t>(t) * h);
  std::vector<float> proj(static_cast<size_t>(t) * h);
  for (int64_t layer = 0; layer < p.num_hidden_layers; ++layer) {
    const FlintLayerWeights& lw = w.layers[static_cast<size_t>(layer)];
    const bool is_sliding = p.IsSlidingLayer(layer);
    const bool has_rope = p.LayerHasRope(layer);

    // ---- attention: h = attn(ln1(x)) + x ----
    RmsNormBf16(x.data(), lw.input_layernorm, t, h, p.rms_norm_eps,
                normed.data());
    std::vector<float> q(static_cast<size_t>(t) * hq * dh);
    std::vector<float> k(static_cast<size_t>(t) * hkv * dh);
    std::vector<float> v(static_cast<size_t>(t) * hkv * dh);
    MatmulBf16(normed.data(), lw.attn.q_proj.w, t, q.data());
    MatmulBf16(normed.data(), lw.attn.k_proj.w, t, k.data());
    MatmulBf16(normed.data(), lw.attn.v_proj.w, t, v.data());

    // Per-head q/k RMSNorm over the head dim, BEFORE RoPE. Kept f32 past the
    // norm (the head stream; the score path is the architecture's own f32).
    {
      const uint16_t* qw = Bf16Data(lw.attn.q_norm);
      const uint16_t* kw = Bf16Data(lw.attn.k_norm);
      auto head_norm = [&](float* t4, int64_t rows, const uint16_t* nw) {
        for (int64_t r = 0; r < rows; ++r) {
          float* head = t4 + r * dh;
          double ss = 0.0;
          for (int64_t d = 0; d < dh; ++d) ss += double(head[d]) * head[d];
          const float inv =
              static_cast<float>(1.0 / std::sqrt(ss / double(dh) + p.rms_norm_eps));
          for (int64_t d = 0; d < dh; ++d) head[d] *= inv * Bf16(nw[d]);
        }
      };
      head_norm(q.data(), t * hq, qw);
      head_norm(k.data(), t * hkv, kw);
    }

    // RoPE on the SLIDING layers only; the full-attention layers carry no
    // positional encoding at all.
    if (has_rope) {
      RopeNeoxHalfSplit(q.data(), t, hq, dh, positions, p.rope_theta);
      RopeNeoxHalfSplit(k.data(), t, hkv, dh, positions, p.rope_theta);
    }

    // Sinks attention, scores in f32, the per-head sink folded into the
    // softmax denominator. Output rounded to bf16.
    // The buffer is the HEAD layout [t, hq, dh] — what o_proj ([q, h])
    // consumes. Sizing it t*h overflows whenever hq*head_dim > hidden_size
    // (ISSUE-LOCAL-01M5Q1PS772CPTMYJ02AMVZ3Y0; the W1 synthetic config had
    // hq*dh < h and masked the defect).
    std::vector<float> attn_out(static_cast<size_t>(t) * hq * dh);
    {
      const uint16_t* sinks = Bf16Data(lw.attn.sinks);
      const int64_t gqa = hq / hkv;
      for (int64_t i = 0; i < t; ++i) {
        // The sliding window reaches back sliding_window-1 positions.
        const int64_t first =
            is_sliding ? std::max<int64_t>(0, i - (p.sliding_window - 1)) : 0;
        for (int64_t hh = 0; hh < hq; ++hh) {
          const float* qh = &q[(i * hq + hh) * dh];
          const float* kh_base = &k[0];
          const float* vh_base = &v[0];
          const int64_t kvh = hh / gqa;
          float m = kNegInf;
          std::vector<float> s(static_cast<size_t>(i - first + 1));
          for (int64_t j = first; j <= i; ++j) {
            const float* kh = kh_base + (j * hkv + kvh) * dh;
            float dot = 0.0f;
            for (int64_t d = 0; d < dh; ++d) dot += qh[d] * kh[d];
            s[static_cast<size_t>(j - first)] = dot * scale;
            m = std::max(m, s[static_cast<size_t>(j - first)]);
          }
          const float sink = Bf16(sinks[hh]);
          double denom = std::exp(static_cast<double>(sink - m));
          for (float sv : s) denom += std::exp(static_cast<double>(sv - m));
          float* out = &attn_out[(i * hq + hh) * dh];
          for (int64_t d = 0; d < dh; ++d) out[d] = 0.0f;
          for (int64_t j = first; j <= i; ++j) {
            const float wgt = static_cast<float>(
                std::exp(static_cast<double>(s[static_cast<size_t>(j - first)] - m)) /
                denom);
            const float* vh = vh_base + (j * hkv + kvh) * dh;
            for (int64_t d = 0; d < dh; ++d) out[d] += wgt * vh[d];
          }
        }
      }
      for (float& val : attn_out) val = Bf16(ToBf16(val));
    }

    std::vector<float> ao(static_cast<size_t>(t) * h);
    MatmulBf16(attn_out.data(), lw.attn.o_proj.w, t, ao.data());
    for (int64_t i = 0; i < t * h; ++i)
      x[static_cast<size_t>(i)] =
          Bf16(ToBf16(x[static_cast<size_t>(i)] + ao[static_cast<size_t>(i)]));
    if (cpu_dump && layer == 0) {
      double s = 0.0;
      for (float v : x) s += v;
      std::fprintf(stderr, "[FLINT-CPU] layer 0 post-attn x sum=%.6f\n", s);
    }

    // ---- MLP / MoE: out = mlp(ln2(h)) + h ----
    RmsNormBf16(x.data(), lw.post_attention_layernorm, t, h, p.rms_norm_eps,
                normed.data());
    std::vector<float> ff(static_cast<size_t>(t) * h);
    if (p.IsDenseLayer(layer)) {
      const int64_t inter = lw.dense_mlp.gate_proj.w.shape[1];
      std::vector<float> g(static_cast<size_t>(t) * inter);
      std::vector<float> u(static_cast<size_t>(t) * inter);
      MatmulBf16(normed.data(), lw.dense_mlp.gate_proj.w, t, g.data());
      MatmulBf16(normed.data(), lw.dense_mlp.up_proj.w, t, u.data());
      for (int64_t i = 0; i < t * inter; ++i)
        g[static_cast<size_t>(i)] =
            g[static_cast<size_t>(i)] /
            (1.0f + std::exp(-g[static_cast<size_t>(i)])) *
            u[static_cast<size_t>(i)];
      std::vector<float> gi(g);
      for (float& gv : gi) gv = Bf16(ToBf16(gv));
      MatmulBf16(gi.data(), lw.dense_mlp.down_proj.w, t, ff.data());
    } else {
      const int64_t e = p.num_experts;
      const int64_t inter = p.moe_intermediate_size;
      const int64_t top_k = p.num_experts_per_tok;
      // Router logits, f32 (architecture-owned wideness), sigmoid scores.
      std::vector<float> logits(static_cast<size_t>(t) * e);
      {
        // Router is [H, E] Matmul-B bf16; the product accumulates in f32 and
        // STAYS f32 (no bf16 rounding — the score path).
        const uint16_t* rd = Bf16Data(lw.moe.router_gate);
        for (int64_t i = 0; i < t; ++i) {
          for (int64_t n = 0; n < e; ++n) logits[static_cast<size_t>(i) * e + n] = 0.0f;
          for (int64_t kk = 0; kk < h; ++kk) {
            const float xv = normed[static_cast<size_t>(i) * h + kk];
            const uint16_t* rrow = rd + kk * e;
            for (int64_t n = 0; n < e; ++n)
              logits[static_cast<size_t>(i) * e + n] += xv * Bf16(rrow[n]);
          }
        }
      }
      for (int64_t i = 0; i < t; ++i) {
        // sigmoid -> top-k -> renormalise (expert_weights_norm).
        std::vector<std::pair<float, int64_t>> score(static_cast<size_t>(e));
        for (int64_t n = 0; n < e; ++n) {
          const float lg = logits[static_cast<size_t>(i) * e + n];
          score[static_cast<size_t>(n)] = {
              1.0f / (1.0f + std::exp(-lg)), n};
        }
        std::partial_sort(score.begin(), score.begin() + top_k, score.end(),
                          [](const auto& a, const auto& b) {
                            return a.first > b.first;
                          });
        double sum = 0.0;
        for (int64_t kk = 0; kk < top_k; ++kk) sum += score[static_cast<size_t>(kk)].first;
        std::vector<float> expert_in(normed.begin() + static_cast<std::ptrdiff_t>(i) * h,
                                     normed.begin() + static_cast<std::ptrdiff_t>(i + 1) * h);
        std::vector<float> combined(h, 0.0f);
        for (int64_t kk = 0; kk < top_k; ++kk) {
          const float wgt = static_cast<float>(score[static_cast<size_t>(kk)].first / sum);
          const FlintExpertWeights& ex =
              lw.moe.experts[static_cast<size_t>(score[static_cast<size_t>(kk)].second)];
          std::vector<float> gg(inter), uu(inter);
          MatmulBf16(expert_in.data(), ex.gate_proj.w, 1, gg.data());
          MatmulBf16(expert_in.data(), ex.up_proj.w, 1, uu.data());
          for (int64_t d = 0; d < inter; ++d)
            gg[static_cast<size_t>(d)] =
                gg[static_cast<size_t>(d)] / (1.0f + std::exp(-gg[static_cast<size_t>(d)])) *
                uu[static_cast<size_t>(d)];
          for (float& gv : gg) gv = Bf16(ToBf16(gv));
          std::vector<float> down(h);
          MatmulBf16(gg.data(), ex.down_proj.w, 1, down.data());
          for (int64_t d = 0; d < h; ++d) combined[static_cast<size_t>(d)] += wgt * down[static_cast<size_t>(d)];
        }
        // The UNGATED shared expert, always added.
        {
          const int64_t sinter = lw.moe.shared_expert.gate_proj.w.shape[1];
          std::vector<float> sg(sinter), su(sinter);
          MatmulBf16(expert_in.data(), lw.moe.shared_expert.gate_proj.w, 1, sg.data());
          MatmulBf16(expert_in.data(), lw.moe.shared_expert.up_proj.w, 1, su.data());
          for (int64_t d = 0; d < sinter; ++d)
            sg[static_cast<size_t>(d)] =
                sg[static_cast<size_t>(d)] / (1.0f + std::exp(-sg[static_cast<size_t>(d)])) *
                su[static_cast<size_t>(d)];
          for (float& gv : sg) gv = Bf16(ToBf16(gv));
          std::vector<float> sdown(h);
          MatmulBf16(sg.data(), lw.moe.shared_expert.down_proj.w, 1, sdown.data());
          for (int64_t d = 0; d < h; ++d) combined[static_cast<size_t>(d)] += sdown[static_cast<size_t>(d)];
        }
        for (int64_t d = 0; d < h; ++d)
          ff[static_cast<size_t>(i) * h + d] = Bf16(ToBf16(combined[static_cast<size_t>(d)]));
      }
    }
    for (int64_t i = 0; i < t * h; ++i)
      x[static_cast<size_t>(i)] =
          Bf16(ToBf16(x[static_cast<size_t>(i)] + ff[static_cast<size_t>(i)]));
    if (cpu_dump && layer == 0) {
      double s = 0.0;
      for (float v : x) s += v;
      std::fprintf(stderr, "[FLINT-CPU] layer 0 post-mlp x sum=%.6f\n", s);
    }
  }

  // Final norm + untied lm_head. Logits accumulate f32 and STAY f32 (the
  // sampler contract).
  RmsNormBf16(x.data(), w.final_norm, t, h, p.rms_norm_eps, normed.data());
  const int64_t vocab = p.vocab_size;
  std::vector<float> logits(static_cast<size_t>(t) * vocab, 0.0f);
  {
    const uint16_t* ld = Bf16Data(w.lm_head);
    for (int64_t i = 0; i < t; ++i) {
      for (int64_t kk = 0; kk < h; ++kk) {
        const float xv = normed[static_cast<size_t>(i) * h + kk];
        const uint16_t* lrow = ld + kk * vocab;
        for (int64_t n = 0; n < vocab; ++n)
          logits[static_cast<size_t>(i) * vocab + n] += xv * Bf16(lrow[n]);
      }
    }
  }
  if (cpu_dump) {
    double s = 0.0;
    for (int64_t n = 0; n < vocab; ++n) s += logits[static_cast<size_t>(n)];
    std::fprintf(stderr, "[FLINT-CPU] logits tok0 sum=%.6f\n", s);
  }
  return logits;
}

// ---- registry wiring (the qwen3_moe_registry.cpp seam) ----

namespace {

inline constexpr ModelInfo kFlintInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,  // hybrid ATTENTION, not hybrid GDN — one KV family
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class FlintLoadedModel final : public LoadedModel {
 public:
  FlintLoadedModel(const ModelRegistration& registration, FlintWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const FlintWeights& weights() const { return weights_; }

  // The W0 Tenstorrent device context (spec .agents/specs/flint-tenstorrent.md):
  // built once at prepare on the Tenstorrent queue, lazily otherwise. The
  // context borrows the queue it was built on; the engine keeps that queue
  // alive for the model's lifetime.
  FlintTTDeviceContext& tt_context(vt::Queue& queue) {
    if (!tt_ctx_) {
      vt::Backend& be = vt::GetBackend(queue.device.type);
      tt_ctx_ = BuildFlintTTDeviceContext(be, queue, weights_);
    }
    return *tt_ctx_;
  }

 private:
  FlintWeights weights_;
  std::unique_ptr<FlintTTDeviceContext> tt_ctx_;
};

std::unique_ptr<LoadedModel> LoadFlintForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  // The flint GGUF arm (.agents/specs/flint-gguf-arm.md, issue
  // ISSUE-LOCAL-01M5PRSC2E4VR54HBSWFPMN37E): the published containers load
  // into the same bf16-owned FlintWeights, forward unchanged.
  if (source.kind == ModelSource::Kind::kGguf) {
    if (source.gguf == nullptr) {
      throw std::runtime_error(
          "FlintForCausalLM: the model source says GGUF but carries no "
          "file. See .agents/specs/flint-gguf-arm.md");
    }
    return std::make_unique<FlintLoadedModel>(
        registration, LoadFlintFromGguf(*source.gguf, config));
  }
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "FlintForCausalLM: unsupported model source kind (the landed arms "
        "are safetensors and GGUF)");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<FlintLoadedModel>(
      registration,
      LoadFlintWeights(*source.safetensors, config));
}

void PrepareFlintForCausalLM(LoadedModel& model, const HfConfig& config,
                             vt::Queue& queue) {
  // The W0 Tenstorrent arm materializes the device context eagerly at
  // prepare — the resident slice uploads once, off the first forward (the
  // kolibri1_registry.cpp:316 seam).
  if (queue.device.type != vt::DeviceType::kTENSTORRENT) return;
  (void)config;
  auto& m = ModelAs<FlintLoadedModel>(model, "FlintForCausalLM");
  (void)m.tt_context(queue);
}

ForwardLogits ForwardFlintForCausalLM(LoadedModel& model,
                                      const ModelForwardInput& input) {
  auto& flint = ModelAs<FlintLoadedModel>(model, "FlintForCausalLM");
  // The device dispatch (the kolibri1_registry.cpp:328 seam): the CPU row
  // owns the CPU reference arm (unchanged); the Tenstorrent W0 arm owns the
  // device forward; every other device is refused by name here, before any
  // model code runs.
  switch (input.queue.device.type) {
    case vt::DeviceType::kCPU:
      break;
    case vt::DeviceType::kTENSTORRENT:
      break;
    default:
      throw std::runtime_error(
          std::string("FlintForCausalLM: the ") +
          vt::DeviceTypeName(input.queue.device.type) +
          " forward arm is not implemented. The landed arms are the CPU "
          "reference row (MODEL-TEXT-FLINT, spec "
          ".agents/specs/flint-cpu-gguf.md) and the Tenstorrent W0 device "
          "arm (spec .agents/specs/flint-tenstorrent.md); the " +
          vt::DeviceTypeName(input.queue.device.type) +
          " arm is a separate owed row.");
  }
  if (input.attn_meta.num_reqs != 1) {
    throw std::runtime_error(
        "FlintForCausalLM: the W0 forward arms serve exactly ONE "
        "sequence per step (got " +
        std::to_string(input.attn_meta.num_reqs) +
        "); the paged engine arm is a later wave of MODEL-TEXT-FLINT");
  }
  std::vector<float> logits;
  if (input.queue.device.type == vt::DeviceType::kTENSTORRENT) {
    logits = ForwardFlintTT(input.token_ids, input.positions,
                            flint.weights(), flint.tt_context(input.queue));
  } else {
    logits = FlintModel::Forward(input.token_ids, input.positions,
                                 flint.weights(), input.config);
  }
  return HostLogits(std::move(logits), input.config.vocab_size);
}

const ModelFactory kFlintFactory{
    .parse_config = &ParseFlintConfig,
    .load_weights = &LoadFlintForCausalLM,
    .prepare = &PrepareFlintForCausalLM,
    .forward = &ForwardFlintForCausalLM,
    .make_kv_cache = &MakeFlintKVCache,
    .is_dense_model = false,
    .consumes_device_token_ids = false,
    .streams_routed_experts = false,
};

}  // namespace

REGISTER_VLLM_MODEL(flint, "FlintForCausalLM", kFlintFactory, kFlintInfo)

// The checked accessor over the registry's loaded model (the TT W0 test and
// the device waves read the weights back out of the LoadedModel they are
// handed — the kolibri1_registry.cpp:386 seam).
const FlintWeights& FlintLoadedModelWeights(LoadedModel& model) {
  return ModelAs<FlintLoadedModel>(model, "FlintForCausalLM").weights();
}

}  // namespace vllm

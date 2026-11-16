// Weight loader: Gemma4 E4B PLE / 12B dense BF16 / 26B-A4B MoE BF16 fused experts.
#include <cmath>
// Experts are mmap-borrowed (30GB host cannot hold full BF16 MoE).
#include "vllm/model_executor/models/gemma4.h"

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/gemma4_moe.h"
#include "vt/dtype.h"
#include "gemma4_ct_weights.h"

#include <cmath>
#include <cstring>

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::MakeOwned;

const nlohmann::json& TextCfg(const nlohmann::json& raw) {
  const auto it = raw.find("text_config");
  if (it != raw.end() && it->is_object()) return *it;
  return raw;
}
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number_integer()) return fallback;
  return it->get<int64_t>();
}

struct LayerTopo {
  std::vector<bool> is_full;
  std::vector<bool> is_shared;
  std::vector<int64_t> head_dim;
  std::vector<int64_t> kv_target;
};

LayerTopo MakeTopo(const HfConfig& cfg) {
  const nlohmann::json& raw = TextCfg(cfg.raw);
  const int64_t L = cfg.num_hidden_layers;
  const int64_t head_dim_sliding = cfg.head_dim;
  const int64_t head_dim_full = RawInt(raw, "global_head_dim", cfg.head_dim);
  const int64_t num_shared = RawInt(raw, "num_kv_shared_layers", 0);
  const int64_t first_shared = L - num_shared;
  LayerTopo t;
  t.is_full.assign(static_cast<size_t>(L), false);
  const auto it = raw.find("layer_types");
  if (it != raw.end() && it->is_array()) {
    for (int64_t l = 0; l < L && static_cast<size_t>(l) < it->size(); ++l)
      t.is_full[static_cast<size_t>(l)] =
          it->at(static_cast<size_t>(l)).is_string() &&
          it->at(static_cast<size_t>(l)).get<std::string>() == "full_attention";
  }
  t.is_shared.assign(static_cast<size_t>(L), false);
  t.head_dim.assign(static_cast<size_t>(L), head_dim_sliding);
  t.kv_target.assign(static_cast<size_t>(L), -1);
  for (int64_t l = 0; l < L; ++l) {
    const bool full = t.is_full[static_cast<size_t>(l)];
    t.head_dim[static_cast<size_t>(l)] = full ? head_dim_full : head_dim_sliding;
    if (num_shared > 0 && l >= first_shared) {
      t.is_shared[static_cast<size_t>(l)] = true;
      int64_t target = -1;
      for (int64_t p = first_shared - 1; p >= 0; --p) {
        if (t.is_full[static_cast<size_t>(p)] == full) {
          target = p;
          break;
        }
      }
      t.kv_target[static_cast<size_t>(l)] = target;
    }
  }
  return t;
}

// Borrow BF16 tensor from mmap (no host copy). Keepalive = shards shared_ptr.
OwnedTensor BorrowBf16(const StTensor& t, std::shared_ptr<const void> owner,
                       const std::vector<int64_t>& shape_override = {}) {
  VT_CHECK(t.dtype == "BF16", "gemma4: expected BF16 borrow");
  std::vector<int64_t> shape = shape_override.empty() ? t.shape : shape_override;
  OwnedTensor o;
  o.dtype = vt::DType::kBF16;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "gemma4: rank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  VT_CHECK(static_cast<size_t>(n) * 2 == t.nbytes, "gemma4: borrow size mismatch");
  o.bytes = OwnedBytes::Borrow(t.data, t.nbytes, std::move(owner));
  return o;
}

OwnedTensor BorrowBytes(const StTensor& t, vt::DType dt, size_t elem_size,
                        std::shared_ptr<const void> owner) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(t.shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "gemma4: rank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = t.shape[static_cast<size_t>(i)];
    n *= t.shape[static_cast<size_t>(i)];
  }
  VT_CHECK(static_cast<size_t>(n) * elem_size == t.nbytes, "gemma4: borrow bytes size");
  o.bytes = OwnedBytes::Borrow(t.data, t.nbytes, std::move(owner));
  return o;
}

// FP8 + channel scale → owned BF16 raw-NK [N,K]
OwnedTensor LoadFp8ChannelToBf16RawNk(const TensorResolver& get, const std::string& base) {
  const StTensor& w = get(base + ".weight");
  const StTensor& s = get(base + ".weight_scale");
  VT_CHECK(w.dtype == "F8_E4M3", "gemma4 fp8: expected F8_E4M3 for " + base);
  VT_CHECK(s.dtype == "BF16", "gemma4 fp8: expected BF16 scale for " + base);
  VT_CHECK(w.shape.size() == 2, "gemma4 fp8: rank-2 weight");
  const int64_t N = w.shape[0], K = w.shape[1];
  VT_CHECK(s.shape[0] == N, "gemma4 fp8: scale N");
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {N, K});
  o.nk = true;
  // `s.data` points into the mmap'd safetensors payload at an arbitrary byte
  // offset, so a `const uint16_t*` onto it is undefined to form or load through
  // (issue #627), and DequantFp8ChannelToBf16 takes a typed scale pointer. Copy
  // the N-element scale row into an aligned buffer first — N is the output
  // channel count, negligible next to the N*K dequant it feeds.
  std::vector<uint16_t> scale(static_cast<size_t>(N));
  VT_CHECK(s.nbytes >= scale.size() * sizeof(uint16_t),
           "gemma4 fp8: scale tensor too small for " + base);
  std::memcpy(scale.data(), s.data, scale.size() * sizeof(uint16_t));
  DequantFp8ChannelToBf16(w.data, scale.data(), N, K,
                          reinterpret_cast<uint16_t*>(o.bytes.data()));
  MaybeReleaseSourcePages(w.data, w.nbytes);
  MaybeReleaseSourcePages(s.data, s.nbytes);
  return o;
}

void FuseRouter(Gemma4MoeLayerWeights& m, int64_t E, int64_t H) {
  VT_CHECK(m.router_proj.nk && m.router_proj.shape[0] == E && m.router_proj.shape[1] == H,
           "gemma4 moe: router_proj shape");
  const float rsqrt_h = 1.f / std::sqrt(static_cast<float>(H));
  const auto* sc = reinterpret_cast<const uint16_t*>(m.router_scale.bytes.data());
  const auto* ps = reinterpret_cast<const uint16_t*>(m.router_proj.bytes.data());
  m.router_proj_fused = MakeOwned(vt::DType::kBF16, {E, H});
  m.router_proj_fused.nk = true;
  auto* pd = reinterpret_cast<uint16_t*>(m.router_proj_fused.bytes.data());
  for (int64_t e = 0; e < E; ++e)
    for (int64_t j = 0; j < H; ++j)
      pd[e * H + j] =
          vt::F32ToBF16(vt::BF16ToF32(ps[e * H + j]) * vt::BF16ToF32(sc[j]) * rsqrt_h);
}

Gemma4MoeLayerWeights LoadMoeCommonRouter(const TensorResolver& get, const std::string& base,
                                          int64_t /*E*/, int64_t top_k, int64_t moe_I) {
  Gemma4MoeLayerWeights m;
  m.enabled = true;
  m.top_k = static_cast<int>(top_k);
  m.moe_intermediate = moe_I;
  m.router_scale = LoadBf16Direct(get, base + "router.scale");
  m.router_proj = LoadMergedBf16RawNK(get, {base + "router.proj.weight"});
  m.per_expert_scale = LoadBf16Direct(get, base + "router.per_expert_scale");
  m.pre_feedforward_layernorm_2 =
      LoadBf16Direct(get, base + "pre_feedforward_layernorm_2.weight");
  m.post_feedforward_layernorm_1 =
      LoadBf16Direct(get, base + "post_feedforward_layernorm_1.weight");
  m.post_feedforward_layernorm_2 =
      LoadBf16Direct(get, base + "post_feedforward_layernorm_2.weight");
  return m;
}

Gemma4MoeLayerWeights LoadMoeBf16(
    const TensorResolver& get, const std::string& base, int64_t E, int64_t top_k,
    int64_t moe_I, int64_t H, std::shared_ptr<const void> owner) {
  Gemma4MoeLayerWeights m = LoadMoeCommonRouter(get, base, E, top_k, moe_I);
  FuseRouter(m, E, H);
  const StTensor& gu = get(base + "experts.gate_up_proj");
  const StTensor& dn = get(base + "experts.down_proj");
  VT_CHECK(gu.shape.size() == 3 && gu.shape[0] == E && gu.shape[1] == 2 * moe_I &&
               gu.shape[2] == H,
           "gemma4 moe: gate_up expected [E,2I,H]");
  VT_CHECK(dn.shape.size() == 3 && dn.shape[0] == E && dn.shape[1] == H &&
               dn.shape[2] == moe_I,
           "gemma4 moe: down expected [E,H,I]");
  m.experts.gate_up = BorrowBf16(gu, owner);
  m.experts.down = BorrowBf16(dn, owner);
  m.experts.is_fp8 = false;
  m.experts.num_experts = E;
  m.experts.intermediate = moe_I;
  m.experts.hidden = H;
  return m;
}

// Firworks / per-expert FP8 MoELinear export.
Gemma4MoeLayerWeights LoadMoeFp8PerExpert(
    const TensorResolver& get, const std::string& base, int64_t E, int64_t top_k,
    int64_t moe_I, int64_t H, std::shared_ptr<const void> owner) {
  Gemma4MoeLayerWeights m = LoadMoeCommonRouter(get, base, E, top_k, moe_I);
  FuseRouter(m, E, H);
  m.experts.is_fp8 = true;
  m.experts.num_experts = E;
  m.experts.intermediate = moe_I;
  m.experts.hidden = H;
  m.experts.fp8.resize(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    const std::string eb = base + "experts." + std::to_string(e) + ".";
    auto& ex = m.experts.fp8[static_cast<size_t>(e)];
    const StTensor& gw = get(eb + "gate_proj.weight");
    const StTensor& gs = get(eb + "gate_proj.weight_scale");
    const StTensor& uw = get(eb + "up_proj.weight");
    const StTensor& us = get(eb + "up_proj.weight_scale");
    const StTensor& dw = get(eb + "down_proj.weight");
    const StTensor& ds = get(eb + "down_proj.weight_scale");
    VT_CHECK(gw.dtype == "F8_E4M3" && gw.shape.size() == 2 && gw.shape[0] == moe_I &&
                 gw.shape[1] == H,
             "gemma4 fp8 expert gate shape");
    VT_CHECK(dw.dtype == "F8_E4M3" && dw.shape[0] == H && dw.shape[1] == moe_I,
             "gemma4 fp8 expert down shape");
    ex.gate_w = BorrowBytes(gw, vt::DType::kI8, 1, owner);
    ex.gate_s = BorrowBytes(gs, vt::DType::kBF16, 2, owner);
    ex.up_w = BorrowBytes(uw, vt::DType::kI8, 1, owner);
    ex.up_s = BorrowBytes(us, vt::DType::kBF16, 2, owner);
    ex.down_w = BorrowBytes(dw, vt::DType::kI8, 1, owner);
    ex.down_s = BorrowBytes(ds, vt::DType::kBF16, 2, owner);
  }
  return m;
}

Gemma4LayerWeights LoadGemma4Layer(
    const TensorResolver& get, const std::unordered_set<std::string>& names,
    int64_t layer, const LayerTopo& topo, bool load_ple, int64_t num_q_heads,
    int64_t num_kv_heads_sliding, int64_t num_kv_heads_full, bool enable_moe,
    int64_t num_experts, int64_t top_k, int64_t moe_I, int64_t H,
    std::shared_ptr<const void> owner) {
  const std::string base =
      "model.language_model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  Gemma4LayerWeights w;
  w.is_full_attention = topo.is_full[static_cast<size_t>(layer)];
  w.is_kv_shared = topo.is_shared[static_cast<size_t>(layer)];
  w.head_dim = topo.head_dim[static_cast<size_t>(layer)];
  w.kv_target_layer = topo.kv_target[static_cast<size_t>(layer)];
  w.num_kv_heads =
      w.is_full_attention ? num_kv_heads_full : num_kv_heads_sliding;

  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");
  w.pre_feedforward_layernorm =
      LoadBf16Direct(get, base + "pre_feedforward_layernorm.weight");
  w.post_feedforward_layernorm =
      LoadBf16Direct(get, base + "post_feedforward_layernorm.weight");

  if (load_ple) {
    w.per_layer_input_gate =
        LoadMergedBf16RawNK(get, {base + "per_layer_input_gate.weight"});
    w.per_layer_projection =
        LoadMergedBf16RawNK(get, {base + "per_layer_projection.weight"});
    w.post_per_layer_input_norm =
        LoadBf16Direct(get, base + "post_per_layer_input_norm.weight");
  }
  if (names.count(base + "layer_scalar"))
    w.layer_scalar = LoadBf16Direct(get, base + "layer_scalar");

  const bool fp8_attn = names.count(sa + "q_proj.weight_scale") > 0;
  const std::string v_name = sa + "v_proj.weight";
  w.k_eq_v = names.count(v_name) == 0;
  if (fp8_attn) {
    OwnedTensor q = LoadFp8ChannelToBf16RawNk(get, sa + "q_proj");
    OwnedTensor k = LoadFp8ChannelToBf16RawNk(get, sa + "k_proj");
    OwnedTensor v = w.k_eq_v ? k : LoadFp8ChannelToBf16RawNk(get, sa + "v_proj");
    const int64_t nq = q.shape[0], nk = k.shape[0], nv = v.shape[0];
    w.attn.qkv_proj = MakeOwned(vt::DType::kBF16, {nq + nk + nv, H});
    w.attn.qkv_proj.nk = true;
    auto* dst = reinterpret_cast<uint16_t*>(w.attn.qkv_proj.bytes.data());
    std::memcpy(dst, q.bytes.data(), q.bytes.size());
    std::memcpy(dst + nq * H, k.bytes.data(), k.bytes.size());
    std::memcpy(dst + (nq + nk) * H, v.bytes.data(), v.bytes.size());
    w.attn.o_proj = LoadFp8ChannelToBf16RawNk(get, sa + "o_proj");
  } else if (w.k_eq_v) {
    w.attn.qkv_proj = LoadMergedBf16RawNK(
        get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "k_proj.weight"});
    w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  } else {
    w.attn.qkv_proj = LoadMergedBf16RawNK(
        get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
    w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  }
  w.attn.q_norm = LoadBf16Direct(get, sa + "q_norm.weight");
  w.attn.k_norm = LoadBf16Direct(get, sa + "k_norm.weight");

  if (w.attn.qkv_proj.rank >= 1 && num_q_heads > 0 && w.num_kv_heads > 0) {
    const int64_t rows = w.attn.qkv_proj.shape[0];
    const int64_t denom = num_q_heads + 2 * w.num_kv_heads;
    if (denom > 0 && rows % denom == 0) {
      const int64_t dh = rows / denom;
      if (dh > 0) w.head_dim = dh;
    }
  }

  if (names.count(mlp + "gate_proj.weight_scale")) {
    OwnedTensor g = LoadFp8ChannelToBf16RawNk(get, mlp + "gate_proj");
    OwnedTensor u = LoadFp8ChannelToBf16RawNk(get, mlp + "up_proj");
    const int64_t Ig = g.shape[0];
    w.mlp.gate_up_proj = MakeOwned(vt::DType::kBF16, {Ig + u.shape[0], H});
    w.mlp.gate_up_proj.nk = true;
    auto* dst = reinterpret_cast<uint16_t*>(w.mlp.gate_up_proj.bytes.data());
    std::memcpy(dst, g.bytes.data(), g.bytes.size());
    std::memcpy(dst + Ig * H, u.bytes.data(), u.bytes.size());
    w.mlp.down_proj = LoadFp8ChannelToBf16RawNk(get, mlp + "down_proj");
  } else {
    w.mlp.gate_up_proj =
        LoadMergedBf16RawNK(get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
    w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  }

  if (enable_moe) {
    if (names.count(base + "experts.gate_up_proj")) {
      w.moe = LoadMoeBf16(get, base, num_experts, top_k, moe_I, H, owner);
    } else if (names.count(base + "experts.0.gate_proj.weight")) {
      w.moe = LoadMoeFp8PerExpert(get, base, num_experts, top_k, moe_I, H, owner);
    }
  }
  return w;
}


Gemma4Weights LoadCt(const TensorResolver& get,
                    const std::unordered_set<std::string>& names,
                    const HfConfig& cfg, std::shared_ptr<const void> owner) {
  gemma4_ct::Reader reader(get, cfg.raw);
  const auto& text = TextCfg(cfg.raw);
  VT_CHECK(RawInt(text, "hidden_size_per_layer_input", 0) == 0,
           "gemma4 CT: per-layer embeddings are not implemented");
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const int64_t E = RawInt(text, "num_experts", 0);
  const int64_t EI = RawInt(text, "moe_intermediate_size", 0);
  const int64_t K = RawInt(text, "top_k_experts", 8);
  const bool moe = RawBool(text, "enable_moe_block", false);
  VT_CHECK(!moe || (E > 0 && EI > 0 && K > 0 && K <= E),
           "gemma4 CT: invalid expert dimensions");
  const LayerTopo topo = MakeTopo(cfg);
  Gemma4Weights w;
  w.compact_ct = true;
  w.shards_keepalive = std::move(owner);
  w.tie_word_embeddings = RawBool(text, "tie_word_embeddings", true);
  // Retain F16 storage for the tied output head; Embedding converts selected
  // rows by value into the existing BF16 activation stream.
  w.embed_tokens = reader.Plain("model.language_model.embed_tokens.weight",
                                {cfg.vocab_size, H}, true);
  w.final_norm = reader.Plain("model.language_model.norm.weight", {H});
  if (!w.tie_word_embeddings) {
    // Keep torch Linear [vocab,H] orientation for the output head.
    w.lm_head = reader.Linear("lm_head", cfg.vocab_size, H);
  }
  for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
    const std::string b = "model.language_model.layers." + std::to_string(l) + ".";
    Gemma4LayerWeights v;
    v.is_full_attention = topo.is_full[static_cast<size_t>(l)];
    v.is_kv_shared = topo.is_shared[static_cast<size_t>(l)];
    v.head_dim = topo.head_dim[static_cast<size_t>(l)];
    v.kv_target_layer = topo.kv_target[static_cast<size_t>(l)];
    v.num_kv_heads = v.is_full_attention
                        ? RawInt(text, "num_global_key_value_heads", cfg.num_key_value_heads)
                        : cfg.num_key_value_heads;
    const int64_t qn = cfg.num_attention_heads * v.head_dim;
    const int64_t kn = v.num_kv_heads * v.head_dim;
    v.input_layernorm = reader.Plain(b + "input_layernorm.weight", {H});
    v.post_attention_layernorm = reader.Plain(b + "post_attention_layernorm.weight", {H});
    v.pre_feedforward_layernorm = reader.Plain(b + "pre_feedforward_layernorm.weight", {H});
    v.post_feedforward_layernorm = reader.Plain(b + "post_feedforward_layernorm.weight", {H});
    if (names.count(b + "layer_scalar"))
      v.layer_scalar = reader.Plain(b + "layer_scalar", {1});
    v.k_eq_v = names.count(b + "self_attn.v_proj.weight_packed") == 0 &&
               names.count(b + "self_attn.v_proj.weight") == 0;
    v.attn.qkv_parts.push_back(reader.Linear(b + "self_attn.q_proj", qn, H));
    v.attn.qkv_parts.push_back(reader.Linear(b + "self_attn.k_proj", kn, H));
    v.attn.qkv_parts.push_back(v.k_eq_v
                                  ? v.attn.qkv_parts[1]
                                  : reader.Linear(b + "self_attn.v_proj", kn, H));
    v.attn.o_proj = reader.Linear(b + "self_attn.o_proj", H, qn);
    v.attn.q_norm = reader.Plain(b + "self_attn.q_norm.weight", {v.head_dim});
    v.attn.k_norm = reader.Plain(b + "self_attn.k_norm.weight", {v.head_dim});
    v.mlp.gate_up_parts.push_back(reader.Linear(b + "mlp.gate_proj", I, H));
    v.mlp.gate_up_parts.push_back(reader.Linear(b + "mlp.up_proj", I, H));
    v.mlp.down_proj = reader.Linear(b + "mlp.down_proj", H, I);
    if (moe) {
      auto& m = v.moe;
      m.enabled = true;
      m.top_k = static_cast<int>(K);
      m.moe_intermediate = EI;
      m.router_scale = reader.Plain(b + "router.scale", {H});
      m.router_proj = reader.Linear(b + "router.proj", E, H);
      VT_CHECK(m.router_proj.dtype == vt::DType::kBF16,
               "gemma4 CT: quantized router is not implemented");
      m.per_expert_scale = reader.Plain(b + "router.per_expert_scale", {E});
      m.pre_feedforward_layernorm_2 = reader.Plain(b + "pre_feedforward_layernorm_2.weight", {H});
      m.post_feedforward_layernorm_1 = reader.Plain(b + "post_feedforward_layernorm_1.weight", {H});
      m.post_feedforward_layernorm_2 = reader.Plain(b + "post_feedforward_layernorm_2.weight", {H});
      FuseRouter(m, E, H);
      auto& ex = m.experts;
      ex.num_experts = E;
      ex.intermediate = EI;
      ex.hidden = H;
      ex.compact.reserve(static_cast<size_t>(E));
      for (int64_t e = 0; e < E; ++e) {
        const std::string eb = b + "experts." + std::to_string(e) + ".";
        Gemma4QuantExpertMats x;
        x.gate_up.push_back(reader.Linear(eb + "gate_proj", EI, H));
        x.gate_up.push_back(reader.Linear(eb + "up_proj", EI, H));
        x.down = reader.Linear(eb + "down_proj", H, EI);
        ex.compact.push_back(std::move(x));
      }
    }
    w.layers.push_back(std::move(v));
  }
  return w;
}

Gemma4Weights LoadImpl(const std::vector<SafetensorsFile>& shards,
                       const HfConfig& config,
                       std::shared_ptr<const void> shards_owner) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  std::unordered_set<std::string> names;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      where[name] = &shard;
      names.insert(name);
    }
  }
  const TensorResolver get = [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "gemma4: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0, "gemma4: num_hidden_layers");
  const auto* quant = layers::compressed_tensors::QuantizationConfigOf(config.raw);
  if (quant && quant->value("quant_method", std::string()) == "compressed-tensors" &&
      quant->value("format", std::string()) == "pack-quantized")
    return LoadCt(get, names, config, std::move(shards_owner));

  const nlohmann::json& text = TextCfg(config.raw);
  const int64_t H = config.hidden_size;
  const int64_t ple_cfg = RawInt(text, "hidden_size_per_layer_input", 0);
  const bool load_ple =
      ple_cfg > 0 &&
      names.count("model.language_model.embed_tokens_per_layer.weight") > 0;
  const bool enable_moe =
      RawBool(text, "enable_moe_block", false) ||
      names.count("model.language_model.layers.0.experts.gate_up_proj") > 0 ||
      names.count("model.language_model.layers.0.experts.0.gate_proj.weight") > 0 ||
      names.count("model.language_model.layers.0.router.proj.weight") > 0;
  const int64_t num_experts = RawInt(text, "num_experts", 0);
  const int64_t top_k = RawInt(text, "top_k_experts", 8);
  const int64_t moe_I = RawInt(text, "moe_intermediate_size", 0);

  if (enable_moe) {
    VT_CHECK(shards_owner != nullptr,
             "gemma4 MoE BF16 requires safetensors_owned keepalive for expert mmap");
  }

  const LayerTopo topo = MakeTopo(config);
  Gemma4Weights w;
  w.tie_word_embeddings = RawBool(text, "tie_word_embeddings", true);
  w.embed_tokens = LoadBf16Direct(get, "model.language_model.embed_tokens.weight");
  if (load_ple) {
    w.embed_tokens_per_layer =
        LoadBf16Direct(get, "model.language_model.embed_tokens_per_layer.weight");
    w.per_layer_model_projection = LoadMergedBf16RawNK(
        get, {"model.language_model.per_layer_model_projection.weight"});
    w.per_layer_projection_norm =
        LoadBf16Direct(get, "model.language_model.per_layer_projection_norm.weight");
  }
  w.final_norm = LoadBf16Direct(get, "model.language_model.norm.weight");
  if (!w.tie_word_embeddings && names.count("lm_head.weight"))
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");

  // Keep shards alive for borrowed experts
  w.shards_keepalive = shards_owner;

  const int64_t Hq = config.num_attention_heads;
  const int64_t Hkv_slide = config.num_key_value_heads;
  const int64_t Hkv_full = RawInt(text, "num_global_key_value_heads", Hkv_slide);
  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    w.layers.push_back(LoadGemma4Layer(get, names, l, topo, load_ple, Hq, Hkv_slide,
                                       Hkv_full, enable_moe, num_experts, top_k, moe_I,
                                       H, shards_owner));
  }
  return w;
}

}  // namespace

Gemma4Weights LoadGemma4ForConditionalGenerationWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  // Dense 12B path: no owner needed. MoE will throw if experts present without owner.
  return LoadImpl(shards, config, nullptr);
}

Gemma4Weights LoadGemma4ForConditionalGenerationWeightsOwned(
    std::shared_ptr<const std::vector<SafetensorsFile>> shards,
    const HfConfig& config) {
  VT_CHECK(shards != nullptr, "gemma4: null shards");
  return LoadImpl(*shards, config, std::shared_ptr<const void>(shards, shards.get()));
}

}  // namespace vllm

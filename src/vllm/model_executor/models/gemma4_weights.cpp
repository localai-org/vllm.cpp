// Weight loader: Gemma4 E4B PLE / 12B dense BF16 / 26B-A4B MoE BF16 fused experts.
#include <cmath>
#include <cstdio>
// Experts are mmap-borrowed (30GB host cannot hold full BF16 MoE).
#include "vllm/model_executor/models/gemma4.h"

#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/gemma4_moe.h"
#include "vt/dtype.h"

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

// vLLM gemma4_mm.py::Gemma4MultimodalEmbedder at a7c23ac96d:
// biasless projection from encoder output to text hidden, with weightless norm.
Gemma4ModalityWeights LoadModalityProjector(
    const TensorResolver& get, const HfConfig& config, const char* key,
    const char* prefix, bool audio) {
  Gemma4ModalityWeights w;
  const auto it = config.raw.find(key);
  if (it == config.raw.end() || it->is_null()) return w;
  const std::string context = std::string("gemma4: ") + key;
  VT_CHECK(it->is_object(), context + " must be an object or null");
  const auto& cfg = *it;
  const auto hidden = cfg.find("hidden_size");
  VT_CHECK(hidden != cfg.end() && hidden->is_number_integer() &&
               *hidden > 0 && *hidden <= INT64_MAX,
           context + " hidden_size must be a positive integer");
  int64_t width = hidden->get<int64_t>();
  const auto eps = cfg.find("rms_norm_eps");
  VT_CHECK(eps != cfg.end() && eps->is_number() &&
               std::isfinite(eps->get<double>()) && eps->get<double>() > 0,
           context + " rms_norm_eps must be positive and finite");
  if (audio) {
    const auto output = cfg.find("output_proj_dims");
    if (output != cfg.end() && !output->is_null()) {
      VT_CHECK(output->is_number_integer() && *output >= 0 && *output <= INT64_MAX,
               context + " output_proj_dims must be a nonnegative integer or null");
      if (*output != 0) width = output->get<int64_t>();
    }
  }
  VT_CHECK(config.hidden_size > 0, context + " text hidden_size must be positive");
  const std::string name = std::string(prefix) + ".embedding_projection.weight";
  const auto& tensor = get(name);
  VT_CHECK((tensor.shape == std::vector<int64_t>{config.hidden_size, width}),
           "gemma4: " + name + " must have shape [text_hidden, encoder_output]");
  w.config = cfg;
  w.embedding_projection = LoadBf16Direct(get, name);
  return w;
}

// Transformers a08ace4b Gemma4AudioSubSampleConvProjection. Host-F32
// reference inventory only: no Conformer execution or encoder-hook claim.
void LoadAudioFrontend(const TensorResolver& get, Gemma4Weights& w) {
  const auto& cfg = w.audio.config;
  if (cfg.is_null()) return;
  Gemma4AudioFrontendConfig c{};
  c.hidden_size = cfg.at("hidden_size").get<int64_t>(); // projector validated
  const auto channels = cfg.value("subsampling_conv_channels", nlohmann::json{128, 32});
  VT_CHECK(channels.is_array() && channels.size() == 2,
           "gemma4: audio_config subsampling_conv_channels must contain two integers");
  for (const auto& v : channels)
    VT_CHECK(v.is_number_integer() && v > 0 && v <= INT64_MAX,
             "gemma4: audio_config subsampling_conv_channels must be positive integers");
  c.sub_ch0 = channels[0].get<int64_t>();
  c.sub_ch1 = channels[1].get<int64_t>();
  VT_CHECK(c.sub_ch0 >= 4,
           "gemma4: audio_config subsampling_conv_channels produces zero input width");
  const double eps = cfg.at("rms_norm_eps").get<double>();
  c.rms_norm_eps = static_cast<float>(eps);
  VT_CHECK(std::isfinite(c.rms_norm_eps) && c.rms_norm_eps > 0,
           "gemma4: audio_config rms_norm_eps must be representable as positive finite F32");
  // Check every F32 destination size before any lookup, multiplication or allocation.
  const auto count = [](std::initializer_list<int64_t> dims) {
    int64_t n = 1;
    for (auto d : dims) {
      VT_CHECK(d > 0 && n <= INT64_MAX / 4 / d &&
                   static_cast<uint64_t>(n) <= SIZE_MAX / sizeof(float) / static_cast<uint64_t>(d),
               "gemma4: audio_config frontend tensor size overflow");
      n *= d;
    }
    return n;
  };
  count({c.sub_ch0, 1, 3, 3});
  count({c.sub_ch1, c.sub_ch0, 3, 3});
  c.input_width = count({c.sub_ch0 / 4, c.sub_ch1});
  count({c.hidden_size, c.input_width});
  const auto load = [&](const char* suffix, const std::vector<int64_t>& shape) {
    const std::string name = std::string("model.audio_tower.subsample_conv_projection.") + suffix;
    const auto& t = get(name);
    VT_CHECK(t.shape == shape, "gemma4: " + name + " incorrect shape");
    VT_CHECK(t.dtype == "BF16", "gemma4: " + name + " requires BF16");
    auto owned = LoadBf16Direct(get, name);
    std::vector<float> values(owned.bytes.size() / sizeof(uint16_t));
    for (size_t i = 0; i < values.size(); ++i) {
      uint16_t bits;
      std::memcpy(&bits, owned.bytes.data() + i * sizeof(bits), sizeof(bits));
      values[i] = vt::BF16ToF32(bits);
    }
    return values;
  };
  auto& a = w.audio_frontend_weights;
  a.sub0_conv = load("layer0.conv.weight", {c.sub_ch0, 1, 3, 3});
  a.sub0_norm = load("layer0.norm.weight", {c.sub_ch0});
  a.sub1_conv = load("layer1.conv.weight", {c.sub_ch1, c.sub_ch0, 3, 3});
  a.sub1_norm = load("layer1.norm.weight", {c.sub_ch1});
  a.input_proj = load("input_proj_linear.weight", {c.hidden_size, c.input_width});
  w.audio_frontend_config = c;
}

// Transformers a08ace4b Gemma4Audio{Layer,Model,Config}. This is owned
// host-F32 reference storage widened from the header's BF16, not device execution.
void LoadAudioBlocks(const TensorResolver& get, Gemma4Weights& w, int64_t text_hidden) {
  const auto& cfg = w.audio.config;
  if (cfg.is_null()) return;
  multimodal::Gemma4AudioConfig c;
  const auto integer = [&](const char* key, int64_t fallback, bool zero = false) {
    const auto v = cfg.value(key, nlohmann::json(fallback));
    VT_CHECK(v.is_number_integer() && v >= (zero ? 0 : 1) && v <= INT64_MAX,
             std::string("gemma4: audio_config ") + key + " invalid integer");
    return v.get<int64_t>();
  };
  const auto number = [&](const char* key, double fallback, bool positive) {
    const auto v = cfg.value(key, nlohmann::json(fallback));
    VT_CHECK(v.is_number() && std::isfinite(v.get<double>()) &&
                 std::isfinite(static_cast<float>(v.get<double>())) &&
                 (!positive || static_cast<float>(v.get<double>()) > 0),
             std::string("gemma4: audio_config ") + key + " invalid finite F32 value");
    return static_cast<float>(v.get<double>());
  };
  c.hidden_size = w.audio_frontend_config->hidden_size;
  c.sub_ch0 = w.audio_frontend_config->sub_ch0;
  c.sub_ch1 = w.audio_frontend_config->sub_ch1;
  c.rms_norm_eps = w.audio_frontend_config->rms_norm_eps;
  c.text_hidden_size = text_hidden;
  c.num_layers = integer("num_hidden_layers", 12);
  c.num_heads = integer("num_attention_heads", 8);
  c.head_dim = c.hidden_size / c.num_heads;
  VT_CHECK(c.head_dim > 0, "gemma4: audio_config num_attention_heads exceeds hidden_size");
  c.conv_kernel_size = integer("conv_kernel_size", 5);
  c.chunk_size = integer("attention_chunk_size", 12);
  c.context_left = integer("attention_context_left", 13);
  c.context_right = integer("attention_context_right", 0, true);
  VT_CHECK(c.context_left - 1 <= INT64_MAX - c.chunk_size &&
               c.context_right <= INT64_MAX - c.chunk_size - (c.context_left - 1),
           "gemma4: audio_config attention context overflow");
  c.residual_weight = number("residual_weight", 0.5, false);
  c.attention_logit_cap = number("attention_logit_cap", 50, true);
  c.attention_invalid_logits_value = number("attention_invalid_logits_value", -1e9, false);
  c.gradient_clipping = number("gradient_clipping", 1e10, true);
  const auto act = cfg.value("hidden_act", nlohmann::json("silu"));
  VT_CHECK(act.is_string(), "gemma4: audio_config hidden_act must be a string");
  c.hidden_act = act.get<std::string>(); // retention only; no activation dispatch here
  const auto clipped = cfg.value("use_clipped_linears", nlohmann::json(true));
  VT_CHECK(clipped.is_boolean(), "gemma4: audio_config use_clipped_linears must be boolean");
  c.use_clipped_linears = clipped.get<bool>();
  c.output_proj_dims = c.hidden_size;
  if (cfg.contains("output_proj_dims") && !cfg.at("output_proj_dims").is_null() &&
      cfg.at("output_proj_dims") != 0)
    c.output_proj_dims = integer("output_proj_dims", c.hidden_size);
  const auto count = [](std::initializer_list<int64_t> dims) {
    uint64_t n = 1;
    for (auto d : dims) {
      VT_CHECK(d > 0 && n <= static_cast<uint64_t>(INT64_MAX) / 4 / d &&
                   n <= SIZE_MAX / sizeof(float) / static_cast<uint64_t>(d),
               "gemma4: audio_config block tensor size overflow");
      n *= d;
    }
    return n;
  };
  const auto h = c.hidden_size;
  count({4,h,h,c.num_layers}); // includes expansion and all layer sizes before allocation
  count({h,1,c.conv_kernel_size,c.num_layers});
  count({c.output_proj_dims,h});
  count({c.num_layers,static_cast<int64_t>(sizeof(multimodal::Gemma4AudioLayerWeights))});
  const auto load = [&](const std::string& name, const std::vector<int64_t>& shape) {
    const auto& t = get(name);
    VT_CHECK(t.shape == shape, "gemma4: " + name + " incorrect shape");
    VT_CHECK(t.dtype == "BF16", "gemma4: " + name + " requires BF16");
    const auto owned = LoadBf16Direct(get, name);
    std::vector<float> values(owned.bytes.size()/2);
    for (size_t i=0;i<values.size();++i) {
      uint16_t bits; std::memcpy(&bits,owned.bytes.data()+2*i,2);
      values[i]=vt::BF16ToF32(bits);
    }
    return values;
  };
  const auto linear = [&](const std::string& name, int64_t out, int64_t in) {
    multimodal::ClipLinear v;
    v.w = load(name+".linear.weight",{out,in});
    if(c.use_clipped_linears) {
      const auto bound = [&](const char* suffix) {
        const auto full=name+suffix;
        const auto f=load(full,{}).at(0);
        VT_CHECK(!std::isnan(f), "gemma4: " + full + " must not be NaN");
        return f;
      };
      v.clip.in_min=bound(".input_min"); v.clip.in_max=bound(".input_max");
      v.clip.out_min=bound(".output_min"); v.clip.out_max=bound(".output_max");
      VT_CHECK(v.clip.in_min<=v.clip.in_max, "gemma4: "+name+" input_min exceeds input_max");
      VT_CHECK(v.clip.out_min<=v.clip.out_max, "gemma4: "+name+" output_min exceeds output_max");
    }
    return v;
  };
  auto& a=w.audio_frontend_weights;
  a.layers.resize(c.num_layers);
  for(int64_t i=0;i<c.num_layers;++i) {
    const auto p="model.audio_tower.layers."+std::to_string(i)+".";
    auto& l=a.layers[i];
    const auto ff = [&](const char* suffix, multimodal::Gemma4AudioFFWeights& f) {
      const auto n=p+suffix;
      f.ffw1=linear(n+".ffw_layer_1",4*h,h);
      f.ffw2=linear(n+".ffw_layer_2",h,4*h);
      f.pre_ln=load(n+".pre_layer_norm.weight",{h});
      f.post_ln=load(n+".post_layer_norm.weight",{h});
    };
    ff("feed_forward1",l.ff1); ff("feed_forward2",l.ff2);
    const auto q=c.num_heads*c.head_dim;
    l.attn.q_proj=linear(p+"self_attn.q_proj",q,h);
    l.attn.k_proj=linear(p+"self_attn.k_proj",q,h);
    l.attn.v_proj=linear(p+"self_attn.v_proj",q,h);
    l.attn.post=linear(p+"self_attn.post",h,h);
    l.attn.relative_k_proj=load(p+"self_attn.relative_k_proj.weight",{q,h});
    l.attn.per_dim_scale=load(p+"self_attn.per_dim_scale",{c.head_dim});
    l.lconv.linear_start=linear(p+"lconv1d.linear_start",2*h,h);
    l.lconv.linear_end=linear(p+"lconv1d.linear_end",h,h);
    l.lconv.depthwise=load(p+"lconv1d.depthwise_conv1d.weight",{h,1,c.conv_kernel_size});
    l.lconv.pre_ln=load(p+"lconv1d.pre_layer_norm.weight",{h});
    l.lconv.conv_norm=load(p+"lconv1d.conv_norm.weight",{h});
    l.norm_pre_attn=load(p+"norm_pre_attn.weight",{h});
    l.norm_post_attn=load(p+"norm_post_attn.weight",{h});
    l.norm_out=load(p+"norm_out.weight",{h});
  }
  a.output_proj_w=load("model.audio_tower.output_proj.weight",{c.output_proj_dims,h});
  a.output_proj_b=load("model.audio_tower.output_proj.bias",{c.output_proj_dims});
  w.audio_config=std::move(c);
}

// Pinned Transformers a08ace4b modeling_gemma4.py:579-588.
// This retains patch inventory; encoder/hook wiring remains owed.
multimodal::Gemma4VisionWeights LoadVisionPatches(
    const TensorResolver& get, const nlohmann::json& cfg) {
  multimodal::Gemma4VisionWeights w;
  if (cfg.is_null()) return w;
  const auto positive = [&](const char* key) {
    const auto it = cfg.find(key);
    VT_CHECK(it != cfg.end() && it->is_number_integer() && *it > 0 && *it <= INT64_MAX,
             std::string("gemma4: vision_config ") + key + " must be a positive integer");
    return it->get<int64_t>();
  };
  const auto h = positive("hidden_size");
  const auto patch = positive("patch_size");
  const auto positions = positive("position_embedding_size");
  VT_CHECK(patch <= INT64_MAX / 3 / patch,
           "gemma4: vision_config patch_size overflow");
  const int64_t dim = 3 * patch * patch;
  VT_CHECK(h <= INT64_MAX / 2 / dim,
           "gemma4: vision_config patch_size tensor size overflow");
  VT_CHECK(positions <= INT64_MAX / 8 / h,
           "gemma4: vision_config position_embedding_size tensor size overflow");
  const std::string prefix = "model.vision_tower.patch_embedder.";
  const auto load = [&](const std::string& suffix, const std::vector<int64_t>& shape) {
    const auto name = prefix + suffix;
    VT_CHECK(get(name).shape == shape, "gemma4: " + name + " incorrect shape");
    return LoadBf16Direct(get, name);
  };
  auto input = load("input_proj.weight", {h, dim});
  w.input_proj.resize(input.bytes.size() / sizeof(uint16_t));
  std::memcpy(w.input_proj.data(), input.bytes.data(), input.bytes.size());
  auto position = load("position_embedding_table", {2, positions, h});
  w.position_embedding_table.resize(position.bytes.size() / sizeof(uint16_t));
  // Existing representation's documented host position-lookup F32 exception.
  for (size_t i = 0; i < w.position_embedding_table.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, position.bytes.data() + i * sizeof(bits), sizeof(bits));
    w.position_embedding_table[i] = vt::BF16ToF32(bits);
  }
  return w;
}

// Transformers a08ace4b modeling_gemma4.py:168-215,927-1002,1036-1038.
// Retain learned block inventory; clipping is loaded separately below.
void LoadVisionBlocks(const TensorResolver& get, const nlohmann::json& cfg,
                      multimodal::Gemma4VisionWeights& w) {
  if (cfg.is_null()) return;
  const auto positive = [&](const char* key) {
    const auto it = cfg.find(key);
    VT_CHECK(it != cfg.end() && it->is_number_integer() && *it > 0 && *it <= INT64_MAX,
             std::string("gemma4: vision_config ") + key + " must be a positive integer");
    return it->get<int64_t>();
  };
  const int64_t h = positive("hidden_size");
  const int64_t depth = positive("num_hidden_layers");
  const int64_t heads = positive("num_attention_heads");
  const int64_t kv_heads = positive("num_key_value_heads");
  const int64_t hd = positive("head_dim");
  const int64_t intermediate = positive("intermediate_size");
  VT_CHECK(heads <= INT64_MAX / hd && kv_heads <= INT64_MAX / hd,
           "gemma4: vision_config head dimension overflow");
  VT_CHECK(heads % kv_heads == 0, "gemma4: vision_config head divisibility");
  const int64_t q = heads * hd, kv = kv_heads * hd;
  using Block = multimodal::Gemma4VisionBlockWeights;
  VT_CHECK(static_cast<uint64_t>(depth) <= SIZE_MAX / sizeof(Block),
           "gemma4: vision_config num_hidden_layers overflow");
  const auto matrix_size = [](int64_t rows, int64_t cols) {
    VT_CHECK(rows <= INT64_MAX / 2 / cols &&
                 static_cast<uint64_t>(rows) <= SIZE_MAX / 2 / static_cast<uint64_t>(cols),
             "gemma4: vision_config block tensor size overflow");
  };
  matrix_size(q, h); matrix_size(kv, h); matrix_size(intermediate, h);
  VT_CHECK(hd <= INT64_MAX / 2, "gemma4: vision_config norm size overflow");
  // Validate each tensor before materialization. Do not reserve an untrusted
  // layer count: a missing first layer must fail without a huge allocation.
  for (int64_t layer = 0; layer < depth; ++layer) {
    const auto prefix = "model.vision_tower.encoder.layers." + std::to_string(layer) + ".";
    const auto load = [&](const char* suffix, std::vector<int64_t> shape) {
      const auto name = prefix + suffix;
      VT_CHECK(get(name).shape == shape, "gemma4: " + name + " incorrect shape");
      auto tensor = LoadBf16Direct(get, name);
      std::vector<uint16_t> values(tensor.bytes.size() / sizeof(uint16_t));
      std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
      return values;
    };
    Block b;
    b.q_proj = load("self_attn.q_proj.linear.weight", {q, h});
    b.k_proj = load("self_attn.k_proj.linear.weight", {kv, h});
    b.v_proj = load("self_attn.v_proj.linear.weight", {kv, h});
    b.o_proj = load("self_attn.o_proj.linear.weight", {h, q});
    b.gate_proj = load("mlp.gate_proj.linear.weight", {intermediate, h});
    b.up_proj = load("mlp.up_proj.linear.weight", {intermediate, h});
    b.down_proj = load("mlp.down_proj.linear.weight", {h, intermediate});
    b.input_ln = load("input_layernorm.weight", {h});
    b.post_attn_ln = load("post_attention_layernorm.weight", {h});
    b.pre_ff_ln = load("pre_feedforward_layernorm.weight", {h});
    b.post_ff_ln = load("post_feedforward_layernorm.weight", {h});
    b.q_norm = load("self_attn.q_norm.weight", {hd});
    b.k_norm = load("self_attn.k_norm.weight", {hd});
    w.blocks.push_back(std::move(b));
  }
}

// Source: pinned Transformers Gemma4VisionConfig and Gemma4ClippableLinear.
// Called after dimension/byte validation by the patch and block loaders.
multimodal::Gemma4VisionConfig LoadVisionConfig(const nlohmann::json& cfg,
                                               int64_t text_hidden) {
  multimodal::Gemma4VisionConfig c;
  c.hidden_size = cfg.at("hidden_size").get<int64_t>();
  c.num_heads = cfg.at("num_attention_heads").get<int64_t>();
  c.num_kv_heads = cfg.at("num_key_value_heads").get<int64_t>();
  c.depth = cfg.at("num_hidden_layers").get<int64_t>();
  c.intermediate_size = cfg.at("intermediate_size").get<int64_t>();
  c.head_dim = cfg.at("head_dim").get<int64_t>();
  c.patch_size = cfg.at("patch_size").get<int64_t>();
  c.position_embedding_size = cfg.at("position_embedding_size").get<int64_t>();
  c.text_hidden_size = text_hidden;
  const auto boolean = [&](const char* key) {
    auto it = cfg.find(key);
    if (it == cfg.end()) return false;
    VT_CHECK(it->is_boolean(), std::string("gemma4: vision_config ") + key);
    return it->get<bool>();
  };
  c.use_clipped_linears = boolean("use_clipped_linears");
  VT_CHECK(!boolean("standardize"), "gemma4: vision_config standardize unsupported");
  VT_CHECK(!boolean("attention_bias"), "gemma4: vision_config attention_bias unsupported");
  VT_CHECK(!cfg.contains("hidden_activation") || cfg["hidden_activation"] == "gelu_pytorch_tanh",
           "gemma4: vision_config hidden_activation unsupported");
  if (cfg.contains("pooling_kernel_size")) {
    const auto& k = cfg["pooling_kernel_size"];
    VT_CHECK(k.is_number_integer() && k > 0 && k <= INT64_MAX,
             "gemma4: vision_config pooling_kernel_size must be a positive integer");
    c.pooling_kernel_size = k.get<int64_t>();
  }
  VT_CHECK(c.pooling_kernel_size <= INT64_MAX / c.pooling_kernel_size,
           "gemma4: vision_config pooling_kernel_size overflow");
  const double eps = cfg.at("rms_norm_eps").get<double>();
  VT_CHECK(eps <= std::numeric_limits<float>::max() &&
               eps >= std::numeric_limits<float>::denorm_min(),
           "gemma4: vision_config rms_norm_eps float range");
  c.rms_norm_eps = static_cast<float>(eps);
  if (cfg.contains("attention_dropout")) {
    const auto& d = cfg["attention_dropout"];
    VT_CHECK(d.is_number() && std::isfinite(d.get<double>()) &&
                 d.get<double>() >= 0 && d.get<double>() < 1,
             "gemma4: vision_config attention_dropout range");
    c.attention_dropout = d.get<float>();
  }
  if (cfg.contains("rope_parameters") && !cfg["rope_parameters"].is_null()) {
    const auto& rope = cfg["rope_parameters"];
    VT_CHECK(rope.is_object() && rope.value("rope_type", nlohmann::json("default")) == "default",
             "gemma4: vision_config rope_parameters type unsupported");
    const auto theta = rope.value("rope_theta", nlohmann::json(100.0));
    VT_CHECK(theta.is_number() && std::isfinite(theta.get<double>()) && theta.get<double>() > 0,
             "gemma4: vision_config rope_parameters theta must be positive and finite");
    c.rope_theta = theta.get<double>();
  }
  return c;
}

void LoadVisionClips(const TensorResolver& get, multimodal::Gemma4VisionWeights& w) {
  using Block = multimodal::Gemma4VisionBlockWeights;
  const std::pair<const char*, multimodal::Clip Block::*> entries[] = {
      {"self_attn.q_proj", &Block::q_clip}, {"self_attn.k_proj", &Block::k_clip},
      {"self_attn.v_proj", &Block::v_clip}, {"self_attn.o_proj", &Block::o_clip},
      {"mlp.gate_proj", &Block::gate_clip}, {"mlp.up_proj", &Block::up_clip},
      {"mlp.down_proj", &Block::down_clip}};
  for (size_t l = 0; l < w.blocks.size(); ++l) {
    const auto prefix = "model.vision_tower.encoder.layers." + std::to_string(l) + ".";
    for (const auto& [projection, member] : entries) {
      const auto scalar = [&](const char* suffix) {
        const auto name = prefix + projection + "." + suffix;
        const auto& t = get(name);
        VT_CHECK(t.shape.empty() && t.dtype == "BF16",
                 "gemma4: " + name + " must be BF16 scalar []");
        uint16_t bits;
        std::memcpy(&bits, t.data, sizeof(bits));
        const float value = vt::BF16ToF32(bits);
        VT_CHECK(!std::isnan(value), "gemma4: " + name + " is NaN");
        return value;
      };
      auto& clip = w.blocks[l].*member;
      clip.in_min = scalar("input_min");
      clip.in_max = scalar("input_max");
      clip.out_min = scalar("output_min");
      clip.out_max = scalar("output_max");
      VT_CHECK(clip.in_min <= clip.in_max && clip.out_min <= clip.out_max,
               "gemma4: clip bounds order: " + prefix + projection);
    }
  }
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
  const auto tie = text.find("tie_word_embeddings");
  VT_CHECK(tie == text.end() || tie->is_null() || tie->is_boolean(),
           "gemma4: tie_word_embeddings must be boolean or null");
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
  if (!w.tie_word_embeddings) {
    // vLLM gemma4.py at a7c23ac96d: ParallelLMHead(vocab_size, hidden_size)
    // is tied only when configured. Missing weights must not trigger a tied
    // fallback in Forward; preserve the shared transpose loader for valid heads.
    const StTensor& head = get("lm_head.weight");
    VT_CHECK((head.shape == std::vector<int64_t>{config.vocab_size, H}),
             "gemma4: lm_head.weight must have shape [vocab_size, hidden_size]");
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.vision = LoadModalityProjector(get, config, "vision_config", "model.embed_vision", false);
  w.audio = LoadModalityProjector(get, config, "audio_config", "model.embed_audio", true);
  LoadAudioFrontend(get, w);
  LoadAudioBlocks(get, w, H);
  w.vision_patch_weights = LoadVisionPatches(get, w.vision.config);
  LoadVisionBlocks(get, w.vision.config, w.vision_patch_weights);
  if (!w.vision.config.is_null()) {
    w.vision_config = LoadVisionConfig(w.vision.config, H);
    if (w.vision_config->use_clipped_linears) LoadVisionClips(get, w.vision_patch_weights);
  }

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

Gemma4SpecialTokenIds::Gemma4SpecialTokenIds(const HfConfig& config) {
  const auto& raw = config.raw;
  audio_token_id = raw.value("audio_token_id", int32_t(0));
  boa_token_id = raw.value("boa_token_id", int32_t(0));
  boi_token_id = raw.value("boi_token_id", int32_t(0));
  eoa_token_id = raw.value("eoa_token_id", int32_t(0));
  eoa_token_index = raw.value("eoa_token_index", int32_t(0));
  eoi_token_id = raw.value("eoi_token_id", int32_t(0));
  image_token_id = raw.value("image_token_id", int32_t(0));
  video_token_id = raw.value("video_token_id", int32_t(0));
  const auto& text = raw.contains("text_config") && raw["text_config"].is_object()
                          ? raw["text_config"] : raw;
  bos_token_id = text.value("bos_token_id", int32_t(0));
  pad_token_id = text.value("pad_token_id", int32_t(0));
  vocab_size = text.value("vocab_size", int32_t(0));
  eos_token_ids.clear();
  if (raw.contains("eos_token_id") && raw["eos_token_id"].is_array()) {
    for (const auto& v : raw["eos_token_id"])
      eos_token_ids.push_back(v.get<int32_t>());
  }
}

std::vector<int32_t> Gemma4VideoReplacement(
    const std::vector<double>& timestamps,
    const std::vector<int32_t>& num_soft_tokens_per_frame,
    int32_t boi_token_id, int32_t eoi_token_id, int32_t video_token_id,
    const std::function<std::vector<int32_t>(std::string_view)>& encode) {
  VT_CHECK(timestamps.size() == num_soft_tokens_per_frame.size(),
           "gemma4: video repl length mismatch");
  std::vector<int32_t> all_token_ids;
  for (size_t i = 0; i < timestamps.size(); ++i) {
    const double ts = timestamps[i];
    const int32_t n_tokens = num_soft_tokens_per_frame[i];
    const int minutes = static_cast<int>(ts / 60.0);
    const int seconds = static_cast<int>(std::fmod(ts, 60.0));
    char ts_buf[8];
    std::snprintf(ts_buf, sizeof(ts_buf), "%02d:%02d", minutes, seconds);
    std::string prefix =
        (i > 0) ? (std::string(" ") + ts_buf + " ") : (std::string(ts_buf) + " ");
    auto ts_token_ids = encode(prefix);
    all_token_ids.insert(all_token_ids.end(), ts_token_ids.begin(),
                         ts_token_ids.end());
    all_token_ids.push_back(boi_token_id);
    all_token_ids.insert(all_token_ids.end(), n_tokens, video_token_id);
    all_token_ids.push_back(eoi_token_id);
  }
  return all_token_ids;
}

}  // namespace vllm

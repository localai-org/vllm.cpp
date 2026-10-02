// Cohere2MoeForCausalLM weight loader (bf16 safetensors).
//
// Mirrors Cohere2MoeForCausalLM.hf_to_vllm_mapper and load_weights
// (vllm/model_executor/models/cohere2_moe.py:477-545 @ a7c23ac96d):
//   * q/k/v stack into one qkv owner (orig_to_new_stacked :480-482);
//   * the dense and shared-expert gate/up stack into one gate_up owner
//     (:484-487), and so does each routed expert's pair, which FusedMoE keeps as
//     one w13 parameter per expert;
//   * `lm_head.*` is dropped (orig_to_new_prefix :489): the logits read the
//     tied `embed_tokens` (:537-541).
// Every other shipped name must be one this model claims, or the load is refused
// by name, the way AutoWeightsLoader raises on an unexpected weight.
#include "vllm/model_executor/models/cohere2_moe.h"

#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::MakeOwned;

std::string ShapeStr(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) out += (i ? ", " : "") + std::to_string(s[i]);
  return out + "]";
}

// Name -> shard, with the dtype and shape of each claimed tensor checked before
// any byte is read. Only BF16 is accepted: the loaders below would also
// dequantize an F8_E4M3 shard, and the FP8 sibling checkpoint is owed, not
// silently served through that side door.
class Resolver {
 public:
  explicit Resolver(const std::vector<SafetensorsFile>& shards) {
    for (const SafetensorsFile& shard : shards)
      for (const std::string& name : shard.Names()) where_[name] = &shard;
  }

  bool Has(const std::string& name) const { return where_.count(name) != 0; }

  const StTensor& Expect(const std::string& name, const std::vector<int64_t>& shape) {
    auto it = where_.find(name);
    VT_CHECK(it != where_.end(), "cohere2_moe: tensor not found: " + name);
    const StTensor& t = it->second->Get(name);
    VT_CHECK(t.dtype == "BF16", "cohere2_moe: " + name + " has dtype " + t.dtype +
                                    "; only the bf16 checkpoint is supported "
                                    "(the quantized arms are owed)");
    VT_CHECK(t.shape == shape, "cohere2_moe: " + name + " has shape " +
                                   ShapeStr(t.shape) + ", expected " + ShapeStr(shape));
    return t;
  }

  TensorResolver Get() const {
    return [this](const std::string& name) -> const StTensor& {
      auto it = where_.find(name);
      VT_CHECK(it != where_.end(), "cohere2_moe: tensor not found: " + name);
      return it->second->Get(name);
    };
  }

  // Every shipped name is claimed, dropped (lm_head.*) or refused by name.
  // Checked BEFORE any byte is copied, so an unexpected tensor costs no load.
  void RefuseUnclaimed(const std::vector<std::string>& expected) const {
    const std::set<std::string> claimed(expected.begin(), expected.end());
    for (const auto& [name, shard] : where_) {
      (void)shard;
      if (claimed.count(name) != 0) continue;
      if (name.rfind("lm_head.", 0) == 0) continue;  // cohere2_moe.py:489
      VT_CHECK(false, "cohere2_moe: checkpoint tensor " + name +
                          " is not a Cohere2MoeForCausalLM weight (the pinned "
                          "loader has no parameter for it)");
    }
  }

 private:
  std::unordered_map<std::string, const SafetensorsFile*> where_;
};

OwnedTensor Direct(Resolver& r, const std::string& name, const std::vector<int64_t>& shape) {
  r.Expect(name, shape);
  return LoadBf16Direct(r.Get(), name);
}

OwnedTensor Merged(Resolver& r, const std::vector<std::string>& names,
                   const std::vector<std::vector<int64_t>>& shapes) {
  for (size_t i = 0; i < names.size(); ++i) r.Expect(names[i], shapes[i]);
  return LoadMergedBf16RawNK(r.Get(), names);
}

Cohere2MoeMlpWeights LoadMlp(Resolver& r, const std::string& prefix, int64_t H, int64_t I) {
  Cohere2MoeMlpWeights m;
  m.gate_up_proj = Merged(r, {prefix + "gate_proj.weight", prefix + "up_proj.weight"},
                          {{I, H}, {I, H}});
  m.down_proj = Merged(r, {prefix + "down_proj.weight"}, {{H, I}});
  return m;
}

Cohere2MoeLayerWeights LoadLayer(Resolver& r, const Cohere2MoeParams& p, int64_t l) {
  const int64_t H = p.hidden_size;
  const int64_t qdim = p.num_heads * p.head_dim;
  const int64_t kdim = p.num_kv_heads * p.head_dim;
  const std::string base = "model.layers." + std::to_string(l) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  Cohere2MoeLayerWeights w;
  // The ONE norm of the parallel block (cohere2_moe.py:366-367). Weight-only in
  // both arms: neither RMSNorm (:85-94) nor LayerNorm (commandr.py) has a bias.
  w.input_layernorm = Direct(r, base + "input_layernorm.weight", {H});
  w.attn.qkv_proj = Merged(r, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"},
                           {{qdim, H}, {kdim, H}, {kdim, H}});
  w.attn.o_proj = Merged(r, {sa + "o_proj.weight"}, {{H, qdim}});

  w.dense = p.dense[static_cast<size_t>(l)];
  if (w.dense) {
    w.mlp = LoadMlp(r, mlp, H, p.prefix_dense_intermediate_size);
    return w;
  }
  w.moe.router = Merged(r, {mlp + "gate.weight"}, {{p.num_experts, H}});
  w.moe.experts.reserve(static_cast<size_t>(p.num_experts));
  for (int64_t e = 0; e < p.num_experts; ++e)
    w.moe.experts.push_back(
        LoadMlp(r, mlp + "experts." + std::to_string(e) + ".", H, p.intermediate_size));
  if (p.num_shared_experts > 0) {
    w.moe.has_shared = true;
    w.moe.shared = LoadMlp(r, mlp + "shared_experts.", H,
                           p.intermediate_size * p.num_shared_experts);
  }
  return w;
}

}  // namespace

OwnedTensor BuildCohere2MoeRopeCache(const Cohere2MoeParams& params, int64_t rows,
                                     vt::DType dtype) {
  VT_CHECK(rows > 0, "cohere2_moe: rope cache needs at least one row");
  // get_rope(head_dim, max_position, rope_parameters, is_neox_style=False)
  // (cohere2_moe.py:198-203). The CACHE is style-independent; the GPT-J pairing
  // is applied by the forward's RopeFromCache(is_neox_style=false).
  RotaryEmbedding rope(params.head_dim, params.head_dim, rows, params.rope_theta,
                       /*is_neox_style=*/false, dtype);
  const vt::Tensor cache = rope.cos_sin_cache();
  VT_CHECK(cache.rank == 2 && cache.shape[1] == params.head_dim && cache.shape[0] >= rows,
           "cohere2_moe: rope cache shape mismatch");
  OwnedTensor out = MakeOwned(dtype, {rows, params.head_dim});
  std::memcpy(out.bytes.data(), cache.data, out.bytes.size());
  return out;
}

std::vector<std::string> EnumerateCohere2MoeTensors(const Cohere2MoeParams& p) {
  std::vector<std::string> names{"model.embed_tokens.weight"};
  auto mlp = [&names](const std::string& prefix) {
    for (const char* s : {"gate_proj", "up_proj", "down_proj"})
      names.push_back(prefix + s + ".weight");
  };
  for (int64_t l = 0; l < p.num_layers; ++l) {
    const std::string base = "model.layers." + std::to_string(l) + ".";
    names.push_back(base + "input_layernorm.weight");
    for (const char* s : {"q_proj", "k_proj", "v_proj", "o_proj"})
      names.push_back(base + "self_attn." + s + ".weight");
    if (p.dense[static_cast<size_t>(l)]) {
      mlp(base + "mlp.");
      continue;
    }
    names.push_back(base + "mlp.gate.weight");
    for (int64_t e = 0; e < p.num_experts; ++e)
      mlp(base + "mlp.experts." + std::to_string(e) + ".");
    if (p.num_shared_experts > 0) mlp(base + "mlp.shared_experts.");
  }
  names.push_back("model.norm.weight");
  return names;
}

Cohere2MoeLayerWeights LoadCohere2MoeLayerWeights(const std::vector<SafetensorsFile>& shards,
                                                  const Cohere2MoeParams& params,
                                                  int64_t layer) {
  VT_CHECK(layer >= 0 && layer < params.num_layers, "cohere2_moe: layer out of range");
  Resolver r(shards);
  return LoadLayer(r, params, layer);
}

Cohere2MoeWeights LoadCohere2MoeWeights(const std::vector<SafetensorsFile>& shards,
                                        const HfConfig& config) {
  Cohere2MoeWeights w;
  w.params = ParseCohere2MoeParams(config);
  const Cohere2MoeParams& p = w.params;
  Resolver r(shards);
  r.RefuseUnclaimed(EnumerateCohere2MoeTensors(p));
  w.embed_tokens = Direct(r, "model.embed_tokens.weight", {p.vocab_size, p.hidden_size});
  w.layers.reserve(static_cast<size_t>(p.num_layers));
  for (int64_t l = 0; l < p.num_layers; ++l) w.layers.push_back(LoadLayer(r, p, l));
  w.final_norm = Direct(r, "model.norm.weight", {p.hidden_size});
  w.compute_dtype = vt::DType::kBF16;
  w.rope_cos_sin = BuildCohere2MoeRopeCache(p, p.max_position, vt::DType::kBF16);
  return w;
}

}  // namespace vllm

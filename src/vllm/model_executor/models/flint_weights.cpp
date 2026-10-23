// RecursiveSI Flint (`FlintForCausalLM`) — weight loader (MODEL-TEXT-FLINT W1).
//
// Loads the checkpoint safetensors into FlintWeights (flint.h) by the house
// bf16 direct/transposed helpers. Name map (RecursiveSI `config.json` +
// transformers `models/flint/` layout; the llama.cpp PR #30928 converter maps
// the SAME names into GGUF, `conversion/flint.py`):
//   model.embed_tokens.weight                          -> embed_tokens [V,H]
//   model.norm.weight                                  -> final_norm [H]
//   lm_head.weight                                     -> lm_head (untied)
//   model.layers.N.input_layernorm.weight              -> input_layernorm
//   model.layers.N.post_attention_layernorm.weight     -> post_attention_ln
//   model.layers.N.self_attn.{q,k,v,o}_proj.weight     -> attn projections
//   model.layers.N.self_attn.{q,k}_norm.weight         -> per-head q/k RMS
//   model.layers.N.self_attn.softmax_offset.weight     -> per-head sinks [hq]
//   model.layers.N.mlp.gate.weight                     -> router_gate
//   model.layers.N.mlp.experts.E.{gate,up,down}_proj   -> routed experts
//   model.layers.N.mlp.shared_expert.{gate,up,down}    -> shared expert
//   model.layers.N.mlp.{gate,up,down}_proj             -> dense layers 0-1
#include "vllm/model_executor/models/flint.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/dtype.h"

namespace vllm {

namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;

FlintLinear LoadLinearT(const TensorResolver& get, const std::string& name) {
  FlintLinear l;
  l.w = LoadBf16Transposed(get, name);  // [N,K] -> [K,N] Matmul-B
  return l;
}

FlintAttnWeights LoadAttn(const TensorResolver& get, int64_t layer) {
  const std::string sa =
      "model.layers." + std::to_string(layer) + ".self_attn.";
  FlintAttnWeights a;
  a.q_proj = LoadLinearT(get, sa + "q_proj.weight");
  a.k_proj = LoadLinearT(get, sa + "k_proj.weight");
  a.v_proj = LoadLinearT(get, sa + "v_proj.weight");
  a.o_proj = LoadLinearT(get, sa + "o_proj.weight");
  a.q_norm = LoadBf16Direct(get, sa + "q_norm.weight");
  a.k_norm = LoadBf16Direct(get, sa + "k_norm.weight");
  // The per-head sinks. transformers stores `softmax_offset` without a
  // `.weight` suffix; the converter appends one (`conversion/flint.py`
  // modify_tensors). Accept the checkpoint's own spelling.
  a.sinks = LoadBf16Direct(get, sa + "softmax_offset.weight");
  return a;
}

FlintMlpWeights LoadMlp(const TensorResolver& get, const std::string& base) {
  FlintMlpWeights m;
  m.gate_proj = LoadLinearT(get, base + "gate_proj.weight");
  m.up_proj = LoadLinearT(get, base + "up_proj.weight");
  m.down_proj = LoadLinearT(get, base + "down_proj.weight");
  return m;
}

FlintMoeWeights LoadMoe(const TensorResolver& get, const std::string& mlp,
                        int64_t num_experts) {
  FlintMoeWeights m;
  m.router_gate = LoadBf16Transposed(get, mlp + "gate.weight");  // [E,H]->[H,E]
  m.experts.reserve(static_cast<size_t>(num_experts));
  for (int64_t e = 0; e < num_experts; ++e) {
    const std::string ex = mlp + "experts." + std::to_string(e) + ".";
    FlintExpertWeights w;
    w.gate_proj = LoadLinearT(get, ex + "gate_proj.weight");
    w.up_proj = LoadLinearT(get, ex + "up_proj.weight");
    w.down_proj = LoadLinearT(get, ex + "down_proj.weight");
    m.experts.push_back(std::move(w));
  }
  m.shared_expert = LoadMlp(get, mlp + "shared_expert.");
  return m;
}

}  // namespace

std::vector<std::string> EnumerateFlintTensors(const FlintParams& p) {
  std::vector<std::string> names;
  names.push_back("model.embed_tokens.weight");
  names.push_back("model.norm.weight");
  names.push_back("lm_head.weight");
  for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
    const std::string base = "model.layers." + std::to_string(i) + ".";
    const std::string sa = base + "self_attn.";
    names.push_back(base + "input_layernorm.weight");
    names.push_back(base + "post_attention_layernorm.weight");
    for (const char* t : {"q_proj", "k_proj", "v_proj", "o_proj"})
      names.push_back(sa + t + ".weight");
    names.push_back(sa + "q_norm.weight");
    names.push_back(sa + "k_norm.weight");
    names.push_back(sa + "softmax_offset.weight");
    if (p.IsDenseLayer(i)) {
      for (const char* t : {"gate_proj", "up_proj", "down_proj"})
        names.push_back(base + "mlp." + t + ".weight");
    } else {
      names.push_back(base + "mlp.gate.weight");
      for (int64_t e = 0; e < p.num_experts; ++e) {
        const std::string ex =
            base + "mlp.experts." + std::to_string(e) + ".";
        for (const char* t : {"gate_proj", "up_proj", "down_proj"})
          names.push_back(ex + t + ".weight");
      }
      for (const char* t : {"gate_proj", "up_proj", "down_proj"})
        names.push_back(base + "mlp.shared_expert." + t + ".weight");
    }
  }
  return names;
}

FlintAccounting AccountFlintTensors(const FlintParams& p,
                                    const std::vector<std::string>& present) {
  const std::vector<std::string> expected = EnumerateFlintTensors(p);
  FlintAccounting acc;
  std::multiset<std::string> seen(present.begin(), present.end());
  std::set<std::string> counted;
  for (const auto& name : expected) {
    const auto it = seen.find(name);
    if (it == seen.end()) {
      acc.missing.push_back(name);
    } else {
      seen.erase(it);
      counted.insert(name);
    }
  }
  for (const auto& name : present) {
    if (counted.count(name) && seen.count(name))
      acc.duplicated.push_back(name);  // a second copy of an expected name
  }
  for (const auto& leftover : seen)
    if (!counted.count(leftover)) acc.unaccounted.push_back(leftover);
  return acc;
}

FlintWeights LoadFlintWeights(const std::vector<SafetensorsFile>& shards,
                              const HfConfig& config) {
  const FlintParams p = ParseFlintParams(config);

  // Full accounting FIRST: every expected tensor present, none duplicated,
  // none unaccounted.
  std::vector<std::string> present;
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      present.push_back(name);
      where[name] = &shard;
    }
  }
  const FlintAccounting acc = AccountFlintTensors(p, present);
  auto fail = [](const char* kind, const std::vector<std::string>& v) {
    if (v.empty()) return;
    std::string msg = std::string("flint: ") + kind + " tensors: ";
    for (size_t i = 0; i < v.size() && i < 8; ++i)
      msg += (i ? ", " : "") + v[i];
    throw std::runtime_error(msg);
  };
  fail("missing", acc.missing);
  fail("duplicated", acc.duplicated);
  fail("unaccounted", acc.unaccounted);

  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "flint: tensor not found: " + name);
    return it->second->Get(name);
  };

  FlintWeights w;
  w.params = p;
  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  w.lm_head = LoadBf16Transposed(get, "lm_head.weight");  // untied
  w.layers.reserve(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
    const std::string base = "model.layers." + std::to_string(i) + ".";
    const std::string mlp = base + "mlp.";
    FlintLayerWeights l;
    l.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
    l.post_attention_layernorm =
        LoadBf16Direct(get, base + "post_attention_layernorm.weight");
    l.attn = LoadAttn(get, i);
    if (p.IsDenseLayer(i)) {
      l.dense_mlp = LoadMlp(get, mlp);
    } else {
      l.moe = LoadMoe(get, mlp, p.num_experts);
    }
    w.layers.push_back(std::move(l));
  }
  return w;
}

}  // namespace vllm

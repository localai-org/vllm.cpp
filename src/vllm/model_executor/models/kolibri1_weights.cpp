// Kolibri-1 — weight loader (MODEL-TEXT-kolibri-1 W1).
//
// Loads every tensor from the safetensors shards, performs full accounting
// (every on-disk name must be classified), then materializes weights into
// the Kolibri1Weights tree. The checkpoint is FP8 block-quantized
// (weight_block_size [128, 128], weight_scale_inv siblings, spec risk R1);
// the router gate is bf16 (modules_to_not_convert). The MoE name layout —
// experts at `mlp.experts.N`, the shared expert at `mlp.shared_experts`,
// the router bias at `moe.router.expert_bias`, the gate at `mlp.gate` — IS
// the mapper reconciliation (kolibri1.py:258-266).

#include "vllm/model_executor/models/kolibri1_weights.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "vllm/model_executor/models/dense_weight_loaders.h"

namespace vllm {

namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadFp8BlockRaw;
using dense_loaders::MakeOwned;

constexpr int64_t kBlockN = 128;
constexpr int64_t kBlockK = 128;

// ── Tensor index helper (the mimo_v2_weights.cpp pattern) ───────────────────

struct TensorIndex {
  std::unordered_map<std::string, const StTensor*> by_name;
  std::vector<std::string> duplicates;
};

TensorIndex BuildTensorIndex(const std::vector<SafetensorsFile>& shards) {
  TensorIndex idx;
  for (const auto& shard : shards) {
    for (const std::string& name : shard.Names()) {
      auto [it, inserted] = idx.by_name.try_emplace(name, &shard.Get(name));
      if (!inserted) idx.duplicates.push_back(name);
    }
  }
  return idx;
}

struct GetTensor {
  const TensorIndex& idx;
  const StTensor& operator()(const std::string& name) const {
    auto it = idx.by_name.find(name);
    if (it == idx.by_name.end())
      throw std::runtime_error("kolibri1: missing tensor: " + name);
    return *it->second;
  }
};

// ── Quantization config ─────────────────────────────────────────────────────

// The checkpoint's quantization_config must be fp8 block [128, 128] dynamic.
// Anything else is refused by name: no other arm is implemented for the CPU
// path and a silent fall-through would load plausible wrong weights.
void CheckQuantConfig(const nlohmann::json& raw) {
  const auto qit = raw.find("quantization_config");
  VT_CHECK(qit != raw.end() && qit->is_object(),
           "kolibri1: the checkpoint carries no quantization_config, but the "
           "Kolibri-1 checkpoint is FP8 block-quantized (weight_block_size "
           "[128, 128]); a bf16 Kolibri-1 artifact is not an implemented arm");
  const auto& q = *qit;
  VT_CHECK(q.value("quant_method", std::string()) == "fp8",
           "kolibri1: quant_method '" +
               q.value("quant_method", std::string()) +
               "' is not the fp8 this row implements");
  VT_CHECK(q.value("activation_scheme", std::string()) == "dynamic",
           "kolibri1: activation_scheme '" +
               q.value("activation_scheme", std::string()) +
               "' is not the dynamic scheme the Kolibri-1 checkpoint carries");
  const auto bit = q.find("weight_block_size");
  VT_CHECK(bit != q.end() && bit->is_array() && bit->size() == 2,
           "kolibri1: weight_block_size must be a 2-element array, the "
           "[block_n, block_k] grid");
  VT_CHECK((*bit)[0].get<int64_t>() == kBlockN &&
               (*bit)[1].get<int64_t>() == kBlockK,
           "kolibri1: weight_block_size [" +
               std::to_string((*bit)[0].get<int64_t>()) + ", " +
               std::to_string((*bit)[1].get<int64_t>()) +
               "] is not the [128, 128] grid the Kolibri-1 checkpoint "
               "quantizes with");
}

// ── Projection loaders ──────────────────────────────────────────────────────

Kolibri1Projection LoadProjection(const TensorIndex& idx,
                                  const std::string& proj) {
  Kolibri1Projection r;
  GetTensor get{idx};
  const StTensor& w = get(proj + ".weight");
  if (w.dtype == "F8_E4M3") {
    r.fp8_block = LoadFp8BlockRaw(get, proj, kBlockN, kBlockK);
  } else {
    VT_CHECK(w.dtype == "BF16",
             "kolibri1: '" + proj + ".weight' ships dtype " + w.dtype +
                 "; the implemented arms are F8_E4M3 block-quantized and "
                 "BF16 (modules_to_not_convert)");
    r.bf16 = LoadBf16Direct(get, proj + ".weight");
  }
  return r;
}

Kolibri1ExpertWeights LoadExpert(const TensorIndex& idx,
                                 const std::string& base) {
  Kolibri1ExpertWeights e;
  e.gate_proj = LoadProjection(idx, base + ".gate_proj");
  e.up_proj = LoadProjection(idx, base + ".up_proj");
  e.down_proj = LoadProjection(idx, base + ".down_proj");
  return e;
}

}  // namespace

// ── Enumeration ─────────────────────────────────────────────────────────────

namespace {

void AppendLinear(std::vector<Kolibri1Tensor>& out, const std::string& proj,
                  const char* consumer) {
  // fp8-block arm: the weight AND its scale grid are both expected. A bf16
  // (modules_to_not_convert) linear ships only the weight; the router gate
  // is the one such module and is appended directly by its caller.
  out.push_back({proj + ".weight", consumer});
  out.push_back({proj + ".weight_scale_inv", consumer});
}

}  // namespace

std::vector<Kolibri1Tensor> EnumerateKolibri1Tensors(
    const Kolibri1Params& p) {
  std::vector<Kolibri1Tensor> out;
  out.reserve(static_cast<size_t>(2 + p.num_hidden_layers * (4 + 2 + 2 +
                                                            2 + 3 +
                                                            6 * p.num_experts +
                                                            6)));

  out.push_back({"model.embed_tokens.weight", "embedding"});
  out.push_back({"lm_head.weight", "lm_head"});
  out.push_back({"model.norm.weight", "final_norm"});

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string base =
        "model.layers." + std::to_string(l) + ".";
    const char* layer = base.c_str();
    out.push_back({base + "input_layernorm.weight", layer});
    out.push_back({base + "post_attn_norm.weight", layer});
    out.push_back({base + "post_attention_layernorm.weight", layer});
    out.push_back({base + "post_ffn_norm.weight", layer});
    out.push_back({base + "self_attn.q_norm.weight", layer});
    out.push_back({base + "self_attn.k_norm.weight", layer});
    AppendLinear(out, base + "self_attn.q_proj", layer);
    AppendLinear(out, base + "self_attn.k_proj", layer);
    AppendLinear(out, base + "self_attn.v_proj", layer);
    AppendLinear(out, base + "self_attn.o_proj", layer);
    // The router gate is bf16 (modules_to_not_convert): weight only. The
    // router bias lives at layers.N.moe.router.expert_bias — NOTE the REAL
    // checkpoint's layout: the bias sits under `moe.router.`, NOT under
    // `mlp.moe.router.` as the spec's inventory and the mapper substring
    // (".moe.router.expert_bias") both suggest; the mapper's substring match
    // renames either spelling. Captured against the real index in the W1
    // manifest test.
    out.push_back({base + "mlp.gate.weight", layer});
    out.push_back({base + "moe.router.expert_bias", layer});
    for (int64_t e = 0; e < p.num_experts; ++e) {
      // NOTE the REAL checkpoint's layout, captured in the W1 manifest: the
      // routed experts sit at mlp.experts.N and the shared expert at
      // mlp.shared_experts — NOT under mlp.moe.* as the spec's inventory
      // recorded. Only the router bias lives outside mlp (moe.router.).
      const std::string expert = base + "mlp.experts." +
                                 std::to_string(e);
      AppendLinear(out, expert + ".gate_proj", layer);
      AppendLinear(out, expert + ".up_proj", layer);
      AppendLinear(out, expert + ".down_proj", layer);
    }
    AppendLinear(out, base + "mlp.shared_experts.gate_proj", layer);
    AppendLinear(out, base + "mlp.shared_experts.up_proj", layer);
    AppendLinear(out, base + "mlp.shared_experts.down_proj", layer);
  }
  return out;
}

Kolibri1Accounting AccountKolibri1Tensors(
    const Kolibri1Params& p, const std::vector<std::string>& present) {
  const auto expected = EnumerateKolibri1Tensors(p);
  std::unordered_set<std::string> expected_set;
  expected_set.reserve(expected.size());
  for (const auto& t : expected) expected_set.insert(t.name);

  Kolibri1Accounting acc;
  std::unordered_set<std::string> seen;
  for (const auto& name : present) {
    if (expected_set.count(name) == 0) {
      acc.unaccounted.push_back(name);
    } else if (!seen.insert(name).second) {
      acc.duplicated.push_back(name);
    }
  }
  for (const auto& t : expected) {
    if (seen.count(t.name) == 0) acc.missing.push_back(t.name);
  }
  return acc;
}

// ── The load ────────────────────────────────────────────────────────────────

Kolibri1Weights LoadKolibri1Weights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  const Kolibri1Params p = ParseKolibri1Params(config);
  CheckQuantConfig(config.raw);

  const TensorIndex idx = BuildTensorIndex(shards);
  VT_CHECK(idx.duplicates.empty(),
           "kolibri1: duplicated tensor(s) across shards, first: " +
               idx.duplicates.front());

  const Kolibri1Accounting acc = AccountKolibri1Tensors(
      p, [&idx] {
        std::vector<std::string> names;
        names.reserve(idx.by_name.size());
        for (const auto& [name, _] : idx.by_name) names.push_back(name);
        return names;
      }());
  VT_CHECK(acc.missing.empty(),
           "kolibri1: missing tensor(s), first: " +
               (acc.missing.empty() ? std::string("(none)")
                                    : acc.missing.front()));
  VT_CHECK(acc.unaccounted.empty(),
           "kolibri1: unaccounted tensor(s), first: " +
               (acc.unaccounted.empty() ? std::string("(none)")
                                        : acc.unaccounted.front()));

  Kolibri1Weights w;
  w.params = p;
  GetTensor get{idx};

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.lm_head = LoadBf16Direct(get, "lm_head.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");

  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string base =
        "model.layers." + std::to_string(l) + ".";
    Kolibri1LayerWeights& layer = w.layers[static_cast<size_t>(l)];
    layer.is_sliding = p.IsSlidingLayer(l);
    layer.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
    layer.post_attn_norm = LoadBf16Direct(get, base + "post_attn_norm.weight");
    layer.post_attention_layernorm =
        LoadBf16Direct(get, base + "post_attention_layernorm.weight");
    layer.post_ffn_norm = LoadBf16Direct(get, base + "post_ffn_norm.weight");

    layer.attn.q_proj = LoadProjection(idx, base + "self_attn.q_proj");
    layer.attn.k_proj = LoadProjection(idx, base + "self_attn.k_proj");
    layer.attn.v_proj = LoadProjection(idx, base + "self_attn.v_proj");
    layer.attn.o_proj = LoadProjection(idx, base + "self_attn.o_proj");
    layer.attn.q_norm = LoadBf16Direct(get, base + "self_attn.q_norm.weight");
    layer.attn.k_norm = LoadBf16Direct(get, base + "self_attn.k_norm.weight");

    // The router gate: bf16 ONLY. An fp8 router would mean the checkpoint
    // quantized a module its own config excludes.
    const StTensor& gate = get(base + "mlp.gate.weight");
    VT_CHECK(gate.dtype == "BF16",
             "kolibri1: router '" + base +
                 "mlp.gate.weight' ships dtype " + gate.dtype +
                 ", not the BF16 a modules_to_not_convert router is");
    layer.moe.router_gate = LoadBf16Direct(get, base + "mlp.gate.weight");

    const StTensor& bias = get(base + "moe.router.expert_bias");
    VT_CHECK(
        (bias.dtype == "BF16" || bias.dtype == "F32") &&
            bias.shape.size() == 1 && bias.shape[0] == p.num_experts,
        "kolibri1: router bias '" + base +
            "moe.router.expert_bias' must be BF16 or F32 [num_experts]");
    // Upstream allocates e_score_correction_bias fp32 and loads with a
    // converting copy; the checkpoint ships BF16, widened losslessly here.
    layer.moe.e_score_correction_bias =
        MakeOwned(vt::DType::kF32, bias.shape);
    auto* dst = reinterpret_cast<float*>(
        layer.moe.e_score_correction_bias.bytes.data());
    const int64_t count = p.num_experts;
    if (bias.dtype == "BF16") {
      for (int64_t i = 0; i < count; ++i)
        dst[i] = vt::BF16ToF32(
            vt::LoadUnaligned<uint16_t>(bias.data + i * 2));
    } else {
      std::memcpy(dst, bias.data, bias.nbytes);
    }

    layer.moe.experts.resize(static_cast<size_t>(p.num_experts));
    for (int64_t e = 0; e < p.num_experts; ++e) {
      layer.moe.experts[static_cast<size_t>(e)] = LoadExpert(
          idx, base + "mlp.experts." + std::to_string(e));
    }
    layer.moe.shared_experts =
        LoadExpert(idx, base + "mlp.shared_experts");
  }
  return w;
}

}  // namespace vllm

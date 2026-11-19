// vllm.cpp ORIGINAL — the `maxwell_moe` config.json builder TU. See the header
// and docs/bench-evidence/new-candidates-parsers-20261119/maxwell-fields.txt.
#include "vllm/model_executor/models/maxwell_config.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace vllm {
namespace {

namespace fs = std::filesystem;

int64_t ReqJsonInt(const nlohmann::json& raw, const std::string& key,
                   const std::string& source) {
  const auto it = raw.find(key);
  if (it == raw.end() || !it->is_number_integer()) {
    throw std::runtime_error("maxwell config: " + source +
                             " carries no integer field " + key);
  }
  return it->get<int64_t>();
}

bool IsShape(const StTensor& t, const std::string& dtype, int64_t d0,
             int64_t d1) {
  if (!dtype.empty() && t.dtype != dtype) return false;
  if (t.shape.size() != 2) return false;
  return t.shape[0] == d0 && t.shape[1] == d1;
}

}  // namespace

bool IsMaxwellMoeConfig(const HfConfig& config) {
  if (config.model_type == "maxwell_moe") return true;
  for (const std::string& a : config.architectures) {
    if (a == "MaxwellMoeForCausalLM") return true;
  }
  return false;
}

std::string MaxwellForwardRefusal() {
  return std::string(
      "maxwell_moe: the config.json parsed and the shard headers validated "
      "(MaxwellConfigFromJson); the FORWARD is not implemented in this build "
      "and is owed to row NEW-CANDIDATE-PORTS (block-sparse GQA + indexer "
      "arch port). Remote code is never executed by this build. Refused at "
      "the forward boundary — nothing ran.");
}

HfConfig MaxwellConfigFromJson(const HfConfig& config,
                               const std::string& model_dir) {
  if (!IsMaxwellMoeConfig(config)) {
    throw std::runtime_error(
        "maxwell config: model_type must be 'maxwell_moe' (got '" +
        config.model_type + "')");
  }
  HfConfig c = config;
  const nlohmann::json& raw = c.raw;

  // --- standard fields the generic parse already read; consistency here ---
  if (c.hidden_size <= 0 || c.num_hidden_layers <= 0 || c.vocab_size <= 0 ||
      c.num_attention_heads <= 0 || c.num_experts <= 0) {
    throw std::runtime_error(
        "maxwell config: hidden_size/num_hidden_layers/vocab_size/"
        "num_attention_heads/num_experts must all be positive explicit "
        "fields of config.json (remote-code defaults are never applied)");
  }
  const int64_t layers = c.num_hidden_layers;
  const int64_t hidden = c.hidden_size;
  const int64_t experts = c.num_experts;
  const int64_t moe_inter =
      ReqJsonInt(raw, "moe_intermediate_size", "config.json");
  c.head_dim = ReqJsonInt(raw, "attention_head_size", "config.json");
  // GEOMETRY NOTE, not fatal: heads * head_dim (32 * 128 = 4096) is TWICE
  // hidden_size on the pinned artifact, and q_proj.weight [4096, 2048]
  // (the committed walk) confirms the file means it — a hyper-connection /
  // residual-stream expansion the port row owns. Recorded, never guessed.
  c.raw["geometry_notes"] =
      "num_attention_heads * attention_head_size != hidden_size "
      "(q_proj [4096, 2048] confirms; port-row semantics)";
  // mlp_only_layers names the DENSE layers; the rest are MoE.
  std::vector<int64_t> mlp_only;
  if (const auto it = raw.find("mlp_only_layers"); it != raw.end()) {
    for (const auto& e : *it) mlp_only.push_back(e.get<int64_t>());
  }

  // --- shard headers: header reads ONLY, no weight bytes ------------------
  std::vector<fs::path> shards;
  for (const fs::directory_entry& e : fs::directory_iterator(model_dir)) {
    if (e.path().extension() == ".safetensors") shards.push_back(e.path());
  }
  std::sort(shards.begin(), shards.end());
  if (shards.empty()) {
    throw std::runtime_error("maxwell config: no .safetensors shard under " +
                             model_dir);
  }
  c.raw["maxwell_shard_count"] = shards.size();

  // Pass 1: every shard's header read once, kept OPEN — the expert tensors
  // are spread across the shards, so geometry lookups need a per-name handle,
  // not just shard 0. Header reads only; no weight bytes move.
  int64_t tensors_seen = 0;
  std::vector<std::string> names;
  std::vector<SafetensorsFile> files;
  files.reserve(shards.size());
  for (const fs::path& p : shards) {
    files.push_back(SafetensorsFile::Open(p.string()));
    tensors_seen += static_cast<int64_t>(files.back().Names().size());
    names.insert(names.end(), files.back().Names().begin(),
                 files.back().Names().end());
  }
  auto Find = [&](const std::string& name) -> const StTensor* {
    for (const SafetensorsFile& f : files) {
      if (std::find(f.Names().begin(), f.Names().end(), name) !=
          f.Names().end()) {
        return &f.Get(name);
      }
    }
    return nullptr;
  };
  const bool has_final_norm =
      std::find(names.begin(), names.end(), "model.norm.weight") != names.end();
  const bool has_lm_head =
      std::find(names.begin(), names.end(), "lm_head.weight") != names.end();
  c.raw["maxwell_has_final_norm"] = has_final_norm;
  c.raw["maxwell_has_lm_head"] = has_lm_head;

  // Pass 2: geometry against the union of headers.
  {
    const StTensor* embed = Find("model.embed_tokens.weight");
    if (embed == nullptr ||
        !IsShape(*embed, "BF16", c.vocab_size, hidden)) {
      throw std::runtime_error(
          "maxwell config: model.embed_tokens.weight disagrees with "
          "vocab_size/hidden_size");
    }
    for (int64_t l = 0; l < layers; ++l) {
      const std::string pre = "model.layers." + std::to_string(l) + ".";
      // Every layer, dense or MoE, carries these two.
      for (const char* suffix :
           {"input_layernorm.weight", "post_attention_layernorm.weight"}) {
        if (Find(pre + suffix) == nullptr) {
          throw std::runtime_error("maxwell config: layer " +
                                   std::to_string(l) + " has no " + suffix);
        }
      }
      const bool dense =
          std::find(mlp_only.begin(), mlp_only.end(), l) != mlp_only.end();
      if (dense) {
        // Dense layers keep the MoE container's NAME (`block_sparse_moe.w1`)
        // but drop the `experts.N` level — the committed walk shows this for
        // layers 0..2, which is what mlp_only_layers declares.
        const StTensor* w1 = Find(pre + "block_sparse_moe.w1.weight");
        if (w1 == nullptr || !IsShape(*w1, "", c.intermediate_size, hidden)) {
          throw std::runtime_error("maxwell config: dense layer " +
                                   std::to_string(l) +
                                   " block_sparse_moe.w1 disagrees with "
                                   "intermediate_size");
        }
        continue;
      }
      const StTensor* w1 = Find(pre + "block_sparse_moe.experts.0.w1.weight");
      if (w1 == nullptr || !IsShape(*w1, "", moe_inter, hidden)) {
        throw std::runtime_error("maxwell config: MoE layer " +
                                 std::to_string(l) +
                                 " expert w1 disagrees with "
                                 "moe_intermediate_size");
      }
      // Expert tensors are numbered 0..num_experts-1; check BOTH ends so a
      // re-indexed checkpoint cannot satisfy a shape-only probe.
      const std::string last =
          pre + "block_sparse_moe.experts." + std::to_string(experts - 1) +
          ".w1.weight";
      if (Find(last) == nullptr) {
        throw std::runtime_error(
            "maxwell config: layer " + std::to_string(l) + " has no expert " +
            std::to_string(experts - 1) + " tensor (" + last + ")");
      }
    }
  }
  c.raw["maxwell_header_tensors_seen"] = tensors_seen;
  // Recorded, not asserted: `lm_head.weight` presence decides the tie the way
  // the kolibri1 arm reads it off the FILE. The pinned artifact declares
  // tie_word_embeddings=false; if the file disagrees, that is the record.
  c.raw["maxwell_has_final_norm"] = has_final_norm;
  c.raw["maxwell_has_lm_head"] = has_lm_head;

  // --- UNKNOWN: what the remote MaxwellMoeConfig would have defaulted -----
  // Recorded, never guessed. Each entry names the field and why it is
  // unknown without executing configuration_maxwell_moe.MaxwellMoeConfig.
  c.raw["maxwell_unknown"] = nlohmann::json::array({
      "attention_bias — absent from config.json; remote default UNKNOWN",
      "mlp bias conventions — remote default UNKNOWN",
      "router top-k normalization semantics — remote default UNKNOWN",
      "rope_scaling/rope_parameters — config.json carries rope_theta only; "
      "the remote class's scaling defaults are UNKNOWN",
      "mhc (hyper-connection) defaults not expressed in config.json — "
      "UNKNOWN",
  });
  return c;
}

}  // namespace vllm

// A tiny Cohere2MoeForCausalLM checkpoint written to a temporary safetensors
// file, with the LCG weights of scripts/cohere2-moe-ref.py reproduced bit-exactly
// (same integer recurrence, same double mapping, one f32 rounding, then bf16
// round-to-nearest-even). Shared by the forward gate and the runner gate so both
// ask their question of the same bytes the reference ran on.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/cohere2_moe.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace c2m_tiny {

inline std::vector<float> Lcg(uint32_t seed, size_t n, double scale) {
  std::vector<float> out(n);
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    out[i] = static_cast<float>(((static_cast<double>(s >> 8) / 16777216.0) * 2.0 - 1.0) * scale);
  }
  return out;
}

struct Tensor {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype = "BF16";
  std::string bytes;
};

// The shape of a shipped tensor, from its name (the same table the reference's
// tensor_specs builds).
inline std::vector<int64_t> ShapeOf(const std::string& n, const vllm::Cohere2MoeParams& p) {
  const int64_t q = p.num_heads * p.head_dim, kv = p.num_kv_heads * p.head_dim;
  auto ends = [&n](const std::string& s) {
    return n.size() >= s.size() && n.compare(n.size() - s.size(), s.size(), s) == 0;
  };
  if (n == "model.embed_tokens.weight") return {p.vocab_size, p.hidden_size};
  if (ends("layernorm.weight") || n == "model.norm.weight") return {p.hidden_size};
  if (ends("q_proj.weight")) return {q, p.hidden_size};
  if (ends("k_proj.weight") || ends("v_proj.weight")) return {kv, p.hidden_size};
  if (ends("o_proj.weight")) return {p.hidden_size, q};
  if (ends("mlp.gate.weight")) return {p.num_experts, p.hidden_size};
  int64_t inter = p.prefix_dense_intermediate_size;
  if (n.find(".experts.") != std::string::npos) inter = p.intermediate_size;
  if (n.find(".shared_experts.") != std::string::npos)
    inter = p.intermediate_size * p.num_shared_experts;
  if (ends("down_proj.weight")) return {p.hidden_size, inter};
  return {inter, p.hidden_size};  // gate_proj / up_proj
}

inline std::vector<Tensor> SynthTensors(const vllm::Cohere2MoeParams& p) {
  std::vector<Tensor> out;
  const std::vector<std::string> names = vllm::EnumerateCohere2MoeTensors(p);
  for (size_t i = 0; i < names.size(); ++i) {
    const std::string& n = names[i];
    Tensor t{n, ShapeOf(n, p), "BF16", {}};
    size_t numel = 1;
    for (int64_t d : t.shape) numel *= static_cast<size_t>(d);
    const uint32_t seed = static_cast<uint32_t>(1000 + 7919 * i);
    const bool norm = t.shape.size() == 1;
    const bool embed = n == "model.embed_tokens.weight";
    const bool router = n.size() >= 16 && n.compare(n.size() - 16, 16, ".mlp.gate.weight") == 0;
    const double in = static_cast<double>(t.shape.back());
    std::vector<float> v = Lcg(seed, numel,
                               norm ? 0.25 : embed ? 1.0 : router ? 3.0 / std::sqrt(in)
                                                                  : 1.5 / std::sqrt(in));
    if (norm)
      for (float& x : v) x = 1.0F + x;
    t.bytes.resize(numel * 2);
    auto* dst = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (size_t k = 0; k < numel; ++k) dst[k] = vt::F32ToBF16(v[k]);
    out.push_back(std::move(t));
  }
  return out;
}

inline std::string BuildSt(const std::vector<Tensor>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Tensor& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  std::string len(8, '\0');
  for (int i = 0; i < 8; ++i) len[i] = static_cast<char>((header.size() >> (8 * i)) & 0xff);
  return len + header + data;
}

class TempFile {
 public:
  TempFile(const std::string& bytes, const std::string& ext) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("cohere2_moe_" + std::to_string(::getpid()) + "_" + std::to_string(counter++) + ext))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

inline vllm::HfConfig ConfigFromJson(const char* json) {
  return vllm::ParseHfConfig(nlohmann::json::parse(json), "cohere2_moe synthetic config");
}

}  // namespace c2m_tiny

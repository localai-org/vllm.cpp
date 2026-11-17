// Current production per-layer loader, real pinned checkpoint, host-only.
#include <filesystem>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vt/unaligned.h"

namespace {
void CheckHalf(const vllm::OwnedTensor& loaded, const vllm::StTensor& source) {
  VT_CHECK(loaded.dtype == vt::DType::kF16 && loaded.bytes.size() / 2 == source.nbytes / (source.dtype == "F32" ? 4 : 2),
           "qwen parameter check: FP16 size/dtype mismatch");
  for (size_t i = 0; i < loaded.bytes.size() / 2; ++i) {
    const uint16_t expected = source.dtype == "F16"
        ? vt::LoadUnaligned<uint16_t>(source.data + i * 2)
        : vt::F32ToF16(source.dtype == "BF16"
            ? vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(source.data + i * 2))
            : vt::LoadUnaligned<float>(source.data + i * 4));
    VT_CHECK(vt::LoadUnaligned<uint16_t>(loaded.bytes.data() + i * 2) == expected,
             "qwen parameter check: FP16 source conversion mismatch");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    VT_CHECK(argc == 3, "usage: qwen_exl3_parameter_check MODEL_DIR NEW_REPORT");
    const std::filesystem::path model(argv[1]), report_path(argv[2]);
    VT_CHECK(!std::filesystem::exists(report_path), "qwen parameter check: report exists");
    std::ifstream config_file(model / "config.json");
    const auto raw = nlohmann::json::parse(config_file);
    const auto& text = raw.at("text_config");
    const auto types = text.at("layer_types").get<std::vector<std::string>>();
    VT_CHECK(types.size() == 64 && text.at("mamba_ssm_dtype") == "float32" &&
                 raw.at("quantization_config").at("codebook") == "mul1",
             "qwen parameter check: requires pinned target geometry/policy");
    const auto weight_map = vllm::LoadSafetensorsIndex((model / "model.safetensors.index.json").string());
    std::set<std::string> file_names;
    for (const auto& [name, file] : weight_map) { (void)name; file_names.insert(file); }
    std::vector<vllm::SafetensorsFile> shards;
    for (const auto& file : file_names) shards.push_back(vllm::SafetensorsFile::Open((model / file).string()));
    std::map<std::string, const vllm::StTensor*> index;
    std::vector<std::string> names;
    for (const auto& shard : shards) {
      for (const auto& name : shard.Names()) {
        VT_CHECK(index.emplace(name, &shard.Get(name)).second, "qwen parameter check: duplicate tensor " + name);
        names.push_back(name);
      }
    }
    const auto get = [&index](const std::string& name) -> const vllm::StTensor& { return *index.at(name); };
    const auto has = [&index](const std::string& name) { return index.count(name) != 0; };
    const auto backbone = vllm::ResolveQwen3_5BackbonePrefix(names);
    nlohmann::json records = nlohmann::json::array();
    int gdn_layers = 0, attention_layers = 0;
    for (size_t i = 0; i < types.size(); ++i) {
      const auto layer = vllm::LoadQwen3_5DenseLayer(get, has, types[i], static_cast<int64_t>(i), backbone);
      const std::string base = backbone + "layers." + std::to_string(i) + ".";
      CheckHalf(layer.input_layernorm, get(base + "input_layernorm.weight"));
      CheckHalf(layer.post_attention_layernorm, get(base + "post_attention_layernorm.weight"));
      VT_CHECK(layer.mlp.IsExl3(), "qwen parameter check: missing EXL3 MLP");
      nlohmann::json record{{"layer", i}, {"kind", types[i]}, {"norm_dtype", "F16"}};
      if (layer.is_linear_attention) {
        ++gdn_layers;
        const std::string la = base + "linear_attn.";
        CheckHalf(layer.gdn.conv1d_weight, get(la + "conv1d.weight"));
        CheckHalf(layer.gdn.norm_weight, get(la + "norm.weight"));
        const auto& b = get(la + "in_proj_b.weight");
        const auto& a = get(la + "in_proj_a.weight");
        VT_CHECK(b.dtype == "F16" && a.dtype == "F16" && layer.gdn.in_proj_ba.dtype == vt::DType::kF16 &&
                     layer.gdn.in_proj_ba.nk && layer.gdn.in_proj_ba.bytes.size() == b.nbytes + a.nbytes &&
                     std::memcmp(layer.gdn.in_proj_ba.bytes.data(), b.data, b.nbytes) == 0 &&
                     std::memcmp(layer.gdn.in_proj_ba.bytes.data() + b.nbytes, a.data, a.nbytes) == 0,
                 "qwen parameter check: BA half bits/row order mismatch");
        VT_CHECK(layer.gdn.a_log.dtype == vt::DType::kF32 && layer.gdn.dt_bias.dtype == vt::DType::kF32,
                 "qwen parameter check: expected FP32 gate computation parameters");
        record["conv_dtype"] = record["gated_norm_dtype"] = record["ba_dtype"] = "F16";
        record["a_log_dtype"] = record["dt_bias_compute_dtype"] = "F32";
      } else {
        ++attention_layers;
        CheckHalf(layer.attn.q_norm, get(base + "self_attn.q_norm.weight"));
        CheckHalf(layer.attn.k_norm, get(base + "self_attn.k_norm.weight"));
        VT_CHECK(layer.attn.IsExl3(), "qwen parameter check: missing EXL3 attention");
        record["qk_norm_dtype"] = "F16";
      }
      records.push_back(std::move(record));
    }
    VT_CHECK(gdn_layers == 48 && attention_layers == 16, "qwen parameter check: target topology mismatch");
    std::ofstream output(report_path);
    VT_CHECK(output.good(), "qwen parameter check: cannot write report");
    output << nlohmann::json{{"schema", 1}, {"status", "PASS"}, {"model_dir", model.string()},
                             {"gdn_layers", gdn_layers}, {"attention_layers", attention_layers},
                             {"layers", records}, {"scope", "Current native per-layer loader with all64 real target layers; FP16 norm/Conv/BA source conversions only. No whole-model embedding/head/vision load, MTP load, GPU execution, activation or numerical model/state parity."}}.dump(2) << '\n';
    output.close();
    VT_CHECK(output.good(), "qwen parameter check: report write failed");
    std::cout << "PASS: 64 current native target layer loads; 48 GDN / 16 attention; FP16 norms/Conv/BA\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "qwen_exl3_parameter_check: " << error.what() << '\n';
    return 1;
  }
}

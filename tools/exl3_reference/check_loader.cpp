// Focused host qualification of the real native LoadExl3 seam. No GPU,
// dequantization, grouped transform or whole-model forward is performed.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/dense_weight_loaders.h"

namespace {
using Json = nlohmann::json;
using Index = std::map<std::string, const vllm::StTensor*>;

void AddShard(Index& index, const vllm::SafetensorsFile& shard) {
  for (const auto& name : shard.Names()) {
    VT_CHECK(index.emplace(name, &shard.Get(name)).second,
             "exl3 loader check: duplicate tensor across shards: " + name);
  }
}

vllm::Exl3Weight Load(const Index& index, const Json& module) {
  const std::string name = module.at("name");
  const vllm::TensorResolver get = [&index](const std::string& key) -> const vllm::StTensor& {
    const auto it = index.find(key);
    VT_CHECK(it != index.end(), "exl3 loader check: missing tensor: " + key);
    return *it->second;
  };
  const auto has = [&index](const std::string& key) { return index.count(key) != 0; };
  VT_CHECK(vllm::dense_loaders::IsExl3Projection(has, name),
           "exl3 loader check: incomplete EXL3 storage: " + name);
  auto weight = vllm::dense_loaders::LoadExl3(get, has, name);
  const auto& marker = get(name + ".mul1");
  VT_CHECK(marker.dtype == "I32" && marker.shape.empty() && marker.nbytes == 4 &&
               vt::LoadUnaligned<uint32_t>(marker.data) == 0x83DCD12DU,
           "exl3 loader check: expected pinned scalar mul1 marker: " + name);
  VT_CHECK(weight.codebook == 2 && weight.InFeatures() == module.at("k") &&
               weight.OutFeatures() == module.at("n") && weight.Bits() == module.at("bits"),
           "exl3 loader check: native geometry/codebook mismatch: " + name);
  // Full-span alias equality proves the native byte view is bit-preserving;
  // it needs no second copy or read of the complete packed model.
  for (const auto& entry : {std::pair{"trellis", &weight.trellis},
                            std::pair{"suh", &weight.suh}, std::pair{"svh", &weight.svh}}) {
    const auto& source = get(name + "." + entry.first);
    VT_CHECK(entry.second->bytes.borrowed() && entry.second->bytes.data() == source.data &&
                 entry.second->bytes.size() == source.nbytes,
             "exl3 loader check: expected whole-span mapped borrow: " + name);
  }
  return weight;
}

void NegativeChecks(const Index& index, const Json& module) {
  const std::string name = module.at("name");
  const auto rejects = [&module](const Index& bad) {
    bool rejected = false;
    try { (void)Load(bad, module); } catch (const std::runtime_error&) { rejected = true; }
    VT_CHECK(rejected, "exl3 loader check: malformed fixture was accepted");
  };
  Index bad = index;
  bad.erase(name + ".suh");
  rejects(bad);
  uint32_t wrong_multiplier = 0;
  auto marker = *index.at(name + ".mul1");
  marker.data = reinterpret_cast<const uint8_t*>(&wrong_multiplier);
  bad = index;
  bad[name + ".mul1"] = &marker;
  rejects(bad);
  bad = index;
  bad[name + ".mcg"] = index.at(name + ".mul1");
  rejects(bad);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    VT_CHECK(argc == 4, "usage: exl3_loader_check MODEL_DIR REFERENCE_MANIFEST NEW_REPORT");
    const std::filesystem::path model_dir(argv[1]), output(argv[3]);
    VT_CHECK(!std::filesystem::exists(output), "exl3 loader check: report already exists");
    std::ifstream manifest_file(argv[2]);
    VT_CHECK(manifest_file.good(), "exl3 loader check: cannot read manifest");
    const auto manifest = Json::parse(manifest_file);
    const auto& checkpoint = manifest.at("reference_B").at("checkpoint");
    VT_CHECK(checkpoint.at("packed_byte_identity").at("status") == "verified",
             "exl3 loader check: complete payload identity prerequisite not verified");
    const auto& inventory = checkpoint.at("header_inventory");
    const auto& modules = inventory.at("modules");
    VT_CHECK(modules.size() == 409 && inventory.at("mtp_count") == 8,
             "exl3 loader check: requires complete pinned 409-module manifest");
    std::vector<vllm::SafetensorsFile> shards;
    shards.reserve(inventory.at("shards").size());
    for (const auto& shard : inventory.at("shards"))
      shards.push_back(vllm::SafetensorsFile::Open((model_dir / shard.at("name").get<std::string>()).string()));
    Index index;
    for (const auto& shard : shards) AddShard(index, shard);
    VT_CHECK(index.size() == inventory.at("tensor_count"), "exl3 loader check: tensor count mismatch");
    // Exercise duplicate detection without overwriting an existing index entry.
    bool duplicate_rejected = false;
    try { AddShard(index, shards.front()); } catch (const std::runtime_error&) { duplicate_rejected = true; }
    VT_CHECK(duplicate_rejected, "exl3 loader check: duplicate shard was accepted");
    std::set<std::string> actual_modules, expected_modules;
    for (const auto& [name, tensor] : index) {
      (void)tensor;
      if (name.ends_with(".trellis")) actual_modules.insert(name.substr(0, name.size() - 8));
    }
    for (const auto& module : modules)
      VT_CHECK(expected_modules.insert(module.at("name")).second, "exl3 loader check: duplicate manifest module");
    VT_CHECK(actual_modules == expected_modules, "exl3 loader check: native module names differ from manifest");
    NegativeChecks(index, modules.front());
    std::vector<vllm::Exl3Weight> weights;
    weights.reserve(modules.size());
    size_t bytes = 0, mtp = 0;
    Json loaded = Json::array();
    std::vector<uint16_t> first_words;
    std::weak_ptr<const void> mapping = index.at(modules.front().at("name").get<std::string>() + ".trellis")->mapping;
    for (const auto& module : modules) {
      weights.push_back(Load(index, module));
      const auto& weight = weights.back();
      bytes += weight.trellis.bytes.size() + weight.suh.bytes.size() + weight.svh.bytes.size();
      if (module.at("group") == "mtp") ++mtp;
      first_words.push_back(vt::LoadUnaligned<uint16_t>(weight.trellis.bytes.data()));
      loaded.push_back({{"name", weight.name}, {"k", weight.InFeatures()}, {"n", weight.OutFeatures()},
                        {"bits", weight.Bits()}, {"codebook", weight.codebook},
                        {"trellis_bytes", weight.trellis.bytes.size()}, {"mapped_borrow", true}});
    }
    VT_CHECK(mtp == 8, "exl3 loader check: native MTP count mismatch");
    // The borrowed tensors must own the maps after file/index destruction.
    index.clear();
    shards.clear();
    VT_CHECK(!mapping.expired(), "exl3 loader check: borrowed weights lost their map owner");
    for (size_t i = 0; i < weights.size(); ++i)
      VT_CHECK(vt::LoadUnaligned<uint16_t>(weights[i].trellis.bytes.data()) == first_words[i],
               "exl3 loader check: mapped bytes invalid after shard destruction");
    weights.clear();
    VT_CHECK(mapping.expired(), "exl3 loader check: map remained retained after final owner destruction");
    Json result{{"schema", 1}, {"status", "PASS"}, {"model_dir", model_dir.string()},
                {"modules", loaded}, {"module_count", loaded.size()}, {"mtp_count", mtp},
                {"mapped_weight_bytes", bytes}, {"duplicate_shard_rejected", true},
                {"missing_scale_rejected", true}, {"wrong_mul1_marker_rejected", true},
                {"conflicting_markers_rejected", true}, {"mapping_lifetime_checked", true},
                {"scope", "Current native SafetensorsFile/LoadExl3 seam and mapped ownership only. No GPU upload, grouped mapping, full model loader, FP16 boundary or model parity. Complete shard hashes are the prior S0 prerequisite, not rehashed here."}};
    std::ofstream report(output);
    VT_CHECK(report.good(), "exl3 loader check: cannot create report");
    report << result.dump(2) << '\n';
    report.close();
    VT_CHECK(report.good(), "exl3 loader check: report write failed");
    std::cout << "PASS: " << loaded.size() << " native EXL3 mapped modules, " << mtp
              << " MTP; packed bytes/geometry, negative fixtures and map lifetime checked\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "exl3_loader_check: " << error.what() << '\n';
    return 1;
  }
}

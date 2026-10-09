// N1 isolated LayerNorm experiment. This executable never constructs a model
// or changes the registered production operator. All readbacks are diagnostic.
#include "native_layer_norm_boundary_kernels.h"
#include "vt/unaligned.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <cmath>

namespace {
using json = nlohmann::json;
using namespace vt;
namespace fs = std::filesystem;
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
std::vector<unsigned char> Read(const fs::path& p, size_t bytes) {
  Require(fs::file_size(p) == bytes, "input length differs from full-shape contract");
  std::ifstream in(p, std::ios::binary);
  Require(in.good(), "cannot open operand");
  std::vector<unsigned char> result(bytes);
  in.read(reinterpret_cast<char*>(result.data()), bytes);
  Require(in.good(), "cannot read operand");
  return result;
}
struct Queue {
  vt::Queue q = CreateQueue({DeviceType::kXPU, 0});
  ~Queue() { DestroyQueue(q); }
};
struct Buffer {
  vt::Queue& q;
  Tensor t;
  Buffer(vt::Queue& queue, DType dtype, int64_t rows, int64_t width, bool original_shape = false)
      : q(queue), t(original_shape ? Tensor::Contiguous(nullptr, dtype, q.device, {rows, 1, width})
                                  : Tensor::Contiguous(nullptr, dtype, q.device, {rows, width})) {
    t.data = Alloc(q.device, t.Bytes());
  }
  Buffer(const Buffer&) = delete;
  ~Buffer() { Free(q.device, t.data); }
  void Put(const std::vector<unsigned char>& bytes) {
    Require(bytes.size() == t.Bytes(), "upload byte count differs");
    GetBackend(q.device).Copy(q, t.data, bytes.data(), bytes.size());
    GetBackend(q.device).Synchronize(q);
  }
  std::vector<unsigned char> Get() const {
    std::vector<unsigned char> result(t.Bytes());
    GetBackend(q.device).Copy(q, result.data(), t.data, result.size());
    GetBackend(q.device).Synchronize(q);
    return result;
  }
};
void Write(const fs::path& p, const std::vector<unsigned char>& bytes) {
  std::ofstream out(p, std::ios::binary);
  Require(out.good(), "cannot open output");
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  Require(out.good(), "cannot write output");
}
json Metrics(const std::vector<unsigned char>& actual, const std::vector<unsigned char>& ref,
             bool half) {
  Require(actual.size() == ref.size(), "comparison byte count differs");
  const size_t unit = half ? 2 : 4;
  uint64_t different = 0, nonfinite = 0;
  double squared = 0, norm = 0, maximum = 0;
  json witnesses = json::array();
  for (size_t i = 0; i < actual.size() / unit; ++i) {
    float a, b;
    uint32_t abits, bbits;
    if (half) {
      abits = LoadUnaligned<uint16_t>(actual.data() + unit * i);
      bbits = LoadUnaligned<uint16_t>(ref.data() + unit * i);
      a = F16ToF32(abits); b = F16ToF32(bbits);
    } else {
      abits = LoadUnaligned<uint32_t>(actual.data() + unit * i);
      bbits = LoadUnaligned<uint32_t>(ref.data() + unit * i);
      a = LoadUnaligned<float>(actual.data() + unit * i);
      b = LoadUnaligned<float>(ref.data() + unit * i);
    }
    nonfinite += !std::isfinite(a) || !std::isfinite(b);
    double delta = static_cast<double>(a) - b;
    squared += delta * delta; norm += static_cast<double>(b) * b;
    maximum = std::max(maximum, std::abs(delta));
    if (abits != bbits) {
      ++different;
      if (witnesses.size() < 64)
        witnesses.push_back({{"index", i}, {"actual", a}, {"reference", b},
                             {"actual_bits", abits}, {"reference_bits", bbits}});
    }
  }
  return {{"elements", actual.size() / unit}, {"different_elements", different},
          {"storage_exact", different == 0}, {"nonfinite_elements", nonfinite},
          {"max_abs", maximum}, {"rel_l2", std::sqrt(squared / std::max(norm, 1e-30))},
          {"first_differences", witnesses}};
}
std::vector<unsigned char> Column(const std::vector<unsigned char>& bytes, size_t col) {
  std::vector<unsigned char> result(bytes.size() / 6);
  for (size_t row = 0; row < result.size() / 4; ++row)
    std::memcpy(result.data() + 4 * row, bytes.data() + 4 * (row * 6 + col), 4);
  return result;
}
json Run(const fs::path& manifest, const fs::path& output) {
  Require(!fs::exists(output), "new output directory required");
  Require(fs::file_size(manifest) <= 128 * 1024, "manifest bound exceeded");
  json doc;
  std::ifstream in(manifest); in >> doc;
  Require(doc.at("status") == "CAPTURED", "installed R receipt required");
  Require(doc.at("cases").size() == 2, "exactly two preselected full-shape block norms required");
  // Complete host admission before opening the XPU queue.
  const auto first_label = doc.at("cases").at(0).at("label").get<std::string>();
  const auto prefix = first_label.substr(0, first_label.find('-'));
  Require(prefix == "block0" || prefix == "block12" || prefix == "block26",
          "only the preselected block0/12/26 inputs are admitted");
  std::map<std::string, std::map<std::string, std::vector<unsigned char>>> inputs;
  for (const auto& c : doc.at("cases")) {
    const auto label = c.at("label").get<std::string>();
    Require((label == prefix + "-norm1" || label == prefix + "-norm2") && !inputs.contains(label),
            "wrong or repeated frozen norm label");
    Require(c.at("shape") == json::array({768, 1, 1152}) &&
            c.at("strides") == json::array({1152, 1152, 1}) &&
            c.at("epsilon_fp32_bits") == "bd378635", "original shape/epsilon required");
    for (const auto& role : {"input", "reference", "weight", "bias", "mean", "rstd"}) {
      const auto name = c.at("files").at(role).at("file").get<std::string>();
      Require(fs::path(name).filename() == name && !fs::is_symlink(manifest.parent_path() / name),
              "invalid operand path");
      const size_t bytes = std::string(role) == "input" || std::string(role) == "reference" ? 768 * 1152 * 2
          : std::string(role) == "mean" || std::string(role) == "rstd" ? 768 * 4 : 1152 * 2;
      inputs[label][role] = Read(manifest.parent_path() / name, bytes);
    }
  }
  Require(vt::xpu::DeviceCount() > 0, "XPU device required");
  Queue queue;
  auto& q = queue.q;
  auto device = vt::xpu::NativeQueue(q).get_device();
  json report{{"status", "DIAGNOSTIC"}, {"scope", "N1 two full-shape norms, no tower or serving gate"},
              {"device", device.get_info<sycl::info::device::name>()},
              {"driver", device.get_info<sycl::info::device::driver_version>()},
              {"compiler_version", __INTEL_LLVM_COMPILER},
              {"device_max_work_group_size", device.get_info<sycl::info::device::max_work_group_size>()},
              {"cases", json::array()}};
  fs::create_directory(output);
  for (const auto& c : doc.at("cases")) {
    const auto label = c.at("label").get<std::string>();
    const auto& source = inputs.at(label);
    Buffer x(q, DType::kF16, 768, 1152, true), w(q, DType::kF16, 1, 1152), b(q, DType::kF16, 1, 1152),
        y(q, DType::kF16, 768, 1152, true), stats(q, DType::kF32, 768, 6),
        pre(q, DType::kF32, 768, 1152);
    x.Put(source.at("input")); w.Put(source.at("weight")); b.Put(source.at("bias"));
    auto one = w.t.View({1152}), two = b.t.View({1152});
    vt::LayerNorm(q, y.t, x.t, &one, &two, {1e-6f});
    const auto original = y.Get();
    json result{{"label", label}, {"shape", c.at("shape")}, {"strides", c.at("strides")},
                {"epsilon_fp32_bits", c.at("epsilon_fp32_bits")}, {"variants", json::object()},
                {"native_original_vs_R", Metrics(original, source.at("reference"), true)},
                {"native_pointer_alignment_mod64", uintptr_t(x.t.data) % 64}};
    result["native_pointer_alignment_mod64_all"] = {
        {"input", uintptr_t(x.t.data) % 64}, {"output", uintptr_t(y.t.data) % 64},
        {"weight", uintptr_t(w.t.data) % 64}, {"bias", uintptr_t(b.t.data) % 64}};
    Write(output / (label + "-N-original.float16"), original);
    std::map<std::string, std::vector<unsigned char>> values, stat_values;
    auto save = [&](const std::string& name, bool observed, int maximum) {
      const auto raw = y.Get();
      values[name] = raw;
      Write(output / (label + "-" + name + ".float16"), raw);
      auto& entry = result["variants"][name];
      entry["output_vs_R"] = Metrics(raw, source.at("reference"), true);
      if (maximum) {
        entry["kernel_max_work_group_size"] = maximum;
        int lanes = maximum; while (lanes > 1152 / 4 && lanes > 32) lanes /= 2;
        entry["work_group_size"] = lanes;
        entry["sub_group_size"] = 32;
      }
      if (!observed) return;
      auto raw_stats = stats.Get(), raw_pre = pre.Get();
      stat_values[name] = raw_stats;
      Write(output / (label + "-" + name + "-stats.float32"), raw_stats);
      Write(output / (label + "-" + name + "-precast.float32"), raw_pre);
      entry["mean_vs_R"] = Metrics(Column(raw_stats, 0), source.at("mean"), false);
      entry["rstd_vs_R"] = Metrics(Column(raw_stats, 4), source.at("rstd"), false);
      entry["affine_witnesses"] = json::array();
      for (const auto& witness : entry["output_vs_R"]["first_differences"]) {
        size_t index = witness.at("index"), row = index / 1152;
        json item = witness;
        item["row"] = row; item["channel"] = index % 1152;
        item["precast"] = LoadUnaligned<float>(raw_pre.data() + 4 * index);
        item["precast_bits"] = LoadUnaligned<uint32_t>(raw_pre.data() + 4 * index);
        item["statistics_bits"] = json::array();
        for (int col = 0; col < 6; ++col)
          item["statistics_bits"].push_back(LoadUnaligned<uint32_t>(raw_stats.data() + 4 * (row * 6 + col)));
        entry["affine_witnesses"].push_back(item);
      }
    };
    auto* s = static_cast<float*>(stats.t.data);
    auto* p = static_cast<float*>(pre.t.data);
    using namespace vt::xpu::norm_probe;
    Native<false, false>(q, y.t, x.t, &one, &two, {1e-6f}, nullptr, nullptr); save("N-clone", false, 0);
    Native<false, true>(q, y.t, x.t, &one, &two, {1e-6f}, s, p); save("N-observed", true, 0);
    Native<true, false>(q, y.t, x.t, &one, &two, {1e-6f}, nullptr, nullptr); save("N1-materialized", false, 0);
    Native<true, true>(q, y.t, x.t, &one, &two, {1e-6f}, s, p); save("N1-observed", true, 0);
    Native<false, false, true>(q, y.t, x.t, &one, &two, {1e-6f}, nullptr, nullptr); save("N2-rounded-divide", false, 0);
    Native<false, true, true>(q, y.t, x.t, &one, &two, {1e-6f}, s, p); save("N2-observed", true, 0);
    auto maximum = Adapted<false>(q, y.t, x.t, one, two, 1e-6f, nullptr, nullptr); save("D", false, maximum);
    maximum = Adapted<true>(q, y.t, x.t, one, two, 1e-6f, s, p); save("D-observed", true, maximum);
    result["observer_controls"] = {{"N_clone_matches_actual_native", values.at("N-clone") == original},
                                  {"N_probe_matches_actual_native", values.at("N-observed") == original},
                                  {"N1_output_preserved", values.at("N1-observed") == values.at("N1-materialized")},
                                  {"D_output_preserved", values.at("D-observed") == values.at("D")},
                                  {"N2_output_preserved", values.at("N2-observed") == values.at("N2-rounded-divide")},
                                  {"N2_probe_matches_registered_native", values.at("N2-observed") == original}};
    result["candidate1_changes_output"] = values.at("N1-materialized") != original;
    result["candidate1_statistics_vs_N"] = json::object();
    const char* columns[] = {"mean", "M2", "variance", "variance_plus_epsilon", "rstd", "count"};
    for (int col = 0; col < 6; ++col)
      result["candidate1_statistics_vs_N"][columns[col]] = Metrics(Column(stat_values.at("N1-observed"), col),
                                                                  Column(stat_values.at("N-observed"), col), false);
    result["statistics_layout"] = columns;
    result["R_hidden_statistics"] = "M2/variance/variance+eps/precast are not exposed by installed R";
    int lanes = report.at("device_max_work_group_size");
    while (lanes > 1152 / 4 && lanes > 32) lanes /= 2;
    result["N_work_group_size"] = lanes;
    result["N_sub_group_size"] = 32;
    report["cases"].push_back(result);
  }
  return report;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 3, "usage: native-layer-norm-boundary-probe CASES_JSON NEW_OUTPUT_DIR");
    auto report = Run(argv[1], argv[2]);
    std::ofstream out(fs::path(argv[2]) / "native.json");
    out << report.dump(2) << '\n';
    Require(out.good(), "cannot write report");
    std::cout << "N1 two-norm boundary probe DIAGNOSTIC complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "N1 boundary probe FAIL: " << e.what() << '\n';
    return 1;
  }
}

// Bounded isolated phase measurements. The language model is not loaded.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/model_executor/models/qwen3_vl.h"
#include "vllm/multimodal/torchvision_resize.h"
#include "vt/op_provider.h"
#include "vt/unaligned.h"
#include "vt/xpu.h"

namespace {
using Clock = std::chrono::steady_clock;
using json = nlohmann::json;
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Queue {
  vt::Queue value = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  ~Queue() { vt::DestroyQueue(value); }
};
struct Buffer {
  vt::Tensor value;
  Buffer(vt::Device device, int64_t rows, int64_t columns)
      : value(vt::Tensor::Contiguous(nullptr, vt::DType::kF16, device, {rows, columns})) {
    value.data = vt::Alloc(device, value.Bytes());
  }
  ~Buffer() { vt::Free(value.device, value.data); }
  Buffer(const Buffer&) = delete;
};

json Run(const std::filesystem::path& model, const std::filesystem::path& image,
         const std::string& mime) {
  using namespace vllm;
  using namespace vllm::multimodal;
  using namespace vllm::entrypoints::openai;
  for (const char* key : {"VT_XPU_PROFILE", "VT_XPU_HOST_PROFILE", "VT_NATIVE_VISION_TRACE"}) {
    const char* value = std::getenv(key);
    Require(!value || std::string(value) == "0", "trace/profiler flags must be off");
  }
  Require(!std::getenv("VT_PREFIX_SNAPSHOT_TRACE"), "prefix trace flag must be absent");
  Require(mime == "image/png" || mime == "image/jpeg", "PNG/JPEG MIME required");
  const auto size = std::filesystem::file_size(image);
  Require(size > 0 && size <= kMaxImageContainerBytes, "container exceeds native budget");
  std::ifstream input(image, std::ios::binary);
  Require(input.good(), "cannot open image");
  DecodedMedia media{mime, {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()}};
  Require(media.bytes.size() == size, "image read incomplete");
  const auto text_config = LoadHfConfig((model / "config.json").string());
  const auto tower = Qwen3_5DenseVisionConfig(text_config);
  Require(tower.hidden_size == 1152 && tower.num_heads == 16 && tower.head_dim() == 72 &&
          tower.depth == 27 && tower.out_hidden_size == text_config.hidden_size &&
          tower.deepstack_visual_indexes.empty(), "qualified Qwen3.5 tower required");
  auto config = LoadQwen3VLProcessorConfig((model / "preprocessor_config.json").string(),
                                         (model / "config.json").string(), "isolated-native-vision");
  Require(config.patch_size == tower.patch_size && config.merge_size == tower.spatial_merge_size &&
          config.temporal_patch_size == tower.temporal_patch_size, "processor/tower geometry differs");
  config.max_pixels = std::min({config.max_pixels, kNativeQwen3_5MaxImagePixels,
      kNativeQwen3_5MaxImagePatches * config.patch_size * config.patch_size});
  config.torchvision_bicubic_resize = true;
  config.pixel_dtype = ImagePixelDType::kF16;
  config.retain_pixel_values_f32 = false;
  const Qwen3VLImageProcessor processor(config);
  const auto codec = DefaultImageCodec();
  json report{{"status", "FAIL"}, {"scope", "isolated native decode/resize/patch and FP16 XPU encoder; not serving latency or language prefill"},
              {"filename", image.filename().string()}, {"mime", mime}, {"samples", json::array()},
              {"weight_loading_timed", false}, {"capture_readbacks_timed", false},
              {"language_model_loaded", false}, {"driver_memory_included", false}};
  ImageKwargs patches;
  for (int n = 0; n < 4; ++n) {
    auto start = Clock::now();
    const auto decoded = codec(media);
    const double decode_s = Seconds(start);
    const auto shape = SmartResize(decoded.height, decoded.width,
        int64_t(config.patch_size) * config.merge_size, config.min_pixels, config.max_pixels);
    const bool resize = shape[0] != decoded.height || shape[1] != decoded.width;
    start = Clock::now();
    std::vector<uint8_t> resized;
    if (resize) resized = TorchvisionResizeBicubicRgb(decoded.rgb.data(), decoded.height, decoded.width, shape[0], shape[1]);
    const double resize_s = Seconds(start);
    start = Clock::now();
    patches = processor.ProcessImage(resize ? resized.data() : decoded.rgb.data(), shape[0], shape[1]);
    const double patches_s = Seconds(start);
    // Outside timings: splitting the phases must preserve the actual pipeline.
    const auto integrated = processor.ProcessImage(decoded.rgb.data(), decoded.height, decoded.width);
    Require(patches.image_grid_thw == integrated.image_grid_thw &&
            patches.pixel_values_f16 == integrated.pixel_values_f16, "split preprocessing differs from native pipeline");
    report["decoded_hw"] = {decoded.height, decoded.width};
    report["processed_hw"] = shape;
    report["resize_required"] = resize;
    report["samples"].push_back({{"index", n}, {"decode_s", decode_s},
                                {"resize_s", resize_s}, {"normalize_patch_s", patches_s}});
  }
  Require(vt::xpu::DeviceCount() > 0, "native XPU device required");
  auto& backend = vt::GetBackend(vt::DeviceType::kXPU);
  Queue queue;
  const auto memory_before = vt::xpu::GetMemoryInfo();
  const auto index = LoadSafetensorsIndex((model / "model.safetensors.index.json").string());
  std::set<std::string> names;
  for (const auto& [key, filename] : index) if (key.rfind("model.visual.", 0) == 0) names.insert(filename);
  Require(!names.empty(), "visual weights missing from index");
  std::vector<SafetensorsFile> shards;
  for (const auto& name : names) shards.push_back(SafetensorsFile::Open((model / name).string()));
  auto host_weights = LoadQwen3VLVisionWeights(shards, tower);
  auto weights = PrepareVisionDeviceWeights(host_weights, tower, backend, vt::DType::kF16);
  host_weights = {};
  auto workspace = PrepareVisionWorkspace(patches.image_grid_thw, tower, backend, queue.value);
  Buffer pixels(queue.value.device, patches.num_patches, patches.patch_feature_dim);
  backend.Synchronize(queue.value);
  auto start = Clock::now();
  backend.Copy(queue.value, pixels.value.data, patches.pixel_values_f16.data(), pixels.value.Bytes());
  backend.Synchronize(queue.value);
  report["input_upload_s"] = Seconds(start);
  report["grid_thw"] = patches.image_grid_thw;
  report["visual_rows"] = patches.num_patches / tower.merge_unit();
  report["device"] = vt::xpu::DeviceDescription();
  std::vector<unsigned char> first;
  for (int n = 0; n < 4; ++n) {
    start = Clock::now();
    auto output = Qwen3VLVisionForwardDevice(pixels.value, weights, *workspace, backend, queue.value);
    // An isolated benchmark boundary, not a new product wait or polling loop.
    backend.Synchronize(queue.value);
    report["samples"][n]["encoder_s"] = Seconds(start);
    Require(output.tensor().dtype == vt::DType::kF16 &&
            output.tensor().shape[0] == patches.num_patches / tower.merge_unit() &&
            output.tensor().shape[1] == tower.out_hidden_size, "wrong encoder output");
    const auto memory = vt::xpu::GetMemoryInfo();
    report["samples"][n]["backend_live_bytes"] = memory.allocated_bytes;
    report["samples"][n]["backend_peak_bytes"] = memory.peak_allocated_bytes;
    std::vector<unsigned char> bytes(output.tensor().Bytes());
    backend.Copy(queue.value, bytes.data(), output.tensor().data, bytes.size());
    output.RecordUse(queue.value);
    backend.Synchronize(queue.value);
    for (size_t i = 0; i < bytes.size(); i += 2)
      Require(std::isfinite(vt::F16ToF32(vt::LoadUnaligned<uint16_t>(bytes.data() + i))), "nonfinite encoder output");
    if (n == 0) first = std::move(bytes);
    else Require(first == bytes, "repeated encoder output differs");
  }
  Require(vt::GetReferenceTierHits() == 0, "provider reference fallback executed");
  json medians;
  for (const char* key : {"decode_s", "resize_s", "normalize_patch_s", "encoder_s"}) {
    std::vector<double> values;
    for (int n = 1; n < 4; ++n) values.push_back(report["samples"][n][key].get<double>());
    std::sort(values.begin(), values.end());
    medians[key] = values[1];
  }
  report["warm_medians"] = medians;
  report["backend_live_before_bytes"] = memory_before.allocated_bytes;
  report["reference_tier_hits"] = vt::GetReferenceTierHits();
  report["encoder_invocations"] = 4;
  report["encoder_cache_hits"] = 0;
  report["split_preprocessing_exact"] = true;
  report["repeat_encoder_output_exact"] = true;
  report["status"] = "PASS";
  return report;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: native-vision-bench MODEL_DIR IMAGE_PATH image/png|image/jpeg OUTPUT_JSON\n";
    return 2;
  }
  const std::filesystem::path output = argv[4];
  if (std::filesystem::exists(output)) {
    std::cerr << "output exists; preserve previous evidence\n";
    return 2;
  }
  json report;
  int code = 1;
  try { report = Run(argv[1], argv[2], argv[3]); code = 0; }
  catch (const std::exception& error) { report = {{"status", "FAIL"}, {"error", error.what()}}; }
  std::ofstream file(output);
  if (!file.good()) { std::cerr << "cannot write output\n"; return 2; }
  file << report.dump(2) << '\n';
  std::cout << json{{"status", report.value("status", "FAIL")},
                   {"warm_medians", report.value("warm_medians", json::object())},
                   {"error", report.value("error", "")}}.dump() << '\n';
  return code;
}

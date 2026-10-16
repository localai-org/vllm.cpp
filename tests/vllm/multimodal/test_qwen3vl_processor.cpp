// Processor-parity gate (M1 correctness gate). Verifies the C++ Qwen3-VL mm
// input pipeline is BIT/BYTE-identical to the vLLM 0.25.0 oracle fixture
// (captured by scripts/mm/m0_oracle_capture.py): pixel_values (bf16), grid_thw,
// placeholder-expanded prompt ids, and the MultiModalHasher mm-hash.
//
// Golden: tests/vllm/multimodal/fixtures/qwen3vl/{manifest.json,
//   image_rgb_uint8_448x448x3.bin, image_grid_thw_i64.bin, pixel_values_f32.bin}
// Upstream oracle: MULTIMODAL_REGISTRY.create_processor(...).apply for
//   Qwen/Qwen3-VL-4B-Instruct.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/multimodal/qwen3vl_processor.h"
#include "vt/dtype.h"

namespace {

std::string FixDir() { return std::string(MM_FIXTURE_DIR); }

std::vector<uint8_t> ReadBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open fixture: ", path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream f(path);
  REQUIRE_MESSAGE(f.good(), "cannot open fixture: ", path);
  nlohmann::json j;
  f >> j;
  return j;
}

vllm::multimodal::Qwen3VLProcessorConfig ConfigFromManifest(
    const nlohmann::json& m) {
  vllm::multimodal::Qwen3VLProcessorConfig cfg;
  const auto& c = m.at("config");
  cfg.patch_size = c.at("patch_size").get<int>();
  cfg.temporal_patch_size = c.at("temporal_patch_size").get<int>();
  cfg.merge_size = c.at("merge_size").get<int>();
  cfg.image_mean = c.at("image_mean")[0].get<double>();
  cfg.image_std = c.at("image_std")[0].get<double>();
  cfg.image_token_id = c.at("image_token_id").get<int32_t>();
  cfg.vision_start_token_id = c.at("vision_start_token_id").get<int32_t>();
  cfg.vision_end_token_id = c.at("vision_end_token_id").get<int32_t>();
  cfg.model_id = m.at("model_id").get<std::string>();
  return cfg;
}

}  // namespace

TEST_CASE("qwen3vl-processor-parity: pixel_values + grid + hash + expansion") {
  const std::string dir = FixDir();
  const nlohmann::json manifest = ReadJson(dir + "/manifest.json");
  const auto cfg = ConfigFromManifest(manifest);

  const int64_t H = manifest.at("image").at("shape")[0].get<int64_t>();
  const int64_t W = manifest.at("image").at("shape")[1].get<int64_t>();
  const std::vector<uint8_t> rgb =
      ReadBytes(dir + "/image_rgb_uint8_448x448x3.bin");
  REQUIRE(rgb.size() == static_cast<size_t>(H * W * 3));

  vllm::multimodal::Qwen3VLImageProcessor proc(cfg);
  const auto kw = proc.ProcessImage(rgb.data(), H, W);

  SUBCASE("image_grid_thw exact") {
    const auto g = manifest.at("image_grid_thw").at("values");
    CHECK(kw.image_grid_thw[0] == g[0].get<int64_t>());
    CHECK(kw.image_grid_thw[1] == g[1].get<int64_t>());
    CHECK(kw.image_grid_thw[2] == g[2].get<int64_t>());
  }

  SUBCASE("pixel_values bf16 byte-identical to production golden") {
    // Golden file holds the bf16 production values widened to float32.
    const std::vector<uint8_t> gbytes = ReadBytes(dir + "/pixel_values_f32.bin");
    REQUIRE(gbytes.size() == kw.pixel_values_bf16.size() * sizeof(float));
    size_t mismatches = 0;
    for (size_t i = 0; i < kw.pixel_values_bf16.size(); ++i) {
      float g;
      std::memcpy(&g, gbytes.data() + i * sizeof(float), sizeof(float));
      // Golden values are exactly bf16-representable, so F32ToBF16(golden) is
      // the original bf16 code. Compare bf16 codes -> bit-exact, no type-punning.
      if (kw.pixel_values_bf16[i] != vt::F32ToBF16(g)) ++mismatches;
    }
    CHECK(mismatches == 0);
  }

  SUBCASE("mm-hash byte-identical to MultiModalHasher") {
    const std::string ours = proc.HashImage(rgb.data(), H, W);
    CHECK(ours == manifest.at("mm_hash").get<std::string>());
  }

  SUBCASE("placeholder expansion byte-identical to oracle expanded ids") {
    std::vector<int32_t> pre;
    for (const auto& v : manifest.at("pre_expansion_token_ids"))
      pre.push_back(v.get<int32_t>());
    std::vector<int32_t> golden;
    for (const auto& v : manifest.at("expanded_prompt_token_ids"))
      golden.push_back(v.get<int32_t>());

    std::vector<std::array<int64_t, 3>> grids = {kw.image_grid_thw};
    std::vector<std::array<int, 2>> placeholders;
    const auto expanded = vllm::multimodal::ExpandImagePlaceholders(
        pre, cfg.image_token_id, cfg.merge_size, grids, &placeholders);

    CHECK(expanded == golden);
    REQUIRE(placeholders.size() == 1);
    const int64_t expected_n = (kw.image_grid_thw[0] * kw.image_grid_thw[1] *
                                kw.image_grid_thw[2]) /
                               (cfg.merge_size * cfg.merge_size);
    CHECK(placeholders[0][1] == static_cast<int>(expected_n));
    CHECK(manifest.at("n_image_tokens").get<int>() ==
          static_cast<int>(expected_n));
  }
}

TEST_CASE("native vision processor: tensor resize, per-channel normalize and direct FP16") {
  const std::string dir = TV_RESIZE_FIXTURE_DIR;
  const auto manifest = ReadJson(dir + "/manifest.json");
  for (const auto& fixture : manifest.at("processor_cases")) {
    INFO(fixture.at("config_file"));
    auto cfg = vllm::multimodal::LoadQwen3VLProcessorConfig(
        dir + "/" + fixture.at("config_file").get<std::string>(), dir + "/model-config.json", "generated-native-fixture");
    cfg.torchvision_bicubic_resize = true;
    cfg.pixel_dtype = vllm::multimodal::ImagePixelDType::kF16;
    const int64_t h = fixture.at("input_hw")[0], w = fixture.at("input_hw")[1];
    std::vector<uint8_t> rgb(static_cast<size_t>(h * w * 3));
    for (int64_t y = 0; y < h; ++y) {
      for (int64_t x = 0; x < w; ++x) {
        const size_t offset = static_cast<size_t>((y * w + x) * 3);
        rgb[offset] = static_cast<uint8_t>((x * 17 + y * 3) % 256);
        rgb[offset + 1] = static_cast<uint8_t>((x * 5 + y * 11) % 256);
        rgb[offset + 2] = static_cast<uint8_t>((x * 7 + y * 19) % 256);
      }
    }
    auto kw = vllm::multimodal::Qwen3VLImageProcessor(cfg).ProcessImage(rgb.data(), h, w);
    CHECK(kw.pixel_dtype == vllm::multimodal::ImagePixelDType::kF16);
    CHECK(kw.pixel_values_bf16.empty());
    CHECK(kw.image_grid_thw == fixture.at("grid").get<std::array<int64_t, 3>>());
    CHECK(kw.num_patches == fixture.at("patch_shape")[0].get<int64_t>());
    CHECK(kw.patch_feature_dim == fixture.at("patch_shape")[1].get<int64_t>());
    const auto f32 = ReadBytes(dir + "/" + fixture.at("files").at("f32").at("file").get<std::string>());
    const auto f16 = ReadBytes(dir + "/" + fixture.at("files").at("f16").at("file").get<std::string>());
    REQUIRE(f32.size() == kw.pixel_values_f32.size() * sizeof(float));
    REQUIRE(f16.size() == kw.pixel_values_f16.size() * sizeof(uint16_t));
    CHECK(std::memcmp(f32.data(), kw.pixel_values_f32.data(), f32.size()) == 0);
    CHECK(std::memcmp(f16.data(), kw.pixel_values_f16.data(), f16.size()) == 0);
    // The serving selection owns only one production tensor; retaining FP32
    // for comparisons is optional and does not change model input bytes.
    cfg.retain_pixel_values_f32 = false;
    auto compact = vllm::multimodal::Qwen3VLImageProcessor(cfg).ProcessImage(rgb.data(), h, w);
    CHECK(compact.pixel_values_f32.empty());
    CHECK(compact.pixel_values_bf16.empty());
    CHECK((compact.pixel_values_f16 == kw.pixel_values_f16));
    bool differs_from_bf16_intermediate = false;
    for (size_t i = 0; i < kw.pixel_values_f32.size(); ++i)
      differs_from_bf16_intermediate |= kw.pixel_values_f16[i] != vt::F32ToF16(vt::BF16ToF32(vt::F32ToBF16(kw.pixel_values_f32[i])));
    CHECK(differs_from_bf16_intermediate);
  }
}

TEST_CASE("native vision processor: smart resize matches reference and invalid geometry is bounded") {
  const auto manifest = ReadJson(std::string(TV_RESIZE_FIXTURE_DIR) + "/manifest.json");
  using vllm::multimodal::SmartResize;
  for (const auto& fixture : manifest.at("smart_resize_cases")) {
    const auto actual = SmartResize(fixture.at("input_hw")[0], fixture.at("input_hw")[1], 32, 65536, 4194304);
    CHECK(actual == fixture.at("output_hw").get<std::array<int64_t, 2>>());
  }
  CHECK_THROWS_AS(SmartResize(1, 201, 32, 65536, 4194304), std::invalid_argument);
  CHECK_THROWS_AS(SmartResize(100, 100, 0, 65536, 4194304), std::runtime_error);
  CHECK_THROWS_AS(SmartResize(100, 100, 32, 100, 0), std::runtime_error);
  CHECK_THROWS_AS(SmartResize(std::numeric_limits<int64_t>::max(), 100, 32, 65536, 4194304),
                  std::invalid_argument);
  const uint8_t rgb[3] = {1, 2, 3};
  vllm::multimodal::Qwen3VLProcessorConfig cfg;
  cfg.temporal_patch_size = 0;
  CHECK_THROWS(vllm::multimodal::Qwen3VLImageProcessor(cfg).ProcessImage(rgb, 1, 1));
  cfg.temporal_patch_size = 2;
  CHECK_THROWS(vllm::multimodal::Qwen3VLImageProcessor(cfg).ProcessImage(nullptr, 1, 1));
}

// Optional larger captures are local generated images, not model weights or
// public fixtures. A missing directory explicitly skips this capture check.
TEST_CASE("native vision processor: actual pinned EXL3 worker input captures") {
  const char* capture_dir = std::getenv("VLLM_NATIVE_VISION_PROCESSOR_CAPTURE_DIR");
  if (!capture_dir) {
    MESSAGE("SKIP: VLLM_NATIVE_VISION_PROCESSOR_CAPTURE_DIR is not set");
    return;
  }
  const std::string dir = capture_dir;
  vllm::multimodal::Qwen3VLProcessorConfig cfg;
  cfg.max_pixels = 4194304;
  cfg.torchvision_bicubic_resize = true;
  cfg.pixel_dtype = vllm::multimodal::ImagePixelDType::kF16;
  cfg.channel_mean = std::array<double, 3>{0.5, 0.5, 0.5};
  cfg.channel_std = std::array<double, 3>{0.5, 0.5, 0.5};
  for (const std::string name : {"aligned", "resize"}) {
    INFO(name);
    const int64_t h = name == "aligned" ? 384 : 385;
    const int64_t w = name == "aligned" ? 512 : 513;
    const auto rgb = ReadBytes(dir + "/" + name + "-rgb.u8");
    REQUIRE(rgb.size() == static_cast<size_t>(h * w * 3));
    const auto kw = vllm::multimodal::Qwen3VLImageProcessor(cfg).ProcessImage(rgb.data(), h, w);
    const auto f32 = ReadBytes(dir + "/" + name + "-patches.f32");
    const auto f16 = ReadBytes(dir + "/" + name + "-patches.f16");
    REQUIRE(f32.size() == kw.pixel_values_f32.size() * sizeof(float));
    REQUIRE(f16.size() == kw.pixel_values_f16.size() * sizeof(uint16_t));
    CHECK(std::memcmp(f32.data(), kw.pixel_values_f32.data(), f32.size()) == 0);
    CHECK(std::memcmp(f16.data(), kw.pixel_values_f16.data(), f16.size()) == 0);
    CHECK((kw.image_grid_thw == std::array<int64_t, 3>{1, 24, 32}));
  }
}

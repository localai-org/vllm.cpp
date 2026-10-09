#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/multimodal/torchvision_resize.h"

TEST_CASE("native tensor bicubic resize: executed pinned torchvision byte parity") {
  const std::string dir = TV_RESIZE_FIXTURE_DIR;
  std::ifstream manifest_file(dir + "/manifest.json");
  REQUIRE(manifest_file.good());
  nlohmann::json manifest;
  manifest_file >> manifest;
  for (const auto& fixture : manifest.at("cases")) {
    INFO(fixture.at("file"));
    const int64_t h = fixture.at("input_hw")[0];
    const int64_t w = fixture.at("input_hw")[1];
    const int64_t rh = fixture.at("output_hw")[0];
    const int64_t rw = fixture.at("output_hw")[1];
    std::vector<uint8_t> rgb(static_cast<size_t>(h * w * 3));
    for (int64_t y = 0; y < h; ++y) {
      for (int64_t x = 0; x < w; ++x) {
        const size_t offset = static_cast<size_t>((y * w + x) * 3);
        rgb[offset] = static_cast<uint8_t>((x * 17 + y * 3) % 256);
        rgb[offset + 1] = static_cast<uint8_t>((x * 5 + y * 11) % 256);
        rgb[offset + 2] = static_cast<uint8_t>((x * 7 + y * 19) % 256);
      }
    }
    std::ifstream file(dir + "/" + fixture.at("file").get<std::string>(), std::ios::binary);
    REQUIRE(file.good());
    const std::vector<uint8_t> expected{std::istreambuf_iterator<char>(file),
                                         std::istreambuf_iterator<char>()};
    const auto result = vllm::multimodal::TorchvisionResizeBicubicRgb(rgb.data(), h, w, rh, rw);
    REQUIRE(result.size() == expected.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < result.size(); ++i) mismatches += result[i] != expected[i];
    CHECK(mismatches == 0);
  }
}

TEST_CASE("native tensor bicubic resize: reject invalid budgets before allocation") {
  const uint8_t rgb[3] = {1, 2, 3};
  using vllm::multimodal::TorchvisionResizeBicubicRgb;
  CHECK_THROWS(TorchvisionResizeBicubicRgb(nullptr, 1, 1, 1, 1));
  CHECK_THROWS(TorchvisionResizeBicubicRgb(rgb, 0, 1, 1, 1));
  CHECK_THROWS(TorchvisionResizeBicubicRgb(rgb, 1, 1, -1, 1));
  CHECK_THROWS(TorchvisionResizeBicubicRgb(rgb, 1, 32769, 1, 1));
  CHECK_THROWS(TorchvisionResizeBicubicRgb(rgb, 8192, 8192, 1, 1));
  CHECK_THROWS(TorchvisionResizeBicubicRgb(rgb, 1, 1, 8192, 8192));
  CHECK_THROWS(TorchvisionResizeBicubicRgb(rgb, 16384, 1, 1, 16384));
}

#include <doctest/doctest.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/v1/engine/input_processor.h"

namespace {
namespace oai = vllm::entrypoints::openai;
std::vector<uint8_t> ReadFixture(const std::string& name) {
  std::ifstream f(std::string(IMAGE_CODEC_FIXTURE_DIR) + "/" + name,
                  std::ios::binary);
  REQUIRE(f.good());
  return {(std::istreambuf_iterator<char>(f)), {}};
}
}

TEST_CASE("native PNG codec: RGB grayscale and alpha match reference RGB") {
  for (const std::string name : {"rgb", "gray", "rgba", "gray-alpha",
                                 "palette-alpha", "rgb-transparency"}) {
    INFO(name);
    const auto out = oai::DefaultImageCodec()({"image/png", ReadFixture(name + ".png")});
    CHECK(out.width == 64);
    CHECK(out.height == 64);
    CHECK((out.rgb == ReadFixture(name + ".rgb")));
  }
}

TEST_CASE("native image codec: PNG EXIF orientation matches reference") {
  const auto out = oai::DefaultImageCodec()({"image/png", ReadFixture("png-orientation.png")});
  CHECK(out.width == 32);
  CHECK(out.height == 64);
  CHECK((out.rgb == ReadFixture("png-orientation.rgb")));
}

TEST_CASE("native image codec: all EXIF orientations and TIFF byte orders match reference") {
  for (const std::string order : {"little", "big"}) {
    for (int orientation = 1; orientation <= 8; ++orientation) {
      for (const std::string extension : {"png", "jpg"}) {
        const auto name = "exif-" + order + "-" + std::to_string(orientation) + "-" + extension;
        INFO(name);
        const auto media_type = extension == "png" ? "image/png" : "image/jpeg";
        const auto out = oai::DefaultImageCodec()({media_type, ReadFixture(name + "." + extension)});
        CHECK(out.width == (orientation >= 5 ? 32 : 64));
        CHECK(out.height == (orientation >= 5 ? 64 : 32));
        CHECK(out.rgb == ReadFixture(name + ".rgb"));
      }
    }
  }
}

TEST_CASE("native JPEG codec: RGB grayscale and EXIF match reference") {
  for (const std::string name : {"jpeg-rgb", "jpeg-gray", "jpeg-orientation"}) {
    INFO(name);
    const auto out = oai::DefaultImageCodec()({"image/jpeg", ReadFixture(name + ".jpg")});
    CHECK(out.width == (name == "jpeg-orientation" ? 32 : 64));
    CHECK(out.height == 64);
    CHECK((out.rgb == ReadFixture(name + ".rgb")));
  }
  auto truncated = ReadFixture("jpeg-rgb.jpg");
  truncated.resize(truncated.size() / 2);
  CHECK_THROWS_AS(oai::DefaultImageCodec()({"image/jpeg", truncated}), vllm::v1::InputValidationError);
}

TEST_CASE("native image codec: reject large inputs before pixel decode") {
  auto codec = oai::DefaultImageCodec();
  CHECK_THROWS_WITH_AS(codec({"image/png", ReadFixture("oversize-header.png")}),
                      "multimodal image: decoded extent/pixel limit exceeded",
                      vllm::v1::InputValidationError);
  CHECK_THROWS_WITH_AS(codec({"image/png", std::vector<uint8_t>(oai::kMaxImageContainerBytes + 1)}),
                      "multimodal image: container exceeds byte limit",
                      vllm::v1::InputValidationError);
  oai::ChatContentPart part;
  part.url = std::string(((oai::kMaxImageContainerBytes + 2) / 3) * 4 + 257, 'A');
  CHECK_THROWS_WITH_AS(oai::DecodeImageUrlPart(part),
                      "multimodal image: encoded input exceeds byte limit",
                      vllm::v1::InputValidationError);
}

TEST_CASE("native PNG codec: malformed and unsupported media are client errors") {
  auto codec = oai::DefaultImageCodec();
  auto good = ReadFixture("rgb.png");
  auto truncated = good;
  truncated.resize(good.size() / 2);
  CHECK_THROWS_AS(codec({"image/png", truncated}), vllm::v1::InputValidationError);
  auto corrupt = good;
  corrupt[29] ^= 1;  // IHDR CRC, before any image allocation.
  CHECK_THROWS_AS(codec({"image/png", corrupt}), vllm::v1::InputValidationError);
  CHECK_THROWS_AS(codec({"image/png", {1, 2, 3}}), vllm::v1::InputValidationError);
  CHECK_THROWS_AS(codec({"image/jpeg", good}), vllm::v1::InputValidationError);
  CHECK_THROWS_AS(codec({"image/gif", good}), vllm::v1::InputValidationError);
  auto animated = good;
  const std::vector<uint8_t> actl = {0, 0, 0, 8, 'a', 'c', 'T', 'L',
                                    0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0};
  animated.insert(animated.begin() + 33, actl.begin(), actl.end());
  CHECK_THROWS_WITH_AS(codec({"image/png", animated}),
                      "multimodal image: animated PNG is unsupported",
                      vllm::v1::InputValidationError);
}

TEST_CASE("native image input: malformed base64 and URLs are client errors") {
  oai::ChatContentPart part;
  part.type = "image_url";
  for (const std::string uri : {"data:image/png;base64,TWF", "https://example.com/image.png",
                                 "file:///image.png"}) {
    part.url = uri;
    CHECK_THROWS_AS(oai::DecodeImageUrlPart(part), vllm::v1::InputValidationError);
  }
}

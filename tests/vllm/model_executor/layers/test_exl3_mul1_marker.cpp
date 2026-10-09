#include <doctest/doctest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_weight_loaders.h"

namespace {
struct Fixture {
  std::vector<uint8_t> packed = std::vector<uint8_t>(8192);
  std::vector<uint8_t> scale = std::vector<uint8_t>(256);
  // Deliberately unaligned, as safetensors payloads need not align I32 loads.
  std::vector<uint8_t> multiplier = {0, 0x2d, 0xd1, 0xdc, 0x83};
  std::map<std::string, vllm::StTensor> tensors;

  Fixture() {
    const auto add = [this](const std::string& name, const std::string& dtype,
                            std::vector<int64_t> shape, const uint8_t* data, size_t bytes) {
      vllm::StTensor tensor;
      tensor.dtype = dtype;
      tensor.shape = std::move(shape);
      tensor.data = data;
      tensor.nbytes = bytes;
      tensors.emplace("p." + name, std::move(tensor));
    };
    add("trellis", "I16", {8, 8, 64}, packed.data(), packed.size());
    add("suh", "F16", {128}, scale.data(), scale.size());
    add("svh", "F16", {128}, scale.data(), scale.size());
    add("mul1", "I32", {}, multiplier.data() + 1, 4);
  }

  vllm::Exl3Weight Load() const {
    return vllm::dense_loaders::LoadExl3(
        [this](const std::string& name) -> const vllm::StTensor& { return tensors.at(name); },
        [this](const std::string& name) { return tensors.count(name) != 0; }, "p");
  }
};
}  // namespace

TEST_CASE("EXL3 mul1 accepts the exact scalar and one-element multiplier") {
  Fixture fixture;
  CHECK(fixture.Load().codebook == 2);
  fixture.tensors.at("p.mul1").shape = {1};
  CHECK(fixture.Load().codebook == 2);
}

TEST_CASE("EXL3 mul1 refuses a different multiplier before decoding") {
  Fixture fixture;
  fixture.multiplier[1] ^= 1;
  CHECK_THROWS_WITH_AS(fixture.Load(), doctest::Contains("unsupported mul1 multiplier"), std::runtime_error);
}

TEST_CASE("EXL3 mul1 refuses malformed marker storage") {
  Fixture fixture;
  auto& marker = fixture.tensors.at("p.mul1");
  SUBCASE("non-scalar matrix") { marker.shape = {1, 1}; }
  SUBCASE("empty vector") { marker.shape = {0}; }
  SUBCASE("multiple values") { marker.shape = {2}; }
  SUBCASE("short payload") { marker.nbytes = 3; }
  SUBCASE("long payload") { marker.nbytes = 8; }
  SUBCASE("null payload") { marker.data = nullptr; }
  SUBCASE("wrong dtype") { marker.dtype = "F32"; }
  CHECK_THROWS_AS(fixture.Load(), std::runtime_error);
}

TEST_CASE("EXL3 mul1 refuses a second codebook marker") {
  Fixture fixture;
  fixture.tensors.emplace("p.mcg", fixture.tensors.at("p.mul1"));
  CHECK_THROWS_WITH_AS(fixture.Load(), doctest::Contains("BOTH"), std::runtime_error);
}

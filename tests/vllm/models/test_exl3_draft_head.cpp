#include <doctest/doctest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>

#include "vllm/model_executor/models/qwen3_5_mtp.h"

namespace {
using vllm::OwnedTensor;

OwnedTensor Own(vt::DType dtype, std::initializer_list<int64_t> shape) {
  OwnedTensor out;
  out.dtype = dtype;
  out.rank = static_cast<int>(shape.size());
  size_t bytes = vt::SizeOf(dtype);
  size_t dim = 0;
  for (const auto n : shape) { out.shape[dim++] = n; bytes *= n; }
  out.bytes.resize(bytes);
  return out;
}

vllm::Exl3Weight Head() {
  vllm::Exl3Weight out;
  out.name = "lm_head";
  out.codebook = 2;
  out.trellis = Own(vt::DType::kI8, {8, 15520, 192});  // K128, full N248320, 6bpw
  out.suh = Own(vt::DType::kF16, {128});
  out.svh = Own(vt::DType::kF16, {248320});
  for (size_t i = 0; i < out.trellis.bytes.size(); ++i)
    out.trellis.bytes.data()[i] = static_cast<uint8_t>((i * 13 + i / 192 + i / (15520 * 192)) % 251);
  for (size_t i = 0; i < out.svh.bytes.size(); ++i)
    out.svh.bytes.data()[i] = static_cast<uint8_t>((i * 7 + i / 256) % 251);
  std::fill(out.suh.bytes.begin(), out.suh.bytes.end(), 0x38);
  return out;
}

nlohmann::json Subset() {
  std::vector<int32_t> blocks(512);
  std::iota(blocks.begin(), blocks.end(), 0);
  std::reverse(blocks.begin(), blocks.end());  // preserve order; never sort storage
  blocks.front() = 1939;
  return { {"block_size", 128}, {"n_blocks", 512}, {"tokens", 65536}, {"blocks", blocks} };
}

void CheckSelectedBytes(const vllm::Exl3Weight& target, const nlohmann::json& subset) {
  const auto compact = vllm::BuildExl3DraftHead(target, subset, 248320);
  CHECK(compact.target_vocab == 248320);
  CHECK(compact.weight.InFeatures() == 128);
  CHECK(compact.weight.OutFeatures() == 65536);
  CHECK(compact.weight.Bits() == 6);
  CHECK(compact.weight.codebook == 2);
  CHECK(compact.weight.trellis.bytes.size() == 128 * 65536 * 6 / 8);
  CHECK(compact.weight.suh.bytes == target.suh.bytes);
  CHECK(compact.token_ids.dtype == vt::DType::kI32);
  REQUIRE(compact.token_ids.bytes.size() == 65536 * sizeof(int32_t));
  bool exact_packed = true, exact_signs = true, exact_ids = true;
  for (size_t b = 0; b < 512; ++b) {
    const size_t source = subset.at("blocks").at(b).get<size_t>();
    for (size_t row = 0; row < 8; ++row)
      exact_packed &= std::memcmp(compact.weight.trellis.bytes.data() + (row * 512 + b) * 1536,
          target.trellis.bytes.data() + (row * 1940 + source) * 1536, 1536) == 0;
    exact_signs &= std::memcmp(compact.weight.svh.bytes.data() + b * 256,
                              target.svh.bytes.data() + source * 256, 256) == 0;
    for (size_t lane = 0; lane < 128; ++lane) {
      int32_t id;
      std::memcpy(&id, compact.token_ids.bytes.data() + (b * 128 + lane) * 4, 4);
      exact_ids &= id == static_cast<int32_t>(source * 128 + lane);
    }
  }
  CHECK(exact_packed);
  CHECK(exact_signs);
  CHECK(exact_ids);
  CHECK(target.OutFeatures() == 248320);
  CHECK(target.trellis.bytes.size() == 128 * 248320 * 6 / 8);
}
}  // namespace

TEST_CASE("EXL3 compact draft retains complete packed blocks and global-ID order") {
  const auto target = Head();
  CheckSelectedBytes(target, Subset());
}

TEST_CASE("EXL3 compact draft validates every map before allocating a head") {
  auto target = Head();
  auto spec = Subset();
  SUBCASE("wrong count") { spec["blocks"].erase(0); }
  SUBCASE("duplicate") { spec["blocks"][511] = spec["blocks"][510]; }
  SUBCASE("negative") { spec["blocks"][0] = -1; }
  SUBCASE("outside vocabulary") { spec["blocks"][0] = 1940; }
  SUBCASE("huge unsigned") { spec["blocks"][0] = UINT64_MAX; }
  SUBCASE("fraction") { spec["blocks"][0] = 1.0; }
  SUBCASE("boolean") { spec["blocks"][0] = true; }
  SUBCASE("wrong block size") { spec["block_size"] = 64; }
  SUBCASE("wrong token count") { spec["tokens"] = 65535; }
  SUBCASE("wrong declared block count") { spec["n_blocks"] = 511; }
  SUBCASE("wrong codebook") { target.codebook = 1; }
  SUBCASE("retired packed source") { target.trellis.bytes.Reset(); }
  CHECK_THROWS_AS(vllm::BuildExl3DraftHead(target, spec, 248320), std::runtime_error);
}

TEST_CASE("EXL3 compact draft uses the actual pinned production subset"
          * doctest::skip(std::getenv("VT_EXL3_DRAFT_VOCAB") == nullptr)) {
  std::ifstream file(std::getenv("VT_EXL3_DRAFT_VOCAB"));
  REQUIRE(file.good());
  nlohmann::json subset;
  file >> subset;
  REQUIRE(subset.at("blocks").size() == 512);
  CHECK(subset.at("blocks").front() == 0);
  CHECK(subset.at("blocks").back() == 1938);
  // Historical corpus metadata is not a runtime logit mask: the producer
  // selects whole blocks then slices to the target config's actual vocabulary.
  CHECK(subset.at("vocab") == 248077);
  CheckSelectedBytes(Head(), subset);
}

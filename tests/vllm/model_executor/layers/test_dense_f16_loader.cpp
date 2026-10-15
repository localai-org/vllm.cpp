#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <map>
#include <vector>

#include "vllm/model_executor/models/dense_weight_loaders.h"

namespace {
struct Fixture {
  std::shared_ptr<std::vector<uint8_t>> owner;
  vllm::StTensor tensor;

  template <typename T>
  Fixture(const char* dtype, const std::vector<T>& values)
      : owner(std::make_shared<std::vector<uint8_t>>(1 + values.size() * sizeof(T))) {
    std::memcpy(owner->data() + 1, values.data(), values.size() * sizeof(T));
    tensor.dtype = dtype;
    tensor.shape = {static_cast<int64_t>(values.size())};
    tensor.data = owner->data() + 1;  // intentionally unaligned
    tensor.nbytes = values.size() * sizeof(T);
  }

  vllm::OwnedTensor Load(const std::vector<int64_t>& shape = {}) const {
    return vllm::dense_loaders::LoadF16Direct(
        [this](const std::string&) -> const vllm::StTensor& { return tensor; }, "weight", shape);
  }
};
}  // namespace

TEST_CASE("FP16 loader retains half bits without BF16 rounding") {
  const std::vector<uint16_t> bits{0x3c01, 0xbc01, 0x0001, 0x8000};
  Fixture fixture("F16", bits);
  const auto loaded = fixture.Load({2, 2});
  CHECK(loaded.dtype == vt::DType::kF16);
  CHECK(loaded.shape[0] == 2);
  CHECK(loaded.shape[1] == 2);
  CHECK(std::memcmp(loaded.bytes.data(), bits.data(), loaded.bytes.size()) == 0);
  CHECK_FALSE(loaded.bytes.borrowed());
}

TEST_CASE("FP16 loader converts BF16 and F32 directly at unaligned addresses") {
  SUBCASE("BF16") {
    Fixture fixture("BF16", std::vector<uint16_t>{0x3f80, 0xbf00, 0x3f81});
    const auto loaded = fixture.Load();
    const std::vector<uint16_t> expected{0x3c00, 0xb800, 0x3c08};
    CHECK(std::memcmp(loaded.bytes.data(), expected.data(), loaded.bytes.size()) == 0);
  }
  SUBCASE("F32 avoids BF16 intermediate") {
    Fixture fixture("F32", std::vector<float>{1.0009765625F, -1.0009765625F, 0.5F});
    const auto loaded = fixture.Load();
    const std::vector<uint16_t> expected{0x3c01, 0xbc01, 0x3800};
    CHECK(std::memcmp(loaded.bytes.data(), expected.data(), loaded.bytes.size()) == 0);
  }
}

TEST_CASE("FP16 loader mapped borrow owns its source after resolver destruction") {
  vllm::detail::SetLoadDirectUploadOverrideForTesting(true);
  struct Reset {
    ~Reset() { vllm::detail::SetLoadDirectUploadOverrideForTesting(std::nullopt); }
  } reset;
  vllm::OwnedTensor loaded;
  std::weak_ptr<const void> owner;
  const uint8_t* pointer = nullptr;
  {
    Fixture fixture("F16", std::vector<uint16_t>{0x3c01, 0xbc01});
    fixture.tensor.mapping = fixture.owner;
    owner = fixture.owner;
    pointer = fixture.tensor.data;
    loaded = fixture.Load();
    CHECK(loaded.bytes.borrowed());
    CHECK(loaded.bytes.data() == pointer);
  }
  CHECK_FALSE(owner.expired());
  CHECK(vt::LoadUnaligned<uint16_t>(loaded.bytes.data()) == 0x3c01);
  loaded.bytes.Reset();
  CHECK(owner.expired());
}

TEST_CASE("FP16 loader refuses invalid dtype, byte span and reshape before reading") {
  Fixture fixture("F16", std::vector<uint16_t>{0x3c00, 0x3c00});
  std::vector<int64_t> shape;
  SUBCASE("integer source") { fixture.tensor.dtype = "I16"; }
  SUBCASE("short payload") { fixture.tensor.nbytes = 3; }
  SUBCASE("null payload") { fixture.tensor.data = nullptr; }
  SUBCASE("zero dimension") { fixture.tensor.shape = {0}; }
  SUBCASE("negative dimension") { fixture.tensor.shape = {-1}; }
  SUBCASE("overflow") { fixture.tensor.shape = {std::numeric_limits<int64_t>::max(), 2}; }
  SUBCASE("reshape changes count") { shape = {3}; }
  SUBCASE("too many axes") { shape.assign(vt::kMaxRank + 1, 1); }
  CHECK_THROWS_AS(fixture.Load(shape), std::runtime_error);
}

TEST_CASE("FP16 merged BA preserves row order and half precision across source dtypes") {
  Fixture b("F16", std::vector<uint16_t>{0x3c01, 0xbc01});
  Fixture a("BF16", std::vector<uint16_t>{0x3f81, 0xbf00});
  b.tensor.shape = a.tensor.shape = {1, 2};
  const std::map<std::string, const vllm::StTensor*> tensors{{"b", &b.tensor}, {"a", &a.tensor}};
  const auto get = [&tensors](const std::string& name) -> const vllm::StTensor& { return *tensors.at(name); };
  const auto merged = vllm::dense_loaders::LoadMergedF16RawNK(get, {"b", "a"});
  CHECK(merged.dtype == vt::DType::kF16);
  CHECK(merged.nk);
  CHECK(merged.shape[0] == 2);
  CHECK(merged.shape[1] == 2);
  const std::vector<uint16_t> expected{0x3c01, 0xbc01, 0x3c08, 0xb800};
  CHECK(std::memcmp(merged.bytes.data(), expected.data(), merged.bytes.size()) == 0);
  a.tensor.shape = {2, 1};
  CHECK_THROWS_AS(vllm::dense_loaders::LoadMergedF16RawNK(get, {"b", "a"}), std::runtime_error);
}

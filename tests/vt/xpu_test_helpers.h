#pragma once
#include <doctest/doctest.h>
#include "vt/backend.h"
#include "vt/ops.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>

namespace xpu_test {
struct Queue {
  vt::Queue q;
  explicit Queue(vt::DeviceType type) : q(vt::CreateQueue({type, 0})) {}
  ~Queue() { vt::DestroyQueue(q); }
};
struct Buffer {
  vt::Queue& q;
  vt::Tensor tensor;
  size_t bytes;
  Buffer(vt::Queue& queue, vt::DType type, std::initializer_list<int64_t> shape) : q(queue) {
    tensor = vt::Tensor::Contiguous(nullptr, type, q.device, shape);
    bytes = tensor.Bytes();
    tensor.data = vt::Alloc(q.device, std::max(bytes, size_t{1}));
  }
  ~Buffer() { vt::Free(q.device, tensor.data); }
  Buffer(const Buffer&) = delete;
  void upload(const void* source, bool odd_address = false) {
    std::vector<unsigned char> unaligned;
    if (odd_address) {
      unaligned.resize(bytes + 1);
      std::memcpy(unaligned.data() + 1, source, bytes);
      source = unaligned.data() + 1;
    }
    auto& backend = vt::GetBackend(q.device);
    backend.Copy(q, tensor.data, source, bytes);
    backend.Synchronize(q);
  }
  std::vector<unsigned char> download() const {
    std::vector<unsigned char> result(bytes);
    auto& backend = vt::GetBackend(q.device);
    backend.Copy(q, result.data(), tensor.data, bytes);
    backend.Synchronize(q);
    return result;
  }
  void put(const std::vector<float>& values) {
    REQUIRE(values.size() * vt::SizeOf(tensor.dtype) == bytes);
    if (tensor.dtype == vt::DType::kF32) { upload(values.data()); return; }
    REQUIRE((tensor.dtype == vt::DType::kF16 || tensor.dtype == vt::DType::kBF16));
    std::vector<uint16_t> bits(values.size());
    for (size_t i = 0; i < values.size(); ++i)
      bits[i] = tensor.dtype == vt::DType::kF16 ? vt::F32ToF16(values[i]) : vt::F32ToBF16(values[i]);
    upload(bits.data());
  }
  std::vector<float> floats() const {
    auto raw = download();
    std::vector<float> values(bytes / vt::SizeOf(tensor.dtype));
    for (size_t i = 0; i < values.size(); ++i) {
      if (tensor.dtype == vt::DType::kF32) std::memcpy(&values[i], raw.data() + 4 * i, 4);
      else { uint16_t bits; std::memcpy(&bits, raw.data() + 2 * i, 2);
        values[i] = tensor.dtype == vt::DType::kF16 ? vt::F16ToF32(bits) : vt::BF16ToF32(bits); }
    }
    return values;
  }
};
inline std::vector<float> Values(size_t size, int salt = 0, float scale = 0.01f) {
  std::vector<float> result(size);
  for (size_t i = 0; i < size; ++i) result[i] = (int((i * 7 + salt) % 41) - 20) * scale;
  return result;
}
inline void Close(const std::vector<float>& actual, const std::vector<float>& expected,
                   float relative, float absolute = 1e-6f) {
  REQUIRE(actual.size() == expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
        std::abs(actual[i] - expected[i]) > absolute + relative * std::abs(expected[i])) {
      CAPTURE(i);
      CAPTURE(actual[i]);
      CAPTURE(expected[i]);
      FAIL("numerical mismatch");
    }
  }
}
inline void SameBytes(const std::vector<unsigned char>& actual,
                      const std::vector<unsigned char>& expected) {
  REQUIRE(actual.size() == expected.size());
  const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
  const auto offset = mismatch.first - actual.begin();
  CAPTURE(offset);
  REQUIRE(mismatch.first == actual.end());
}
}  // namespace xpu_test

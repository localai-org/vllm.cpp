#include <doctest/doctest.h>
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/xpu/xpu_common.h"
#include "vt/xpu/xpu_exl3.h"
#include <array>
#include <cstring>
#include <vector>

namespace {
struct DeviceBuffer {
  vt::Queue& q;
  void* data;
  DeviceBuffer(vt::Queue& queue, size_t bytes) : q(queue), data(vt::Alloc(q.device, bytes)) {}
  ~DeviceBuffer() { vt::Free(q.device, data); }
};
struct Queue {
  vt::Queue q = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  ~Queue() { vt::DestroyQueue(q); }
};
}

TEST_CASE("XPU EXL3 decoder: all 65536 codewords for mul1, MCG and 3INST") {
  Queue owner; auto& q = owner.q;
  DeviceBuffer output(q, 65536 * sizeof(float));
  auto* ptr = static_cast<float*>(output.data);
  std::vector<float> actual(65536);
  for (int cb : {2, 0, 1}) {
    vt::xpu::NativeQueue(q).parallel_for(sycl::range<1>(65536), [=](sycl::id<1> id) {
      ptr[id[0]] = vt::xpu::exl3::Decode(static_cast<uint16_t>(id[0]), cb);
    });
    vt::GetBackend(q.device).Copy(q, actual.data(), output.data, actual.size() * sizeof(float));
    vt::GetBackend(q.device).Synchronize(q);
    for (int i = 0; i < 65536; ++i) {
      CAPTURE(cb);
      CAPTURE(i);
      const float expected = vt::Exl3DecodeCodeword(static_cast<uint16_t>(i), cb);
      REQUIRE(std::memcmp(&actual[i], &expected, sizeof(float)) == 0);
    }
  }
}

TEST_CASE("XPU EXL3 bit windows and permutation: bits 1-8, wrap, odd-byte upload") {
  Queue owner; auto& q = owner.q;
  for (int bits = 1; bits <= 8; ++bits) {
    CAPTURE(bits);
    const size_t bytes = 32 * bits;
    // Pack an independently generated bitstream through its uint32 view.
    // The last 'bits' bits in each codeword are the bits newly emitted for t.
    std::vector<unsigned char> host(bytes + 1, 0);
    auto* tile = host.data() + 1;
    std::vector<unsigned> stream(bytes * 8);
    uint32_t seed = 0x5a47bb01u;
    for (size_t i = 0; i < stream.size(); ++i) {
      seed = seed * 1664525u + 1013904223u;
      stream[i] = seed >> 31;
      const size_t word = i / 32, shift = 31 - i % 32;
      tile[word * 4 + shift / 8] |= stream[i] << (shift % 8);
    }
    DeviceBuffer input(q, bytes), output(q, 256 * sizeof(uint16_t)), permutation(q, 512 * sizeof(int));
    auto* src = static_cast<const unsigned char*>(input.data);
    auto* dst = static_cast<uint16_t*>(output.data);
    auto* perm = static_cast<int*>(permutation.data);
    vt::GetBackend(q.device).Copy(q, input.data, tile, bytes);
    vt::xpu::NativeQueue(q).parallel_for(sycl::range<1>(256), [=](sycl::id<1> id) {
      const int t = id[0];
      dst[t] = vt::xpu::exl3::Codeword(src, bits, t);
      const int logical = vt::xpu::exl3::RowMajor(t);
      perm[t] = logical;
      perm[256 + t] = vt::xpu::exl3::Fragment(logical / 16, logical % 16);
    });
    std::array<uint16_t, 256> actual{};
    std::array<int, 512> mapping{};
    vt::GetBackend(q.device).Copy(q, actual.data(), output.data, sizeof(actual));
    vt::GetBackend(q.device).Copy(q, mapping.data(), permutation.data, sizeof(mapping));
    vt::GetBackend(q.device).Synchronize(q);
    for (int t = 0; t < 256; ++t) {
      CAPTURE(t);
      unsigned expected = 0;
      for (int j = 0; j < 16; ++j)
        expected = (expected << 1) | stream[(t * bits + bits - 16 + j + stream.size()) % stream.size()];
      REQUIRE(actual[t] == expected);
      REQUIRE(actual[t] == vt::Exl3TileCodeword(tile, bits, t));
      REQUIRE(mapping[t] == vt::Exl3TileRowMajorIndex(t));
      REQUIRE(mapping[256 + t] == t);
    }
  }
}

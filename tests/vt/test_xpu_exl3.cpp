#include "exl3_fixture.h"
#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/xpu/xpu_exl3_strategy.h"
#include <array>
#include <cstring>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;
using xpu_test::SameBytes;

std::vector<uint16_t> Input(int64_t elements) {
  exl3_test::Rng rng;
  std::vector<uint16_t> result(elements);
  for (auto& v : result) v = vt::F32ToF16(rng.next(0.15f));
  return result;
}
std::vector<unsigned char> Gemm(vt::Queue& q, const exl3_test::Exl3Fixture& fixture,
                                int rows, DType output, bool alias, int cb,
                                bool unaligned = false, bool output_alias = false) {
  const auto k = fixture.k, n = fixture.n;
  Buffer a(q, DType::kF16, {rows, k}), ah(q, DType::kF16, {rows, k});
  const size_t packed_size = fixture.trellis.size() * sizeof(fixture.trellis[0]);
  Buffer b(q, DType::kI8, {int64_t(packed_size) + int(unaligned)});
  Buffer suh(q, DType::kF16, {k}), svh(q, DType::kF16, {n});
  Buffer c(q, output, {rows, n});
  a.upload(Input(rows * k).data());
  std::vector<unsigned char> packed_bytes(packed_size + int(unaligned));
  std::memcpy(packed_bytes.data() + int(unaligned), fixture.trellis.data(), packed_size);
  b.upload(packed_bytes.data(), true);
  auto packed = vt::Tensor::Contiguous(static_cast<unsigned char*>(b.tensor.data) + int(unaligned),
      DType::kI8, q.device, {k / 16, n / 16, 32 * fixture.bits});
  suh.upload(fixture.suh.data(), true); svh.upload(fixture.svh.data(), true);
  auto& scratch = alias ? a.tensor : ah.tensor;
  auto result_tensor = c.tensor;
  if (output_alias) {
    REQUIRE(c.bytes <= scratch.Bytes());
    result_tensor.data = scratch.data;
  }
  vt::Exl3Gemm(q, result_tensor, a.tensor, packed, suh.tensor, svh.tensor, scratch,
               vt::Exl3GemmArgs{fixture.bits, cb});
  std::vector<unsigned char> result(c.bytes);
  vt::GetBackend(q.device).Copy(q, result.data(), result_tensor.data, result.size());
  vt::GetBackend(q.device).Synchronize(q);
  if (output == DType::kF32) {
    Buffer bf(q, DType::kBF16, {rows, n});
    vt::CastBf16(q, bf.tensor, result_tensor);
    const auto converted = bf.download();
    result.insert(result.end(), converted.begin(), converted.end());
    return result;
  }
  return result;
}
}

TEST_CASE("XPU EXL3 strategy: cache identity, measured regimes and bounded growth") {
  using namespace vt::xpu::exl3;
  const StrategyDomain domain{57891, "1.17.39758+10", "1.17",
      "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)", kKernelVersion};
  StrategyCache cache;
  const Shape decode{3, 17408, 5120, 1, DType::kF32};
  CHECK(cache.Get(domain, decode) == Strategy::kFused);
  CHECK(cache.Get(domain, decode) == Strategy::kFused);
  CHECK(cache.Size() == 1);
  // Head6 is selected separately: small-batch fusion won, M=1 did not.
  CHECK(cache.Get(domain, {6, 5120, 248320, 1, DType::kF32}) == Strategy::kPacked);
  CHECK(cache.Get(domain, {6, 5120, 248320, 5, DType::kF32}) == Strategy::kFused);
  for (int m : {128, 129, 512, 2048, 6656}) {
    CHECK(cache.Get(domain, {3, 17408, 5120, m, DType::kF32}) ==
          (m >= 512 ? Strategy::kPrefillAllRows : Strategy::kPrefill));
    CHECK(cache.Get(domain, {6, 5120, 248320, m, DType::kF32}) == Strategy::kPrefill);
  }
  for (const Shape unknown : {Shape{3, 17408, 5120, 17, DType::kF32},
       Shape{3, 17408, 5120, 127, DType::kF32}, Shape{3, 17408, 5120, 6657, DType::kF32},
       Shape{3, 17408, 5120, 128, DType::kF16},
       Shape{3, 17408, 5120, 1, DType::kF16}, Shape{3, 17408, 256, 1, DType::kF32},
       Shape{7, 17408, 5120, 1, DType::kF32}})
    CHECK(cache.Get(domain, unknown) == Strategy::kPacked);
  for (int field = 0; field < 5; ++field) {
    auto changed = domain;
    if (field == 0) changed.device_id++;
    if (field == 1) changed.driver += "+new";
    if (field == 2) changed.runtime += "+new";
    if (field == 3) changed.compiler += "+new";
    if (field == 4) changed.kernel += "+new";
    CHECK(cache.Get(changed, decode) == Strategy::kPacked);
    CHECK(cache.Get(changed, {3, 17408, 5120, 128, DType::kF32}) == Strategy::kPacked);
  }
  for (int64_t m = 21; m < 600; ++m) {
    CHECK(cache.Get(domain, {3, 17408, 5120, m, DType::kF32}) ==
          (m >= 512 ? Strategy::kPrefillAllRows :
           m >= 128 ? Strategy::kPrefill : Strategy::kPacked));
    CHECK(cache.Size() <= 512);
  }
  CHECK(cache.Get(domain, decode) == Strategy::kFused);
}

TEST_CASE("XPU EXL3 Had128: F16/F32, pre/post scales, in-place, model width") {
  Queue host(vt::DeviceType::kCPU), device(vt::DeviceType::kXPU);
  for (auto dtype : {DType::kF16, DType::kF32}) for (int rows : {1, 3})
    for (int cols : {128, 256, 5120}) for (int mode = 0; mode < 3; ++mode)
      for (bool alias : {false, true}) {
        CAPTURE(dtype);
        CAPTURE(rows);
        CAPTURE(cols);
        CAPTURE(mode);
        CAPTURE(alias);
        const auto input = Input(rows * cols);
        std::vector<float> input32(input.size());
        for (size_t i = 0; i < input.size(); ++i) input32[i] = vt::F16ToF32(input[i]);
        std::vector<uint16_t> scales(cols);
        for (int i = 0; i < cols; ++i) scales[i] = vt::F32ToF16(i % 3 ? 0.513f : -0.739f);
        std::vector<unsigned char> expected;
        for (auto* q : {&host.q, &device.q}) {
          Buffer in(*q, dtype, {rows, cols}), out(*q, dtype, {rows, cols}), sc(*q, DType::kF16, {cols});
          in.upload(dtype == DType::kF16 ? static_cast<const void*>(input.data()) : input32.data());
          sc.upload(scales.data(), true);
          vt::Exl3HadArgs args{mode == 1 ? &sc.tensor : nullptr, mode == 2 ? &sc.tensor : nullptr, 0.73f};
          vt::Exl3HadR128(*q, alias ? in.tensor : out.tensor, in.tensor, args);
          auto actual = (alias ? in : out).download();
          if (q == &host.q) expected = actual; else SameBytes(actual, expected);
        }
      }
}

TEST_CASE("XPU EXL3 GEMM: bits 1-8, tail M, scratch alias, F16 and F32 to BF16") {
  Queue host(vt::DeviceType::kCPU), device(vt::DeviceType::kXPU);
  for (int bits = 1; bits <= 8; ++bits) for (int rows : {1, 2, 4, 5, 8, 16, 17, 20})
    for (auto dtype : {DType::kF16, DType::kF32}) {
      CAPTURE(bits);
      CAPTURE(rows);
      CAPTURE(dtype);
      auto fixture = exl3_test::MakeFixture(256, 384, bits, 0x542113u + bits);
      const auto expected = Gemm(host.q, fixture, rows, dtype, false, 2);
      SameBytes(Gemm(device.q, fixture, rows, dtype, rows % 2 != 0, 2), expected);
    }
  for (int cb : {0, 1}) {
    CAPTURE(cb);
    auto fixture = exl3_test::MakeFixture(128, 128, 4, 0x673523u);
    SameBytes(Gemm(device.q, fixture, 3, DType::kF32, true, cb),
              Gemm(host.q, fixture, 3, DType::kF32, false, cb));
  }
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, device.q.device.type));
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == vt::xpu::GetMemoryInfo().exl3_workspace_bytes);
}

TEST_CASE("XPU EXL3 GEMM: unaligned packed weights and output overlapping input scratch") {
  Queue host(vt::DeviceType::kCPU), device(vt::DeviceType::kXPU);
  for (int bits : {3, 4, 5, 6}) for (auto dtype : {DType::kF16, DType::kF32})
    for (bool unaligned : {false, true}) for (bool output_alias : {false, true}) {
      CAPTURE(bits);
      CAPTURE(dtype);
      CAPTURE(unaligned);
      CAPTURE(output_alias);
      auto fixture = exl3_test::MakeFixture(512, 256, bits, 0x125719u);
      SameBytes(Gemm(device.q, fixture, 5, dtype, true, 2, unaligned, output_alias),
                Gemm(host.q, fixture, 5, dtype, true, 2, unaligned, output_alias));
    }
}

TEST_CASE("XPU EXL3 cast fusion: explicit F16 input and F32 to BF16 output rounding is preserved") {
  Queue gpu(vt::DeviceType::kXPU);
  for (int bits : {3, 4, 5, 6}) for (int rows : {1, 5, 129})
    for (auto input_type : {DType::kF16, DType::kBF16, DType::kF32})
      for (auto output_type : {DType::kF16, DType::kBF16, DType::kF32}) {
        CAPTURE(bits);
        CAPTURE(rows);
        CAPTURE(input_type);
        CAPTURE(output_type);
        constexpr int k = 256, n = 640;
        const auto fixture = exl3_test::MakeFixture(k, n, bits, 0x421903u + bits);
        Buffer in(gpu.q, input_type, {rows, k}), half(gpu.q, DType::kF16, {rows, k});
        Buffer had(gpu.q, DType::kF16, {rows, k}), fused_had(gpu.q, DType::kF16, {rows, k});
        Buffer packed(gpu.q, DType::kI8, {k / 16, n / 16, bits * 32});
        Buffer suh(gpu.q, DType::kF16, {k}), svh(gpu.q, DType::kF16, {n});
        Buffer raw(gpu.q, output_type == DType::kF16 ? DType::kF16 : DType::kF32, {rows, n});
        Buffer reference(gpu.q, output_type, {rows, n}), fused(gpu.q, output_type, {rows, n});
        auto values = xpu_test::Values(rows * k, 7, .17f);
        values[0] = -0.f; values[1] = 0x1p-24f; values[2] = 0x1.001p0f;
        values[3] = -0x1.003p0f; values[4] = 0x1p-25f;
        in.put(values); packed.upload(fixture.trellis.data());
        suh.upload(fixture.suh.data()); svh.upload(fixture.svh.data());
        if (input_type == DType::kF16) vt::Copy(gpu.q, half.tensor, in.tensor);
        else vt::CastF16(gpu.q, half.tensor, in.tensor);
        vt::Exl3Gemm(gpu.q, raw.tensor, half.tensor, packed.tensor, suh.tensor, svh.tensor, had.tensor, {bits, 2});
        if (output_type == DType::kBF16) vt::CastBf16(gpu.q, reference.tensor, raw.tensor);
        else vt::Copy(gpu.q, reference.tensor, raw.tensor);
        vt::Exl3GemmArgs args{bits, 2}; args.fuse_casts = true;
        vt::Exl3Gemm(gpu.q, fused.tensor, in.tensor, packed.tensor, suh.tensor, svh.tensor, fused_had.tensor, args);
        SameBytes(fused_had.download(), had.download());
        SameBytes(fused.download(), reference.download());
      }
  CHECK(vt::GetReferenceTierHits() == 0);
}

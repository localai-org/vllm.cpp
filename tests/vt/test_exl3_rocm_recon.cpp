// The ROCm arm of OpId::kExl3ReconstructGemm — spec
// .agents/specs/rocm-exl3-gemv.md (Slice B), row BACKEND-ROCM.
//
// THE BOUND IS TIER 3 (1.0e-3 relative RMS), matching the CUDA arm's unfused
// case in test_exl3_gemm.cpp, because the arithmetic is the same chain:
// had_r_128(f16, suh) → reconstruct to f16 → f32-accumulate GEMM →
// had_r_128(svh). Two differences from the CUDA arm the suite names:
//
//   WIDTHS. The scalar decoder serves every width upstream defines (bits 1..8
//   over all three codebooks), so this arm's cases are wider than the CUDA
//   arm's seven-pair instantiation — (4,0), the stock turboderp body width
//   that checkpoint is quantized at, is gated here directly.
//
//   M >= 1024. The CUDA arm switches to its FUSED kernel there; this arm has
//   no fused path (the file's header says why), so the SAME unfused chain is
//   exercised at M=1024 to prove the dispatch never degrades into a refusal.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3Fixture;
using exl3_test::Exl3ChainF64;
using exl3_test::MakeFixture;
using exl3_test::Rng;
using exl3_test::Rms;

bool HasRocmRecon() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kROCM);
    return vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kROCM);
  } catch (const std::runtime_error&) {
    return false;
  }
}

// One reconstruct+GEMM call on the ROCm device, in the persistent-scratch
// spelling a model forward uses (vt::Exl3ReconstructGemm without w_scratch).
std::vector<float> RocmRecon(vt::Backend& be, const Exl3Fixture& f,
                             const std::vector<uint16_t>& a_h, int64_t m, int64_t k,
                             int64_t n, const vt::Exl3GemmArgs& args) {
  vt::Queue dq = be.CreateQueue();
  const size_t ab = a_h.size() * sizeof(uint16_t);
  const size_t bb = f.trellis.size() * sizeof(uint16_t);
  const size_t cb_bytes = static_cast<size_t>(m * n) * sizeof(float);
  void* d_a = be.Alloc(ab);
  void* d_ah = be.Alloc(ab);
  void* d_b = be.Alloc(bb);
  void* d_suh = be.Alloc(f.suh.size() * 2);
  void* d_svh = be.Alloc(f.svh.size() * 2);
  void* d_c = be.Alloc(cb_bytes);
  be.Copy(dq, d_a, a_h.data(), ab);
  be.Copy(dq, d_b, f.trellis.data(), bb);
  be.Copy(dq, d_suh, f.suh.data(), f.suh.size() * 2);
  be.Copy(dq, d_svh, f.svh.data(), f.svh.size() * 2);

  vt::Tensor ta = vt::Tensor::Contiguous(d_a, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tb =
      vt::Tensor::Contiguous(d_b, vt::DType::kI8, dq.device, {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
  vt::Tensor tc = vt::Tensor::Contiguous(d_c, vt::DType::kF32, dq.device, {m, n});
  vt::Exl3ReconstructGemm(dq, tc, ta, tb, tsuh, tsvh, tah, args);
  be.Synchronize(dq);
  std::vector<float> out(static_cast<size_t>(m * n), 0.0f);
  be.Copy(dq, out.data(), d_c, cb_bytes);
  be.Synchronize(dq);
  for (void* p : {d_a, d_ah, d_b, d_suh, d_svh, d_c}) be.Free(p);
  be.DestroyQueue(dq);
  return out;
}

double RelRms(const std::vector<float>& got, const std::vector<double>& ref) {
  double sq = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - ref[i];
    sq += d * d;
  }
  return std::sqrt(sq / static_cast<double>(ref.size())) / Rms(ref);
}

}  // namespace

// M=256 is above upstream's 144 reconstruct threshold (where a model's
// dispatch selects this op) and below the CUDA arm's 1024 fused threshold, so
// the chain under test is had(suh) → reconstruct → hipBLAS → had(svh).
TEST_CASE("exl3 rocm recon: unfused arm matches the f64 reference at tier 3") {
  if (!HasRocmRecon()) {
    MESSAGE(
        "SKIPPED, no ROCm device: rocm-exl3-gemv slice B parity is PENDING. Reproduce "
        "with: ctest --test-dir build-hip -R test_exl3_rocm_recon -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);

  struct Arm {
    int bits;
    int codebook;
    std::string what;
  };
  const Arm arms[] = {
      {3, 0, "a stock exl3 body"},
      {4, 0, "the turboderp 4bpw body (this row's checkpoint)"},
      {6, 0, "a stock exl3 lm_head"},
      {3, 1, "the DeepSeek-V4 artifact"},
      {4, 1, "a 4-bit mcg tower"},
      {3, 2, "the Qwen3.8-27B mul1 MLP"},
      {4, 2, "the Qwen3.8-27B mul1 GDN tower"},
      {5, 2, "the Qwen3.8-27B mul1 5-bit tensor"},
      {6, 2, "the Qwen3.8-27B mul1 lm_head"},
  };

  const int64_t m = 256, k = 256, n = 256;
  for (const Arm& arm : arms) {
    CAPTURE(arm.bits);
    CAPTURE(arm.codebook);
    const Exl3Fixture f = MakeFixture(k, n, arm.bits, 0x1D0C0DEu + arm.bits);
    Rng rng;
    rng.s = 0xC0FFEEu + arm.bits;
    std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
    std::vector<float> a_f(static_cast<size_t>(m * k));
    for (size_t i = 0; i < a_h.size(); ++i) {
      a_h[i] = vt::F32ToF16(rng.next(1.0f));
      a_f[i] = vt::F16ToF32(a_h[i]);
    }
    vt::Exl3GemmArgs args;
    args.bits = arm.bits;
    args.codebook = arm.codebook;
    const std::vector<float> got = RocmRecon(be, f, a_h, m, k, n, args);
    const std::vector<double> ref = Exl3ChainF64(f, a_f, m, arm.codebook);
    REQUIRE(Rms(ref) > 0.0);
    const double rel = RelRms(got, ref);
    MESSAGE("reconstruct+hipBLAS bits ", arm.bits, " cb ", arm.codebook, " (", arm.what,
            "): rel_rms = ", rel);
    CHECK(rel <= 1.0e-3);
  }
}

// M=145 is upstream's exact dispatch boundary + 1 (exl3.py:10,135
// AUTO_RECONSTRUCT_THRESHOLD=144): the smallest M a model routes here, and not
// a Hadamard-friendly power of two. Spec gate 5 names it.
TEST_CASE("exl3 rocm recon: m=145, one row past the dispatch threshold") {
  if (!HasRocmRecon()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  const int64_t m = 145, k = 256, n = 256;
  const Exl3Fixture f = MakeFixture(k, n, 4, 0x51011CE5u);
  Rng rng;
  rng.s = 0xB0ADC0DEu;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  std::vector<float> a_f(static_cast<size_t>(m * k));
  for (size_t i = 0; i < a_h.size(); ++i) {
    a_h[i] = vt::F32ToF16(rng.next(1.0f));
    a_f[i] = vt::F16ToF32(a_h[i]);
  }
  vt::Exl3GemmArgs args;
  args.bits = 4;
  args.codebook = 0;
  const std::vector<float> got = RocmRecon(be, f, a_h, m, k, n, args);
  const std::vector<double> ref = Exl3ChainF64(f, a_f, m, 0);
  REQUIRE(Rms(ref) > 0.0);
  const double rel = RelRms(got, ref);
  MESSAGE("reconstruct+hipBLAS m=145 (4,0): rel_rms = ", rel);
  CHECK(rel <= 1.0e-3);
}

// The CUDA arm switches to its FUSED kernel at M>=1024; this arm has only the
// unfused chain, so the SAME path is exercised there — the point is that the
// dispatch stays correct (and below tier 3), not that it found a fast kernel.
TEST_CASE("exl3 rocm recon: M=1024 stays correct on the unfused path") {
  if (!HasRocmRecon()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  const int64_t m = 1024, k = 256, n = 256;
  const Exl3Fixture f = MakeFixture(k, n, 4, 0x27D4EB2Fu);
  Rng rng;
  rng.s = 0x165667B1u;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  std::vector<float> a_f(static_cast<size_t>(m * k));
  for (size_t i = 0; i < a_h.size(); ++i) {
    a_h[i] = vt::F32ToF16(rng.next(1.0f));
    a_f[i] = vt::F16ToF32(a_h[i]);
  }
  vt::Exl3GemmArgs args;
  args.bits = 4;
  args.codebook = 0;
  const std::vector<float> got = RocmRecon(be, f, a_h, m, k, n, args);
  const std::vector<double> ref = Exl3ChainF64(f, a_f, m, 0);
  REQUIRE(Rms(ref) > 0.0);
  const double rel = RelRms(got, ref);
  MESSAGE("reconstruct+hipBLAS M=1024 (4,0): rel_rms = ", rel);
  CHECK(rel <= 1.0e-3);
}

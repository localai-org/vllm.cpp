// The EXL3 m<=8 GEMV arm on ROCm — spec .agents/specs/rocm-exl3-gemv.md,
// row BACKEND-ROCM, issue .agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M3QJSFATGWXMMQKTVDG6BG5G.md.
//
// THE BOUND IS TIER 3c, NOT THE BYTE GATE. test_exl3_rocm.cpp pins
// Exl3GemmK BYTE-equal to the CPU arm because that kernel reproduces the
// host's accumulation ORDER. This arm accumulates in fp16 fragments and folds
// to f32 every FOLD k-tiles (exl3_gemv_kernel.cuh:37-52) — the same numeric
// arm as the CUDA GEMV, which spec quant-exl3-perf bounds at 6.0e-3 relative
// RMS. The emulated mma_m16n8k16_f16 reproduces that accumulation order
// exactly (its __hfma2 lane-pair fold IS CUDA's mma.f16.f16.f16.f16), so the
// bound transfers unchanged. `VT_ROCM_EXL3_WMMA=1` swaps in the gfx11 WMMA
// path, which carries its OWN numerics and is evaluation-only; this suite is
// written to the default arm.
//
// DISPATCH IS GATED IN BOTH DIRECTIONS, because this backend's two arms are
// distinguishable BYTE-FOR-BYTE. force_gemv = 0 selects Exl3GemmK, whose
// output is byte-identical to the CPU arm — so a decline result must be
// byte-equal to the CPU reference, and a GEMV result must not be. That is a
// sharper reach check than the CUDA suite's, whose two f32/f16 arms already
// differed incidentally.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::MakeHalfFixture;
using exl3_test::Rng;
using exl3_test::UlpF16;

bool HasRocmExl3() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kROCM);
    return vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM);
  } catch (const std::runtime_error&) {
    return false;
  }
}

std::vector<uint16_t> CpuArm(const Exl3Fixture& f, vt::Queue& hq,
                             const std::vector<uint16_t>& a, int64_t m, int codebook,
                             /*bool half=*/bool use_half = false) {
  const int64_t k = f.k, n = f.n;
  std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);
  std::vector<uint16_t> a_had_h(static_cast<size_t>(m * k), 0);
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = codebook;
  args.half = use_half;
  vt::Tensor ta = vt::Tensor::Contiguous(const_cast<uint16_t*>(a.data()), vt::DType::kF16,
                                       hq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(a_had_h.data(), vt::DType::kF16, hq.device, {m, k});
  vt::Tensor tc = vt::Tensor::Contiguous(out.data(), vt::DType::kF16, hq.device, {m, n});
  vt::Tensor tb =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.trellis.data()), vt::DType::kI8,
                             hq.device,
                             {k / 16, n / 16, 32 * f.bits + (use_half ? 16 : 0)});
  vt::Tensor tsuh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.suh.data()),
                                           vt::DType::kF16, hq.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.svh.data()),
                                           vt::DType::kF16, hq.device, {n});
  vt::Exl3Gemm(hq, tc, ta, tb, tsuh, tsvh, tah, args);
  return out;
}

std::vector<uint16_t> RocmArm(vt::Backend& be, const Exl3Fixture& f,
                              const std::vector<uint16_t>& a, int64_t m, int codebook,
                              int force_gemv, /*bool half=*/bool use_half = false) {
  vt::Queue dq = be.CreateQueue();
  const int64_t k = f.k, n = f.n;
  const size_t ab = a.size() * sizeof(uint16_t);
  const size_t bb = f.trellis.size() * sizeof(uint16_t);
  std::vector<uint16_t> out(static_cast<size_t>(m * n), 0);

  void* d_a = be.Alloc(ab);
  void* d_ah = be.Alloc(ab);
  void* d_b = be.Alloc(bb);
  void* d_suh = be.Alloc(f.suh.size() * 2);
  void* d_svh = be.Alloc(f.svh.size() * 2);
  void* d_c = be.Alloc(out.size() * 2);
  be.Copy(dq, d_a, a.data(), ab);
  be.Copy(dq, d_b, f.trellis.data(), bb);
  be.Copy(dq, d_suh, f.suh.data(), f.suh.size() * 2);
  be.Copy(dq, d_svh, f.svh.data(), f.svh.size() * 2);

  vt::Tensor ta = vt::Tensor::Contiguous(d_a, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tb = vt::Tensor::Contiguous(
      d_b, vt::DType::kI8, dq.device, {k / 16, n / 16, 32 * f.bits + (use_half ? 16 : 0)});
  vt::Tensor tsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
  vt::Tensor tc = vt::Tensor::Contiguous(d_c, vt::DType::kF16, dq.device, {m, n});
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = codebook;
  args.half = use_half;
  args.force_gemv = force_gemv;
  vt::Exl3Gemm(dq, tc, ta, tb, tsuh, tsvh, tah, args);
  be.Synchronize(dq);
  be.Copy(dq, out.data(), d_c, out.size() * 2);
  be.Synchronize(dq);
  be.Free(d_a);
  be.Free(d_ah);
  be.Free(d_c);
  be.Free(d_b);
  be.Free(d_suh);
  be.Free(d_svh);
  be.DestroyQueue(dq);
  return out;
}

}  // namespace

// Every instantiated arm meets tier 3c against the CPU arm, and every arm is
// FAR from the same bits decoded under a different codebook — the
// discrimination check that keeps a mis-threaded `cb` from reporting green.
// Upstream's selector table is instantiated: 4 bpw all three codebooks, 3 bpw
// cb 1/2; cb 0 at 3 bpw is refused upstream (exl3_gemv.cu:113) and here.
TEST_CASE("exl3 rocm gemv: every instantiated arm meets tier 3c") {
  if (!HasRocmExl3()) {
    MESSAGE(
        "SKIPPED, no ROCm device: rocm-exl3-gemv's tier-3c bound is PENDING. Reproduce with: "
        "ctest --test-dir build-hip -R test_exl3_rocm_gemv -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  struct Arm {
    int bits, cb;
    int64_t k, n, m;
    std::string what;
  };
  // Shapes: k=2048 n=4096 resolves the envelope to cfg 0 on every cc bucket
  // (the `size_k <= 2048 && size_n <= 8192` branch), n=8320 resolves mode 2 to
  // cfg 1 (wide). m=1 exercises mmode 0, m=8 mmode 1 — different compiled
  // kernels (ROWS 1 vs 8), so both get a run.
  const Arm kArms[] = {
      {4, 0, 2048, 4096, 1, "(4,0) narrow m=1"},  // the stock turboderp arm
      {4, 1, 2048, 4096, 1, "(4,1) narrow m=1"},
      {4, 2, 2048, 4096, 1, "(4,2) narrow m=1"},
      {3, 1, 2048, 4096, 1, "(3,1) narrow m=1"},
      {3, 2, 2048, 4096, 1, "(3,2) narrow m=1"},
      {4, 2, 2048, 8320, 1, "(4,2) wide m=1"},
      // Wide-arm boundaries: n=8192 sits exactly on the narrow/wide envelope
      // cut (size_k <= 2048 && size_n <= 8192 resolves cfg 0); n=8064 is the
      // largest narrow n.
      {4, 2, 2048, 8192, 1, "(4,2) n=8192 boundary m=1"},
      {4, 2, 2048, 8064, 1, "(4,2) n=8064 narrow edge m=1"},
      // k=8960 = 8*1024 + 768: a non-1024-multiple k feeds the K-tail path
      // both m=1 arms take past the main unrolled stride.
      {4, 2, 8960, 4096, 1, "(4,2) k-remainder m=1"},
      {3, 2, 8960, 4096, 1, "(3,2) k-remainder m=1"},
      // Qwen3.8-27B serving shapes: n/cols past the co-resident cap (grid
      // policy) and the long-k down projection.
      {3, 2, 5120, 17408, 1, "(3,2) gate/up shape m=1"},
      {3, 2, 17408, 5120, 1, "(3,2) down shape m=1"},
      {4, 2, 5120, 10240, 1, "(4,2) in_proj_qkv shape m=1"},
      // n >= 12288 at 4 bpw: the shapes the opt-in fused in-had arm
      // (Exl3GemvM1K4Fused, VT_EXL3_FUSED_HAD=1 — the
      // test_exl3_rocm_gemv_fused ctest entry) serves.
      {4, 2, 5120, 12288, 1, "(4,2) attn_q shape m=1"},
      {4, 2, 5120, 17408, 1, "(4,2) gate/up shape m=1"},
      {4, 0, 2048, 4096, 8, "(4,0) narrow m=8"},
      {3, 2, 2048, 4096, 8, "(3,2) narrow m=8"},  // lands on the batched Exl3GemvMK3 arm
      {4, 0, 2048, 4096, 2, "(4,0) narrow m=2"},   // smallest batched MK3 m
      {4, 0, 2048, 4096, 5, "(4,0) narrow m=5"},   // odd m inside the MK3 rows range
  };

  for (const Arm& arm : kArms) {
    CAPTURE(arm.what);
    CAPTURE(arm.bits);
    CAPTURE(arm.cb);
    const int64_t k = arm.k, n = arm.n, m = arm.m;
    Exl3Fixture f = MakeFixture(k, n, arm.bits, 0x5EEDu);
    std::vector<uint16_t> a(static_cast<size_t>(m * k));
    Rng rng;
    for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

    const int sibling_cb = arm.cb == 2 ? 1 : 2;
    const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, arm.cb);
    const std::vector<uint16_t> sib = CpuArm(f, hq, a, m, sibling_cb);

    // FORCED — force_gemv=1, upstream's own direct-entry lever, so a device
    // whose heuristic declined cannot report a transcription result as this
    // arm's.
    const std::vector<uint16_t> got = RocmArm(be, f, a, m, arm.cb, /*force_gemv=*/1);

    double sq = 0.0, rq = 0.0, worst = 0.0, sq_sib = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double r = vt::F16ToF32(ref[i]);
      const double s = vt::F16ToF32(sib[i]);
      const double g = vt::F16ToF32(got[i]);
      sq += (g - r) * (g - r);
      sq_sib += (g - s) * (g - s);
      rq += r * r;
      worst = std::max(worst, std::fabs(g - r));
    }
    const double rms_ref = std::sqrt(rq / static_cast<double>(got.size()));
    const double rel = std::sqrt(sq / static_cast<double>(got.size())) / rms_ref;
    const double rel_sib = std::sqrt(sq_sib / static_cast<double>(got.size())) / rms_ref;
    MESSAGE(arm.what, " tier 3c: relative RMS ", rel, ", worst elementwise ", worst,
            ", relative RMS against codebook ", sibling_cb, " ", rel_sib);
    CHECK(rel <= 6.0e-3);
    CHECK(worst <= 64.0 * UlpF16(rms_ref));
    CHECK(rms_ref > 0.0);
    CHECK(rel_sib > 100.0 * 6.0e-3);
  }

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

// Dispatch pinned in BOTH directions — possible because this backend's two
// arms are byte-distinguishable. force_gemv=0 is Exl3GemmK and MUST reproduce
// the CPU arm byte-for-byte; force_gemv=1 is the GEMV and MUST differ from it
// (the fp16 fold order is not the host's f32 order, so a byte-equal GEMV
// result means the arm never ran).
TEST_CASE("exl3 rocm gemv: force_gemv selects arms byte-detectably") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  Exl3Fixture f = MakeFixture(2048, 4096, 4, 0xC0DEu);
  const int64_t m = 1;
  std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
  Rng rng;
  for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

  const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, /*codebook=*/0);
  const std::vector<uint16_t> off = RocmArm(be, f, a, m, 0, /*force_gemv=*/0);
  const std::vector<uint16_t> on = RocmArm(be, f, a, m, 0, /*force_gemv=*/1);

  size_t off_equal = 0, on_diff = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (off[i] == ref[i]) ++off_equal;
    if (on[i] != ref[i]) ++on_diff;
  }
  MESSAGE("force_gemv=0 byte-equal to CPU arm: ", off_equal, " of ", ref.size());
  MESSAGE("force_gemv=1 differs from CPU arm at ", on_diff, " of ", ref.size(), " outputs");
  CHECK(off_equal == ref.size());      // the transcription is byte-exact
  // The forced arm provably ran if ANY output differs: the m1 arm accumulates
  // in f32 (v_dot2_f32_f16) and lands within fp16 rounding of the CPU
  // reference on ~99% of elements, so the old `> ref.size()/2` threshold —
  // calibrated to the fp16-fragment arm's error profile — fails on a
  // strictly-more-accurate arm. >0 is the right discriminator.
  CHECK(on_diff > 0);

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

// NOTHING LANDS DEAD. The production call leaves force_gemv at -1, so the
// envelope decides. At k=2048 n=4096 the `size_k <= 2048 && size_n <= 8192`
// branch resolves to cfg 0 on every bucket — including Exl3Cc::kAda, which is
// what Exl3GemvTryLaunchRocm passes for RDNA3 — with NO occupancy input, so a
// default-mode launch on this device must produce the GEMV arm's bytes.
TEST_CASE("exl3 rocm gemv: the default dispatch reaches the GEMV at mode 1") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);

  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAda, 1, 2048, 4096, 4, 0,
                                 /*mode=*/1, /*narrow_coresident=*/0) == 0);

  Exl3Fixture f = MakeFixture(2048, 4096, 4, 0xC0DEu);
  const int64_t m = 1;
  std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
  Rng rng;
  for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

  const std::vector<uint16_t> unforced = RocmArm(be, f, a, m, 0, /*force_gemv=*/-1);
  const std::vector<uint16_t> forced = RocmArm(be, f, a, m, 0, /*force_gemv=*/1);
  size_t same = 0;
  for (size_t i = 0; i < unforced.size(); ++i)
    if (unforced[i] == forced[i]) ++same;
  MESSAGE("(4,0) reached UNFORCED at mode 1: ", same, " of ", unforced.size(),
          " outputs byte-equal to the forced launch");
  CHECK(same == unforced.size());
}

// ─── spec rocm-exl3-gemv-residual: the three residual shapes ────────────────
//
// The rocprofv3 trace that filed ISSUE-LOCAL-01M3RM2TM98FY569AZ6CEASBD3 found
// Exl3GemmK still owned 91% of decode kernel time on this checkpoint because
// (a) the upstream selector declines k=4096/n=4096/cb1 at 4 bpw, (b) the 6 bpw
// lm_head is outside the GEMV envelope, and (c) m in (8, 144] had no arm at
// all. These cases pin each fix against a DETECTABLE wrong arm: on this
// backend the transcription is byte-equal to the CPU arm, so a GEMV or recon
// result that differs from it proves the fast path ran, and equality to the
// forced launch proves WHICH fast path ran.

TEST_CASE("exl3 rocm gemv: upstream-declined (4,1) shape takes the forced cfg") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  // The exact decline the trace recorded: k=4096 n=4096 (4,1) falls through
  // every branch of Exl3GemvSelectConfig at mode 1 (size_n/32=128 >
  // narrow_coresident, size_n < 8192) — upstream returns -1 and the CUDA arm
  // would run its tensor-core GEMM. Assert the selector STILL returns -1 (the
  // function is kept upstream-verbatim) and that the unforced launch now
  // lands the GEMV arm anyway, byte-equal to the forced one.
  CHECK(vt::Exl3GemvSelectConfig(vt::Exl3Cc::kAda, 1, 4096, 4096, 4, 1,
                                 /*mode=*/1, /*narrow_coresident=*/0) == -1);

  Exl3Fixture f = MakeFixture(4096, 4096, 4, 0x9EEDEDu);
  const int64_t m = 1;
  std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
  Rng rng;
  for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

  const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, /*codebook=*/1);
  const std::vector<uint16_t> scalar = RocmArm(be, f, a, m, 1, /*force_gemv=*/0);
  const std::vector<uint16_t> unforced = RocmArm(be, f, a, m, 1, /*force_gemv=*/-1);
  const std::vector<uint16_t> forced = RocmArm(be, f, a, m, 1, /*force_gemv=*/1);

  size_t scalar_eq = 0, unf_eq_forced = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (scalar[i] == ref[i]) ++scalar_eq;
    if (unforced[i] == forced[i]) ++unf_eq_forced;
  }
  MESSAGE("(4,1) k=4096 n=4096: scalar byte-equal to CPU ", scalar_eq, " of ", ref.size(),
          "; unforced == forced at ", unf_eq_forced);
  CHECK(scalar_eq == ref.size());      // the transcription is still byte-exact
  CHECK(unf_eq_forced == ref.size());  // the fallback cfg reached the GEMV arm

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

TEST_CASE("exl3 rocm gemv: mid-m chunked GEMV meets tier 3c") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  // m in (8, 32] exercises the 8-row chunk loop — 9 is the smallest mid-m,
  // 21 the recorded prefill shape, 32 the cap's edge. m=33 routes to the
  // reconstruct arm and is asserted NOT byte-equal to the transcription the
  // same way (recon's own suite holds its 1.0e-3 bound).
  for (const int64_t m : {9, 21, 32, 33}) {
    CAPTURE(m);
    Exl3Fixture f = MakeFixture(2048, 4096, 4, 0x51CE00u + static_cast<uint32_t>(m));
    std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
    Rng rng;
    for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

    const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, /*codebook=*/1);
    const std::vector<uint16_t> scalar = RocmArm(be, f, a, m, 1, /*force_gemv=*/0);
    const std::vector<uint16_t> got = RocmArm(be, f, a, m, 1, /*force_gemv=*/-1);

    size_t scalar_eq = 0, fast_diff = 0;
    double sq = 0.0, rq = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (scalar[i] == ref[i]) ++scalar_eq;
      if (got[i] != scalar[i]) ++fast_diff;
      const double r = vt::F16ToF32(ref[i]);
      const double g = vt::F16ToF32(got[i]);
      sq += (g - r) * (g - r);
      rq += r * r;
    }
    const double rel =
        std::sqrt(sq / static_cast<double>(ref.size())) /
        std::sqrt(rq / static_cast<double>(ref.size()));
    MESSAGE("m=", m, " (4,1): rel RMS ", rel,
            ", fast arm differs from the transcription at ", fast_diff, " of ",
            ref.size());
    CHECK(scalar_eq == ref.size());               // force_gemv=0 is still the byte arm
    CHECK(rel <= (m <= 32 ? 6.0e-3 : 1.0e-2));    // chunked dot / recon bound
    // The dot arm accumulates in f32, so it is byte-NEARER the transcription
    // than the fp16-fragment GEMV — a handful of differing outputs, not half
    // of them, is what proves a fast arm ran.
    CHECK(fast_diff > 0);
  }

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

TEST_CASE("exl3 rocm gemv: 6 bpw takes the reconstruct arm at every m") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  // The checkpoint's lm_head is 6 bpw cb 1 — outside upstream's GEMV envelope
  // entirely. The residual spec routes it to the reconstruct+hipBLAS arm
  // inside Exl3GemmKernelRocm, whose bound is 1.0e-3 rel RMS; m=1 is decode,
  // m=8 the GEMV boundary, m=21 the recorded prefill shape.
  for (const int64_t m : {1, 8, 21}) {
    CAPTURE(m);
    Exl3Fixture f = MakeFixture(256, 4096, 6, 0x6EAD0u + static_cast<uint32_t>(m));
    std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
    Rng rng;
    for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

    const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, /*codebook=*/1);
    const std::vector<uint16_t> scalar = RocmArm(be, f, a, m, 1, /*force_gemv=*/0);
    const std::vector<uint16_t> got = RocmArm(be, f, a, m, 1, /*force_gemv=*/-1);

    size_t scalar_eq = 0, fast_diff = 0;
    double sq = 0.0, rq = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (scalar[i] == ref[i]) ++scalar_eq;
      if (got[i] != scalar[i]) ++fast_diff;
      const double r = vt::F16ToF32(ref[i]);
      const double g = vt::F16ToF32(got[i]);
      sq += (g - r) * (g - r);
      rq += r * r;
    }
    const double rel =
        std::sqrt(sq / static_cast<double>(ref.size())) /
        std::sqrt(rq / static_cast<double>(ref.size()));
    MESSAGE("bits=6 cb=1 m=", m, ": rel RMS ", rel,
            ", dot arm differs from the transcription at ", fast_diff, " of ",
            ref.size());
    CHECK(scalar_eq == ref.size());  // bits 6 is still byte-exact off the fast path
    CHECK(rel <= 1.0e-2);            // f32 accumulation, well inside tier 3
    CHECK(fast_diff > 0);
  }

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

TEST_CASE("exl3 rocm gemv: 5 bpw takes the coalesced dot arm") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  // The Qwen3.8-27B-SC_4.00bpw checkpoint stores 66 tensors (2.36 GB,
  // including the lm_head) at 5 bpw cb 2 — outside upstream's GEMV envelope
  // (bits 2..4). The coalesced Exl3DotKImpl<5> arm is the fast path; its f32
  // accumulation puts it inside the same bound as the 6 bpw row above.
  // m=1 is decode (the hot lm_head shape), m=8 the GEMV boundary, m=21 a
  // prefill shape that chunked dispatch slices.
  for (const int64_t m : {1, 8, 21}) {
    CAPTURE(m);
    Exl3Fixture f = MakeFixture(256, 4096, 5, 0x5EED0u + static_cast<uint32_t>(m));
    std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
    Rng rng;
    for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

    const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, /*codebook=*/2);
    const std::vector<uint16_t> scalar = RocmArm(be, f, a, m, 2, /*force_gemv=*/0);
    const std::vector<uint16_t> got = RocmArm(be, f, a, m, 2, /*force_gemv=*/-1);

    size_t scalar_eq = 0, fast_diff = 0;
    double sq = 0.0, rq = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (scalar[i] == ref[i]) ++scalar_eq;
      if (got[i] != scalar[i]) ++fast_diff;
      const double r = vt::F16ToF32(ref[i]);
      const double g = vt::F16ToF32(got[i]);
      sq += (g - r) * (g - r);
      rq += r * r;
    }
    const double rel =
        std::sqrt(sq / static_cast<double>(ref.size())) /
        std::sqrt(rq / static_cast<double>(ref.size()));
    MESSAGE("bits=5 cb=2 m=", m, ": rel RMS ", rel,
            ", dot arm differs from the transcription at ", fast_diff, " of ",
            ref.size());
    CHECK(scalar_eq == ref.size());  // bits 5 is still byte-exact off the fast path
    CHECK(rel <= 1.0e-2);            // f32 accumulation, well inside tier 3
    CHECK(fast_diff > 0);
  }

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

namespace {

// Raw 16/32-bit output words of one ROCm Exl3Gemm call with the given a / c
// dtypes (a holds the a_dt bit patterns).
std::vector<uint32_t> RocmArmDt(vt::Backend& be, const Exl3Fixture& f,
                                const std::vector<uint16_t>& a, vt::DType a_dt,
                                vt::DType c_dt, int64_t m, int codebook) {
  vt::Queue dq = be.CreateQueue();
  const int64_t k = f.k, n = f.n;
  const size_t ab = a.size() * sizeof(uint16_t);
  const size_t bb = f.trellis.size() * sizeof(uint16_t);
  const size_t cw = c_dt == vt::DType::kF32 ? 4 : 2;
  std::vector<uint8_t> out(static_cast<size_t>(m * n) * cw, 0);

  void* d_a = be.Alloc(ab);
  void* d_ah = be.Alloc(ab);
  void* d_b = be.Alloc(bb);
  void* d_suh = be.Alloc(f.suh.size() * 2);
  void* d_svh = be.Alloc(f.svh.size() * 2);
  void* d_c = be.Alloc(out.size());
  be.Copy(dq, d_a, a.data(), ab);
  be.Copy(dq, d_b, f.trellis.data(), bb);
  be.Copy(dq, d_suh, f.suh.data(), f.suh.size() * 2);
  be.Copy(dq, d_svh, f.svh.data(), f.svh.size() * 2);

  vt::Tensor ta = vt::Tensor::Contiguous(d_a, a_dt, dq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tb =
      vt::Tensor::Contiguous(d_b, vt::DType::kI8, dq.device, {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
  vt::Tensor tc = vt::Tensor::Contiguous(d_c, c_dt, dq.device, {m, n});
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = codebook;
  vt::Exl3Gemm(dq, tc, ta, tb, tsuh, tsvh, tah, args);
  be.Synchronize(dq);
  be.Copy(dq, out.data(), d_c, out.size());
  be.Synchronize(dq);
  be.Free(d_a);
  be.Free(d_ah);
  be.Free(d_c);
  be.Free(d_b);
  be.Free(d_suh);
  be.Free(d_svh);
  be.DestroyQueue(dq);

  std::vector<uint32_t> words(static_cast<size_t>(m * n));
  for (size_t i = 0; i < words.size(); ++i) {
    if (cw == 4) {
      std::memcpy(&words[i], out.data() + 4 * i, 4);
    } else {
      uint16_t h;
      std::memcpy(&h, out.data() + 2 * i, 2);
      words[i] = h;
    }
  }
  return words;
}

}  // namespace

// The folded casts: on ROCm Exl3Gemm admits a bf16 a (widened and
// f16-rounded inside the input Hadamard) and a bf16 c (RN-rounded from the
// f32 output Hadamard). Every m the dispatcher can see — decode, the GEMV
// chunk loop, the internal reconstruct window (32, 144] and above — must
// match the f16 pipeline. Where both dtype mixes reach the same kernels
// (m <= 32) that match is byte-exact: bf16 -> f16 staging then f32 out ->
// bf16 is the same rounding chain as the fold. Above, the f16 inputs take
// reconstruct + hipBLAS and bf16 inputs the dtype-aware transcription, so
// the bound is tier 3c against the CPU arm. A bf16 buffer reinterpreted
// as f16 (the reverted e1982b8aa defect) lands at relative RMS ~1e3.
TEST_CASE("exl3 rocm gemv: bf16 a/c fold matches the f16 pipeline at every m") {
  if (!HasRocmExl3()) {
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  struct Arm {
    int bits, cb;
  };
  for (const Arm arm : {Arm{3, 2}, Arm{4, 2}}) {
    for (const int64_t m : {1, 2, 8, 16, 32, 33, 64, 144, 145, 160}) {
      CAPTURE(arm.bits);
      CAPTURE(m);
      Exl3Fixture f = MakeFixture(1024, 1536, arm.bits,
                                  0xBF16u + static_cast<uint32_t>(m * 16 + arm.bits));
      std::vector<uint16_t> a_bf(static_cast<size_t>(m * f.k));
      std::vector<uint16_t> a_f16(a_bf.size());
      Rng rng;
      for (size_t i = 0; i < a_bf.size(); ++i) {
        a_bf[i] = vt::F32ToBF16(rng.next(1.0f));
        a_f16[i] = vt::F32ToF16(vt::BF16ToF32(a_bf[i]));
      }

      const std::vector<uint16_t> ref = CpuArm(f, hq, a_f16, m, arm.cb);
      const std::vector<uint32_t> f16_f32 =
          RocmArmDt(be, f, a_f16, vt::DType::kF16, vt::DType::kF32, m, arm.cb);
      const std::vector<uint32_t> bf_f32 =
          RocmArmDt(be, f, a_bf, vt::DType::kBF16, vt::DType::kF32, m, arm.cb);
      const std::vector<uint32_t> bf_bf =
          RocmArmDt(be, f, a_bf, vt::DType::kBF16, vt::DType::kBF16, m, arm.cb);

      size_t in_diff = 0, out_diff = 0;
      double sq_f32 = 0.0, sq_bf = 0.0, rq = 0.0;
      for (size_t i = 0; i < ref.size(); ++i) {
        float y;
        std::memcpy(&y, &f16_f32[i], 4);
        if (bf_f32[i] != f16_f32[i]) ++in_diff;
        if (bf_bf[i] != vt::F32ToBF16(y)) ++out_diff;
        float g32, gbf;
        std::memcpy(&g32, &bf_f32[i], 4);
        gbf = vt::BF16ToF32(static_cast<uint16_t>(bf_bf[i]));
        const double r = vt::F16ToF32(ref[i]);
        sq_f32 += (g32 - r) * (g32 - r);
        sq_bf += (gbf - r) * (gbf - r);
        rq += r * r;
      }
      const double rms_ref = std::sqrt(rq / static_cast<double>(ref.size()));
      const double rel_f32 =
          std::sqrt(sq_f32 / static_cast<double>(ref.size())) / rms_ref;
      const double rel_bf = std::sqrt(sq_bf / static_cast<double>(ref.size())) / rms_ref;
      MESSAGE("(", arm.bits, ",", arm.cb, ") m=", m, ": bf16-in vs f16-in differ at ",
              in_diff, ", bf16-out vs RN(f32) at ", out_diff, " of ", ref.size(),
              "; rel RMS vs CPU bf16->f32 ", rel_f32, ", bf16->bf16 ", rel_bf);
      CHECK(rms_ref > 0.0);
      CHECK(rel_f32 <= 6.0e-3);
      CHECK(rel_bf <= 6.0e-3);
      if (m <= 32) {
        CHECK(in_diff == 0);
        CHECK(out_diff == 0);
      }
    }
  }

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

// ── HALF-INTEGER rates (K+0.5, mul1) — BACKEND-ROCM frac rates ──────────────

TEST_CASE("exl3 rocm gemv: half-integer (K+0.5) frac tensors decode through Exl3GemmK and the dot arm") {
  if (!HasRocmExl3()) {
    MESSAGE(
        "SKIPPED, no ROCm device: the frac-rate (K+0.5) arms are PENDING. Reproduce with: "
        "ctest --test-dir build-hip -R test_exl3_rocm_gemv -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue hq = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();

  // KA 2 and 3 run the dq8_half fast port; KA 5 exercises the width-generic
  // scalar arm. The GEMV itself must decline a frac tensor (no frac GEMV arm,
  // spec ## Owed) — force_gemv=1 pushes past that decline to the DOT arm, and
  // force_gemv=0 pins the Exl3GemmK transcription, which is byte-exact.
  for (const int ka : {2, 3, 5}) {
    for (const int64_t m : {1, 4}) {
      CAPTURE(ka);
      CAPTURE(m);
      const Exl3Fixture f = MakeHalfFixture(256, 128, ka, 0x0DD5u + static_cast<uint32_t>(ka));
      std::vector<uint16_t> a(static_cast<size_t>(m * f.k));
      Rng rng;
      for (auto& v : a) v = vt::F32ToF16(rng.next(1.0f));

      const std::vector<uint16_t> ref = CpuArm(f, hq, a, m, /*codebook=*/2, /*half=*/true);
      const std::vector<uint16_t> scalar =
          RocmArm(be, f, a, m, 2, /*force_gemv=*/0, /*half=*/true);
      const std::vector<uint16_t> dot =
          RocmArm(be, f, a, m, 2, /*force_gemv=*/1, /*half=*/true);

      size_t scalar_eq = 0;
      double sq = 0.0, rq = 0.0;
      for (size_t i = 0; i < ref.size(); ++i) {
        if (scalar[i] == ref[i]) ++scalar_eq;
        const double r = vt::F16ToF32(ref[i]);
        const double g = vt::F16ToF32(dot[i]);
        sq += (g - r) * (g - r);
        rq += r * r;
      }
      const double rel = std::sqrt(sq / static_cast<double>(ref.size())) /
                         std::sqrt(rq / static_cast<double>(ref.size()));
      MESSAGE("KA=", ka, " m=", m, ": Exl3GemmK byte-equal ", scalar_eq, " of ",
              ref.size(), "; dot arm rel RMS ", rel);
      CHECK(scalar_eq == ref.size());  // the frac transcription is byte-exact
      CHECK(rel <= 6.0e-3);            // the dot arm's own bound (f32 accumulation)

      // Decode discrimination: the same frac bytes through the integer-KA
      // reading must NOT reproduce the frac result — a frac tensor silently
      // decoded as integer KA is exactly the failure the flag exists to bar.
      const Exl3Fixture fi = MakeFixture(256, 128, ka, 0x0DD5u + static_cast<uint32_t>(ka));
      const std::vector<uint16_t> wrong =
          CpuArm(fi, hq, a, m, /*codebook=*/2, /*half=*/false);
      int same = 0;
      for (size_t i = 0; i < ref.size(); ++i)
        if (wrong[i] == ref[i]) ++same;
      CHECK(same < static_cast<int>(ref.size()) / 2);
    }
  }

  // An out-of-envelope frac call refuses: KA 8 at the seam.
  {
    const Exl3Fixture f = MakeHalfFixture(128, 128, 8, 0xBADu);
    std::vector<uint16_t> a(static_cast<size_t>(f.k), vt::F32ToF16(0.5f));
    std::string msg;
    try {
      (void)CpuArm(f, hq, a, 1, /*codebook=*/2, /*half=*/true);
      FAIL("exl3 rocm gemv: KA=8 did NOT throw");
    } catch (const std::exception& e) {
      msg = e.what();
    }
    CHECK(msg.find("K+0.5") != std::string::npos);
  }

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

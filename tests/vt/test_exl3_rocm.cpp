// The ROCm arm of the two ops an EXL3 checkpoint ran on the CPU reference tier
// — row BACKEND-ROCM-EXL3, issue #2433, spec .agents/specs/backend-rocm-exl3.md.
//
// WHAT THIS SUITE GATES, AND WHY THE BOUND IS ZERO.
//
// `src/vt/rocm/rocm_exl3.hip` is transcribed from the PORTABLE CPU reference
// (`cpu_exl3_kernels.cpp`, `cpu_exl3_dequant.cpp`) rather than ported from
// `cuda_exl3.cu`, whose 90 KiB shared-memory budget, `m16n8k16` mma, `ldmatrix`
// and `cp.async` have no gfx1151 form. Because it is a transcription of that
// reference and uses no matrix cores and no split-K reduction, all three steps
// run the same IEEE f32 operations on the same values in the same ORDER — so
// the gate here is BYTE equality with the CPU arm, and not the 1.0e-3 RMS bound
// `test_exl3_gemm`'s CUDA cases carry. A tolerance here would be slack this arm
// has not earned and does not need.
//
// The comparison is therefore INTEGER: the fp16 (or f32) output words are
// compared as bit patterns. That is what makes it a decode gate as well as a
// GEMM gate. A single mis-decoded codeword changes an f32 accumulation and the
// stored bits with it, and the codebook argument is proven to REACH the device
// decode by the discrimination case below, which requires the three codebooks
// to disagree with each other on one trellis.
//
// Every device case SKIPS when no ROCm backend is registered, says so, and
// STILL ASSERTS the precondition it skipped on. `assertions: 0` printed under
// `SUCCESS!` is a skip wearing a pass, and this family has already paid for
// that once.
#include <doctest/doctest.h>

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
using exl3_test::Rng;

bool HasRocm() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kROCM);
    return vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM);
  } catch (const std::runtime_error&) {
    return false;
  }
}

bool HasRocmCast() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kROCM);
    return vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kROCM);
  } catch (const std::runtime_error&) {
    return false;
  }
}

vt::Queue CpuQueue() { return vt::GetBackend(vt::DeviceType::kCPU).CreateQueue(); }

// A `const char*` handed to doctest's MESSAGE stringifies as a BOOL and prints
// "1", which turns an honest skip notice into noise. std::string prints.
const std::string kNoDevice =
    "SKIPPED, no ROCm device: BACKEND-ROCM-EXL3 device parity is PENDING. "
    "strix:gpu0 (gfx1151, ROCm 7.2.4) is the only AMD device on the fleet. "
    "Reproduce with: rc run --device strix:gpu0 -- "
    "ctest --test-dir build-hip -R test_exl3_rocm -V";

// One EXL3 linear evaluated on the CPU arm, which is this suite's oracle AND
// this file's donor. `c_f32` selects upstream's `c_fp32` arm.
std::vector<uint8_t> CpuGemm(const Exl3Fixture& f, const std::vector<uint16_t>& a_h, int64_t m,
                             int codebook, bool c_f32) {
  vt::Queue hq = CpuQueue();
  const int64_t k = f.k, n = f.n;
  std::vector<uint16_t> a_had(a_h.size(), 0);
  std::vector<uint16_t> a_copy = a_h;
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = codebook;
  std::vector<uint8_t> out(static_cast<size_t>(m * n) * (c_f32 ? 4 : 2), 0);
  vt::Tensor ta = vt::Tensor::Contiguous(a_copy.data(), vt::DType::kF16, hq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(a_had.data(), vt::DType::kF16, hq.device, {m, k});
  vt::Tensor tc = vt::Tensor::Contiguous(
      out.data(), c_f32 ? vt::DType::kF32 : vt::DType::kF16, hq.device, {m, n});
  vt::Tensor tb =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.trellis.data()), vt::DType::kI8, hq.device,
                             {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.suh.data()), vt::DType::kF16, hq.device, {k});
  vt::Tensor tsvh =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.svh.data()), vt::DType::kF16, hq.device, {n});
  vt::Exl3Gemm(hq, tc, ta, tb, tsuh, tsvh, tah, args);
  return out;
}

// The same linear on the ROCm device. `alias_a_had` runs the input transform IN
// PLACE over `A`, which the op's contract permits and which no CPU call here can
// exercise, because in-place is a device-buffer property.
std::vector<uint8_t> RocmGemm(const Exl3Fixture& f, const std::vector<uint16_t>& a_h, int64_t m,
                              int codebook, bool c_f32, bool alias_a_had) {
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue dq = cb.CreateQueue();
  const int64_t k = f.k, n = f.n;
  const size_t ab = a_h.size() * sizeof(uint16_t);
  const size_t bb = f.trellis.size() * sizeof(uint16_t);
  const size_t cb_bytes = static_cast<size_t>(m * n) * (c_f32 ? 4u : 2u);
  void* d_a = cb.Alloc(ab);
  void* d_ah = alias_a_had ? d_a : cb.Alloc(ab);
  void* d_b = cb.Alloc(bb);
  void* d_suh = cb.Alloc(f.suh.size() * sizeof(uint16_t));
  void* d_svh = cb.Alloc(f.svh.size() * sizeof(uint16_t));
  void* d_c = cb.Alloc(cb_bytes);
  cb.Copy(dq, d_a, a_h.data(), ab);
  cb.Copy(dq, d_b, f.trellis.data(), bb);
  cb.Copy(dq, d_suh, f.suh.data(), f.suh.size() * sizeof(uint16_t));
  cb.Copy(dq, d_svh, f.svh.data(), f.svh.size() * sizeof(uint16_t));

  vt::Tensor ta = vt::Tensor::Contiguous(d_a, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tb =
      vt::Tensor::Contiguous(d_b, vt::DType::kI8, dq.device, {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
  vt::Tensor tc = vt::Tensor::Contiguous(
      d_c, c_f32 ? vt::DType::kF32 : vt::DType::kF16, dq.device, {m, n});
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = codebook;
  // This suite gates the BYTE-exact transcription. force_gemv = 0 pins step 2
  // to Exl3GemmK so the m<=8 GEMV arm (rocm_exl3_gemv.hip, its own 6.0e-3 gate
  // in test_exl3_rocm_gemv.cpp) cannot change the arm under test.
  args.force_gemv = 0;
  vt::Exl3Gemm(dq, tc, ta, tb, tsuh, tsvh, tah, args);
  cb.Synchronize(dq);
  std::vector<uint8_t> out(cb_bytes, 0);
  cb.Copy(dq, out.data(), d_c, cb_bytes);
  cb.Synchronize(dq);
  cb.Free(d_a);
  if (!alias_a_had) cb.Free(d_ah);
  cb.Free(d_b);
  cb.Free(d_suh);
  cb.Free(d_svh);
  cb.Free(d_c);
  cb.DestroyQueue(dq);
  return out;
}

int FirstDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  if (a.size() != b.size()) return 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) return static_cast<int>(i);
  return -1;
}

}  // namespace

// ─── R1: kCastF16 ────────────────────────────────────────────────────────────

TEST_CASE("exl3 rocm: CastF16 is registered on ROCm, like its two siblings") {
  // The op-table half of #2433's first reference-tier hit. `kCastBf16` and
  // `kCastF32` have six backends each; `kCastF16` had two. This case is what
  // reds if the registrar entry is deleted while the kernel still compiles.
  if (!HasRocmCast()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kROCM));
    return;
  }
  CHECK(vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kROCM));
  CHECK(vt::OpRegistered(vt::OpId::kCastBf16, vt::DeviceType::kROCM));
  CHECK(vt::OpRegistered(vt::OpId::kCastF32, vt::DeviceType::kROCM));
}

TEST_CASE("exl3 rocm: CastF16 is BYTE-identical to the CPU arm, from f32 and from bf16") {
  if (!HasRocmCast()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kCastF16, vt::DeviceType::kROCM));
    return;
  }
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue dq = cb.CreateQueue();
  vt::Queue hq = CpuQueue();
  const int64_t rows = 5, cols = 71;  // deliberately not a multiple of the block
  const int64_t nelem = rows * cols;

  Rng rng;
  rng.s = 0x9E3779B9u;
  std::vector<float> src_f32(static_cast<size_t>(nelem));
  for (auto& v : src_f32) v = rng.next(6.0f);
  // Values a plain conversion agrees on are not the interesting ones. These are:
  // an exact fp16 tie (rounds to EVEN), a subnormal, and an overflow.
  src_f32[0] = 1.0f + 1.0f / 2048.0f;      // exactly halfway between two fp16
  src_f32[1] = 3.0f + 3.0f / 2048.0f;      // the other parity of the same tie
  src_f32[2] = 5.0e-8f;                    // below the fp16 subnormal floor
  src_f32[3] = 1.0e-6f;                    // an fp16 subnormal
  src_f32[4] = 1.0e6f;                     // overflows to +inf
  src_f32[5] = -0.0f;                      // signed zero survives

  std::vector<uint16_t> src_bf16(static_cast<size_t>(nelem));
  for (int64_t i = 0; i < nelem; ++i) src_bf16[i] = vt::F32ToBF16(src_f32[i]);

  for (int arm = 0; arm < 2; ++arm) {
    const bool from_f32 = arm == 0;
    CAPTURE(from_f32);
    const size_t src_bytes =
        static_cast<size_t>(nelem) * (from_f32 ? sizeof(float) : sizeof(uint16_t));
    const void* src = from_f32 ? static_cast<const void*>(src_f32.data())
                               : static_cast<const void*>(src_bf16.data());
    const vt::DType sdt = from_f32 ? vt::DType::kF32 : vt::DType::kBF16;

    std::vector<uint16_t> host_out(static_cast<size_t>(nelem), 0);
    vt::Tensor hi =
        vt::Tensor::Contiguous(const_cast<void*>(src), sdt, hq.device, {rows, cols});
    vt::Tensor ho =
        vt::Tensor::Contiguous(host_out.data(), vt::DType::kF16, hq.device, {rows, cols});
    vt::CastF16(hq, ho, hi);

    void* d_in = cb.Alloc(src_bytes);
    void* d_out = cb.Alloc(static_cast<size_t>(nelem) * sizeof(uint16_t));
    cb.Copy(dq, d_in, src, src_bytes);
    vt::Tensor di = vt::Tensor::Contiguous(d_in, sdt, dq.device, {rows, cols});
    vt::Tensor dof = vt::Tensor::Contiguous(d_out, vt::DType::kF16, dq.device, {rows, cols});
    vt::CastF16(dq, dof, di);
    cb.Synchronize(dq);
    std::vector<uint16_t> dev_out(static_cast<size_t>(nelem), 0xFFFF);
    cb.Copy(dq, dev_out.data(), d_out, dev_out.size() * sizeof(uint16_t));
    cb.Synchronize(dq);
    cb.Free(d_in);
    cb.Free(d_out);

    int mismatches = 0;
    for (size_t i = 0; i < host_out.size(); ++i)
      if (host_out[i] != dev_out[i]) ++mismatches;
    CHECK(mismatches == 0);
    // Not vacuous: the cast must have produced something other than the poison
    // the device buffer was read back into.
    bool any_nonpoison = false;
    for (uint16_t v : dev_out)
      if (v != 0xFFFF) any_nonpoison = true;
    CHECK(any_nonpoison);
  }
  cb.DestroyQueue(dq);
}

// ─── R2: kExl3Gemm ───────────────────────────────────────────────────────────

TEST_CASE("exl3 rocm: exl3_gemm is registered on ROCm") {
  if (!HasRocm()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  CHECK(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
}

TEST_CASE("exl3 rocm: exl3_gemm is BYTE-identical to the CPU arm on every codebook and width") {
  if (!HasRocm()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  struct Arm {
    int bits;
    int codebook;
    int64_t m;
    bool c_f32;
    const char* what;
  };
  // Every (bits, codebook) pair the CUDA arm instantiates, plus the row counts
  // that select each grid arm: m = 3 is one workgroup row-block, m = 20 is two,
  // and m = 1 is the decode shape a served token actually has.
  const Arm arms[] = {
      {3, 0, 1, false, "a stock exl3 body at batch 1, the shape a decode step has"},
      {3, 0, 20, false, "the same, over two row-blocks of the grid"},
      {6, 0, 3, false, "a stock exl3 lm_head"},
      {3, 1, 3, false, "the SparkInfer DeepSeek-V4 marker"},
      {3, 1, 3, true, "the same, into upstream's c_fp32 arm"},
      {4, 2, 3, false, "the Qwen3.8-27B mul1 GDN tower and attention, 270 of its 409"},
      {5, 2, 3, false, "its 5-bit tensor, and all 36 of the draft"},
      {6, 2, 3, false, "its mul1 lm_head"},
  };
  for (const Arm& arm : arms) {
    CAPTURE(arm.bits);
    CAPTURE(arm.codebook);
    CAPTURE(arm.m);
    CAPTURE(arm.c_f32);
    const int64_t k = 256, n = 128;
    const Exl3Fixture f = MakeFixture(k, n, arm.bits, 0x1D0C0DEu + arm.bits * 7 + arm.codebook);
    Rng rng;
    rng.s = 0xC0FFEEu + arm.bits;
    std::vector<uint16_t> a_h(static_cast<size_t>(arm.m * k));
    for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));

    const std::vector<uint8_t> host = CpuGemm(f, a_h, arm.m, arm.codebook, arm.c_f32);
    const std::vector<uint8_t> dev = RocmGemm(f, a_h, arm.m, arm.codebook, arm.c_f32, false);
    const int diff = FirstDiff(host, dev);
    // std::string, not the bare `const char*`: doctest stringifies a char* as a
    // BOOL and would print "1" where the arm's description belongs.
    MESSAGE("bits ", arm.bits, " cb ", arm.codebook, " m ", arm.m, " (",
            std::string(arm.what), "): first differing byte = ", diff);
    // -1 means "no differing byte". INTEGER equality over the stored bits, not a
    // tolerance: see the head of this file for why this arm can claim it.
    CHECK(diff == -1);
    // Not vacuous: an all-zero output would compare equal and prove nothing.
    bool nonzero = false;
    for (uint8_t b : host) nonzero = nonzero || b != 0;
    REQUIRE(nonzero);
  }
}

TEST_CASE("exl3 rocm: the codebook argument REACHES the device decode") {
  // The byte gate above is also the decode gate — a single mis-decoded codeword
  // moves an f32 accumulation and the stored bits with it. That argument holds
  // only if the device actually READS `args.codebook`; a kernel that hardcoded
  // one codebook would pass every row above whose codebook happened to be that
  // one. So: one trellis, three codebooks, and the three results must DIFFER
  // from each other on the device exactly as they differ on the host.
  if (!HasRocm()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  const int64_t m = 2, k = 256, n = 128;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x51ED2A17u);
  Rng rng;
  rng.s = 0x2545F491u;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));

  std::vector<std::vector<uint8_t>> dev;
  for (int cb = 0; cb < 3; ++cb) {
    const std::vector<uint8_t> d = RocmGemm(f, a_h, m, cb, false, false);
    CHECK(FirstDiff(CpuGemm(f, a_h, m, cb, false), d) == -1);
    dev.push_back(d);
  }
  CHECK(FirstDiff(dev[0], dev[1]) != -1);
  CHECK(FirstDiff(dev[0], dev[2]) != -1);
  CHECK(FirstDiff(dev[1], dev[2]) != -1);
}

TEST_CASE("exl3 rocm: the input transform runs IN PLACE when a_had aliases A") {
  // `Exl3Gemm`'s contract permits `a_had` to alias `a`, and upstream says the
  // same of its own kernel ("Works inplace if y == x", hadamard.cu:86). No CPU
  // call can exercise it meaningfully; on a device it is a claim about barriers,
  // because every lane must finish LOADING its 128-block before any lane stores.
  if (!HasRocm()) {
    MESSAGE(kNoDevice);
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kROCM));
    return;
  }
  const int64_t m = 3, k = 256, n = 128;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x2C1B3C6Du);
  Rng rng;
  rng.s = 0x85EBCA6Bu;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));
  const std::vector<uint8_t> host = CpuGemm(f, a_h, m, 1, false);
  const std::vector<uint8_t> aliased = RocmGemm(f, a_h, m, 1, false, /*alias_a_had=*/true);
  CHECK(FirstDiff(host, aliased) == -1);
}

TEST_CASE("exl3 rocm: an out-of-range codebook is refused BY NAME, not decoded") {
  // Upstream's `decode_3inst<cb>` has arms for 0, 1 and 2 and falls off the end
  // for anything else, and a device kernel has no VT_CHECK — so the refusal has
  // to happen on the host before the launch. Reaching this needs no device: the
  // seam check in `vt::Exl3Gemm` fires first and is what a CPU-only run gates.
  vt::Queue hq = CpuQueue();
  const int64_t m = 1, k = 128, n = 128;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x7FEDCBA9u);
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k), vt::F32ToF16(0.5f));
  std::vector<uint16_t> a_had(a_h.size(), 0);
  std::vector<uint16_t> c(static_cast<size_t>(m * n), 0);
  vt::Exl3GemmArgs args;
  args.bits = 3;
  args.codebook = 3;
  vt::Tensor ta = vt::Tensor::Contiguous(a_h.data(), vt::DType::kF16, hq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(a_had.data(), vt::DType::kF16, hq.device, {m, k});
  vt::Tensor tc = vt::Tensor::Contiguous(c.data(), vt::DType::kF16, hq.device, {m, n});
  vt::Tensor tb =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.trellis.data()), vt::DType::kI8, hq.device,
                             {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.suh.data()), vt::DType::kF16, hq.device, {k});
  vt::Tensor tsvh =
      vt::Tensor::Contiguous(const_cast<uint16_t*>(f.svh.data()), vt::DType::kF16, hq.device, {n});
  std::string msg;
  try {
    vt::Exl3Gemm(hq, tc, ta, tb, tsuh, tsvh, tah, args);
  } catch (const std::exception& e) {
    msg = e.what();
  }
  CHECK(msg.find("codebook") != std::string::npos);
  CHECK(msg.find("3") != std::string::npos);
}

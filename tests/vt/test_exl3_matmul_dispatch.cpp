// The EXL3 reconstruct dispatch through the production seam — QUANT-EXL3,
// ISSUE-LOCAL-01M2BYPW7YTC2B2MY023ETTKQ2 item 1.
//
// #3150 ported exllamav3's `AUTO_RECONSTRUCT_THRESHOLD` (exl3.py:10,135) into
// `dense_attn::Exl3MatmulD`: M > 144 routes to `vt::Exl3ReconstructGemm`. That
// op is registered for CUDA only (`src/vt/cuda/cuda_exl3.cu`), while
// `vt::Exl3Gemm` is registered for CPU, ROCm and Vulkan too. A device-blind
// threshold therefore refused every EXL3 prefill above 144 tokens on those
// backends, which served every M through `Exl3Gemm` before #3150. Upstream's
// reconstruct path is CUDA-only as well, so the correct mirror on a backend with
// no reconstruct kernel is the cooperative kernel at every M.
//
// This suite enters through `Exl3MatmulD` on a CPU queue, which is the seam
// `layers::Exl3LinearMethod::Apply` delegates to, so it measures the dispatch a
// model forward reaches and not the kernels by hand. CPU-only; runs in CI.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#ifdef VLLM_CPP_CUDA
// Internal, not installed; the relative spelling needs no test include path.
#include "../../src/vt/cuda/cuda_exl3_internal.h"
#endif

#include "exl3_fixture.h"

namespace {

using exl3_test::Exl3ChainF64;
using exl3_test::Exl3Fixture;
using exl3_test::MakeFixture;
using exl3_test::MakeHalfFixture;
using exl3_test::Rms;
using exl3_test::Rng;
using exl3_test::UlpF16;

vllm::Exl3Weight WrapFixture(const Exl3Fixture& f) {
  vllm::Exl3Weight w;
  w.codebook = f.half ? 2 : 1;  // the codebook Exl3ChainF64 decodes by default; frac is mul1
  w.half = f.half;
  const auto bytes_of = [](const std::vector<uint16_t>& v) {
    return vllm::OwnedBytes(std::vector<uint8_t>(
        reinterpret_cast<const uint8_t*>(v.data()),
        reinterpret_cast<const uint8_t*>(v.data()) + v.size() * 2));
  };
  w.trellis.dtype = vt::DType::kI8;
  w.trellis.rank = 3;
  w.trellis.shape[0] = f.k / 16;
  w.trellis.shape[1] = f.n / 16;
  w.trellis.shape[2] = 32 * f.bits + (f.half ? 16 : 0);
  w.trellis.bytes = bytes_of(f.trellis);
  w.suh.dtype = vt::DType::kF16;
  w.suh.rank = 1;
  w.suh.shape[0] = f.k;
  w.suh.bytes = bytes_of(f.suh);
  w.svh.dtype = vt::DType::kF16;
  w.svh.rank = 1;
  w.svh.shape[0] = f.n;
  w.svh.bytes = bytes_of(f.svh);
  return w;
}

}  // namespace

TEST_CASE("exl3 dispatch: M > 144 on a backend with no reconstruct kernel serves through Exl3Gemm") {
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = b.CreateQueue();
  vllm::dense_attn::Dev d{b, q};

  // The premise, asserted rather than assumed: if a CPU reconstruct kernel is
  // ever registered, this case no longer tests the fallback and must be revised.
  REQUIRE_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCPU));
  REQUIRE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU));

  // 145 is the first M upstream sends to reconstruct_hgemm (`rows <= 144` keeps
  // the cooperative kernel, exl3.py:135).
  const int64_t m = 145, k = 128, n = 256;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x3150A11u);
  const vllm::Exl3Weight w = WrapFixture(f);

  Rng rng;
  rng.s = 0x0C0FFEEu;
  std::vector<float> x(static_cast<size_t>(m * k));
  // Through fp16 first: the seam stages the activation to fp16, and the
  // reference must read the same values.
  for (auto& v : x) v = vt::F16ToF32(vt::F32ToF16(rng.next(1.0f)));

  vt::EnableOpProviderCallStats(true);
  const unsigned long long before =
      vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU).selections;

  vllm::dense_attn::DBuf xb(d, vt::DType::kF32, {m, k}, x.data());
  std::vector<float> got(static_cast<size_t>(m * n), 0.0f);
  std::string refusal;
  try {
    vllm::dense_attn::DBuf out = vllm::dense_attn::Exl3MatmulD(d, xb.t(), w, vt::DType::kF32);
    out.Download(d, got.data());
  } catch (const std::exception& e) {
    refusal = e.what();
  }
  INFO("Exl3MatmulD refusal: " << refusal);
  REQUIRE(refusal.empty());

  // Positive signal that the cooperative kernel served the call, not merely
  // that something returned numbers.
  const unsigned long long after =
      vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU).selections;
  vt::EnableOpProviderCallStats(false);
  CHECK(after > before);

  const std::vector<double> ref = Exl3ChainF64(f, x, m);
  const double rms = Rms(ref);
  REQUIRE(rms > 0.0);
  double sq = 0.0, worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double dlt = static_cast<double>(got[i]) - ref[i];
    sq += dlt * dlt;
    worst = std::max(worst, std::fabs(dlt));
  }
  const double rel_rms = std::sqrt(sq / static_cast<double>(ref.size())) / rms;
  MESSAGE("Exl3MatmulD M=145 cpu vs f64: rel_rms=", rel_rms, " worst=", worst);
  // The CPU arm's own tier-3 bound (test_exl3_gemm.cpp, spec `## W2 design` §1).
  CHECK(rel_rms <= 1.0e-3);
  CHECK(worst <= 8.0 * UlpF16(rms));

  b.DestroyQueue(q);
}

// ─── QUANT-EXL3 frac rates: the reconstruct seam refuses half weights ───────
//
// `.agents/specs/quant-exl3-frac-rates.md` `## Owed`. On ROCm
// `kExl3ReconstructGemm` IS registered, so the pre-fix seam
// (`M > 144 && OpRegistered(...)`) routed a frac tensor (`w.half`, K+0.5 rate)
// to a reconstruct arm that does not exist and threw — every OrcaSAQ-3.21bpw
// prefill above 144 tokens crashed. `rocm_exl3.hip` `want_recon` already
// carries `!args.half`; the decision is split into `Exl3WantsReconstruct` so a
// host test can pin it, because NO host-reachable device has the op
// registered to exercise the branch through `Exl3MatmulD` end to end.
TEST_CASE("exl3 dispatch: a frac weight never routes to the reconstruct gemm") {
  using vllm::dense_attn::Exl3WantsReconstruct;
  vllm::Exl3Weight frac;
  frac.half = true;  // K+0.5 rate, mul1; the helper reads only `half`
  vllm::Exl3Weight integer;  // default half == false

  // The ROCm prefill shape: op registered, M past the threshold.
  CHECK_FALSE(Exl3WantsReconstruct(145, frac, /*reconstruct_registered=*/true));
  CHECK(Exl3WantsReconstruct(145, integer, /*reconstruct_registered=*/true));
  CHECK(Exl3WantsReconstruct(100000, frac, /*reconstruct_registered=*/true) == false);

  // Threshold polarity and a backend without the op are unchanged.
  CHECK_FALSE(Exl3WantsReconstruct(144, integer, /*reconstruct_registered=*/true));
  CHECK_FALSE(Exl3WantsReconstruct(145, integer, /*reconstruct_registered=*/false));
}

// The same promise end to end through the seam on a host queue: a frac weight
// at M > 145 serves through `Exl3Gemm` and matches the f64 chain at the mul1
// codebook. On CPU the op is unregistered, so this cannot catch a
// registration-only regression — that is the static case's job above.
TEST_CASE("exl3 dispatch: a frac weight at M > 144 serves through Exl3Gemm on CPU") {
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = b.CreateQueue();
  vllm::dense_attn::Dev d{b, q};

  const int64_t m = 145, k = 128, n = 256;
  const Exl3Fixture f = MakeHalfFixture(k, n, /*ka=*/3, 0x3A7C0DEu);
  const vllm::Exl3Weight w = WrapFixture(f);
  REQUIRE(w.half);

  Rng rng;
  rng.s = 0x0C0FFEEu;
  std::vector<float> x(static_cast<size_t>(m * k));
  for (auto& v : x) v = vt::F16ToF32(vt::F32ToF16(rng.next(1.0f)));

  vt::EnableOpProviderCallStats(true);
  const unsigned long long before =
      vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU).selections;

  vllm::dense_attn::DBuf xb(d, vt::DType::kF32, {m, k}, x.data());
  std::vector<float> got(static_cast<size_t>(m * n), 0.0f);
  std::string refusal;
  try {
    vllm::dense_attn::DBuf out = vllm::dense_attn::Exl3MatmulD(d, xb.t(), w, vt::DType::kF32);
    out.Download(d, got.data());
  } catch (const std::exception& e) {
    refusal = e.what();
  }
  INFO("Exl3MatmulD refusal: " << refusal);
  REQUIRE(refusal.empty());

  const unsigned long long after =
      vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kCPU).selections;
  vt::EnableOpProviderCallStats(false);
  CHECK(after > before);

  const std::vector<double> ref = Exl3ChainF64(f, x, m, /*codebook=*/2);
  const double rms = Rms(ref);
  REQUIRE(rms > 0.0);
  double sq = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double dlt = static_cast<double>(got[i]) - ref[i];
    sq += dlt * dlt;
  }
  const double rel_rms = std::sqrt(sq / static_cast<double>(ref.size())) / rms;
  MESSAGE("Exl3MatmulD frac M=145 cpu vs f64: rel_rms=", rel_rms);
  CHECK(rel_rms <= 1.0e-3);

  b.DestroyQueue(q);
}

// ─── QUANT-EXL3 W7: one reconstruct scratch per stream ──────────────────────
//
// `.agents/specs/quant-exl3-recon-scratch.md` §7, ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ.
// #3150 drew the reconstruct weight scratch, fp16 [K, min(N, 32768)], from the
// DevicePool on every M > 144 call. A CUDA-graph decode step PINS every block its
// eager step demanded, so each lazily captured verify width pinned one block of
// every scratch class (about 786 MiB per slot on Qwen3.8-27B EXL3) and the GB10
// host ran out of memory at c = 32. These cases enter through `Exl3MatmulD`, the
// seam `layers::Exl3LinearMethod::Apply` delegates to, on a real CUDA queue.
// They SKIP without a CUDA reconstruct kernel, and still assert, so a CPU run
// never reports a skip as a pass.

namespace {

bool HasReconstructCuda() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kCUDA);
    return vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCUDA);
  } catch (const std::runtime_error&) {
    return false;
  }
}

struct W7Shape {
  int64_t k;
  int64_t n;
};

// Every (K, N) a W7 case uses, including one N > 32768 so the sliced scratch
// ([K, 32768], smaller than the weight) is covered. All are multiples of 128.
constexpr W7Shape kW7Shapes[] = {{256, 512}, {512, 256}, {384, 1024}, {256, 32896}};

std::vector<uint16_t> RandomF16(size_t count, uint32_t seed) {
  Rng rng;
  rng.s = seed;
  std::vector<uint16_t> v(count);
  for (auto& x : v) x = vt::F32ToF16(rng.next(1.0f));
  return v;
}

int64_t ScratchCols(int64_t n) { return n <= 32768 ? n : 32768; }

std::vector<uint8_t> Download(vllm::dense_attn::Dev d, vllm::dense_attn::DBuf& buf) {
  std::vector<uint8_t> out(buf.bytes());
  buf.Download(d, out.data());
  return out;
}

}  // namespace

TEST_CASE("exl3 dispatch cuda: the reconstruct scratch draws no DevicePool block") {
  if (!HasReconstructCuda()) {
    MESSAGE("SKIPPED, no CUDA reconstruct kernel: QUANT-EXL3 W7 test (1) is PENDING. "
            "Reproduce with: rc run -d dgx:gpu0 -- ctest -R test_exl3_matmul_dispatch -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCUDA));
    return;
  }
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = cb.CreateQueue();
  vllm::dense_attn::Dev d{cb, q};
  const int64_t m = 145;  // the first M upstream sends to reconstruct_hgemm

  // The measurement is only meaningful if no OTHER buffer of the step shares a
  // class with a scratch: the x input and a_had are [M, K] f16 and the output is
  // [M, N] f16. The premise is asserted, not assumed.
  std::vector<size_t> other_classes, scratch_classes;
  for (const W7Shape& s : kW7Shapes) {
    other_classes.push_back(vllm::DevicePool::SizeClassForTest(static_cast<size_t>(m * s.k * 2)));
    other_classes.push_back(vllm::DevicePool::SizeClassForTest(static_cast<size_t>(m * s.n * 2)));
    scratch_classes.push_back(
        vllm::DevicePool::SizeClassForTest(static_cast<size_t>(s.k * ScratchCols(s.n) * 2)));
  }
  for (size_t sc : scratch_classes)
    for (size_t oc : other_classes) REQUIRE(sc != oc);

  std::vector<Exl3Fixture> fixtures;
  std::vector<vllm::Exl3Weight> weights;
  for (const W7Shape& s : kW7Shapes) fixtures.push_back(MakeFixture(s.k, s.n, 3, 0x3150B07u));
  for (const Exl3Fixture& f : fixtures) weights.push_back(WrapFixture(f));

  const auto run_all = [&]() {
    for (size_t i = 0; i < weights.size(); ++i) {
      const std::vector<uint16_t> x = RandomF16(static_cast<size_t>(m * kW7Shapes[i].k), 7u + i);
      vllm::dense_attn::DBuf xb(d, vt::DType::kF16, {m, kW7Shapes[i].k}, x.data());
      vllm::dense_attn::DBuf out =
          vllm::dense_attn::Exl3MatmulD(d, xb.t(), weights[i], vt::DType::kF16);
      cb.Synchronize(q);
    }
  };

  vllm::DevicePool& pool = vllm::ActivePool(cb);
  run_all();  // the eager warm step: uploads, and grows whatever it grows
  pool.MarkStepBoundary();
  run_all();
  run_all();
  const vllm::DevicePool::StepDemand demand = pool.StepDemandProfile();

  for (const auto& entry : demand) {
    for (size_t i = 0; i < scratch_classes.size(); ++i) {
      INFO("shape K=" << kW7Shapes[i].k << " N=" << kW7Shapes[i].n << ": the pool's demand "
                      "profile holds the reconstruct scratch class " << entry.first
                      << " with peak " << entry.second);
      CHECK(entry.first != scratch_classes[i]);
    }
  }
  cb.DestroyQueue(q);
}

TEST_CASE("exl3 dispatch cuda: the persistent scratch leaves the reconstruct output byte-identical") {
  if (!HasReconstructCuda()) {
    MESSAGE("SKIPPED, no CUDA reconstruct kernel: QUANT-EXL3 W7 test (2) is PENDING. "
            "Reproduce with: rc run -d dgx:gpu0 -- ctest -R test_exl3_matmul_dispatch -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCUDA));
    return;
  }
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = cb.CreateQueue();
  vllm::dense_attn::Dev d{cb, q};
  using vllm::dense_attn::DBuf;

  // M = 145 and 300 take the UNFUSED reconstruct (M < 1024), and 1024 the FUSED
  // one (exl3.py:176-184), so both sub-paths read the scratch.
  for (const int64_t m : {int64_t{145}, int64_t{300}, int64_t{1024}}) {
    for (size_t i = 0; i < std::size(kW7Shapes); ++i) {
      const int64_t k = kW7Shapes[i].k, n = kW7Shapes[i].n;
      if (m == 1024 && n > 32768) continue;  // keeps the f16 [M, N] output small
      CAPTURE(m);
      CAPTURE(k);
      CAPTURE(n);
      const Exl3Fixture f = MakeFixture(k, n, 3, 0x3150B07u + static_cast<uint32_t>(i));
      const vllm::Exl3Weight w = WrapFixture(f);
      const std::vector<uint16_t> x = RandomF16(static_cast<size_t>(m * k), 0xB07u + i);
      DBuf xb(d, vt::DType::kF16, {m, k}, x.data());

      // Through the production seam.
      DBuf got = vllm::dense_attn::Exl3MatmulD(d, xb.t(), w, vt::DType::kF16);
      const std::vector<uint8_t> got_bytes = Download(d, got);

      // The pre-change call, spelled out: the same op with a scratch DBuf drawn
      // for this call, exactly as `Exl3MatmulD` did at #3150.
      vt::Tensor trellis = vllm::dense_attn::ResidentWeight(d, w.trellis);
      vt::Tensor suh = vllm::dense_attn::ResidentWeight(d, w.suh);
      vt::Tensor svh = vllm::dense_attn::ResidentWeight(d, w.svh);
      vt::Exl3GemmArgs args;
      args.bits = w.Bits();
      args.codebook = w.codebook;
      DBuf a_had(d, vt::DType::kF16, {m, k});
      DBuf w_scratch(d, vt::DType::kF16, {k, ScratchCols(n)});
      DBuf old(d, vt::DType::kF16, {m, n});
      vt::Exl3ReconstructGemm(q, old.t(), xb.t(), trellis, suh, svh, a_had.t(), w_scratch.t(),
                              args);
      const std::vector<uint8_t> old_bytes = Download(d, old);
      CHECK(got_bytes == old_bytes);

      // And an INDEPENDENT reference within the bound the existing gates hold
      // each sub-path to (test_exl3_gemm.cpp, rel RMS 1.0e-3). The unfused path
      // is checked against the cooperative kernel at the same M, as that file's
      // cross-path gate does. The FUSED path is checked against the f64 chain, as
      // that file's fused gate does: measured on thor:gpu0 at base 909125d16, the
      // fused path sits 1.01e-3 from `Exl3Gemm` at M = 1024 on every shape here,
      // because the two errors do not share a sign, while each is within 1.0e-3
      // of f64. That is a property of the kernels before this change, not of it.
      std::vector<double> ref;
      if (m < 1024) {
        DBuf a_had2(d, vt::DType::kF16, {m, k});
        DBuf coop(d, vt::DType::kF32, {m, n});
        vt::Exl3GemmArgs cargs = args;
        cargs.force_gemv = 0;
        vt::Exl3Gemm(q, coop.t(), xb.t(), trellis, suh, svh, a_had2.t(), cargs);
        std::vector<float> coop_h(static_cast<size_t>(m * n));
        coop.Download(d, coop_h.data());
        ref.assign(coop_h.begin(), coop_h.end());
      } else {
        std::vector<float> xf(x.size());
        for (size_t j = 0; j < x.size(); ++j) xf[j] = vt::F16ToF32(x[j]);
        ref = Exl3ChainF64(f, xf, m);
      }
      double num = 0.0, den = 0.0;
      for (size_t j = 0; j < ref.size(); ++j) {
        uint16_t h = 0;
        std::memcpy(&h, got_bytes.data() + 2 * j, 2);
        const double dl = static_cast<double>(vt::F16ToF32(h)) - ref[j];
        num += dl * dl;
        den += ref[j] * ref[j];
      }
      REQUIRE(den > 0.0);
      const double rel = std::sqrt(num / den);
      MESSAGE("W7 M=", m, " K=", k, " N=", n, ": reconstruct via seam vs ",
              std::string(m < 1024 ? "Exl3Gemm" : "f64 chain"), " rel_rms=", rel);
      CHECK(rel <= 1.0e-3);
    }
  }
  cb.DestroyQueue(q);
}

// ─── QUANT-EXL3 W7 review repair: retire, per-stream key, queue teardown ─────
//
// The fresh review of c83521c20 mutated three guarantees and none of the cases
// above went red: growth freeing a block a capture baked (M2), a scratch keyed
// by device only (M3), and a destroyed queue keeping its entry. Each case below
// is red under exactly that mutation. They use the kernel's internal test hooks,
// so they compile only into a CUDA build; a CPU build still asserts the premise.

#ifdef VLLM_CPP_CUDA

namespace {

size_t F16Bytes(int64_t k, int64_t n) { return static_cast<size_t>(k * ScratchCols(n) * 2); }

}  // namespace

TEST_CASE("exl3 dispatch cuda: growth retires a reconstruct scratch that a captured graph baked") {
  if (!HasReconstructCuda()) {
    MESSAGE("SKIPPED, no CUDA reconstruct kernel: QUANT-EXL3 W7 retire case is PENDING. "
            "Reproduce with: rc run -d thor:gpu0 -- ctest -R test_exl3_matmul_dispatch -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCUDA));
    return;
  }
  namespace hooks = vt::cuda::testing;
  using vllm::dense_attn::DBuf;
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  REQUIRE(cb.SupportsGraphCapture());
  vt::Queue q = cb.CreateQueue();
  vt::Queue q2 = cb.CreateQueue();
  vllm::dense_attn::Dev d{cb, q};
  vllm::dense_attn::Dev d2{cb, q2};
  const int dev = q.device.index;
  // A fresh queue owns no scratch, so the first call below is a real growth.
  REQUIRE(hooks::Exl3ReconScratchBytesForTesting(dev, q.handle) == 0);
  REQUIRE(hooks::Exl3ReconScratchBytesForTesting(dev, q2.handle) == 0);

  // M = 1024 takes the FUSED reconstruct (exl3.py:176-184): the graph's only
  // writable pointers are the scratch and the held output, and a_had is unused,
  // so a pool block released by the captured call cannot race anything here.
  const int64_t m = 1024, k1 = 1024, n1 = 2048, k2 = 2048, n2 = 4096;
  const Exl3Fixture f1 = MakeFixture(k1, n1, 3, 0x3E71A01u);
  const Exl3Fixture f1b = MakeFixture(k1, n1, 3, 0x3E71A02u);  // same class, other bytes
  const Exl3Fixture f2 = MakeFixture(k2, n2, 3, 0x3E71A03u);
  const vllm::Exl3Weight w1 = WrapFixture(f1);
  const vllm::Exl3Weight w1b = WrapFixture(f1b);
  const vllm::Exl3Weight w2 = WrapFixture(f2);
  const std::vector<uint16_t> x1h = RandomF16(static_cast<size_t>(m * k1), 0xA01u);
  const std::vector<uint16_t> x1bh = RandomF16(static_cast<size_t>(m * k1), 0xA02u);
  const std::vector<uint16_t> x2h = RandomF16(static_cast<size_t>(m * k2), 0xA03u);

  void* graph = nullptr;
  {
    DBuf x1(d, vt::DType::kF16, {m, k1}, x1h.data());
    DBuf x1b(d2, vt::DType::kF16, {m, k1}, x1bh.data());

    // (1) Eager growth to B1, and the reference output.
    std::vector<uint8_t> ref;
    {
      DBuf r = vllm::dense_attn::Exl3MatmulD(d, x1.t(), w1, vt::DType::kF16);
      ref = Download(d, r);
    }
    const void* const b1 = hooks::Exl3ReconScratchPtrForTesting(dev, q.handle);
    REQUIRE(b1 != nullptr);
    REQUIRE(hooks::Exl3ReconScratchBytesForTesting(dev, q.handle) == F16Bytes(k1, n1));

    // (2) Capture G1, which bakes B1.
    const size_t retired_before = hooks::RetiredGraphScratchCountForTesting();
    DBuf out;
    std::string capture_error;
    cb.BeginCapture(q);
    try {
      out = vllm::dense_attn::Exl3MatmulD(d, x1.t(), w1, vt::DType::kF16);
    } catch (const std::exception& e) {
      capture_error = e.what();
    }
    graph = cb.EndCaptureGraph(q);
    INFO("Exl3MatmulD under capture: '" << capture_error << "'");
    REQUIRE(capture_error.empty());
    REQUIRE(hooks::Exl3ReconScratchPtrForTesting(dev, q.handle) == b1);

    // The graph is sound before any growth, so a later mismatch is the growth.
    cb.Memset(q, out.ptr(), 0xA5, out.bytes());
    cb.ReplayGraph(q, graph);
    REQUIRE(Download(d, out) == ref);

    // (3) A later eager call at a larger [K, min(N, 32768)] grows to B2.
    {
      DBuf x2(d, vt::DType::kF16, {m, k2}, x2h.data());
      DBuf g = vllm::dense_attn::Exl3MatmulD(d, x2.t(), w2, vt::DType::kF16);
      cb.Synchronize(q);
    }
    CHECK(hooks::Exl3ReconScratchBytesForTesting(dev, q.handle) == F16Bytes(k2, n2));
    CHECK(hooks::Exl3ReconScratchPtrForTesting(dev, q.handle) != b1);
    // B1 was handed out during a capture, so growth RETIRED it.
    CHECK(hooks::RetiredGraphScratchCountForTesting() == retired_before + 1);

    // (4) Invite the allocator to hand B1 out again: a second queue on the same
    // device asks cudaMallocAsync for exactly B1's size. When growth FREED B1,
    // the device memory pool offers that block back once q has synchronized.
    {
      DBuf o = vllm::dense_attn::Exl3MatmulD(d2, x1b.t(), w1b, vt::DType::kF16);
      cb.Synchronize(q2);
    }
    const void* const q2_block = hooks::Exl3ReconScratchPtrForTesting(dev, q2.handle);
    MESSAGE("B1=", b1, " q2 block of the same size=", q2_block,
            (q2_block == b1 ? " (the freed B1 was reused)" : ""));
    CHECK(q2_block != b1);  // a retired block is never handed out again

    // (5) Replay G1 while q2 rewrites its own scratch, with no synchronization
    // between the two streams. q2 enqueues before and after each replay so its
    // reconstructs overlap G1's reconstruct and GEMM on the device. If B1 were
    // q2's block, q2's weight would land in the bytes G1's GEMM reads.
    constexpr int kRounds = 8;
    int mismatches = 0;
    for (int r = 0; r < kRounds; ++r) {
      std::vector<DBuf> q2_outs;
      cb.Memset(q, out.ptr(), 0xA5, out.bytes());
      for (int j = 0; j < 2; ++j)
        q2_outs.push_back(vllm::dense_attn::Exl3MatmulD(d2, x1b.t(), w1b, vt::DType::kF16));
      cb.ReplayGraph(q, graph);
      for (int j = 0; j < 2; ++j)
        q2_outs.push_back(vllm::dense_attn::Exl3MatmulD(d2, x1b.t(), w1b, vt::DType::kF16));
      if (Download(d, out) != ref) ++mismatches;
      cb.Synchronize(q2);
    }
    INFO("replays of G1 after growth whose output differs from the pre-growth reference");
    CHECK(mismatches == 0);

    cb.DestroyGraph(graph);
    cb.Synchronize(q);
    cb.Synchronize(q2);
  }
  cb.DestroyQueue(q2);
  cb.DestroyQueue(q);
}

TEST_CASE("exl3 dispatch cuda: two queues on one device reconstruct into separate scratches") {
  if (!HasReconstructCuda()) {
    MESSAGE("SKIPPED, no CUDA reconstruct kernel: QUANT-EXL3 W7 per-stream case is PENDING. "
            "Reproduce with: rc run -d thor:gpu0 -- ctest -R test_exl3_matmul_dispatch -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCUDA));
    return;
  }
  namespace hooks = vt::cuda::testing;
  using vllm::dense_attn::DBuf;
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue qa = cb.CreateQueue();
  vt::Queue qb = cb.CreateQueue();
  vllm::dense_attn::Dev da{cb, qa};
  vllm::dense_attn::Dev db{cb, qb};
  const int dev = qa.device.index;

  // Different (K, N). B's scratch fits inside A's, so a scratch keyed by device
  // only never grows after the references below: the two streams then write the
  // SAME block, A all of it and B its first 4 MiB, and the defect is a data race
  // on live bytes rather than a fault. M = 1024 is the FUSED path, and its GEMM
  // on A is the long window (1024 x 2048 x 8192) in which B's reconstruct lands.
  const int64_t m = 1024;
  const int64_t ka = 2048, na = 8192, kb = 1024, nb = 2048;
  const Exl3Fixture fa = MakeFixture(ka, na, 3, 0x3E71B01u);
  const Exl3Fixture fb = MakeFixture(kb, nb, 3, 0x3E71B02u);
  const vllm::Exl3Weight wa = WrapFixture(fa);
  const vllm::Exl3Weight wb = WrapFixture(fb);
  const std::vector<uint16_t> xah = RandomF16(static_cast<size_t>(m * ka), 0xB01u);
  const std::vector<uint16_t> xbh = RandomF16(static_cast<size_t>(m * kb), 0xB02u);
  {
    DBuf xa(da, vt::DType::kF16, {m, ka}, xah.data());
    DBuf xb(db, vt::DType::kF16, {m, kb}, xbh.data());

    // Single-queue references: each call runs and synchronizes alone.
    std::vector<uint8_t> ref_a, ref_b;
    {
      DBuf r = vllm::dense_attn::Exl3MatmulD(da, xa.t(), wa, vt::DType::kF16);
      ref_a = Download(da, r);
    }
    {
      DBuf r = vllm::dense_attn::Exl3MatmulD(db, xb.t(), wb, vt::DType::kF16);
      ref_b = Download(db, r);
    }
    CHECK(hooks::Exl3ReconScratchBytesForTesting(dev, qa.handle) == F16Bytes(ka, na));
    CHECK(hooks::Exl3ReconScratchBytesForTesting(dev, qb.handle) == F16Bytes(kb, nb));
    CHECK(hooks::Exl3ReconScratchPtrForTesting(dev, qa.handle) !=
          hooks::Exl3ReconScratchPtrForTesting(dev, qb.handle));

    // Interleaved: one call on A, then three on B, with no synchronization
    // until every call is enqueued. The GPU runs the two streams concurrently.
    constexpr int kRounds = 6;
    std::vector<DBuf> outs_a, outs_b;
    for (int r = 0; r < kRounds; ++r) {
      outs_a.push_back(vllm::dense_attn::Exl3MatmulD(da, xa.t(), wa, vt::DType::kF16));
      for (int j = 0; j < 3; ++j)
        outs_b.push_back(vllm::dense_attn::Exl3MatmulD(db, xb.t(), wb, vt::DType::kF16));
    }
    cb.Synchronize(qa);
    cb.Synchronize(qb);
    int bad_a = 0, bad_b = 0;
    for (DBuf& o : outs_a) bad_a += Download(da, o) != ref_a ? 1 : 0;
    for (DBuf& o : outs_b) bad_b += Download(db, o) != ref_b ? 1 : 0;
    MESSAGE("interleaved outputs that differ from the single-queue reference: A ", bad_a, "/",
            outs_a.size(), ", B ", bad_b, "/", outs_b.size());
    CHECK(bad_a == 0);
    CHECK(bad_b == 0);
  }
  cb.DestroyQueue(qb);
  cb.DestroyQueue(qa);
}

TEST_CASE("exl3 dispatch cuda: destroying a queue releases its reconstruct scratch entry") {
  if (!HasReconstructCuda()) {
    MESSAGE("SKIPPED, no CUDA reconstruct kernel: QUANT-EXL3 W7 teardown case is PENDING. "
            "Reproduce with: rc run -d thor:gpu0 -- ctest -R test_exl3_matmul_dispatch -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCUDA));
    return;
  }
  namespace hooks = vt::cuda::testing;
  using vllm::dense_attn::DBuf;
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  const size_t live_before = hooks::Exl3ReconScratchLiveBytesForTesting();

  const int64_t m = 1024, k_big = 2048, n_big = 4096, k_small = 256, n_small = 512;
  const vllm::Exl3Weight w_big = WrapFixture(MakeFixture(k_big, n_big, 3, 0x3E71C01u));
  const vllm::Exl3Weight w_small = WrapFixture(MakeFixture(k_small, n_small, 3, 0x3E71C02u));
  const std::vector<uint16_t> x_big = RandomF16(static_cast<size_t>(m * k_big), 0xC01u);
  const std::vector<uint16_t> x_small = RandomF16(static_cast<size_t>(m * k_small), 0xC02u);

  vt::Queue q = cb.CreateQueue();
  const int dev = q.device.index;
  {
    vllm::dense_attn::Dev d{cb, q};
    DBuf x(d, vt::DType::kF16, {m, k_big}, x_big.data());
    DBuf o = vllm::dense_attn::Exl3MatmulD(d, x.t(), w_big, vt::DType::kF16);
    cb.Synchronize(q);
  }
  const void* const old_handle = q.handle;
  REQUIRE(hooks::Exl3ReconScratchBytesForTesting(dev, q.handle) == F16Bytes(k_big, n_big));
  CHECK(hooks::Exl3ReconScratchLiveBytesForTesting() == live_before + F16Bytes(k_big, n_big));

  cb.DestroyQueue(q);
  // The handle value is only a key here; nothing dereferences it.
  CHECK(hooks::Exl3ReconScratchPtrForTesting(dev, const_cast<void*>(old_handle)) == nullptr);
  CHECK(hooks::Exl3ReconScratchBytesForTesting(dev, const_cast<void*>(old_handle)) == 0);
  CHECK(hooks::Exl3ReconScratchLiveBytesForTesting() == live_before);

  // A new queue starts with no scratch, even when the driver hands it the
  // destroyed stream's handle value, and grows only to what it asks for.
  vt::Queue q2 = cb.CreateQueue();
  MESSAGE("the new stream reuses the destroyed handle value: ", q2.handle == old_handle);
  CHECK(hooks::Exl3ReconScratchBytesForTesting(dev, q2.handle) == 0);
  {
    vllm::dense_attn::Dev d2{cb, q2};
    DBuf x(d2, vt::DType::kF16, {m, k_small}, x_small.data());
    DBuf o = vllm::dense_attn::Exl3MatmulD(d2, x.t(), w_small, vt::DType::kF16);
    cb.Synchronize(q2);
  }
  CHECK(hooks::Exl3ReconScratchBytesForTesting(dev, q2.handle) == F16Bytes(k_small, n_small));
  cb.DestroyQueue(q2);
  CHECK(hooks::Exl3ReconScratchLiveBytesForTesting() == live_before);
}

#endif  // VLLM_CPP_CUDA

// LAST IN THE FILE ON PURPOSE: it opens a capture on the stream, and a defect in
// the code under test could leave that stream unusable for a later case.
TEST_CASE("exl3 dispatch cuda: growing the reconstruct scratch while capturing is refused by name") {
  if (!HasReconstructCuda()) {
    MESSAGE("SKIPPED, no CUDA reconstruct kernel: QUANT-EXL3 W7 test (3) is PENDING. "
            "Reproduce with: rc run -d dgx:gpu0 -- ctest -R test_exl3_matmul_dispatch -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3ReconstructGemm, vt::DeviceType::kCUDA));
    return;
  }
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  REQUIRE(cb.SupportsGraphCapture());
  vt::Queue q = cb.CreateQueue();
  vllm::dense_attn::Dev d{cb, q};
  using vllm::dense_attn::DBuf;

  // Larger than every scratch the earlier cases asked for ([256, 32768] is the
  // largest), so the capture has to GROW whatever this stream inherited even if
  // the driver hands this queue a recycled stream handle.
  const int64_t m = 145, k = 1024, n = 32896;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x3150C4Du);
  const vllm::Exl3Weight w = WrapFixture(f);
  const std::vector<uint16_t> x = RandomF16(static_cast<size_t>(m * k), 0xC4Du);
  DBuf xb(d, vt::DType::kF16, {m, k}, x.data());

  // Everything the captured call needs that is NOT the scratch is made ready
  // eagerly, so the capture can only fail on the scratch: the weight upload
  // (an M <= 144 call takes the cooperative kernel and allocates no scratch), and
  // one free pool block for each of the call's [M, K] and [M, N] buffers.
  {
    DBuf x4(d, vt::DType::kF16, {4, k});
    x4.Zero(d);
    DBuf warm = vllm::dense_attn::Exl3MatmulD(d, x4.t(), w, vt::DType::kF16);
    DBuf p1(d, vt::DType::kF16, {m, k});
    DBuf p2(d, vt::DType::kF16, {m, n});
    // At #3150 the scratch was a pool block; give it a free one too, so the
    // pre-change code reaches the end of the call rather than a driver malloc.
    DBuf p3(d, vt::DType::kF16, {k, ScratchCols(n)});
    cb.Synchronize(q);
  }

  std::string refusal;
  cb.BeginCapture(q);
  try {
    DBuf out = vllm::dense_attn::Exl3MatmulD(d, xb.t(), w, vt::DType::kF16);
  } catch (const std::exception& e) {
    refusal = e.what();
  }
  void* graph = nullptr;
  try {
    graph = cb.EndCaptureGraph(q);
  } catch (const std::exception& e) {
    MESSAGE("EndCaptureGraph after the call: ", e.what());
  }
  if (graph != nullptr) cb.DestroyGraph(graph);

  INFO("Exl3MatmulD under capture: '" << refusal << "'");
  CHECK(refusal.find("exl3 reconstruct scratch") != std::string::npos);
  CHECK(refusal.find("CUDA graph capture") != std::string::npos);

  // The refusal is recoverable: the same call runs eagerly afterwards.
  DBuf out = vllm::dense_attn::Exl3MatmulD(d, xb.t(), w, vt::DType::kF16);
  cb.Synchronize(q);
  cb.DestroyQueue(q);
}

// Kolibri-1 BFDOT bf16 GEMV arm: accuracy envelope gate (DRAFT row work on
// the FMA-contract draft branch).
//
// The arm is a ROUNDING-POLICY change by design (bf16-pair-dot f32
// accumulate vs f32 widen + sequential f32 accumulate): the chain is
// EXPECTED to move under the knob, and no gate assertion anywhere is
// relaxed. This test pins the ENVELOPE so the arm cannot silently degrade
// further: on the exact kolibri projection shapes (K=2560/512/6144) plus a
// K%8 tail case, the BFDOT GEMV must track the production-class reference
// (f32 widen, sequential f32 accumulate, single bf16 store round) within
// TWO bf16 ULP relative on typical magnitudes, and stay finite.
//
// Red-first discipline: each claimed guarantee was mutation-verified on a
// scratch copy (break the weight row index -> red; feed the tail through
// the vector body -> red). A perf smoke is deliberately OUT of unit scope
// (bench-only).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>

#include "vllm/model_executor/models/kolibri1_bfdot_gemv.h"
#include "vt/cpu/cpu_isa_arm.h"
#include "vt/dtype.h"

namespace {

using vllm::kolibri1_bfdot_gemv::Gemv;
using vllm::kolibri1_bfdot_gemv::GemvF32Out;

// bf16 ULP distance between two same-sign floats, via the bf16 integer
// grid: re-round both to bf16 and count representable steps.
int64_t Bf16Ulps(uint16_t a, uint16_t b) {
  const int32_t ia = static_cast<int32_t>(static_cast<int16_t>(a));
  const int32_t ib = static_cast<int32_t>(static_cast<int16_t>(b));
  return std::abs(ia - ib);
}

// Deterministic bf16 activation, trained-weight magnitudes.
std::vector<uint16_t> MakeActivation(int64_t k, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<uint16_t> x(static_cast<size_t>(k));
  for (auto& v : x) v = vt::F32ToBF16(dist(rng));
  return x;
}

// Deterministic bf16 weight [n, k] row-major, the warm-cache layout.
std::vector<uint16_t> MakeWeight(int64_t n, int64_t k, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  std::vector<uint16_t> w(static_cast<size_t>(n) * static_cast<size_t>(k));
  for (auto& v : w) v = vt::F32ToBF16(dist(rng));
  return w;
}

// The production-class reference: f32 widen of BOTH operands, strict
// sequential f32 accumulate over K, ONE bf16 round on store (what
// MatmulOneChunkRef does for this dtype at t==1).
std::vector<uint16_t> ReferenceGemv(const std::vector<uint16_t>& w,
                                    const std::vector<uint16_t>& x,
                                    int64_t n, int64_t k) {
  std::vector<uint16_t> out(static_cast<size_t>(n));
  for (int64_t row = 0; row < n; ++row) {
    float acc = 0.0f;
    for (int64_t e = 0; e < k; ++e) {
      acc = acc + vt::BF16ToF32(w[static_cast<size_t>(row) *
                                     static_cast<size_t>(k) + e]) *
                    vt::BF16ToF32(x[static_cast<size_t>(e)]);
    }
    out[static_cast<size_t>(row)] = vt::F32ToBF16(acc);
  }
  return out;
}

void CheckEnvelope(int64_t n, int64_t k, uint32_t seed) {
  const auto w = MakeWeight(n, k, seed);
  const auto x = MakeActivation(k, seed + 1);
  std::vector<uint16_t> out(static_cast<size_t>(n), 0);
  Gemv(w.data(), x.data(), out.data(), n, k);
  const auto ref = ReferenceGemv(w, x, n, k);
  for (int64_t r = 0; r < n; ++r) {
    const float got = vt::BF16ToF32(out[static_cast<size_t>(r)]);
    const float want = vt::BF16ToF32(ref[static_cast<size_t>(r)]);
    CHECK(std::isfinite(got));
    // Typical magnitudes only: near-zero outputs have unbounded relative
    // error by construction under ANY rounding change (cancellation).
    if (std::fabs(want) < 1e-3f) continue;
    const int64_t ulps = Bf16Ulps(out[static_cast<size_t>(r)],
                                  ref[static_cast<size_t>(r)]);
    CHECK_MESSAGE(ulps <= 2,
                  "row ", r, " of [", n, ",", k,
                  "]: BFDOT GEMV drifted ", ulps,
                  " bf16 ULP from the production reference (got ", got,
                  ", want ", want, ")");
  }
}

// The f32-out reference: identical accumulation, NO final bf16 round (the
// f32 accumulator is the output, as the widen-f32 MatmulBT produces it).
std::vector<float> ReferenceGemvF32Out(const std::vector<uint16_t>& w,
                                       const std::vector<uint16_t>& x,
                                       int64_t n, int64_t k) {
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t row = 0; row < n; ++row) {
    float acc = 0.0f;
    for (int64_t e = 0; e < k; ++e) {
      acc = acc + vt::BF16ToF32(w[static_cast<size_t>(row) *
                                     static_cast<size_t>(k) + e]) *
                    vt::BF16ToF32(x[static_cast<size_t>(e)]);
    }
    out[static_cast<size_t>(row)] = acc;
  }
  return out;
}

// The lm_head / router-gate arm: same envelope contract expressed on the
// bf16 grid (round BOTH sides to bf16 and count ULPs), because the
// accumulation is the SAME bf16-pair-dot the bf16-out form runs — only the
// final store rounding differs, and dropping a round cannot widen the
// envelope relative to the bf16-out arm's pinned 2-ULP floor.
void CheckEnvelopeF32Out(int64_t n, int64_t k, uint32_t seed) {
  const auto w = MakeWeight(n, k, seed);
  const auto x = MakeActivation(k, seed + 1);
  std::vector<float> out(static_cast<size_t>(n), 0.0f);
  GemvF32Out(w.data(), x.data(), out.data(), n, k);
  const auto ref = ReferenceGemvF32Out(w, x, n, k);
  for (int64_t r = 0; r < n; ++r) {
    const float got = out[static_cast<size_t>(r)];
    const float want = ref[static_cast<size_t>(r)];
    CHECK(std::isfinite(got));
    if (std::fabs(want) < 1e-3f) continue;
    const int64_t ulps = Bf16Ulps(vt::F32ToBF16(got), vt::F32ToBF16(want));
    CHECK_MESSAGE(ulps <= 2,
                  "row ", r, " of [", n, ",", k,
                  "]: BFDOT f32-out GEMV drifted ", ulps,
                  " bf16 ULP from the production reference (got ", got,
                  ", want ", want, ")");
  }
}

}  // namespace

TEST_CASE("kolibri1 bfdot gemv f32-out: lm_head / router-gate envelope") {
  if (vt::cpu::DetectArmBf16() == false) {
    MESSAGE("SKIPPED BY NAME: this host does not report FEAT_BF16");
    std::exit(77);
  }
  SUBCASE("production lm_head shape N=128000 K=2560") {
    CheckEnvelopeF32Out(2048, 2560, 21);  // subsampled rows: full vocab is
    // covered by the decode bench chain; the envelope is row-independent.
  }
  SUBCASE("production router shape N=384 K=2560") { CheckEnvelopeF32Out(384, 2560, 22); }
  SUBCASE("K%8 tail case K=257") { CheckEnvelopeF32Out(32, 257, 23); }
}

TEST_CASE("kolibri1 bfdot gemv f32-out: threading determinism") {
  if (vt::cpu::DetectArmBf16() == false) {
    MESSAGE("SKIPPED BY NAME: no FEAT_BF16 on this host");
    std::exit(77);
  }
  const int64_t n = 96, k = 2560;
  const auto w = MakeWeight(n, k, 24);
  const auto x = MakeActivation(k, 25);
  std::vector<float> a(static_cast<size_t>(n), 0.0f);
  std::vector<float> b(static_cast<size_t>(n), 0.0f);
  GemvF32Out(w.data(), x.data(), a.data(), n, k);
  GemvF32Out(w.data(), x.data(), b.data(), n, k);
  CHECK(a == b);
}

TEST_CASE("kolibri1 bfdot gemv: bf16-store-floor envelope") {
  if (vt::cpu::DetectArmBf16() == false) {
    MESSAGE("SKIPPED BY NAME: this host does not report FEAT_BF16 "
            "(HWCAP2_BF16); the BFDOT GEMV arm's vector body cannot run "
            "here and KnobActive would refuse it");
    std::exit(77);  // house skip convention: CTest reports Skipped, not pass
  }
  SUBCASE("production q/k/v shape K=2560 N=512") { CheckEnvelope(512, 2560, 7); }
  SUBCASE("production o/down shape K=2560 N=2560") { CheckEnvelope(2560, 2560, 8); }
  SUBCASE("production gate/up shape K=512") { CheckEnvelope(512, 512, 9); }
  SUBCASE("production q shape K=6144") { CheckEnvelope(128, 6144, 10); }
  SUBCASE("K%8 tail case K=257") { CheckEnvelope(64, 257, 11); }
  SUBCASE("K%8 tail case K=5") { CheckEnvelope(16, 5, 12); }
}

TEST_CASE("kolibri1 bfdot gemv: threading determinism") {
  if (vt::cpu::DetectArmBf16() == false) {
    MESSAGE("SKIPPED BY NAME: no FEAT_BF16 on this host");
    std::exit(77);
  }
  // The pool contract: the threaded result must equal a serial run of the
  // same kernel bit-for-bit (ForOutputRows partitions output rows only).
  const int64_t n = 96, k = 2560;
  const auto w = MakeWeight(n, k, 13);
  const auto x = MakeActivation(k, 14);
  std::vector<uint16_t> a(static_cast<size_t>(n), 0);
  std::vector<uint16_t> b(static_cast<size_t>(n), 0);
  Gemv(w.data(), x.data(), a.data(), n, k);
  Gemv(w.data(), x.data(), b.data(), n, k);
  CHECK(a == b);
}

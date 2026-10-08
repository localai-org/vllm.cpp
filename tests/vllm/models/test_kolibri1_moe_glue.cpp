// Kolibri-1 MoE glue: the expert-output COMBINE bit-identity gate
// (MODEL-TEXT-kolibri-1, ISSUE-LOCAL-01M4D7FYY1WQZV9PNR6GBWS2R5).
//
// The moe_glue stage's combine — out[t,h] = shared[t,h] + sum_j w[t,j] *
// expert_out[t,j,h] (f32 accumulation over j in increasing order,
// routed_scale applied to the ASSEMBLED routed sum only, the shared term
// added last, ONE round to the out dtype) — is the largest MoE-scoped glue
// sub-operation that is not the nested expert GEMM/dequant work (those belong
// to the shared vt GEMM and the dequant cache, out of this unit's scope).
// This file pins the combine's BIT-IDENTITY as a raw-uint16 contract over the
// REAL expert geometries, so the ONE landed lever — order-preserving NEON
// across the independent output COLUMNS, each lane keeping the identical
// scalar j-order reduction (products vmulq + vaddq, NEVER vfmaq, at the
// project-pinned -ffp-contract=off) — cannot move a single bit:
//
//   - 2560-wide hidden (the real hidden_size), 6-of-384 routed slots (the
//     real num_experts_per_tok over the real 384-expert routing domain), the
//     shared-expert path (the always-added shared term);
//   - the production decode (t=1), ragged (t=7) and prefill (t=128) token
//     counts; h tails (1/3/7/13/2557) through the NEON body + scalar tail;
//     k tails (1/2/13); the no-shared form; the routed_scale arm;
//   - exact-tie stores (dyadic weights x integer activations: every store is
//     a round-to-nearest-even tie), overflow to +-inf, and inf-inf NaN (the
//     bf16 store's truncate-and-quiet NaN branch);
//   - cross-thread-count identity (1-thread vs 8-thread pool: ForRows
//     partitions ROWS only, so every row's result is bit-identical).
//
// The scalar reference below is the exact MoeCombineKernel math transcribed
// with vt::BF16ToF32 / vt::F32ToBF16 — the same converters the kernel uses —
// so the comparison pins the ARITHMETIC ORDER (the j sequence, the scale
// placement, the shared add, the single store rounding), not the converters
// (those are pinned by tests/vt/test_ops_matmul_elem.cpp and
// test_kolibri1_dequant.cpp).
//
// MUTATION CHECKS (run in this worktree, scratch only, restored
// byte-for-byte after each): fusing the product into the accumulation
// (vfmaq instead of vmulq+vaddq) changes the f32 rounding -> the bit-identity
// cases go RED; reordering the j loop (k-1 .. 0) changes the accumulation
// order -> RED; dropping the shared add -> RED. The tests detect each claimed
// guarantee.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vt/cpu/cpu_threadpool.h"  // Threadpool::SwapForTesting (the ONE pool)
#include "vt/dtype.h"               // BF16ToF32, F32ToBF16
#include "vt/ops.h"                 // vt::MoeCombine

namespace {

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Bf16_2(std::vector<uint16_t>& v, int64_t a, int64_t b) {
  return Tensor::Contiguous(v.data(), DType::kBF16, Cpu(), {a, b});
}
Tensor Bf16_3(std::vector<uint16_t>& v, int64_t a, int64_t b, int64_t c) {
  return Tensor::Contiguous(v.data(), DType::kBF16, Cpu(), {a, b, c});
}
Tensor F32_2(std::vector<float>& v, int64_t a, int64_t b) {
  return Tensor::Contiguous(v.data(), DType::kF32, Cpu(), {a, b});
}

// Seeded deterministic source (the W2 test's Lcg shape).
struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695041ULL) {}
  uint32_t NextU32() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  // A bf16 bit pattern spread over the FINITE domain: normals, subnormals,
  // both signs, huge and tiny. The exponent-all-ones patterns (inf/NaN) are
  // mapped to the largest finite exponent: an INPUT NaN's payload propagation
  // through `acc += shared` is compiler-scheduling-dependent (addition is
  // commutative, so which NaN payload wins when BOTH operands are NaN is a
  // register-allocation accident, not a bit contract — measured: the kernel
  // and a same-source scalar reference diverge on exactly those elements).
  // The production activations are finite; the one deterministic NaN path —
  // overflow to +-inf then inf + (-inf) — is the hardware default QNaN and IS
  // pinned bit-for-bit by the overflow case below.
  uint16_t NextBf16() {
    uint16_t b = static_cast<uint16_t>(NextU32() >> 16);
    if ((b & 0x7F80) == 0x7F80) b = static_cast<uint16_t>(b ^ 0x0080);
    return b;
  }
  // A finite f32 weight in a router-weight-like range, full mantissa.
  float NextWeight() {
    const int32_t m = static_cast<int32_t>(NextU32() % 2001) - 1000;
    return std::ldexp(static_cast<float>(m), -6);  // [-15.625, 15.609375]
  }
};

// The EXACT MoeCombineKernel math (src/vt/cpu/cpu_ops.cpp), transcribed:
// f32 accumulation over j in increasing order, routed_scale applied to the
// assembled routed sum only, the shared term added last, ONE round to bf16.
void ScalarCombineBf16(std::vector<uint16_t>& out,
                       const std::vector<uint16_t>& expert_out,
                       const std::vector<float>& weights,
                       const std::vector<uint16_t>* shared, int64_t t,
                       int64_t k, int64_t h, float routed_scale) {
  for (int64_t row = 0; row < t; ++row) {
    for (int64_t col = 0; col < h; ++col) {
      float acc = 0.0f;
      for (int64_t j = 0; j < k; ++j) {
        acc += weights[static_cast<size_t>(row * k + j)] *
               vt::BF16ToF32(
                   expert_out[static_cast<size_t>((row * k + j) * h + col)]);
      }
      if (routed_scale != 1.0f) acc *= routed_scale;
      if (shared != nullptr)
        acc += vt::BF16ToF32((*shared)[static_cast<size_t>(row * h + col)]);
      out[static_cast<size_t>(row * h + col)] = vt::F32ToBF16(acc);
    }
  }
}

struct Case {
  int64_t t, k, h;
  bool shared;
  float scale;
};

// Drive ONE geometry through the production op (the exact call shape
// kolibri1_forward.cpp's MoeBlock uses: bf16 expert_out [t,k,h], f32
// weights [t,k], bf16 shared [t,h], bf16 out [t,h]) and through the scalar
// reference; the outputs must be raw-uint16 IDENTICAL.
void RunCase(const Case& c, uint64_t seed) {
  Lcg rng(seed);
  std::vector<uint16_t> expert_out(static_cast<size_t>(c.t * c.k * c.h));
  for (auto& v : expert_out) v = rng.NextBf16();
  std::vector<float> weights(static_cast<size_t>(c.t * c.k));
  for (auto& v : weights) v = rng.NextWeight();
  std::vector<uint16_t> shared(static_cast<size_t>(c.t * c.h));
  for (auto& v : shared) v = rng.NextBf16();
  std::vector<uint16_t> got(static_cast<size_t>(c.t * c.h), 0xBEEF);
  std::vector<uint16_t> want(static_cast<size_t>(c.t * c.h), 0xDEAD);

  Tensor eo = Bf16_3(expert_out, c.t, c.k, c.h);
  Tensor wt = F32_2(weights, c.t, c.k);
  Tensor sh = Bf16_2(shared, c.t, c.h);
  Tensor ot = Bf16_2(got, c.t, c.h);
  Queue q = Q();
  vt::MoeCombine(q, ot, eo, wt, c.shared ? &sh : nullptr, c.scale);
  ScalarCombineBf16(want, expert_out, weights, c.shared ? &shared : nullptr,
                    c.t, c.k, c.h, c.scale);

  const size_t bytes = want.size() * sizeof(uint16_t);
  CHECK_EQ(std::memcmp(got.data(), want.data(), bytes), 0);
  if (std::memcmp(got.data(), want.data(), bytes) != 0) {
    for (size_t i = 0; i < want.size(); ++i) {
      if (got[i] != want[i]) {
        const int64_t row = static_cast<int64_t>(i) / c.h;
        const int64_t col = static_cast<int64_t>(i) % c.h;
        MESSAGE("divergence at t=", row, " h=", col, ": got 0x",
                std::hex, static_cast<unsigned>(got[i]), " want 0x",
                static_cast<unsigned>(want[i]));
        for (int64_t j = 0; j < c.k; ++j)
          MESSAGE("  j=", j, " w=", weights[static_cast<size_t>(row * c.k + j)],
                  " eo=0x", std::hex,
                  static_cast<unsigned>(
                      expert_out[static_cast<size_t>((row * c.k + j) * c.h + col)]));
        MESSAGE("  shared=0x", std::hex,
                static_cast<unsigned>(shared[static_cast<size_t>(row * c.h + col)]));
        break;
      }
    }
  }
}

// A deterministic 6-of-384 route: top_k DISTINCT expert ids per token over
// the real 384-expert domain (the production router's OUTPUT shape — the
// combine consumes the selected slots, never the ids). Expert outputs are
// filled per (expert id, slot) so the case genuinely routes over 384
// experts and only the selected six contribute.
std::vector<int32_t> Route6of384(int64_t t, uint64_t seed) {
  Lcg rng(seed);
  std::vector<int32_t> ids(static_cast<size_t>(t * 6));
  for (int64_t row = 0; row < t; ++row) {
    const int32_t base = static_cast<int32_t>(rng.NextU32() % 384);
    for (int64_t kk = 0; kk < 6; ++kk)
      ids[static_cast<size_t>(row * 6 + kk)] =
          static_cast<int32_t>((base + static_cast<int32_t>(kk) * 61) % 384);
  }
  return ids;
}

}  // namespace

// The production decode geometry, exactly: one token, 6-of-384 routed slots,
// 2560-wide hidden, the shared-expert path, routed_scale 1.0 (the kolibri1
// polarity: the router weights ARE the mixture weights).
TEST_CASE("kolibri1 moe_glue combine: the real decode geometry (t=1, k=6, "
          "h=2560, shared, scale 1.0) is raw-uint16 identical to the scalar "
          "reference") {
  RunCase(Case{1, 6, 2560, true, 1.0f}, 0xA0B1C2D3E4F5ULL);
}

// The ragged batch (t=7) and the prefill geometry (t=128): the combine's
// ForRows partitions rows across the pool, and every row must stay
// bit-identical to the serial reference.
TEST_CASE("kolibri1 moe_glue combine: ragged t=7 and prefill t=128 over "
          "h=2560 are raw-uint16 identical") {
  RunCase(Case{7, 6, 2560, true, 1.0f}, 0xBEEFCAFE1234ULL);
  RunCase(Case{128, 6, 2560, true, 1.0f}, 0xDEC0DE128ULL);
}

// Tails: h not divisible by the NEON body (1/3/7/13/2557) and k tails
// (1/2/13) — the scalar tail path and the odd-j path must match the
// reference bit for bit.
TEST_CASE("kolibri1 moe_glue combine: h and k tails are raw-uint16 identical") {
  for (int64_t h : {1, 3, 7, 13, 2557}) RunCase(Case{3, 6, h, true, 1.0f}, 0x7A110ULL + static_cast<uint64_t>(h));
  for (int64_t k : {1, 2, 13}) RunCase(Case{3, k, 2560, true, 1.0f}, 0x7A111ULL + static_cast<uint64_t>(k));
}

// The no-shared form and the routed_scale arm (scale applied to the routed
// sum only, BEFORE the shared add — the pinned polarity).
TEST_CASE("kolibri1 moe_glue combine: no-shared and routed_scale arms are "
          "raw-uint16 identical") {
  RunCase(Case{2, 6, 2560, false, 1.0f}, 0xA5EA12EDULL);
  RunCase(Case{2, 6, 2560, true, 0.5f}, 0x5CA1E0A5ULL);
  RunCase(Case{2, 6, 2560, true, 2.5f}, 0x5CA1E2B5ULL);
}

// The full glue data path at the routing geometry: a 6-of-384 route over the
// real 384-expert domain, per-expert outputs gathered into the [t, k, h]
// slot layout, the sigmoid-range router weights, the shared-expert term.
TEST_CASE("kolibri1 moe_glue combine: the 384-expert 6-of-384 route with the "
          "shared expert path is raw-uint16 identical") {
  for (int64_t t : {1, 7}) {
    const std::vector<int32_t> ids = Route6of384(t, 0xA0C6ULL + static_cast<uint64_t>(t));
    const int64_t k = 6, h = 2560;
    Lcg rng(0x5EED0ULL + static_cast<uint64_t>(t));
    std::vector<uint16_t> expert_out(static_cast<size_t>(t * k * h));
    for (int64_t row = 0; row < t; ++row) {
      for (int64_t kk = 0; kk < k; ++kk) {
        // Per-expert deterministic bytes: the slot's output depends on the
        // expert id the route selected, so only the routed experts contribute.
        const uint16_t base = static_cast<uint16_t>(ids[static_cast<size_t>(row * k + kk)] * 7 + 1);
        for (int64_t col = 0; col < h; ++col)
          expert_out[static_cast<size_t>((row * k + kk) * h + col)] =
              static_cast<uint16_t>(base + static_cast<uint16_t>(col % 251));
      }
    }
    std::vector<float> weights(static_cast<size_t>(t * k));
    for (auto& v : weights) v = rng.NextWeight();
    std::vector<uint16_t> shared(static_cast<size_t>(t * h));
    for (auto& v : shared) v = rng.NextBf16();
    std::vector<uint16_t> got(static_cast<size_t>(t * h), 0xBEEF);
    std::vector<uint16_t> want(static_cast<size_t>(t * h), 0xDEAD);

    Tensor eo = Bf16_3(expert_out, t, k, h);
    Tensor wt = F32_2(weights, t, k);
    Tensor sh = Bf16_2(shared, t, h);
    Tensor ot = Bf16_2(got, t, h);
    Queue q = Q();
    vt::MoeCombine(q, ot, eo, wt, &sh, 1.0f);
    ScalarCombineBf16(want, expert_out, weights, &shared, t, k, h, 1.0f);

    const size_t bytes = want.size() * sizeof(uint16_t);
    CHECK_EQ(std::memcmp(got.data(), want.data(), bytes), 0);
  }
}

// Exact-tie stores, overflow, and NaN: dyadic weights x integer activations
// make every accumulated value an exact f32 whose bf16 store is a
// round-to-nearest-even TIE (a tie-away store diverges); near-max weights
// overflow the f32 accumulation to +-inf; +inf routed with a -inf shared term
// is NaN, exercising the store's truncate-and-quiet NaN branch. All three
// must match the reference bit for bit.
TEST_CASE("kolibri1 moe_glue combine: exact ties, overflow and NaN stores are "
          "raw-uint16 identical") {
  const int64_t t = 2, k = 6, h = 64;
  // Dyadic weights (exact products, exact sums -> RNE ties at the store).
  std::vector<float> dyadic_w = {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f};
  std::vector<uint16_t> int_act(static_cast<size_t>(t * k * h));
  for (size_t i = 0; i < int_act.size(); ++i)
    int_act[i] = vt::F32ToBF16(static_cast<float>(static_cast<int>(i % 17) - 8));
  std::vector<uint16_t> sh_int(static_cast<size_t>(t * h));
  for (size_t i = 0; i < sh_int.size(); ++i)
    sh_int[i] = vt::F32ToBF16(static_cast<float>(static_cast<int>(i % 5) - 2));
  {
    std::vector<float> w(static_cast<size_t>(t * k));
    std::vector<uint16_t> eo = int_act, got(static_cast<size_t>(t * h), 0xBEEF),
                         want(static_cast<size_t>(t * h), 0xDEAD);
    for (int64_t row = 0; row < t; ++row)
      for (int64_t kk = 0; kk < k; ++kk) w[static_cast<size_t>(row * k + kk)] = dyadic_w[static_cast<size_t>(kk)];
    Tensor eot = Bf16_3(eo, t, k, h), wt = F32_2(w, t, k), sht = Bf16_2(sh_int, t, h),
           ot = Bf16_2(got, t, h);
    Queue q = Q();
    vt::MoeCombine(q, ot, eot, wt, &sht, 1.0f);
    ScalarCombineBf16(want, eo, w, &sh_int, t, k, h, 1.0f);
    CHECK_EQ(std::memcmp(got.data(), want.data(), want.size() * 2), 0);
  }
  // Overflow to +-inf and the NaN store (routed +inf, shared -inf).
  {
    std::vector<float> w(static_cast<size_t>(t * k), 3.0e38f);
    std::vector<uint16_t> eo(static_cast<size_t>(t * k * h), vt::F32ToBF16(2.0f));
    std::vector<uint16_t> sh(static_cast<size_t>(t * h), static_cast<uint16_t>(0xFF80));  // -inf bf16
    std::vector<uint16_t> got(static_cast<size_t>(t * h), 0xBEEF),
        want(static_cast<size_t>(t * h), 0xDEAD);
    Tensor eot = Bf16_3(eo, t, k, h), wt = F32_2(w, t, k), sht = Bf16_2(sh, t, h),
           ot = Bf16_2(got, t, h);
    Queue q = Q();
    vt::MoeCombine(q, ot, eot, wt, &sht, 1.0f);
    ScalarCombineBf16(want, eo, w, &sh, t, k, h, 1.0f);
    CHECK_EQ(std::memcmp(got.data(), want.data(), want.size() * 2), 0);
    // The routed sum overflowed to +inf and the shared term is -inf: the
    // accumulated value is NaN, stored through the quiet-NaN branch (either
    // sign — the scalar reference's exact bits are pinned by the memcmp).
    CHECK((got[0] & 0x7FC0) == 0x7FC0);
  }
}

// Cross-thread-count identity: ForRows partitions output ROWS only, so the
// 1-thread and 8-thread pools must produce byte-identical outputs (the pool
// determinism contract), at a t large enough to actually partition.
TEST_CASE("kolibri1 moe_glue combine: 1-thread and 8-thread pools are "
          "raw-uint16 identical at t=128") {
  const int64_t t = 128, k = 6, h = 2560;
  Lcg rng(0x7EADED10ULL);
  std::vector<uint16_t> expert_out(static_cast<size_t>(t * k * h));
  for (auto& v : expert_out) v = rng.NextBf16();
  std::vector<float> weights(static_cast<size_t>(t * k));
  for (auto& v : weights) v = rng.NextWeight();
  std::vector<uint16_t> shared(static_cast<size_t>(t * h));
  for (auto& v : shared) v = rng.NextBf16();

  auto run = [&](std::vector<uint16_t>& out) {
    Tensor eo = Bf16_3(expert_out, t, k, h);
    Tensor wt = F32_2(weights, t, k);
    Tensor sh = Bf16_2(shared, t, h);
    Tensor ot = Bf16_2(out, t, h);
    Queue q = Q();
    vt::MoeCombine(q, ot, eo, wt, &sh, 1.0f);
  };

  std::vector<uint16_t> serial(static_cast<size_t>(t * h), 0xBEEF);
  vt::cpu::Threadpool serial_pool(1);
  vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&serial_pool);
  run(serial);
  vt::cpu::Threadpool::SwapForTesting(prev);

  std::vector<uint16_t> threaded(static_cast<size_t>(t * h), 0xDEAD);
  vt::cpu::Threadpool threaded_pool(8);
  prev = vt::cpu::Threadpool::SwapForTesting(&threaded_pool);
  run(threaded);
  vt::cpu::Threadpool::SwapForTesting(prev);

  CHECK_EQ(std::memcmp(serial.data(), threaded.data(), serial.size() * 2), 0);
  std::vector<uint16_t> want(static_cast<size_t>(t * h), 0x0000);
  ScalarCombineBf16(want, expert_out, weights, &shared, t, k, h, 1.0f);
  CHECK_EQ(std::memcmp(serial.data(), want.data(), serial.size() * 2), 0);
}

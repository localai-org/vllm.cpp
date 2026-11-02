// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RopeArgs;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }

// Deterministic fill: the same values on every run and at every thread count.
std::vector<float> Fill(int64_t n, uint32_t seed) {
  std::vector<float> v(static_cast<size_t>(n));
  uint32_t s = seed * 2654435761u + 1u;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;  // [-1, 1)
  }
  return v;
}

// Independent Neox reference over [T, heads, D]: rotates the first `rot` dims
// of every head in pair form and leaves dims [rot, D) untouched. The cos/sin
// factors are computed here from the same formula, never by calling the kernel.
std::vector<float> NeoxReference(const std::vector<float>& in,
                                 const std::vector<int32_t>& pos, int64_t t,
                                 int64_t heads, int64_t d, int rot, double base) {
  std::vector<float> out = in;
  const int half = rot / 2;
  for (int64_t i = 0; i < t; ++i) {
    for (int64_t hh = 0; hh < heads; ++hh) {
      const size_t off = static_cast<size_t>((i * heads + hh) * d);
      for (int j = 0; j < half; ++j) {
        const double freq = std::pow(base, -2.0 * j / rot);
        const double angle = static_cast<double>(pos[static_cast<size_t>(i)]) * freq;
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>(std::sin(angle));
        const float x = in[off + static_cast<size_t>(j)];
        const float y = in[off + static_cast<size_t>(j + half)];
        out[off + static_cast<size_t>(j)] = x * c - y * s;
        out[off + static_cast<size_t>(j + half)] = x * s + y * c;
      }
    }
  }
  return out;
}

// Sentinels on both sides of a logical tensor. 32 words (four heads at D=8)
// covers the widest out-of-range write the two GQA-split mutants make: 16
// floats before k when the branch test flips to `hh < hk`, and 30 floats after
// k when the `hh - hq` shift is dropped. The small cases need only 4 words, but
// one guard width keeps every kernel-running case on the same harness. A
// mutant that leaves the logical extent writes a guard word instead of the
// heap, so the assertion reports it and the process exits normally.
constexpr int64_t kRopeGuard = 32;

template <typename T>
T GuardPattern(size_t i, uint32_t seed);

template <>
float GuardPattern<float>(size_t i, uint32_t seed) {
  uint32_t s = seed + static_cast<uint32_t>(i) * 2654435761u;
  s = s * 1664525u + 1013904223u;
  return (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;  // [-1, 1)
}

template <>
uint16_t GuardPattern<uint16_t>(size_t i, uint32_t seed) {
  uint32_t s = seed + static_cast<uint32_t>(i) * 2654435761u;
  s = s * 1664525u + 1013904223u;
  // bf16 with exponent 0x7F: a finite value in [1, 2).
  return static_cast<uint16_t>(((s >> 16) & 0x8000u) | 0x3F80u | ((s >> 8) & 0x007Fu));
}

template <typename T>
struct Guarded {
  std::vector<T> data;
  std::vector<T> before;
  std::vector<T> after;
};

template <typename T>
Guarded<T> MakeGuarded(const std::vector<T>& logical, uint32_t seed) {
  const size_t n = logical.size();
  const size_t guard = static_cast<size_t>(kRopeGuard);
  Guarded<T> g;
  g.data.resize(n + 2 * guard);
  for (size_t i = 0; i < g.data.size(); ++i) g.data[i] = GuardPattern<T>(i, seed);
  std::copy(logical.begin(), logical.end(), g.data.begin() + kRopeGuard);
  g.before.assign(g.data.begin(), g.data.begin() + guard);
  g.after.assign(g.data.end() - guard, g.data.end());
  return g;
}

template <typename T>
void CheckGuards(const Guarded<T>& g, size_t logical, const char* name) {
  const size_t guard = static_cast<size_t>(kRopeGuard);
  for (size_t i = 0; i < guard; ++i) {
    CHECK_MESSAGE(std::memcmp(&g.data[i], &g.before[i], sizeof(T)) == 0,
                  std::string(name) << " before-guard word " << i << " changed");
  }
  for (size_t i = 0; i < guard; ++i) {
    CHECK_MESSAGE(std::memcmp(&g.data[guard + logical + i], &g.after[i], sizeof(T)) == 0,
                  std::string(name) << " after-guard word " << i << " changed");
  }
}
}  // namespace

TEST_CASE("rope neox rotates pairs at pos=1, full rotary") {
  // 1 token, 1 head, head_dim=4, rotary_dim=4, base=10000
  // inv_freq = [1, 10000^(-0.5)=0.01]
  // pairs: (0,2) angle=1.0; (1,3) angle=0.01
  // q = [1, 0, 0, 1]:
  //   q0' = 1*cos(1) - 0*sin(1) = 0.540302
  //   q2' = 1*sin(1) + 0*cos(1) = 0.841471
  //   q1' = 0*cos(.01) - 1*sin(.01) = -0.00999983
  //   q3' = 0*sin(.01) + 1*cos(.01) = 0.99995
  Guarded<float> qs = MakeGuarded(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f}, 501);
  Guarded<float> ks = MakeGuarded(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f}, 502);
  std::vector<int32_t> pos = {1};
  Tensor tq = Tensor::Contiguous(qs.data.data() + kRopeGuard, DType::kF32, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(ks.data.data() + kRopeGuard, DType::kF32, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 4});
  CHECK(qs.data[kRopeGuard + 0] == doctest::Approx(0.540302f));
  CHECK(qs.data[kRopeGuard + 1] == doctest::Approx(-0.00999983f));
  CHECK(qs.data[kRopeGuard + 2] == doctest::Approx(0.841471f));
  CHECK(qs.data[kRopeGuard + 3] == doctest::Approx(0.99995f));
  CHECK(ks.data[kRopeGuard + 0] == doctest::Approx(0.540302f));  // k rotated identically
  CheckGuards(qs, 4, "q");
  CheckGuards(ks, 4, "k");
}

TEST_CASE("rope bf16 in-place: same pos=1 golden within bf16 eps") {
  // Same golden as the f32 case; q/k stored as bf16 and rotated in place.
  // bf16(1.0)=0x3F80, bf16(0.0)=0x0000 are exact.
  Guarded<uint16_t> qs = MakeGuarded(std::vector<uint16_t>{0x3F80, 0x0000, 0x0000, 0x3F80}, 503);
  Guarded<uint16_t> ks = MakeGuarded(std::vector<uint16_t>{0x3F80, 0x0000, 0x0000, 0x3F80}, 504);
  std::vector<int32_t> pos = {1};
  Tensor tq = Tensor::Contiguous(qs.data.data() + kRopeGuard, DType::kBF16, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(ks.data.data() + kRopeGuard, DType::kBF16, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 4});
  CHECK(vt::BF16ToF32(qs.data[kRopeGuard + 0]) == doctest::Approx(0.540302f).epsilon(0.01));
  CHECK(vt::BF16ToF32(qs.data[kRopeGuard + 1]) == doctest::Approx(-0.00999983f).epsilon(0.01));
  CHECK(vt::BF16ToF32(qs.data[kRopeGuard + 2]) == doctest::Approx(0.841471f).epsilon(0.01));
  CHECK(vt::BF16ToF32(qs.data[kRopeGuard + 3]) == doctest::Approx(0.99995f).epsilon(0.01));
  CHECK(vt::BF16ToF32(ks.data[kRopeGuard + 0]) == doctest::Approx(0.540302f).epsilon(0.01));
  CheckGuards(qs, 4, "q");
  CheckGuards(ks, 4, "k");
}

TEST_CASE("rope rejects mismatched q/k dtypes") {
  std::vector<float> qbuf(4, 0.0f);
  std::vector<uint16_t> kbuf(4, 0);
  std::vector<int32_t> pos = {0};
  Tensor tq = Tensor::Contiguous(qbuf.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(kbuf.data(), DType::kBF16, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 4}), std::runtime_error);
}

TEST_CASE("rope pos=0 is identity") {
  Guarded<float> qs = MakeGuarded(std::vector<float>{0.3f, -0.7f, 1.1f, 2.2f}, 505);
  Guarded<float> ks = MakeGuarded(std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}, 506);
  std::vector<int64_t> pos = {0};
  Tensor tq = Tensor::Contiguous(qs.data.data() + kRopeGuard, DType::kF32, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(ks.data.data() + kRopeGuard, DType::kF32, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI64, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 4});
  CHECK(qs.data[kRopeGuard + 0] == doctest::Approx(0.3f));
  CHECK(qs.data[kRopeGuard + 1] == doctest::Approx(-0.7f));
  CHECK(qs.data[kRopeGuard + 2] == doctest::Approx(1.1f));
  CHECK(qs.data[kRopeGuard + 3] == doctest::Approx(2.2f));
  CheckGuards(qs, 4, "q");
  CheckGuards(ks, 4, "k");
}

TEST_CASE("partial rope leaves tail dims untouched") {
  // head_dim=4, rotary_dim=2: only pair (0,1) rotates; dims 2,3 untouched
  Guarded<float> qs = MakeGuarded(std::vector<float>{1.0f, 0.0f, 5.0f, 6.0f}, 507);
  Guarded<float> ks = MakeGuarded(std::vector<float>{0.0f, 0.0f, 7.0f, 8.0f}, 508);
  std::vector<int32_t> pos = {1};
  Tensor tq = Tensor::Contiguous(qs.data.data() + kRopeGuard, DType::kF32, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(ks.data.data() + kRopeGuard, DType::kF32, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 2});
  CHECK(qs.data[kRopeGuard + 0] == doctest::Approx(0.540302f));   // cos(1)
  CHECK(qs.data[kRopeGuard + 1] == doctest::Approx(0.841471f));   // sin(1)
  CHECK(qs.data[kRopeGuard + 2] == 5.0f);
  CHECK(qs.data[kRopeGuard + 3] == 6.0f);
  CHECK(ks.data[kRopeGuard + 2] == 7.0f);
  CheckGuards(qs, 4, "q");
  CheckGuards(ks, 4, "k");
}

TEST_CASE("rope validates rotary_dim") {
  std::vector<float> buf(4, 0.0f);
  std::vector<int32_t> pos = {0};
  Tensor tq = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tk = Tensor::Contiguous(buf.data(), DType::kF32, Cpu(), {1, 1, 4});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {1});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 3}), std::runtime_error);  // odd
  CHECK_THROWS_AS(vt::RopeNeox(q, tq, tk, tp, RopeArgs{10000.0f, 8}), std::runtime_error);  // > D
}

TEST_CASE("rope neox GQA: every q/k slot matches an independent reference exactly") {
  // T=10, Hq=4, Hk=2, D=8, rotary_dim=6. Query and key head counts differ, so
  // the kernel's head-boundary comparison decides which buffer a head belongs
  // to. rotary_dim < D leaves dims 6 and 7 unrotated, which pins each head's
  // stride: a head routed to the wrong base offset moves the tail as well.
  constexpr int64_t kT = 10, kHq = 4, kHk = 2, kD = 8;
  constexpr int kRot = 6;
  constexpr double kBase = 10000.0;
  std::vector<int32_t> pos = {1, 4, 7, 10, 13, 16, 19, 22, 25, 28};
  // Guards on both sides of q and of k turn a mutant write outside the logical
  // tensor into an assertion instead of heap corruption. The GQA split is the
  // seam under test, and both known mutants index outside the logical buffer
  // before the slot comparison can report them. A guard write is itself a
  // failure, so the sentinels are compared, never merely allocated.
  const std::vector<float> q_in = Fill(kT * kHq * kD, 101);
  const std::vector<float> k_in = Fill(kT * kHk * kD, 202);
  const std::vector<float> q_ref = NeoxReference(q_in, pos, kT, kHq, kD, kRot, kBase);
  const std::vector<float> k_ref = NeoxReference(k_in, pos, kT, kHk, kD, kRot, kBase);
  Guarded<float> qs = MakeGuarded(q_in, 303);
  Guarded<float> ks = MakeGuarded(k_in, 404);
  const size_t q_slots = static_cast<size_t>(kT * kHq * kD);
  const size_t k_slots = static_cast<size_t>(kT * kHk * kD);
  Tensor tq = Tensor::Contiguous(qs.data.data() + kRopeGuard, DType::kF32, Cpu(), {kT, kHq, kD});
  Tensor tk = Tensor::Contiguous(ks.data.data() + kRopeGuard, DType::kF32, Cpu(), {kT, kHk, kD});
  Tensor tp = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {kT});
  Queue q{Cpu(), nullptr};
  vt::RopeNeox(q, tq, tk, tp, RopeArgs{static_cast<float>(kBase), kRot});
  for (size_t i = 0; i < q_slots; ++i) {
    CHECK_MESSAGE(std::bit_cast<uint32_t>(qs.data[kRopeGuard + i]) ==
                      std::bit_cast<uint32_t>(q_ref[i]),
                  "q slot " << i);
  }
  for (size_t i = 0; i < k_slots; ++i) {
    CHECK_MESSAGE(std::bit_cast<uint32_t>(ks.data[kRopeGuard + i]) ==
                      std::bit_cast<uint32_t>(k_ref[i]),
                  "k slot " << i);
  }
  CheckGuards(qs, q_slots, "q");
  CheckGuards(ks, k_slots, "k");
}

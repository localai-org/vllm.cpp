// SCRATCH W2 diagnostic (row BACKEND-TENSTORRENT, spec
// .agents/specs/flint-tt-w2.md, issue
// ISSUE-LOCAL-01M5T8070WQ8RJYXZXD5ZWN5KZ): the ttnn::matmul tf32 micro-probe.
//
// The substrate record (src/vt/tenstorrent/tenstorrent_ops.cpp, commit
// b67b30fd lineage) says ttnn::matmul truncates f32 operands to tf32 on this
// pin. This probe measures the deviation vs a host-double reference on
// IDENTICAL bf16-rounded operand VALUES, staged two ways:
//
//   arm F: operands kF32 (the bf16-rounded values widened)
//   arm B: operands kBF16 (the same bf16 values)
//
// First-run finding (2026-10-25, P150, M=8 K=N=2560): arm F and arm B are
// BIT-IDENTICAL and BOTH sit at rel_rms 2.62e-2 vs host double. bf16-valued
// f32 truncates to tf32 exactly, so no operand truncation is measurable;
// the loss is in the ACCUMULATION. The shape sweep below characterises it
// (M rows, K depth) so the audit can name which in-model sites take the
// lossy path.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

#include <doctest/doctest.h>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vt/ops.h"

namespace {

using vllm::dense_attn::DBuf;
using vllm::dense_attn::Dev;

uint16_t ToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t lsb = (u >> 16) & 1;
  u += 0x7fff + lsb;
  return static_cast<uint16_t>(u >> 16);
}

float FromBf16(uint16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Stats {
  double max_abs = 0.0;
  double rms = 0.0;
  double rel_rms = 0.0;
};

Stats Compare(const std::vector<float>& got, const std::vector<double>& ref) {
  Stats s;
  double se = 0.0, sr = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - ref[i];
    s.max_abs = std::max(s.max_abs, std::fabs(d));
    se += d * d;
    sr += ref[i] * ref[i];
  }
  const double n = static_cast<double>(ref.size());
  s.rms = std::sqrt(se / n);
  s.rel_rms = s.rms / std::sqrt(sr / n);
  return s;
}

// One [M,K]x[K,N] probe; both arms against host double. Returns {F, B}.
std::pair<Stats, Stats> Probe(Dev& d, int64_t M, int64_t K, int64_t N,
                              std::mt19937& rng) {
  std::normal_distribution<float> nd(0.0f, 0.5f);
  std::vector<uint16_t> xb(static_cast<size_t>(M * K)),
      wb(static_cast<size_t>(K * N));
  for (auto& v : xb) v = ToBf16(nd(rng));
  for (auto& v : wb) v = ToBf16(nd(rng) * 0.02f);
  std::vector<float> xf(static_cast<size_t>(M * K)),
      wf(static_cast<size_t>(K * N));
  for (size_t i = 0; i < xb.size(); ++i) xf[i] = FromBf16(xb[i]);
  for (size_t i = 0; i < wb.size(); ++i) wf[i] = FromBf16(wb[i]);

  std::vector<double> ref(static_cast<size_t>(M * N), 0.0);
  for (int64_t m = 0; m < M; ++m)
    for (int64_t k = 0; k < K; ++k) {
      const double a =
          static_cast<double>(xf[static_cast<size_t>(m * K + k)]);
      for (int64_t n = 0; n < N; ++n)
        ref[static_cast<size_t>(m * N + n)] +=
            a * static_cast<double>(wf[static_cast<size_t>(k * N + n)]);
    }

  std::vector<float> outF(static_cast<size_t>(M * N));
  {
    DBuf a(d, vt::DType::kF32, {M, K}, xf.data());
    DBuf b(d, vt::DType::kF32, {K, N}, wf.data());
    DBuf c(d, vt::DType::kF32, {M, N});
    vt::Matmul(d.q, c.t(), a.t(), b.t());
    c.Download(d, outF.data());
  }
  std::vector<float> outB(static_cast<size_t>(M * N));
  {
    DBuf a(d, vt::DType::kBF16, {M, K}, xb.data());
    DBuf b(d, vt::DType::kBF16, {K, N}, wb.data());
    DBuf c(d, vt::DType::kF32, {M, N});
    vt::Matmul(d.q, c.t(), a.t(), b.t());
    c.Download(d, outB.data());
  }
  return {Compare(outF, ref), Compare(outB, ref)};
}

}  // namespace

TEST_CASE("SCRATCH dbg matmul tf32 micro (W2)") {
  if (vt::TryGetBackend(vt::DeviceType::kTENSTORRENT) == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Backend& ttb = *vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue ttq = ttb.CreateQueue();
  Dev d{ttb, ttq};

  std::mt19937 rng(7);
  const int64_t shapes[][3] = {
      {1, 64, 64},    {1, 512, 512},  {1, 2560, 2560},
      {8, 64, 64},    {8, 512, 512},  {8, 2560, 2560},
      {8, 2048, 512}, {8, 512, 2048}, {5, 2560, 2560},
  };
  bool identical_everywhere = true;
  for (const auto& sh : shapes) {
    const auto [sf, sb] = Probe(d, sh[0], sh[1], sh[2], rng);
    MESSAGE("M=" << sh[0] << " K=" << sh[1] << " N=" << sh[2]
                 << "  F: max_abs=" << sf.max_abs
                 << " rel_rms=" << sf.rel_rms
                 << "  B: max_abs=" << sb.max_abs
                 << " rel_rms=" << sb.rel_rms);
    if (std::fabs(sf.rel_rms - sb.rel_rms) > 1e-9)
      identical_everywhere = false;
  }
  MESSAGE("arms bit-identical across shapes: " << identical_everywhere);
}
// SCRATCH W2 diagnostic, part 2 (spec .agents/specs/flint-tt-w2.md):
// decompose the measured 2.6e-2 rel_rms into the MULTIPLY path and the
// ACCUMULATE path.
//
//   probe A (multiply only): x = e_k (one 1.0 at position k, rest 0), weight
//     bf16 random. ref row = weight row k exactly. Any deviation is the
//     multiply/store path with NO accumulation.
//   probe B (accumulate only): x = all 1.0, weight bf16 random. ref row =
//     per-column sums of the weight; deviation isolates the reduction.
//   probe C (integer exactness): x and w small integers; ref must be
//     BIT-EXACT. A deviation here means the data path corrupts values, not
//     rounding.

namespace {

std::vector<float> Run(Dev& d, const std::vector<float>& x,
                       const std::vector<float>& w, int64_t M, int64_t K,
                       int64_t N) {
  std::vector<float> out(static_cast<size_t>(M * N));
  DBuf a(d, vt::DType::kF32, {M, K}, x.data());
  DBuf b(d, vt::DType::kF32, {K, N}, w.data());
  DBuf c(d, vt::DType::kF32, {M, N});
  vt::Matmul(d.q, c.t(), a.t(), b.t());
  c.Download(d, out.data());
  return out;
}

}  // namespace

TEST_CASE("SCRATCH dbg matmul decompose (W2)") {
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const int64_t M = 1, K = 512, N = 512;

  std::vector<float> w(static_cast<size_t>(K * N));
  for (size_t i = 0; i < w.size(); ++i)
    w[i] = FromBf16(ToBf16(nd(rng) * 0.1f));

  // A: multiply only.
  {
    std::vector<float> x(static_cast<size_t>(M * K), 0.0f);
    x[123] = 1.0f;
    const std::vector<double> ref(w.begin() + 123 * N, w.begin() + 124 * N);
    const Stats s = Compare(Run(d, x, w, M, K, N), ref);
    MESSAGE("A multiply-only: max_abs=" << s.max_abs
            << " rel_rms=" << s.rel_rms);
  }
  // B: accumulate only.
  {
    std::vector<float> x(static_cast<size_t>(M * K), 1.0f);
    std::vector<double> ref(static_cast<size_t>(N), 0.0);
    for (int64_t k = 0; k < K; ++k)
      for (int64_t n = 0; n < N; ++n)
        ref[static_cast<size_t>(n)] +=
            static_cast<double>(w[static_cast<size_t>(k * N + n)]);
    const Stats s = Compare(Run(d, x, w, M, K, N), ref);
    MESSAGE("B accumulate-only: max_abs=" << s.max_abs
            << " rel_rms=" << s.rel_rms);
  }
  // C: integers, must be bit-exact.
  {
    std::vector<float> x(static_cast<size_t>(M * K));
    std::vector<float> wi(static_cast<size_t>(K * N));
    std::vector<double> ref(static_cast<size_t>(N), 0.0);
    for (int64_t k = 0; k < K; ++k) {
      x[static_cast<size_t>(k)] = static_cast<float>((k % 7) - 3);
      for (int64_t n = 0; n < N; ++n) {
        wi[static_cast<size_t>(k * N + n)] =
            static_cast<float>(((k + n) % 5) - 2);
        ref[static_cast<size_t>(n)] += static_cast<double>(
            x[static_cast<size_t>(k)] * wi[static_cast<size_t>(k * N + n)]);
      }
    }
    const std::vector<float> out = Run(d, x, wi, M, K, N);
    double max_abs = 0.0;
    for (int64_t n = 0; n < N; ++n)
      max_abs = std::max(max_abs, std::fabs(
          static_cast<double>(out[static_cast<size_t>(n)]) -
          ref[static_cast<size_t>(n)]));
    MESSAGE("C integer-exact: max_abs=" << max_abs);
  }
}

TEST_CASE("SCRATCH dbg matmul path compare (W2)") {
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(13);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  const int64_t M = 1, K = 64, N = 64;
  std::vector<float> x(static_cast<size_t>(M * K)), w(static_cast<size_t>(K * N));
  for (auto& v : x) v = FromBf16(ToBf16(nd(rng)));
  for (auto& v : w) v = FromBf16(ToBf16(nd(rng) * 0.02f));
  std::vector<double> ref(static_cast<size_t>(M * N), 0.0);
  for (int64_t m = 0; m < M; ++m)
    for (int64_t k = 0; k < K; ++k) {
      const double a = static_cast<double>(x[static_cast<size_t>(m * K + k)]);
      for (int64_t n = 0; n < N; ++n)
        ref[static_cast<size_t>(m * N + n)] +=
            a * static_cast<double>(w[static_cast<size_t>(k * N + n)]);
    }

  // D: the SAME values through MatmulBT ([N,K] weight, transpose on device).
  {
    std::vector<float> wt(static_cast<size_t>(N * K));
    for (int64_t k = 0; k < K; ++k)
      for (int64_t n = 0; n < N; ++n)
        wt[static_cast<size_t>(n * K + k)] =
            w[static_cast<size_t>(k * N + n)];
    std::vector<float> out(static_cast<size_t>(M * N));
    DBuf a(d, vt::DType::kBF16, {M, K}, x.data());
    DBuf b(d, vt::DType::kBF16, {N, K}, wt.data());
    DBuf c(d, vt::DType::kF32, {M, N});
    vt::MatmulBT(d.q, c.t(), a.t(), b.t());
    c.Download(d, out.data());
    const Stats s = Compare(out, ref);
    MESSAGE("D MatmulBT bf16: max_abs=" << s.max_abs
            << " rel_rms=" << s.rel_rms);
  }
  // E: Matmul, bf16 operands, BF16 OUT (does the out dtype hide or add?).
  {
    std::vector<uint16_t> xb(static_cast<size_t>(M * K)),
        wb(static_cast<size_t>(K * N));
    for (size_t i = 0; i < xb.size(); ++i) xb[i] = ToBf16(x[i]);
    for (size_t i = 0; i < wb.size(); ++i) wb[i] = ToBf16(w[i]);
    std::vector<uint16_t> outb(static_cast<size_t>(M * N));
    DBuf a(d, vt::DType::kBF16, {M, K}, xb.data());
    DBuf b(d, vt::DType::kBF16, {K, N}, wb.data());
    DBuf c(d, vt::DType::kBF16, {M, N});
    vt::Matmul(d.q, c.t(), a.t(), b.t());
    c.Download(d, outb.data());
    std::vector<float> out(static_cast<size_t>(M * N));
    for (size_t i = 0; i < out.size(); ++i) out[i] = FromBf16(outb[i]);
    const Stats s = Compare(out, ref);
    MESSAGE("E Matmul bf16 out: max_abs=" << s.max_abs
            << " rel_rms=" << s.rel_rms);

  // F: sanity in THIS process — the tc1 arm (Matmul, bf16 in, f32 out).
  {
    std::vector<uint16_t> xb(static_cast<size_t>(M * K)),
        wb(static_cast<size_t>(K * N));
    for (size_t j = 0; j < xb.size(); ++j) xb[j] = ToBf16(x[j]);
    for (size_t j = 0; j < wb.size(); ++j) wb[j] = ToBf16(w[j]);
    std::vector<float> out(static_cast<size_t>(M * N));
    DBuf a(d, vt::DType::kBF16, {M, K}, xb.data());
    DBuf b(d, vt::DType::kBF16, {K, N}, wb.data());
    DBuf c(d, vt::DType::kF32, {M, N});
    vt::Matmul(d.q, c.t(), a.t(), b.t());
    c.Download(d, out.data());
    const Stats s = Compare(out, ref);
    MESSAGE("F Matmul bf16 f32-out: max_abs=" << s.max_abs
            << " rel_rms=" << s.rel_rms);
  }
}}

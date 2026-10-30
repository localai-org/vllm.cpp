// SCRATCH W3 diagnostic, spec .agents/specs/tt-matmul-noise.md, issue
// ISSUE-LOCAL-01M5TB3983MW41T4NPBZ5ZYF0E. Extends the W2 micro into the
// noise MAP: shape (K/M/N) sweeps, tile quantization, the model q/k/v
// shape, the W2 D/E anomaly re-run across shapes, and the dest-acc axis.
// Operator-run under the gpu.lock mutex on the local P150.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <doctest/doctest.h>

#include "vt/ops.h"
#include "vllm/model_executor/models/dense_device_glue.h"

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
  double frac_bad = 0.0;  // outputs off by >50% relative: the garbage class
};

Stats Compare(const std::vector<float>& got, const std::vector<double>& ref) {
  Stats s;
  double se = 0.0, sr = 0.0;
  size_t bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - ref[i];
    s.max_abs = std::max(s.max_abs, std::fabs(d));
    se += d * d;
    sr += ref[i] * ref[i];
    if (std::fabs(ref[i]) > 1e-12 && std::fabs(d) > 0.5 * std::fabs(ref[i]))
      ++bad;
  }
  const double n = static_cast<double>(ref.size());
  s.rms = std::sqrt(se / n);
  s.rel_rms = s.rms / std::sqrt(sr / n);
  s.frac_bad = static_cast<double>(bad) / n;
  return s;
}

std::vector<double> HostRef(const std::vector<float>& x,
                            const std::vector<float>& w, int64_t M,
                            int64_t K, int64_t N) {
  std::vector<double> ref(static_cast<size_t>(M * N), 0.0);
  for (int64_t m = 0; m < M; ++m)
    for (int64_t k = 0; k < K; ++k) {
      const double a = static_cast<double>(x[static_cast<size_t>(m * K + k)]);
      for (int64_t n = 0; n < N; ++n)
        ref[static_cast<size_t>(m * N + n)] +=
            a * static_cast<double>(w[static_cast<size_t>(k * N + n)]);
    }
  return ref;
}


// One bf16-operand [M,K]x[K,N] run through the committed op. bt selects
// MatmulBT ([N,K] weight, transpose on device) over Matmul. f32out selects
// the f32-out arm (fp32_dest_acc_en set) over the bf16-out arm.
std::vector<float> RunOp(Dev& d, const std::vector<uint16_t>& xb,
                         const std::vector<uint16_t>& wb, int64_t M,
                         int64_t K, int64_t N, bool bt, bool f32out) {
  DBuf a(d, vt::DType::kBF16, {M, K}, xb.data());
  // bt takes the weight TRANSPOSED: wt[n*K + k] = wb[k*N + n], staged [N,K].
  std::vector<uint16_t> wt;
  if (bt) {
    wt.resize(wb.size());
    for (int64_t k = 0; k < K; ++k)
      for (int64_t n = 0; n < N; ++n)
        wt[static_cast<size_t>(n * K + k)] =
            wb[static_cast<size_t>(k * N + n)];
  }
  DBuf b = bt ? DBuf(d, vt::DType::kBF16, {N, K}, wt.data())
              : DBuf(d, vt::DType::kBF16, {K, N}, wb.data());
  std::vector<float> out(static_cast<size_t>(M * N));
  if (f32out) {
    DBuf c(d, vt::DType::kF32, {M, N});
    if (bt) {
      vt::MatmulBT(d.q, c.t(), a.t(), b.t());
    } else {
      vt::Matmul(d.q, c.t(), a.t(), b.t());
    }
    c.Download(d, out.data());
  } else {
    DBuf c(d, vt::DType::kBF16, {M, N});
    if (bt) {
      vt::MatmulBT(d.q, c.t(), a.t(), b.t());
    } else {
      vt::Matmul(d.q, c.t(), a.t(), b.t());
    }
    std::vector<uint16_t> outb(static_cast<size_t>(M * N));
    c.Download(d, outb.data());
    for (size_t i = 0; i < out.size(); ++i) out[i] = FromBf16(outb[i]);
  }
  return out;
}

// Random bf16 operands and the host-double reference, in one go.
struct Case {
  std::vector<uint16_t> xb, wb;
  std::vector<double> ref;
};

Case MakeCase(std::mt19937& rng, int64_t M, int64_t K, int64_t N) {
  std::normal_distribution<float> nd(0.0f, 0.5f);
  Case c;
  c.xb.resize(static_cast<size_t>(M * K));
  c.wb.resize(static_cast<size_t>(K * N));
  std::vector<float> xf(static_cast<size_t>(M * K)),
      wf(static_cast<size_t>(K * N));
  for (size_t i = 0; i < c.xb.size(); ++i) {
    c.xb[i] = ToBf16(nd(rng));
    xf[i] = FromBf16(c.xb[i]);
  }
  for (size_t i = 0; i < c.wb.size(); ++i) {
    c.wb[i] = ToBf16(nd(rng) * 0.02f);
    wf[i] = FromBf16(c.wb[i]);
  }
  c.ref = HostRef(xf, wf, M, K, N);
  return c;
}

void Report(const char* tag, int64_t M, int64_t K, int64_t N, const char* arm,
            const Stats& s) {
  MESSAGE(tag << " M=" << M << " K=" << K << " N=" << N << " arm=" << arm
              << " max_abs=" << s.max_abs << " rel_rms=" << s.rel_rms
              << " frac_bad=" << s.frac_bad);
}


// 1: the K sweep. Question: is the ~2.5e-2 per-op noise per-product constant
// (K-independent, the W2 reading), K-dependent (an accumulation signature),
// or shape-quantized?
TEST_CASE("SCRATCH dbg W3 noise map: K sweep") {
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(7);
  const int64_t ks[] = {32, 64, 128, 256, 512, 1024, 2048, 2560};
  for (int64_t K : ks) {
    const Case c = MakeCase(rng, 1, K, 512);
    const Stats sf = Compare(RunOp(d, c.xb, c.wb, 1, K, 512, false, true), c.ref);
    const Stats sb = Compare(RunOp(d, c.xb, c.wb, 1, K, 512, false, false), c.ref);
    Report("Ksweep", 1, K, 512, "f32out", sf);
    Report("Ksweep", 1, K, 512, "bf16out", sb);
  }
}

// 2: the M sweep at K=N=512 (does the noise grow with rows?).
TEST_CASE("SCRATCH dbg W3 noise map: M sweep") {
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(8);
  const int64_t ms[] = {1, 2, 4, 8, 16, 32};
  for (int64_t M : ms) {
    const Case c = MakeCase(rng, M, 512, 512);
    const Stats s = Compare(RunOp(d, c.xb, c.wb, M, 512, 512, false, true), c.ref);
    Report("Msweep", M, 512, 512, "f32out", s);
  }
}

// 3: the N sweep at M=1, K=512.
TEST_CASE("SCRATCH dbg W3 noise map: N sweep") {
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(9);
  const int64_t ns[] = {32, 64, 128, 256, 512, 1024, 2048};
  for (int64_t N : ns) {
    const Case c = MakeCase(rng, 1, 512, N);
    const Stats s = Compare(RunOp(d, c.xb, c.wb, 1, 512, N, false, true), c.ref);
    Report("Nsweep", 1, 512, N, "f32out", s);
  }
}

// 4: tile quantization — tile-multiple vs off-tile dims, and the model's
// q/k/v projection shape (hidden 2048, fused 3x out).
TEST_CASE("SCRATCH dbg W3 noise map: tiles and the model shape") {
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(10);
  const int64_t shapes[][3] = {
      {1, 512, 512}, {1, 512, 520}, {1, 520, 512}, {1, 516, 516},
      {1, 2048, 2048}, {1, 2048, 6144}, {5, 2048, 2048},
  };
  for (const auto& sh : shapes) {
    const Case c = MakeCase(rng, sh[0], sh[1], sh[2]);
    const Stats s = Compare(RunOp(d, c.xb, c.wb, sh[0], sh[1], sh[2], false, true), c.ref);
    Report("tiles", sh[0], sh[1], sh[2], "f32out", s);
  }
}

// 5: the W2 D/E anomaly re-run across shapes — MatmulBT and the bf16-out
// arm on the SAME values as the clean f32-out arm.
TEST_CASE("SCRATCH dbg W3 noise map: path compare") {
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(11);
  const int64_t shapes[][3] = {
      {1, 64, 64}, {1, 512, 512}, {5, 512, 512}, {8, 512, 512},
      {1, 2048, 2048}, {5, 2560, 2560},
  };
  for (const auto& sh : shapes) {
    const Case c = MakeCase(rng, sh[0], sh[1], sh[2]);
    const Stats sf = Compare(RunOp(d, c.xb, c.wb, sh[0], sh[1], sh[2], false, true), c.ref);
    const Stats se = Compare(RunOp(d, c.xb, c.wb, sh[0], sh[1], sh[2], false, false), c.ref);
    const Stats sd = Compare(RunOp(d, c.xb, c.wb, sh[0], sh[1], sh[2], true, true), c.ref);
    const Stats sdb = Compare(RunOp(d, c.xb, c.wb, sh[0], sh[1], sh[2], true, false), c.ref);
    Report("pathcmp-F", sh[0], sh[1], sh[2], "matmul_f32out", sf);
    Report("pathcmp-E", sh[0], sh[1], sh[2], "matmul_bf16out", se);
    Report("pathcmp-D", sh[0], sh[1], sh[2], "bt_f32out", sd);
    Report("pathcmp-D2", sh[0], sh[1], sh[2], "bt_bf16out", sdb);
  }
}

// 6: per-product constancy — multiply-only probes (x = e_k) at the first
// and last K position; a constant per-product noise reads the same at both.
TEST_CASE("SCRATCH dbg W3 noise map: per-product") {
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(12);
  const int64_t K = 512, N = 512;
  std::vector<uint16_t> wb(static_cast<size_t>(K * N));
  std::vector<float> wf(static_cast<size_t>(K * N));
  std::normal_distribution<float> nd(0.0f, 0.02f);
  for (size_t i = 0; i < wb.size(); ++i) {
    wb[i] = ToBf16(nd(rng));
    wf[i] = FromBf16(wb[i]);
  }
  for (int64_t kpos : {int64_t(0), K / 2, K - 1}) {
    std::vector<uint16_t> xb(static_cast<size_t>(K), 0);
    xb[static_cast<size_t>(kpos)] = ToBf16(1.0f);
    std::vector<float> xf(static_cast<size_t>(K), 0.0f);
    xf[static_cast<size_t>(kpos)] = 1.0f;
    std::vector<double> ref(wf.begin() + static_cast<ptrdiff_t>(kpos * N),
                            wf.begin() + static_cast<ptrdiff_t>((kpos + 1) * N));
    const Stats s = Compare(RunOp(d, xb, wb, 1, K, N, false, true), ref);
    MESSAGE("perproduct k=" << kpos << " max_abs=" << s.max_abs
            << " rel_rms=" << s.rel_rms << " frac_bad=" << s.frac_bad);
  }
}


// 7: the exactness CHECK. The host-double reference for exact-bf16 operands
// is f32-accumulation-exact to ~sqrt(K)*2^-24, so the bound is tight: an arm
// that carries the 2.5e-2 per-product multiply noise FAILS this (red on the
// old LoFi arm), the HiFi4 arm PASSES. Since the W3 adoption (developer-
// approved 2026-10-26; docs/bench-evidence/tt-matmul-noise-20261025/
// report.md) the HiFi4 + approx-off + f32-dest arm IS the shipped default,
// so this CHECK runs on the default configuration; it is skipped only under
// VT_TT_MATMUL_MATH_FIDELITY=lofi (the pre-adoption A/B arm, red here by
// the map).
TEST_CASE("SCRATCH dbg W3 exactness check (dest-acc arm)") {
  const char* fid = std::getenv("VT_TT_MATMUL_MATH_FIDELITY");
  if (fid != nullptr && std::string(fid) == "lofi") {
    MESSAGE("skipped: VT_TT_MATMUL_MATH_FIDELITY=lofi selects the "
            "pre-adoption arm, which is red here BY the map (see "
            "docs/bench-evidence/tt-matmul-noise-20261025/)");
    return;
  }
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(21);
  const int64_t shapes[][3] = {
      {5, 512, 512}, {5, 2048, 2048}, {1, 2048, 6144},
  };
  for (const auto& sh : shapes) {
    const Case c = MakeCase(rng, sh[0], sh[1], sh[2]);
    const Stats s =
        Compare(RunOp(d, c.xb, c.wb, sh[0], sh[1], sh[2], false, true), c.ref);
    Report("exactcheck", sh[0], sh[1], sh[2], "f32out", s);
    // localize the residual bad elements (positions hint at the mechanism:
    // tile boundaries -> staging padding; scattered -> upload corruption)
    {
      const std::vector<float> got =
          RunOp(d, c.xb, c.wb, sh[0], sh[1], sh[2], false, true);
      size_t shown = 0;
      for (size_t i = 0; i < c.ref.size() && shown < 8; ++i) {
        if (std::fabs(c.ref[i]) > 1e-12 &&
            std::fabs(static_cast<double>(got[i]) - c.ref[i]) >
                0.5 * std::fabs(c.ref[i])) {
          MESSAGE("bad[" << i << "] row=" << i / sh[2] << " col=" << i % sh[2]
                  << " got=" << got[i] << " ref=" << c.ref[i]);
          ++shown;
        }
      }
    }
    CHECK_MESSAGE(s.rel_rms < 1e-3,
                  "f32-out matmul vs host double rel_rms ", s.rel_rms);
    CHECK_MESSAGE(s.max_abs < 1e-3, "f32-out matmul max_abs ", s.max_abs);
  }
}

// 8: the default bf16-compute arm (bf16 out) on the same values — the map's
// reference for what the multiply does WITHOUT the f32-dest config.
TEST_CASE("SCRATCH dbg W3 exactness check (default arm)") {
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(22);
  const int64_t shapes[][3] = {
      {5, 512, 512}, {5, 2048, 2048}, {1, 2048, 6144},
  };
  for (const auto& sh : shapes) {
    const Case c = MakeCase(rng, sh[0], sh[1], sh[2]);
    const Stats s =
        Compare(RunOp(d, c.xb, c.wb, sh[0], sh[1], sh[2], false, false), c.ref);
    Report("defaultcheck", sh[0], sh[1], sh[2], "bf16out", s);
    // bf16 output rounds: 2^-9 rel bound on the OUTPUT grid, and no garbage.
    CHECK_MESSAGE(s.rel_rms < 2e-2, "bf16-out arm rel_rms ", s.rel_rms);
  }
}

// 9: the exact MatmulBT arm (weight transposed on host, staged [N,K]) on
// the model's shapes — adjudicates the W2 D anomaly with correct data.
// Same gating as case 7: the bound holds on the shipped (HiFi4) arm since
// the adoption; skipped under VT_TT_MATMUL_MATH_FIDELITY=lofi.
TEST_CASE("SCRATCH dbg W3 exactness check (MatmulBT arm)") {
  const char* fid = std::getenv("VT_TT_MATMUL_MATH_FIDELITY");
  if (fid != nullptr && std::string(fid) == "lofi") {
    MESSAGE("skipped: VT_TT_MATMUL_MATH_FIDELITY=lofi selects the "
            "pre-adoption arm, which is red here BY the map");
    return;
  }
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(23);
  const int64_t shapes[][3] = {
      {1, 512, 512}, {5, 2048, 2048}, {8, 512, 2048},
  };
  for (const auto& sh : shapes) {
    const Case c = MakeCase(rng, sh[0], sh[1], sh[2]);
    const Stats s =
        Compare(RunOp(d, c.xb, c.wb, sh[0], sh[1], sh[2], true, true), c.ref);
    Report("btcheck", sh[0], sh[1], sh[2], "bt_f32out", s);
    CHECK_MESSAGE(s.rel_rms < 1e-3, "MatmulBT f32-out rel_rms ", s.rel_rms);
    CHECK_MESSAGE(s.max_abs < 1e-3, "MatmulBT f32-out max_abs ", s.max_abs);
  }
}


}  // namespace

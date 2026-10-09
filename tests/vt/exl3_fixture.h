#pragma once
// Shared fixtures and INDEPENDENT references for the EXL3 test suites —
// MODEL-DSV4-EXL3 W2.
//
// One copy, because `test_exl3_gemm`, `test_exl3_gemv` and `test_exl3_moe` all
// need the same synthetic linear and the same fp16 ulp, and two copies of a
// fixture that must agree is the drift this tree files bugs about. Nothing here
// is under test: the fixture GENERATES data and the references are built from
// DEFINITIONS (`SylvesterH` from popcount parity, `HadRefBlock` from the
// docstring at `hadamard.cu:83-86`), never from the implementation's own
// butterfly, so no case here is a transcription gating its own transcription.
#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace exl3_test {

// The natural-order Sylvester Hadamard: H[i][j] = (-1)^popcount(i & j).
// `util/hadamard.py:34-42` recurses to `hadamard_1.txt` = "+" for 128, which is
// exactly this matrix.
inline int SylvesterH(int i, int j) {
  unsigned v = static_cast<unsigned>(i & j);
  int parity = 0;
  while (v) {
    parity ^= static_cast<int>(v & 1u);
    v >>= 1;
  }
  return parity ? -1 : 1;
}

// One 128-block of `had_r_128` in double precision, straight from the docstring
// at `hadamard.cu:83-86`: y = (x.view(-1,128) @ had_128) * scale / sqrt(128).
inline void HadRefBlock(const double* x, double* y, double r_scale) {
  for (int j = 0; j < 128; ++j) {
    double acc = 0.0;
    for (int i = 0; i < 128; ++i) acc += x[i] * SylvesterH(i, j);
    y[j] = acc * r_scale;
  }
}

struct Rng {
  uint32_t s = 0x243F6A88u;
  float next(float scale) {
    s = s * 1664525u + 1013904223u;
    const float u = (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;
    return u * scale;
  }
};

// fp16 ulp at `v` — the gap between consecutive representables around it.
inline double UlpF16(double v) {
  const uint16_t h = vt::F32ToF16(static_cast<float>(std::fabs(v)));
  const float here = vt::F16ToF32(h);
  const float up = vt::F16ToF32(static_cast<uint16_t>(h + 1));
  return static_cast<double>(up - here);
}

inline double Rms(const std::vector<double>& v) {
  double a = 0.0;
  for (double e : v) a += e * e;
  return std::sqrt(a / static_cast<double>(v.size()));
}

// A synthetic EXL3 linear. The trellis is filled with pseudo-random BITS, which
// is legitimate: every 16-bit codeword decodes to a valid fp16 pair under the
// MCG codebook, so a random bit stream is a valid (if meaningless) quantized
// weight and the kernels must reproduce whatever it decodes to. suh/svh carry a
// real per-channel scale, not just signs, exactly as the artifact does.
struct Exl3Fixture {
  int64_t k = 0;
  int64_t n = 0;
  int bits = 3;
  // HALF-INTEGER rate (K+0.5, mul1): `bits` carries the integer KA and the
  // tile is 16*KA+8 words. Set only by `MakeHalfFixture`.
  bool half = false;
  std::vector<uint16_t> trellis;  // [k/16, n/16, 16*bits (+8 if half)] words
  std::vector<uint16_t> suh;      // [k] fp16 bits
  std::vector<uint16_t> svh;      // [n] fp16 bits
};

inline Exl3Fixture MakeFixture(int64_t k, int64_t n, int bits, uint32_t seed) {
  Exl3Fixture f;
  f.k = k;
  f.n = n;
  f.bits = bits;
  Rng rng;
  rng.s = seed;
  f.trellis.resize(static_cast<size_t>(k / 16 * n / 16 * 16 * bits));
  for (auto& w : f.trellis) {
    rng.s = rng.s * 1664525u + 1013904223u;
    w = static_cast<uint16_t>(rng.s >> 13);
  }
  f.suh.resize(static_cast<size_t>(k));
  for (auto& s : f.suh) s = vt::F32ToF16(rng.next(1.0f) >= 0.0f ? 0.75f : -0.75f);
  f.svh.resize(static_cast<size_t>(n));
  for (auto& s : f.svh) s = vt::F32ToF16(rng.next(1.0f) >= 0.0f ? 1.25f : -1.25f);
  return f;
}

// The half-integer twin (BACKEND-ROCM frac rates): `bits` is the integer KA of
// K = KA + 0.5, every tile is 16*KA+8 uint16 words, and the codebook a caller
// pairs with it is ALWAYS 2 (mul1) — the random bit stream is a valid frac
// trellis for the same reason the integer one is: every codeword decodes to a
// valid fp16 pair under mul1.
inline Exl3Fixture MakeHalfFixture(int64_t k, int64_t n, int ka, uint32_t seed) {
  Exl3Fixture f = MakeFixture(k, n, ka, seed);
  f.half = true;
  f.trellis.resize(static_cast<size_t>(k / 16 * n / 16 * (16 * ka + 8)));
  Rng rng;
  rng.s = seed;
  for (auto& w : f.trellis) {
    rng.s = rng.s * 1664525u + 1013904223u;
    w = static_cast<uint16_t>(rng.s >> 13);
  }
  return f;
}

// The tier-3 REFERENCE: the fused chain evaluated in double.
//   x_had = (x * suh) @ H128 / sqrt(128)   (blockwise over k)
//   y_raw = x_had @ W_inner                (W_inner = reconstruct(trellis))
//   y     = (y_raw @ H128 / sqrt(128)) * svh   (blockwise over n)
inline std::vector<double> Exl3ChainF64(const Exl3Fixture& f,
                                        const std::vector<float>& x_f16_rounded, int64_t m,
                                        int codebook = 1) {
  const int64_t k = f.k, n = f.n;
  std::vector<float> w_inner(static_cast<size_t>(k * n));
  vt::Exl3ReconstructInner(f.trellis.data(), k, n, f.bits, codebook, w_inner.data(), f.half);

  const double inv = 1.0 / std::sqrt(128.0);
  std::vector<double> y(static_cast<size_t>(m * n), 0.0);
  std::vector<double> blk(128), obk(128);
  for (int64_t r = 0; r < m; ++r) {
    std::vector<double> xh(static_cast<size_t>(k));
    for (int64_t b = 0; b < k; b += 128) {
      for (int i = 0; i < 128; ++i)
        blk[static_cast<size_t>(i)] =
            static_cast<double>(x_f16_rounded[static_cast<size_t>(r * k + b + i)]) *
            static_cast<double>(vt::F16ToF32(f.suh[static_cast<size_t>(b + i)]));
      HadRefBlock(blk.data(), obk.data(), inv);
      for (int i = 0; i < 128; ++i) xh[static_cast<size_t>(b + i)] = obk[static_cast<size_t>(i)];
    }
    std::vector<double> raw(static_cast<size_t>(n), 0.0);
    for (int64_t i = 0; i < k; ++i) {
      const double xv = xh[static_cast<size_t>(i)];
      if (xv == 0.0) continue;
      const float* wrow = &w_inner[static_cast<size_t>(i * n)];
      for (int64_t j = 0; j < n; ++j)
        raw[static_cast<size_t>(j)] += xv * static_cast<double>(wrow[j]);
    }
    for (int64_t b = 0; b < n; b += 128) {
      HadRefBlock(&raw[static_cast<size_t>(b)], obk.data(), inv);
      for (int i = 0; i < 128; ++i)
        y[static_cast<size_t>(r * n + b + i)] =
            obk[static_cast<size_t>(i)] *
            static_cast<double>(vt::F16ToF32(f.svh[static_cast<size_t>(b + i)]));
    }
  }
  return y;
}

}  // namespace exl3_test

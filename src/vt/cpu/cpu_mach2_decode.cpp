// QUANT-MACH2-ADDITIVE U1 — the Mach-2 packed-format decode primitives.
//
// PORTED FROM SyzygyResearch/Mach-2-Additive-Medium `decode.py`, read
// 2026-11-16. See `include/vt/mach2_decode.h` for the unit's boundaries and for
// what stays owed.
#include <vt/mach2_decode.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vt::mach2 {
namespace {

// decode.py reads the tape MSB first: bit `i` of the stream is bit `15 - i` of
// the word that carries it.
std::vector<std::uint8_t> TapeBits(const std::uint16_t* words, std::size_t nwords) {
  std::vector<std::uint8_t> bits(nwords * 16);
  for (std::size_t i = 0; i < nwords; ++i) {
    for (int j = 0; j < 16; ++j) bits[i * 16 + static_cast<std::size_t>(j)] =
        static_cast<std::uint8_t>((words[i] >> (15 - j)) & 1u);
  }
  return bits;
}

// One element of the character tables `_hmat` is built from.
struct GfElem {
  int a = 0;
  int b = 0;  // the Z/3[i] coefficient; unused by the Z/5 table
};

// `chi` for radix 12 (the quadratic character over Z/5) and for radix 20 (the
// norm character over Z/3[i]): 0 at the zero element, +1 on squares, -1 elsewhere.
int Chi(int radix, GfElem e) {
  if (radix == 12) {
    const bool zero = (e.a == 0);
    if (zero) return 0;
    const int squares[5] = {1, 4};  // the non-zero quadratic residues mod 5
    return (e.a == squares[0] || e.a == squares[1]) ? 1 : -1;
  }
  if (e.a == 0 && e.b == 0) return 0;
  // Squares of Z/3[i]: (a + bi)^2 = a^2 - b^2 + 2abi, so the square elements are
  // (1,0), (2,0), (0,1), (0,2).
  const bool square = (e.b == 0) || (e.a == 0);
  return square ? 1 : -1;
}

// `sub`: subtraction in the same ring, used to index the character.
GfElem Sub(int radix, GfElem u, GfElem v) {
  if (radix == 12) return GfElem{(u.a - v.a + 5) % 5, 0};
  return GfElem{(u.a - v.a + 3) % 3, (u.b - v.b + 3) % 3};
}

std::vector<GfElem> Elements(int radix) {
  std::vector<GfElem> els;
  if (radix == 12) {
    for (int i = 0; i < 5; ++i) els.push_back(GfElem{i, 0});
    return els;
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) els.push_back(GfElem{a, b});
  }
  return els;
}

bool IsPowerOfTwo(int n) { return n > 0 && (n & (n - 1)) == 0; }

// `_HS`: the hand-written sign tables `decode.py` uses for the spine rotation,
// one row per output component. Transcribed from the published file, where the
// same matrices are asserted orthogonal in-file; `HadamardSpine` relies on that.
//
static const std::int8_t kHs12[12][12] = {
    {1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1},
    {1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1},
    {1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1},
    {1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1},
    {1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1},
    {1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1},
    {1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1},
    {1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1},
    {1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1},
    {1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1},
    {1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1},
};

static const std::int8_t kHs20[20][20] = {
    {1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1},
    {1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1},
    {1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1},
    {1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1},
    {1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1},
    {1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1},
    {1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1},
    {1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1},
    {1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1},
    {1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1},
    {1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1},
    {1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1},
    {1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1},
    {1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1},
    {1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1},
    {1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1},
    {1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1},
    {1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1},
    {1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1},
};

const std::int8_t* SpineTable(int radix) {
  if (radix == 12) return &kHs12[0][0];
  if (radix == 20) return &kHs20[0][0];
  return nullptr;
}

// `matmul(_HS, x)` over the radix axis of a `(blocks, radix, cols)` view, times
// `scale`. The same shape as `ApplyRadix`, but with the spine table and the
// spine normalisation, which is 1/sqrt(cols) rather than 1/sqrt(radix).
void ApplySpineRadix(float* x, std::size_t blocks, int radix, int cols, float scale) {
  const std::int8_t* h = SpineTable(radix);
  if (h == nullptr) return;
  std::vector<float> col(static_cast<std::size_t>(radix));
  std::vector<float> acc(static_cast<std::size_t>(radix));
  for (std::size_t b = 0; b < blocks; ++b) {
    for (int c = 0; c < cols; ++c) {
      for (int i = 0; i < radix; ++i) {
        col[static_cast<std::size_t>(i)] =
            x[(b * static_cast<std::size_t>(radix) + static_cast<std::size_t>(i)) *
                  static_cast<std::size_t>(cols) +
              static_cast<std::size_t>(c)];
      }
      std::fill(acc.begin(), acc.end(), 0.0f);
      for (int i = 0; i < radix; ++i) {
        for (int j = 0; j < radix; ++j) {
          const float hv = static_cast<float>(h[i * radix + j]);
          acc[static_cast<std::size_t>(i)] += hv * col[static_cast<std::size_t>(j)];
        }
      }
      for (int i = 0; i < radix; ++i) {
        x[(b * static_cast<std::size_t>(radix) + static_cast<std::size_t>(i)) *
              static_cast<std::size_t>(cols) +
          static_cast<std::size_t>(c)] = acc[static_cast<std::size_t>(i)] * scale;
      }
    }
  }
}

}  // namespace
namespace detail {

float Bf16RoundOne(float x) {
  std::uint32_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  float out = 0.0f;
  std::memcpy(&out, &u, sizeof(out));
  return out;
}

}  // namespace detail

std::vector<int32_t> WaveMap(int mb, int nb) {
  std::vector<int32_t> idx(static_cast<std::size_t>(mb) * static_cast<std::size_t>(nb), 0);
  // starts: the leftmost tile of each row from the bottom up, then the bottom
  // row from the right to the left.
  std::vector<std::pair<int, int>> starts;
  for (int i = 0; i < mb; ++i) starts.emplace_back(mb - i - 1, nb - 1);
  for (int i = 0; i < nb; ++i) starts.emplace_back(0, nb - i - 1);
  int w = 0;
  for (const auto& start : starts) {
    int jm = start.first;
    int jn = start.second;
    while (jm >= 0 && jm < mb && jn >= 0 && jn < nb) {
      idx[static_cast<std::size_t>(jm) * static_cast<std::size_t>(nb) +
          static_cast<std::size_t>(jn)] = w;
      ++jm;
      --jn;
    }
    ++w;
  }
  return idx;
}

std::vector<uint16_t> TrellisStates(const std::uint16_t* words, std::size_t nwords,
                                    int step_bits, std::int64_t nstep) {
  std::vector<uint16_t> out(static_cast<std::size_t>(std::max<std::int64_t>(nstep, 0)), 0);
  if (nstep <= 0 || words == nullptr || nwords == 0) return out;
  const auto bits = TapeBits(words, nwords);
  const std::size_t need = static_cast<std::size_t>(step_bits) *
                           static_cast<std::size_t>(nstep);
  if (need > bits.size()) return out;  // caller promised enough tape
  // decode.py truncates the stream to step * nstep and appends its first
  // (window - step) bits.
  std::vector<std::uint8_t> stream(bits.begin(), bits.begin() + static_cast<std::ptrdiff_t>(need));
  const std::size_t tail = static_cast<std::size_t>(kExpertWindow - step_bits);
  stream.insert(stream.end(), bits.begin(), bits.begin() + static_cast<std::ptrdiff_t>(tail));
  for (std::int64_t i = 0; i < nstep; ++i) {
    std::uint16_t state = 0;
    for (int j = 0; j < kExpertWindow; ++j) {
      const auto pos = static_cast<std::size_t>(i) * static_cast<std::size_t>(step_bits) +
                       static_cast<std::size_t>(j);
      state = static_cast<std::uint16_t>((state << 1) | (pos < stream.size() ? stream[pos] : 0));
    }
    out[static_cast<std::size_t>(i)] = state;
  }
  return out;
}

void ExpandExpertUnit(const std::uint16_t* states, std::int64_t nstep, const float* lut,
                      float* out) {
  for (std::int64_t i = 0; i < nstep; ++i) {
    const std::size_t row = static_cast<std::size_t>(states[static_cast<std::size_t>(i)]);
    for (int v = 0; v < kExpertValues; ++v) {
      out[static_cast<std::size_t>(i) * kExpertValues + static_cast<std::size_t>(v)] =
          lut[row * kExpertValues + static_cast<std::size_t>(v)];
    }
  }
}

std::vector<float> FoldSpineLut(const float* tlut) {
  std::vector<float> out(static_cast<std::size_t>(kSpineStates) * kSpineValues, 0.0f);
  for (std::int64_t s = 0; s < kSpineStates; ++s) {
    const std::int64_t p = s * (s + 1);
    const auto row = static_cast<std::size_t>((p >> 6) & (kSpineTlutRows - 1));
    const float sign = ((p >> 15) & 1) ? -1.0f : 1.0f;
    for (int v = 0; v < kSpineValues; ++v) {
      const float value = tlut[row * kSpineValues + static_cast<std::size_t>(v)];
      out[static_cast<std::size_t>(s) * kSpineValues + static_cast<std::size_t>(v)] =
          (v == 0) ? sign * value : value;
    }
  }
  return out;
}

void RcGrid(float mx, const std::int32_t* k, std::size_t n, float* out) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = mx * static_cast<float>(k[i]) * (1.0f / 127.0f);
  }
}

std::vector<std::int8_t> UnpackInt5(const std::uint8_t* packed, std::size_t n) {
  std::vector<std::int8_t> out(n, 0);
  if (n % 8 != 0) return out;
  for (std::size_t g = 0; g < n / 8; ++g) {
    std::uint64_t word = 0;
    for (int b = 0; b < 5; ++b) {
      word |= static_cast<std::uint64_t>(packed[g * 5 + static_cast<std::size_t>(b)])
              << (8 * b);
    }
    for (int i = 0; i < 8; ++i) {
      out[g * 8 + static_cast<std::size_t>(i)] =
          static_cast<std::int8_t>(static_cast<int>((word >> (5 * i)) & 31u) - 16);
    }
  }
  return out;
}

void DequantGrouped(const std::uint8_t* codes, const float* mn, const float* mx,
                    int bits, int group, std::size_t cols, float* out) {
  const float levels = static_cast<float>((1 << bits) - 1);
  for (std::size_t c = 0; c < cols; ++c) {
    const auto g = static_cast<std::size_t>(static_cast<int>(c) / group);
    const float lo = mn[g];
    const float step = std::max(mx[g] - lo, 1e-8f) * (1.0f / levels);
    out[c] = lo + static_cast<float>(codes[c]) * step;
  }
}

void Bf16Round(const float* in, std::size_t n, float* out) {
  for (std::size_t i = 0; i < n; ++i) out[i] = detail::Bf16RoundOne(in[i]);
}

void Butterflies(float* x, std::size_t rows, int cols) {
  for (std::size_t r = 0; r < rows; ++r) {
    float* row = x + r * static_cast<std::size_t>(cols);
    for (int span = 1; span < cols; span *= 2) {
      for (int base = 0; base < cols; base += 2 * span) {
        for (int i = 0; i < span; ++i) {
          const float a = row[base + i];
          const float b = row[base + span + i];
          row[base + i] = a + b;
          row[base + span + i] = a - b;
        }
      }
    }
  }
}

std::vector<float> RadixMatrixGf(int radix) {
  const auto els = Elements(radix);
  const int n = static_cast<int>(els.size());  // 5 for radix 12, 9 for radix 20
  const int r = 2 * (n + 1);
  if (radix != 12 && radix != 20) return {};
  // Q[i][j] = chi(elem_i - elem_j); C is Q bordered by a row and a column of ones
  // with a zero corner.
  std::vector<std::vector<double>> C(static_cast<std::size_t>(n) + 1,
                                     std::vector<double>(static_cast<std::size_t>(n) + 1, 0.0));
  for (int j = 0; j < n; ++j) C[0][static_cast<std::size_t>(j) + 1] = 1.0;
  for (int i = 0; i < n; ++i) C[static_cast<std::size_t>(i) + 1][0] = 1.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      C[static_cast<std::size_t>(i) + 1][static_cast<std::size_t>(j) + 1] =
          static_cast<double>(Chi(radix, Sub(radix, els[static_cast<std::size_t>(i)],
                                             els[static_cast<std::size_t>(j)])));
    }
  }
  const double A[2][2] = {{1.0, 1.0}, {1.0, -1.0}};
  const double B[2][2] = {{1.0, -1.0}, {-1.0, -1.0}};
  std::vector<float> H(static_cast<std::size_t>(r) * static_cast<std::size_t>(r), 0.0f);
  // H = kron(C, A) + kron(I_{n+1}, B)
  for (int bi = 0; bi <= n; ++bi) {
    for (int bj = 0; bj <= n; ++bj) {
      const double c = C[static_cast<std::size_t>(bi)][static_cast<std::size_t>(bj)];
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          const int row = bi * 2 + i;
          const int col = bj * 2 + j;
          double v = c * A[i][j];
          if (bi == bj) v += B[i][j];
          H[static_cast<std::size_t>(row) * static_cast<std::size_t>(r) +
            static_cast<std::size_t>(col)] = static_cast<float>(v);
        }
      }
    }
  }
  return H;
}

void ApplyRadix(float* x, std::size_t blocks, int radix, int cols, double norm) {
  const auto H = RadixMatrixGf(radix);
  if (H.empty()) return;
  const float scale = static_cast<float>(1.0 / std::sqrt(norm));
  std::vector<float> col(static_cast<std::size_t>(radix));
  std::vector<float> acc(static_cast<std::size_t>(radix));
  for (std::size_t b = 0; b < blocks; ++b) {
    for (int c = 0; c < cols; ++c) {
      for (int i = 0; i < radix; ++i) {
        col[static_cast<std::size_t>(i)] =
            x[(b * static_cast<std::size_t>(radix) + static_cast<std::size_t>(i)) *
                  static_cast<std::size_t>(cols) +
              static_cast<std::size_t>(c)];
      }
      std::fill(acc.begin(), acc.end(), 0.0f);
      for (int i = 0; i < radix; ++i) {
        for (int j = 0; j < radix; ++j) {
          acc[static_cast<std::size_t>(i)] +=
              H[static_cast<std::size_t>(i) * static_cast<std::size_t>(radix) +
                static_cast<std::size_t>(j)] *
              col[static_cast<std::size_t>(j)];
        }
      }
      for (int i = 0; i < radix; ++i) {
        x[(b * static_cast<std::size_t>(radix) + static_cast<std::size_t>(i)) *
              static_cast<std::size_t>(cols) +
          static_cast<std::size_t>(c)] = acc[static_cast<std::size_t>(i)] * scale;
      }
    }
  }
}

void HadamardExpert(float* x, std::size_t rows, int cols) {
  if (IsPowerOfTwo(cols)) {
    Butterflies(x, rows, cols);
    const float s = static_cast<float>(1.0 / std::sqrt(static_cast<double>(cols)));
    for (std::size_t i = 0; i < rows * static_cast<std::size_t>(cols); ++i) x[i] *= s;
    return;
  }
  const int radix = (cols % 12 == 0 && IsPowerOfTwo(cols / 12)) ? 12 : 20;
  const int inner = cols / radix;
  if (cols % radix != 0 || !IsPowerOfTwo(inner)) return;  // shape not supported
  // The inner stage is a NORMALISED transform of its own: `decode.py` recurses
  // into `hadamard` here, which divides by sqrt(inner). Calling the bare
  // butterflies leaves that factor out and scales the whole result by
  // sqrt(inner) - a factor the differential gate caught as exactly 64x on the
  // real expert shapes (sqrt(128 * 32)).
  if (inner > 1) HadamardExpert(x, rows * static_cast<std::size_t>(radix), inner);
  ApplyRadix(x, rows, radix, inner, static_cast<double>(radix));
}

std::vector<float> ExpertSigns(std::uint64_t seed, int dim) {
  // Ten rounds of counter mixing. `decode.py` runs all `dim` lanes as columns of
  // one array; the lanes never interact, so each is expanded on its own here and
  // the result is bit-identical to the column-wise form.
  constexpr std::uint64_t kPm0 = 0xD2511F53u;
  constexpr std::uint64_t kPm1 = 0xCD9E8D57u;
  constexpr std::uint64_t kPw0 = 0x9E3779B9u;
  constexpr std::uint64_t kPw1 = 0xBB67AE85u;
  constexpr std::uint64_t kMask = 0xFFFFFFFFu;
  // 2*pi / 2^32 and the two exact powers of two the fractions use.
  const float inv2pi = static_cast<float>(6.283185307179586476925286766559 / 4294967296.0);
  constexpr float kUnit = 1.0f / 4294967296.0f;   // 2^-32, exact
  constexpr float kHalfUnit = 1.0f / 8589934592.0f;  // 2^-33, exact
  constexpr float kPi = 3.14159265358979323846f;

  std::vector<float> out(static_cast<std::size_t>(dim), 1.0f);
  for (int j = 0; j < dim; ++j) {
    std::uint64_t k0 = seed & kMask;
    std::uint64_t k1 = (seed >> 32) & kMask;
    std::uint64_t c0 = 0;
    std::uint64_t c1 = 0;
    std::uint64_t c3 = 0;
    std::uint64_t c2 = static_cast<std::uint64_t>(j);  // the counter: the lane index
    for (int r = 0; r < 10; ++r) {
      if (r != 0) {
        k0 = (k0 + kPw0) & kMask;
        k1 = (k1 + kPw1) & kMask;
      }
      // Both products WRAP modulo 2^64 before the high half is taken, which is
      // what numpy's uint64 multiply does and what makes the mixing nonlinear.
      const std::uint64_t p0 = kPm0 * c0;
      const std::uint64_t p1 = kPm1 * c2;
      const std::uint64_t n0 = (p1 >> 32) ^ c1 ^ k0;
      const std::uint64_t n1 = p1 & kMask;
      const std::uint64_t n2 = (p0 >> 32) ^ c3 ^ k1;
      const std::uint64_t n3 = p0 & kMask;
      c0 = n0;
      c1 = n1;
      c2 = n2;
      c3 = n3;
    }
    const float v = static_cast<float>(c1) * inv2pi + inv2pi / 2.0f;
    const float u = static_cast<float>(c0) * kUnit + kHalfUnit;
    out[static_cast<std::size_t>(j)] = (v <= kPi || u == 1.0f) ? 1.0f : -1.0f;
  }
  return out;
}

void DecodeExpertTile(const uint16_t* states, std::int64_t nstep, const float* lut, int mb,
                      int nb, const float* gamma, float wscale, const float* su,
                      const float* sv, float* out) {
  const int td = kTile;
  const int m = mb * td;
  const int n = nb * td;
  if (mb <= 0 || nb <= 0 || nstep <= 0 || states == nullptr || lut == nullptr ||
      gamma == nullptr || su == nullptr || sv == nullptr || out == nullptr) {
    return;
  }
  // The reshape is (mb, nb, td, td) over the gathered stream, so a codeword's four
  // values land in four consecutive COLUMNS of one row; the transpose then makes
  // the tile column the slow axis of the output row.
  for (int bm = 0; bm < mb; ++bm) {
    for (int bn = 0; bn < nb; ++bn) {
      for (int r = 0; r < td; ++r) {
        for (int c = 0; c < td; ++c) {
          const std::int64_t flat =
              (static_cast<std::int64_t>(bm) * nb + bn) * td * td + r * td + c;
          const std::int64_t cw = flat / kExpertValues;
          if (cw >= nstep) return;
          const int v = static_cast<int>(flat % kExpertValues);
          out[(bm * td + r) * n + (bn * td + c)] =
              lut[static_cast<std::size_t>(states[cw]) * kExpertValues +
                  static_cast<std::size_t>(v)];
        }
      }
    }
  }

  // The per-tile gamma, looked up through the wave map because the scales are
  // stored in walk order rather than in tile order.
  const auto map = WaveMap(mb, nb);
  if (static_cast<int>(map.size()) != mb * nb) return;
  for (int bm = 0; bm < mb; ++bm) {
    for (int bn = 0; bn < nb; ++bn) {
      const int g = map[static_cast<std::size_t>(bm * nb + bn)];
      const float scale = gamma[static_cast<std::size_t>(g)];
      for (int r = 0; r < td; ++r) {
        for (int c = 0; c < td; ++c) {
          out[(bm * td + r) * n + (bn * td + c)] *= scale;
        }
      }
    }
  }
  for (std::size_t i = 0; i < static_cast<std::size_t>(m) * static_cast<std::size_t>(n); ++i) {
    out[i] *= wscale;
  }

  // The rotation, then the signs: rows first, then the columns through a
  // transpose (the release decoder writes `hadamard(x.T)`, which is a transform
  // along the other axis).
  HadamardExpert(out, static_cast<std::size_t>(m), n);
  for (int c = 0; c < n; ++c) {
    for (int r = 0; r < m; ++r) out[r * n + c] *= su[c];
  }
  std::vector<float> t(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0f);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) t[static_cast<std::size_t>(c) * static_cast<std::size_t>(m) +
                                     static_cast<std::size_t>(r)] = out[r * n + c];
  }
  HadamardExpert(t.data(), static_cast<std::size_t>(n), m);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      t[static_cast<std::size_t>(c) * static_cast<std::size_t>(m) +
        static_cast<std::size_t>(r)] *= sv[r];
    }
  }
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) out[r * n + c] = t[static_cast<std::size_t>(c) * static_cast<std::size_t>(m) +
                                                     static_cast<std::size_t>(r)];
  }
}

void HadamardSpine(float* x, std::size_t rows, int cols) {
  if (x == nullptr || rows == 0 || cols <= 0) return;
  if (IsPowerOfTwo(cols)) {
    Butterflies(x, rows, cols);
    const float s = static_cast<float>(1.0 / std::sqrt(static_cast<double>(cols)));
    for (std::size_t i = 0; i < rows * static_cast<std::size_t>(cols); ++i) x[i] *= s;
    return;
  }
  const int radix = (cols % 12 == 0 && IsPowerOfTwo(cols / 12)) ? 12 : 20;
  const int inner = cols / radix;
  if (cols % radix != 0 || !IsPowerOfTwo(inner)) return;  // shape not supported
  // The spine branch's own convention: BARE butterflies on the inner block, then
  // the whole product divided by sqrt(cols). The expert branch is the mirror of
  // this, and using either one's scale in the other path is a silent rescale.
  Butterflies(x, rows * static_cast<std::size_t>(radix), inner);
  ApplySpineRadix(x, rows, radix, inner,
                  static_cast<float>(1.0 / std::sqrt(static_cast<double>(cols))));
}

void DecodeNeTensor(const uint16_t* states, std::int64_t nstep, const float* tlut, int mb, int nb,
                    float wscale, const std::int32_t* su, const std::int32_t* sv, float rc_max_m,
                    float rc_max_n, float* out) {
  const int td = kTile;
  const int m = mb * td;
  const int n = nb * td;
  if (mb <= 0 || nb <= 0 || nstep <= 0 || states == nullptr || tlut == nullptr || su == nullptr ||
      sv == nullptr || out == nullptr) {
    return;
  }
  const auto fold = FoldSpineLut(tlut);
  // Two values per codeword, so a codeword fills two consecutive columns of one
  // row - the spine tape spends 8 bits (NE_K * NE_V) on it.
  for (int bm = 0; bm < mb; ++bm) {
    for (int bn = 0; bn < nb; ++bn) {
      for (int r = 0; r < td; ++r) {
        for (int c = 0; c < td; ++c) {
          const std::int64_t flat =
              (static_cast<std::int64_t>(bm) * nb + bn) * td * td + r * td + c;
          const std::int64_t cw = flat / kSpineValues;
          if (cw >= nstep) return;
          const int v = static_cast<int>(flat % kSpineValues);
          out[(bm * td + r) * n + (bn * td + c)] =
              fold[static_cast<std::size_t>(states[cw]) * kSpineValues + static_cast<std::size_t>(v)];
        }
      }
    }
  }
  const std::size_t total = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  for (std::size_t i = 0; i < total; ++i) out[i] *= wscale;

  // Row rotation and its signs, then the same on the other axis. `np.sign` of an
  // int is zero at zero, so a zero entry zeroes its line rather than leaving it.
  HadamardSpine(out, static_cast<std::size_t>(m), n);
  for (int c = 0; c < n; ++c) {
    const float sgn = static_cast<float>((su[c] > 0) - (su[c] < 0));
    for (int r = 0; r < m; ++r) out[r * n + c] *= sgn;
  }
  std::vector<float> t(total, 0.0f);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      t[static_cast<std::size_t>(c) * static_cast<std::size_t>(m) +
        static_cast<std::size_t>(r)] = out[r * n + c];
    }
  }
  HadamardSpine(t.data(), static_cast<std::size_t>(n), m);
  for (int r = 0; r < m; ++r) {
    const float sgn = static_cast<float>((sv[r] > 0) - (sv[r] < 0));
    for (int c = 0; c < n; ++c) {
      t[static_cast<std::size_t>(c) * static_cast<std::size_t>(m) +
        static_cast<std::size_t>(r)] *= sgn;
    }
  }
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      out[r * n + c] = t[static_cast<std::size_t>(c) * static_cast<std::size_t>(m) +
                         static_cast<std::size_t>(r)];
    }
  }

  // The decoder rounds here, before the two rescales, and a byte gate depends on
  // that order.
  for (std::size_t i = 0; i < total; ++i) out[i] = detail::Bf16RoundOne(out[i]);

  std::vector<float> row_rc(static_cast<std::size_t>(m));
  std::vector<float> col_rc(static_cast<std::size_t>(n));
  std::vector<std::int32_t> mag_m(static_cast<std::size_t>(m));
  std::vector<std::int32_t> mag_n(static_cast<std::size_t>(n));
  for (int r = 0; r < m; ++r) mag_m[static_cast<std::size_t>(r)] = sv[r] < 0 ? -sv[r] : sv[r];
  for (int c = 0; c < n; ++c) mag_n[static_cast<std::size_t>(c)] = su[c] < 0 ? -su[c] : su[c];
  RcGrid(rc_max_m, mag_m.data(), static_cast<std::size_t>(m), row_rc.data());
  RcGrid(rc_max_n, mag_n.data(), static_cast<std::size_t>(n), col_rc.data());
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      out[r * n + c] = (row_rc[static_cast<std::size_t>(r)] * out[r * n + c]) *
                       col_rc[static_cast<std::size_t>(c)];
    }
  }
}

}  // namespace vt::mach2

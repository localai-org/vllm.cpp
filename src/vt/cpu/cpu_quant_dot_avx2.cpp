// QUANT-GGUF-CIQ-GEMM work row G5 — x86 AVX2 K-quant `vec_dot` tier.
//
// Ported from the registered llama.cpp b10451 source,
// `ggml/src/ggml-cpu/arch/x86/quants.c`:
//   :2018 `ggml_vec_dot_q4_K_q8_K`   (the `__AVX2__` arm)
//   :2406 `ggml_vec_dot_q6_K_q8_K`   (the `__AVX2__` arm)
//   :3364 `ggml_vec_dot_iq3_s_q8_K`  (the `__AVX2__` arm)
// with `hsum_float_8` (:42) and the scale-shuffle tables (:495-530).
//
// Upstream's `_mm*_fmadd_ps` calls become `_mm*_add_ps(_mm*_mul_ps(...))`
// because this tree compiles with global `-ffp-contract=off`: the portable
// tier never fuses a multiply and an add, and the AVX2 tier must not change
// that rule to gain speed. The integer reductions — which carry the work —
// are upstream's verbatim.
//
// This translation unit alone is built with `-mavx2 -mf16c`, exactly like the
// existing elementwise AVX2 tier. Entry is gated on
// `DetectX86IsaCaps().avx2`, so a pre-AVX2 x86 host keeps the portable tier in
// cpu_quant_dot.cpp.

#include <immintrin.h>

#include <cstdlib>
#include <cstring>

#include "cpu_isa_x86.h"
#include "cpu_quant_blocks.h"
#include "cpu_quant_iq_tables.h"
#include "vt/dtype.h"
#include "vt/quant.h"

namespace vt::cpu {
namespace {

// quants.c:42 — horizontally add 8 floats.
inline float HsumFloat8(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}

// quants.c:495 — q4_K's per-64 scale/min shuffle. Entry i selects one of the
// eight 32-byte lanes; the table repeats each scale byte 32 times so one
// `vpshufb` broadcasts eight scales.
inline __m256i GetScaleShuffleK4(int i) {
  static const uint8_t kShuffle[256] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
  };
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kShuffle) + i);
}

// quants.c:518 — q6_K's 16-entry int8 scale broadcast, one 16-byte lane each.
inline __m128i GetScaleShuffle(int i) {
  static const uint8_t kShuffle[128] = {
      0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
      2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,
      4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,
      6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,
      8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,
      10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
      12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
      14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
  };
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(kShuffle) + i);
}

inline __m256i SetM128I(__m128i high, __m128i low) {
  return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
}

// quants.c:2018 — `ggml_vec_dot_q4_K_q8_K`, `__AVX2__` arm.
void VecDotQ4_KQ8_KAvx2(int n, float* s, size_t bs, const void* vx, size_t bx,
                        const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_q4_K_q8_K avx2: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_q4_K_q8_K avx2 supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockQ4_K* x = static_cast<const BlockQ4_K*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);
  const int nb = n / kQK_K;

  static const uint32_t kMask1 = 0x3f3f3f3f;
  static const uint32_t kMask2 = 0x0f0f0f0f;
  static const uint32_t kMask3 = 0x03030303;
  uint32_t utmp[4];

  const __m256i m4 = _mm256_set1_epi8(0xF);

  __m256 acc = _mm256_setzero_ps();
  __m128 acc_m = _mm_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * F16ToF32(x[i].d);
    const float dmin = -y[i].d * F16ToF32(x[i].dmin);

    std::memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kMask2) | (((utmp[1] >> 6) & kMask3) << 4);
    const uint32_t uaux = utmp[1] & kMask1;
    utmp[1] = (utmp[2] & kMask2) | (((utmp[0] >> 6) & kMask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kMask1;

    const uint8_t* q4 = x[i].qs;
    const int8_t* q8 = y[i].qs;

    const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
        _mm_set_epi32(static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                      static_cast<int>(utmp[1]), static_cast<int>(utmp[0])));

    const __m256i q8sums = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(y[i].bsums));
    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                       _mm256_extracti128_si256(q8sums, 1));
    const __m128i prod =
        _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
    acc_m = _mm_add_ps(
        _mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);

    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    const __m256i scales = SetM128I(sc128, sc128);

    __m256i sumi = _mm256_setzero_si256();

    for (int j = 0; j < kQK_K / 64; ++j) {
      const __m256i scale_l = _mm256_shuffle_epi8(scales, GetScaleShuffleK4(2 * j + 0));
      const __m256i scale_h = _mm256_shuffle_epi8(scales, GetScaleShuffleK4(2 * j + 1));

      const __m256i q4bits = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(q4));
      q4 += 32;
      const __m256i q4l = _mm256_and_si256(q4bits, m4);
      const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

      const __m256i q8l = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
      p16l = _mm256_madd_epi16(scale_l, p16l);

      const __m256i q8h = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
      p16h = _mm256_madd_epi16(scale_h, p16h);
      const __m256i sumj = _mm256_add_epi32(p16l, p16h);

      sumi = _mm256_add_epi32(sumi, sumj);
    }

    const __m256 vd = _mm256_set1_ps(d);
    acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);
  }

  acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
  acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

  *s = HsumFloat8(acc) + _mm_cvtss_f32(acc_m);
}

// quants.c:2406 — `ggml_vec_dot_q6_K_q8_K`, `__AVX2__` arm.
void VecDotQ6_KQ8_KAvx2(int n, float* s, size_t bs, const void* vx, size_t bx,
                        const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_q6_K_q8_K avx2: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_q6_K_q8_K avx2 supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockQ6_K* x = static_cast<const BlockQ6_K*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);
  const int nb = n / kQK_K;

  const __m256i m3 = _mm256_set1_epi8(3);
  const __m256i m15 = _mm256_set1_epi8(15);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * F16ToF32(x[i].d);

    const uint8_t* q4 = x[i].ql;
    const uint8_t* qh = x[i].qh;
    const int8_t* q8 = y[i].qs;

    const __m256i q8sums =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[i].bsums));
    const __m128i scales =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[i].scales));
    const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
    const __m256i q8sclsub = _mm256_slli_epi32(
        _mm256_madd_epi16(q8sums, scales_16), 5);

    __m256i sumi = _mm256_setzero_si256();
    int is = 0;

    for (int j = 0; j < kQK_K / 128; ++j) {
      const __m256i q4bits1 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
      q4 += 32;
      const __m256i q4bits2 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
      q4 += 32;
      const __m256i q4bitsH =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
      qh += 32;

      const __m256i q4h_0 =
          _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
      const __m256i q4h_1 = _mm256_slli_epi16(
          _mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
      const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
      const __m256i q4h_3 = _mm256_srli_epi16(
          _mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

      const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
      const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
      const __m256i q4_2 = _mm256_or_si256(
          _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
      const __m256i q4_3 = _mm256_or_si256(
          _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

      const __m256i q8_0 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      const __m256i q8_1 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      const __m256i q8_2 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      const __m256i q8_3 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;

      __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
      __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
      __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
      __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

      const __m128i scale_0 = _mm_shuffle_epi8(scales, GetScaleShuffle(is + 0));
      const __m128i scale_1 = _mm_shuffle_epi8(scales, GetScaleShuffle(is + 1));
      const __m128i scale_2 = _mm_shuffle_epi8(scales, GetScaleShuffle(is + 2));
      const __m128i scale_3 = _mm_shuffle_epi8(scales, GetScaleShuffle(is + 3));
      is += 4;

      p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
      p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
      p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
      p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
    }

    sumi = _mm256_sub_epi32(sumi, q8sclsub);
    acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d),
                                      _mm256_cvtepi32_ps(sumi)),
                        acc);
  }

  *s = HsumFloat8(acc);
}

// quants.c:3364 — `ggml_vec_dot_iq3_s_q8_K`, `__AVX2__` arm.
void VecDotIQ3_SQ8_KAvx2(int n, float* s, size_t bs, const void* vx, size_t bx,
                         const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0, "vec_dot_iq3_s_q8_K avx2: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_iq3_s_q8_K avx2 supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockIQ3_S* x = static_cast<const BlockIQ3_S*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);
  const int nb = n / kQK_K;

  static const uint8_t kMask1[32] = {
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
      0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
      0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
  };
  static const uint8_t kMask2[32] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const __m256i mask1 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(kMask1));
  const __m256i mask2 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(kMask2));

  const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
  const __m256i idx_mask = _mm256_set1_epi32(256);

  typedef union {
    __m256i vec[2];
    uint32_t index[16];
  } index_t;

  index_t idx;

  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < nb; ++i) {
    const float d = F16ToF32(x[i].d) * y[i].d;
    const uint8_t* qs = x[i].qs;
    const uint8_t* qh = x[i].qh;
    const uint16_t* signs = reinterpret_cast<const uint16_t*>(x[i].signs);
    const int8_t* q8 = y[i].qs;
    __m256i sumi1 = _mm256_setzero_si256();
    __m256i sumi2 = _mm256_setzero_si256();
    for (int ib32 = 0; ib32 < kQK_K / 32; ib32 += 2) {
      const __m256i q8_1 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      const __m256i q8_2 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      const __m256i idx_l = _mm256_cvtepu8_epi16(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs)));
      qs += 16;
      idx.vec[0] = _mm256_set1_epi32(qh[ib32 + 0]);
      idx.vec[1] = _mm256_set1_epi32(qh[ib32 + 1]);
      idx.vec[0] = _mm256_and_si256(
          _mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
      idx.vec[1] = _mm256_and_si256(
          _mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
      idx.vec[0] = _mm256_or_si256(
          idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
      idx.vec[1] = _mm256_or_si256(
          idx.vec[1],
          _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

      const __m256i q2_1 = _mm256_set_epi32(
          static_cast<int>(kIq3sGrid[idx.index[7]]),
          static_cast<int>(kIq3sGrid[idx.index[6]]),
          static_cast<int>(kIq3sGrid[idx.index[5]]),
          static_cast<int>(kIq3sGrid[idx.index[4]]),
          static_cast<int>(kIq3sGrid[idx.index[3]]),
          static_cast<int>(kIq3sGrid[idx.index[2]]),
          static_cast<int>(kIq3sGrid[idx.index[1]]),
          static_cast<int>(kIq3sGrid[idx.index[0]]));
      const __m256i q2_2 = _mm256_set_epi32(
          static_cast<int>(kIq3sGrid[idx.index[15]]),
          static_cast<int>(kIq3sGrid[idx.index[14]]),
          static_cast<int>(kIq3sGrid[idx.index[13]]),
          static_cast<int>(kIq3sGrid[idx.index[12]]),
          static_cast<int>(kIq3sGrid[idx.index[11]]),
          static_cast<int>(kIq3sGrid[idx.index[10]]),
          static_cast<int>(kIq3sGrid[idx.index[9]]),
          static_cast<int>(kIq3sGrid[idx.index[8]]));

      __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
      aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
      const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
      const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

      aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
      aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
      const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
      const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

      signs += 4;

      const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
      const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
      const uint16_t ls1 = x[i].scales[ib32 / 2] & 0xf;
      const uint16_t ls2 = x[i].scales[ib32 / 2] >> 4;
      const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1));
      const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1));
      sumi1 = _mm256_add_epi32(sumi1, p1);
      sumi2 = _mm256_add_epi32(sumi2, p2);
    }

    accumf = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(d),
                      _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2))),
        accumf);
  }

  *s = HsumFloat8(accumf);
}

bool Avx2Requested() {
  const char* value = std::getenv("VT_CPU_QUANT_AVX2");
  if (value == nullptr || std::strcmp(value, "auto") == 0) {
    return DetectX86IsaCaps().avx2;
  }
  if (std::strcmp(value, "portable") == 0) {
    return false;
  }
  if (std::strcmp(value, "avx2") == 0) {
    VT_CHECK(DetectX86IsaCaps().avx2,
             "VT_CPU_QUANT_AVX2=avx2 requires an AVX2 host");
    return true;
  }
  VT_CHECK(false, "VT_CPU_QUANT_AVX2 must be auto, portable, or avx2");
  return false;
}

}  // namespace

VecDotFn SelectQuantKVecDot(DType dtype, VecDotFn portable) {
  if (!Avx2Requested()) return portable;
  switch (dtype) {
    case DType::kQ4_K: return &VecDotQ4_KQ8_KAvx2;
    case DType::kQ6_K: return &VecDotQ6_KQ8_KAvx2;
    case DType::kIQ3_S: return &VecDotIQ3_SQ8_KAvx2;
    default: return portable;
  }
}

bool QuantKAvx2Active() {
  return Avx2Requested();
}

}  // namespace vt::cpu

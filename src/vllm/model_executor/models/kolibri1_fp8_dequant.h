// Kolibri-1 fp8-block dequant kernel (MODEL-TEXT-kolibri-1 perf row).
//
// w[n, k] = F8E4M3ToF32(packed[n, k]) * scale_inv[n / block_n, k / block_k],
// stored bf16 (one rounding, same as the loader's converting-copy).
//
// The decode must be BIT-IDENTICAL to the scalar reference
// (vt::F8E4M3ToF32 x scale, vt::F32ToBF16) because the row's token gate
// (tests/vllm/models/test_kolibri1_w3.cpp, 900/900, near-tie adjudication)
// replays fixed argmax chains over bf16 weights. The scalar reference's
// decode is exact arithmetic (mantissa composition + ldexp by an integer
// exponent, bias 7; subnormal mant*2^-9; NaN only at 0x7F/0xFF -> positive
// quiet NaN), so the same values admit an exact closed form on the f32 bit
// pattern, which the NEON path uses:
//
//   normal (exp 1..14, any mantissa, incl. the 0x7E top): f32 bits =
//       sign<<31 | (exp + 120)<<23 | mant<<20
//       (value = (1 + m/8) * 2^(e-7); the f32 bias shift and the /8-to-<<20
//       move are exact integer rewrites of the scalar composition).
//   subnormal (exp == 0): value = m * 2^-9, exact as f32 via
//       cvt(m) * 0x1p-9f (and +/-0.0 for m == 0).
//   NaN (exp == 15 && mant == 7, i.e. 0x7F/0xFF): +qNaN 0x7FC00000, sign
//       IGNORED, exactly as std::numeric_limits<float>::quiet_NaN().
//
// The bf16 store mirrors vt::F32ToBF16 bit for bit, including its NaN
// truncate-to-quiet branch and its round-to-nearest-even carry into the
// exponent.
//
// x86 CI compiles the scalar fallback; every NEON body sits under
// `defined(__aarch64__)` exactly like src/vt/cpu/cpu_matmul_elem.cpp.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_FP8_DEQUANT_H_
#define VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_FP8_DEQUANT_H_

#include <cstdint>

#include "vt/dtype.h"   // F32ToBF16
#include "vt/fp8_kv.h"  // F8E4M3ToF32 (the scalar reference decode)

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace vllm {
namespace kolibri1_fp8 {

// Scalar reference for one chunk: out[i] = F32ToBF16(F8E4M3ToF32(p[i]) * s).
inline void E4m3ScaleToBf16Scalar(const uint8_t* p, int64_t count, float s, uint16_t* out) {
  for (int64_t i = 0; i < count; ++i) {
    out[i] = vt::F32ToBF16(vt::F8E4M3ToF32(p[i]) * s);
  }
}

#if defined(__aarch64__)

// Decode 4 e4m3fn bytes (each widened into a u32 lane) to f32, bit-identical
// to vt::F8E4M3ToF32.
inline float32x4_t E4m3ToF32x4(uint32x4_t w) {
  const uint32x4_t kMantMask = vdupq_n_u32(0x7U);
  const uint32x4_t sign = vshlq_n_u32(vandq_u32(w, vdupq_n_u32(0x80U)), 24);
  const uint32x4_t mag = vandq_u32(w, vdupq_n_u32(0x7FU));
  const uint32x4_t e = vshrq_n_u32(mag, 3);  // mag <= 0x7F, so e = 0..15
  const uint32x4_t m = vandq_u32(mag, kMantMask);
  // NaN: exp == 15 && mant == 7 (0x7F / 0xFF) -> +qNaN, sign ignored.
  const uint32x4_t nan_mask = vandq_u32(vceqq_u32(e, vdupq_n_u32(0xFU)), vceqq_u32(m, kMantMask));
  // Subnormal: exp == 0 -> m * 2^-9, exact.
  const uint32x4_t sub_mask = vceqzq_u32(e);
  // Normal: bits = (e + 120) << 23 | m << 20.
  const uint32x4_t norm_bits =
      vorrq_u32(vshlq_n_u32(vaddq_u32(e, vdupq_n_u32(120U)), 23), vshlq_n_u32(m, 20));
  float32x4_t v = vreinterpretq_f32_u32(norm_bits);
  const float32x4_t sub = vmulq_f32(vcvtq_f32_u32(m), vdupq_n_f32(0x1p-9f));
  v = vbslq_f32(sub_mask, sub, v);
  // The scalar reference returns +qNaN for 0x7F AND 0xFF (sign ignored), so
  // the sign XOR is suppressed on the NaN lanes.
  const uint32x4_t sign_eff = vandq_u32(sign, vmvnq_u32(nan_mask));
  v = vbslq_f32(nan_mask, vreinterpretq_f32_u32(vdupq_n_u32(0x7FC00000U /* +qNaN */)), v);
  return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), sign_eff));
}

// bf16 store, bit-identical to vt::F32ToBF16 on the four f32 lanes.
inline uint16x4_t F32x4ToBf16x4(float32x4_t v) {
  const uint32x4_t u = vreinterpretq_u32_f32(v);
  const uint32x4_t exp_all_ones =
      vceqq_u32(vandq_u32(u, vdupq_n_u32(0x7F800000U)), vdupq_n_u32(0x7F800000U));
  const uint32x4_t mant_nonzero = vceqq_u32(vandq_u32(u, vdupq_n_u32(0x7FFFFFU)), vdupq_n_u32(0U));
  const uint32x4_t nan_mask = vandq_u32(exp_all_ones, vmvnq_u32(mant_nonzero));
  const uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1U));
  // 0x7FFF + lsb: round to nearest even, may carry into the exponent.
  const uint32x4_t rnd = vaddq_u32(vaddq_u32(u, vdupq_n_u32(0x7FFFU)), lsb);
  const uint32x4_t norm = vshrq_n_u32(rnd, 16);
  const uint32x4_t nan_bf = vorrq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(0x40U));
  return vqmovn_u32(vbslq_u32(nan_mask, nan_bf, norm));
}

// Decode 16 consecutive e4m3fn bytes, multiply by the (constant) block scale,
// and store bf16. Bit-identical to E4m3ScaleToBf16Scalar(p, 16, s, out).
inline void E4m3x16ScaleToBf16Neon(const uint8_t* p, float s, uint16_t* out) {
  const uint8x16_t b = vld1q_u8(p);
  const float32x4_t s4 = vdupq_n_f32(s);
  // Widen bytes -> u32 lanes, zero-extension, in lane order.
  const uint16x8_t lo16 = vmovl_u8(vget_low_u8(b));
  const uint16x8_t hi16 = vmovl_u8(vget_high_u8(b));
  const uint32x4_t w0v = vmovl_u16(vget_low_u16(lo16));
  const uint32x4_t w1v = vmovl_u16(vget_high_u16(lo16));
  const uint32x4_t w2v = vmovl_u16(vget_low_u16(hi16));
  const uint32x4_t w3v = vmovl_u16(vget_high_u16(hi16));
  const float32x4_t f0 = vmulq_f32(E4m3ToF32x4(w0v), s4);
  const float32x4_t f1 = vmulq_f32(E4m3ToF32x4(w1v), s4);
  const float32x4_t f2 = vmulq_f32(E4m3ToF32x4(w2v), s4);
  const float32x4_t f3 = vmulq_f32(E4m3ToF32x4(w3v), s4);
  vst1q_u16(out, vcombine_u16(F32x4ToBf16x4(f0), F32x4ToBf16x4(f1)));
  vst1q_u16(out + 8, vcombine_u16(F32x4ToBf16x4(f2), F32x4ToBf16x4(f3)));
}

#endif  // defined(__aarch64__)

// Dequant the row range [n0, n1) of the fp8-block weight into dst (bf16,
// [n, k] row-major). Called per output-row range from the pool; each output
// element depends only on (n, k), so any partition is bit-identical.
inline void DequantRowsBf16(const uint8_t* src, const float* sc, int64_t scale_cols, int64_t n0,
                            int64_t n1, int64_t k_dim, int64_t block_n, int64_t block_k,
                            uint16_t* dst) {
  for (int64_t n = n0; n < n1; ++n) {
    const int64_t sr = n / block_n;
    const float* sc_row = sc + sr * scale_cols;
    const uint8_t* row = src + n * k_dim;
    uint16_t* out = dst + n * k_dim;
    int64_t k = 0;
    while (k < k_dim) {
      const int64_t blk = k / block_k;
      const float s = sc_row[blk];
      // Chunks never cross a block boundary: the scale is constant inside.
      int64_t chunk = block_k - (k % block_k);
      if (chunk > k_dim - k) chunk = k_dim - k;
#if defined(__aarch64__)
      while (chunk >= 16) {
        E4m3x16ScaleToBf16Neon(row + k, s, out + k);
        k += 16;
        chunk -= 16;
      }
#endif
      if (chunk > 0) {
        E4m3ScaleToBf16Scalar(row + k, chunk, s, out + k);
        k += chunk;
      }
    }
  }
}

}  // namespace kolibri1_fp8
}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_FP8_DEQUANT_H_

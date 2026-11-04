// Q6_K x Q8_K NEON DotProd `vec_dot` (FLINT-CPU-DESCENT-2).
//
// The keep-quant arm's UNTIED lm_head is [V, H] Q6_K in the published flint
// Q4_K_M container (`output.weight`, ggml type 14), and the routed-expert
// dot's NEON tier (cpu_quant_dot_q4k_neon.cpp) does not cover it: the head has
// been running the portable scalar kernel (cpu_quant_dot.cpp, quants.c:800)
// behind the same kMatmulBTQuant CIQ GEMM. This TU is its NEON replacement,
// following the exact house shape of cpu_quant_dot_q4k_neon.cpp: ONE
// translation unit compiled wide with `+dotprod`, runtime-gated on the shared
// HWCAP/HWCAP2 dotprod probe, behind the `VT_CPU_Q6K_DOT=auto|portable|neon`
// switch, with the portable kernel as the universal fallback.
//
// BIT-IDENTITY CONTRACT. The row's gate is bit-identity with the scalar
// `VecDotQ6_KQ8_K` on EVERY input, and again the integer core alone does not
// buy it. The scalar accumulates per 16-value chunk: `aux16[l] = q8[8k+l] *
// a[8k+l]` for k = 0..1, `aux32[l] += scales[chunk] * aux16[l]` over 16
// chunks, then per super-block `sums[l] += d * (float)aux32[l]` with a
// non-fused mul-then-add (`-ffp-contract=off` is global), and the final
// `sumf += sums[l]` fold AFTER the block loop. The NEON kernel keeps the same
// 8-lane partition: per chunk, the two int8x8 halves' products are summed in
// int16 (exact: each |product| <= 128*32 = 4096 because the unpacked a is in
// [-32, 31], so the pair fits int16), widened to int32 lanes 0..3 / 4..7,
// scaled, and accumulated in TWO separate accumulators so each lane is
// exactly one scalar aux32[l]. The float stage folds d into the lane BEFORE
// the accumulate, lane-wise, and the final fold extracts lanes in scalar
// order. The 6-bit unpack (ql nibble + qh pair - 32) is also lane-wide but
// purely integer, so it is exact by construction.
//
// The integer magnitudes are the scalar kernel's own: a 16-product chunk pair
// is <= 2 * 128*32 = 8192 in int16, times an 8-bit scale <= 255 in int32 per
// lane and chunk — far inside int32 over 16 chunks.
#include "vt/quant.h"

#if defined(VT_CPU_ARM_Q6K_DOT) && defined(__aarch64__) && \
    defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>

#include <cstdlib>
#include <cstring>

#include "cpu_quant_blocks.h"
#include "vt/cpu/cpu_isa_arm.h"

namespace vt::cpu {
namespace {

bool CpuHasDotProd() {
  return ArmIsaTierSupported(DetectArmIsaCaps(), ArmIsaTier::kDotProd);
}

// One 128-value quarter of the unpack: for l = 0..31 the scalar writes
//   a[l]    = (q4[l]    & 0xF | ((qh[l] >> 0) & 3) << 4) - 32
//   a[l+32] = (q4[l+32] & 0xF | ((qh[l] >> 2) & 3) << 4) - 32
//   a[l+64] = (q4[l]    >> 4  | ((qh[l] >> 4) & 3) << 4) - 32
//   a[l+96] = (q4[l+32] >> 4  | ((qh[l] >> 6) & 3) << 4) - 32
// Purely integer, so the NEON form is exact by construction.
inline void UnpackQuarter(const uint8_t* q4, const uint8_t* qh, int8_t* a) {
  const uint8x16_t ql0 = vld1q_u8(q4);       // scalar l = 0..15
  const uint8x16_t ql1 = vld1q_u8(q4 + 16);  // scalar l = 16..31
  const uint8x16_t ql2 = vld1q_u8(q4 + 32);  // scalar l = 32..47 -> a[l+32]
  const uint8x16_t ql3 = vld1q_u8(q4 + 48);  // scalar l = 48..63 -> a[l+32]
  const uint8x16_t lo0 = vandq_u8(ql0, vdupq_n_u8(0xF));
  const uint8x16_t lo1 = vandq_u8(ql1, vdupq_n_u8(0xF));
  const uint8x16_t lo2 = vandq_u8(ql2, vdupq_n_u8(0xF));
  const uint8x16_t lo3 = vandq_u8(ql3, vdupq_n_u8(0xF));
  const uint8x16_t hi0 = vshrq_n_u8(ql0, 4);
  const uint8x16_t hi1 = vshrq_n_u8(ql1, 4);
  const uint8x16_t hi2 = vshrq_n_u8(ql2, 4);
  const uint8x16_t hi3 = vshrq_n_u8(ql3, 4);
  // qh is 32 bytes per quarter: qh[l] for l = 0..31. The two 16-byte halves
  // feed the low/high output vectors of each plane respectively.
  const uint8x16_t qh0 = vld1q_u8(qh);      // qh[0..15]
  const uint8x16_t qh1 = vld1q_u8(qh + 16); // qh[16..31]
  const uint8x16_t bias = vdupq_n_u8(32);
  const auto unpack = [&](const uint8x16_t& nib, uint8_t shift,
                          const uint8x16_t& qhv) {
    const uint8x16_t p =
        vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, shift), vdupq_n_u8(3)), 4);
    return vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(nib, p), bias));
  };
  vst1q_s8(a, unpack(lo0, 0, qh0));
  vst1q_s8(a + 16, unpack(lo1, 0, qh1));
  vst1q_s8(a + 32, unpack(lo2, 2, qh0));
  vst1q_s8(a + 48, unpack(lo3, 2, qh1));
  vst1q_s8(a + 64, unpack(hi0, 4, qh0));
  vst1q_s8(a + 80, unpack(hi1, 4, qh1));
  vst1q_s8(a + 96, unpack(hi2, 6, qh0));
  vst1q_s8(a + 112, unpack(hi3, 6, qh1));
}

// One 16-value chunk: lane l of the pair (acc_lo lanes 0..3, acc_hi lanes
// 4..7) gains `scale * (q8[l] * a[l] + q8[l+8] * a[l+8])`, the scalar's
// `aux32[l] += scale * aux16[l]` over both int8x8 halves of the chunk.
inline void DotChunk(const int8_t* q8, const int8_t* a, int scale,
                     int32x4_t* acc_lo, int32x4_t* acc_hi) {
  const int8x16_t qv = vld1q_s8(q8);
  const int8x16_t av = vld1q_s8(a);
  const int16x8_t p_lo = vmull_s8(vget_low_s8(qv), vget_low_s8(av));
  const int16x8_t p_hi = vmull_s8(vget_high_s8(qv), vget_high_s8(av));
  const int16x8_t s16 = vaddq_s16(p_lo, p_hi);
  const int32x4_t lo32 = vmovl_s16(vget_low_s16(s16));
  const int32x4_t hi32 = vmovl_s16(vget_high_s16(s16));
  const int32x4_t scale_v = vdupq_n_s32(scale);
  *acc_lo = vaddq_s32(*acc_lo, vmulq_s32(scale_v, lo32));
  *acc_hi = vaddq_s32(*acc_hi, vmulq_s32(scale_v, hi32));
}

// quants.c:800 — ggml_vec_dot_q6_K_q8_K_generic, NEON DotProd arm.
// Bit-identical to the scalar reference by construction (see the header
// comment).
void VecDotQ6KNeon(int n, float* s, size_t bs, const void* vx, size_t bx,
                   const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0,
           "vec_dot_q6_K_q8_K neon: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_q6_K_q8_K neon: supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockQ6_K* x = static_cast<const BlockQ6_K*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);

  const int nb = n / kQK_K;

  int8_t aux8[kQK_K];
  float sums[8];
  std::memset(sums, 0, 8 * sizeof(float));

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* q4 = x[i].ql;
    const uint8_t* qh = x[i].qh;
    const int8_t* q8 = y[i].qs;
    int8_t* a = aux8;
    for (int j = 0; j < kQK_K; j += 128) {
      UnpackQuarter(q4, qh, a);
      a += 128;
      q4 += 64;
      qh += 32;
    }
    a = aux8;
    int is = 0;
    int32x4_t acc_lo = vdupq_n_s32(0);
    int32x4_t acc_hi = vdupq_n_s32(0);
    for (int j = 0; j < kQK_K / 16; ++j) {
      DotChunk(q8, a, x[i].scales[is++], &acc_lo, &acc_hi);
      q8 += 16;
      a += 16;
    }
    // The scalar's float stage in its exact order: per lane,
    // sums[l] += d * (float)aux32[l], non-fused mul-then-add, lane-wise —
    // so each lane stays one scalar accumulator.
    const float d = F16ToF32(x[i].d) * y[i].d;
    const float32x4_t scaled_lo = vmulq_n_f32(vcvtq_f32_s32(acc_lo), d);
    const float32x4_t scaled_hi = vmulq_n_f32(vcvtq_f32_s32(acc_hi), d);
    vst1q_f32(sums, vaddq_f32(vld1q_f32(sums), scaled_lo));
    vst1q_f32(sums + 4, vaddq_f32(vld1q_f32(sums + 4), scaled_hi));
  }
  // The scalar's final fold in its exact lane order.
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
}

}  // namespace

VecDotFn QuantQ6KNeonVecDot() {
  return CpuHasDotProd() ? &VecDotQ6KNeon : nullptr;
}

VecDotFn SelectQuantQ6KVecDot(VecDotFn portable) {
  const char* value = std::getenv("VT_CPU_Q6K_DOT");
  if (value == nullptr || std::strcmp(value, "auto") == 0) {
    const VecDotFn neon = QuantQ6KNeonVecDot();
    return neon != nullptr ? neon : portable;
  }
  if (std::strcmp(value, "portable") == 0) {
    return portable;
  }
  if (std::strcmp(value, "neon") == 0) {
    const VecDotFn selected = QuantQ6KNeonVecDot();
    VT_CHECK(selected != nullptr,
             "VT_CPU_Q6K_DOT=neon requires NEON DotProd OS capability");
    return selected;
  }
  VT_CHECK(false, "VT_CPU_Q6K_DOT must be auto, portable, or neon");
  return portable;
}

bool QuantQ6KNeonActive() { return QuantQ6KNeonVecDot() != nullptr; }

}  // namespace vt::cpu

#else  // !VT_CPU_ARM_Q6K_DOT || !__aarch64__ || !__ARM_FEATURE_DOTPROD

// Non-dotprod (or non-aarch64) build: no NEON tier exists; the dispatch falls
// back to the portable scalar kernel and the explicit getters report inert,
// mirroring cpu_quant_dot_q4k_neon.cpp.
namespace vt::cpu {

VecDotFn QuantQ6KNeonVecDot() { return nullptr; }
VecDotFn SelectQuantQ6KVecDot(VecDotFn portable) { return portable; }
bool QuantQ6KNeonActive() { return false; }

}  // namespace vt::cpu

#endif

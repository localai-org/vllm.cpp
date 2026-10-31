// Q4_K x Q8_K NEON DotProd `vec_dot` (FLINT-NEON-Q4K-DOT).
//
// The m=1 decode routed-expert dot for the Q4_K→Q8_K keep-quant class runs the
// portable scalar kernel (cpu_quant_dot.cpp, quants.c:645) while the attention
// kernel already has its NEON lanes — FLINT-SLIDING-ATTN measured that scalar
// dot as ~55% of the matched-pool routed-expert phase. This TU is the NEON
// replacement, following the exact house shape of cpu_quant_dot_sdot.cpp
// (KERNEL-CPU-A76-Q8-DOT): ONE translation unit compiled wide with
// `+dotprod` so `<arm_neon.h>` exposes `vdotq_s32`, runtime-gated on the
// shared HWCAP/HWCAP2 dotprod probe, behind the
// `VT_CPU_Q4K_DOT=auto|portable|neon` switch, with the portable kernel as the
// universal fallback.
//
// FAILED DRAFTS. A third draft read the nibble plane as one 16-byte sub-block
// read — the nibble plane desynced from the 8x4 transpose contract.
//
// HWCAP HARDENING NOTE. This TU compiles -march=armv8.2-a+dotprod+fp16
// unconditionally on aarch64; auto mode is safe only because
// SelectQuantQ4KVecDot consults CpuHasDotProd (HWCAP). Bypassing the HWCAP
// gate on a non-dotprod host would SIGILL at the first dispatched dot
// (SDOT/UDOT with no fallback). A one-time SDOT probe hardening is a named
// candidate, not in this row.
//
// BIT-IDENTITY CONTRACT. The row's gate is bit-identity with the scalar
// `VecDotQ4_KQ8_K` on EVERY input, and the integer core alone does not buy it:
// the scalar's `aux32[l]` accumulates a STRIDED partition — within each
// 32-value sub-block, `aux32[l] = Σ_j scales[j] * (Σ_{c=0..3} prod at value
// indices l + 8c)` — and the float stage folds `sums[l] += d*(float)aux32[l]`
// per l BEFORE the final sequential `sumf += sums[l]`. A first draft that let
// vdotq lanes group 4 CONSECUTIVE values moved the same products into
// different float accumulators and shifted the final sum by 1-2 ulp (the
// bit-identity suite caught it). The fix is the kPerm32 8x4 transpose below:
// both operands are permuted identically so each vdotq lane sums exactly one
// strided group, and the two 16-byte halves stay in SEPARATE accumulators
// (acc0 lanes = aux32[0..3], acc1 lanes = aux32[4..7], both summed over ALL
// 8 sub-blocks with the sub-block scale applied BEFORE the accumulation —
// exactly the scalar's `aux32[l] += scale * aux16[l]`). Merging the halves
// lane-wise would turn the scalar's single `d * (float)aux32[l]` into two
// float mul-adds and shift the rounding again — the suite caught that too.
//
// The remaining steps copy the scalar order verbatim: per-super-block scales
// decode (kmask1/2/3), the mins bias as a scalar 16-term integer sum, eight
// separate float accumulators with non-fused mul-then-add (`-ffp-contract=off`
// is global), and the final `sumf += sums[l]` fold AFTER the block loop. The
// integer magnitudes are the scalar kernel's own: a strided 4-product group is
// ≤ 4*127*15 = 7620, times a 6-bit scale ≤ 63 ⇒ ≤ ~4.8e5 per term, ≤ ~3.8e6
// per lane over the 8 sub-blocks — far inside int32.
#include "vt/quant.h"

#if defined(VT_CPU_ARM_Q4K_DOT) && defined(__aarch64__) && \
    defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "cpu_quant_blocks.h"
#include "vt/cpu/cpu_isa_arm.h"

namespace vt::cpu {
namespace {

bool CpuHasDotProd() {
  return ArmIsaTierSupported(DetectArmIsaCaps(), ArmIsaTier::kDotProd);
}

inline void DotSubBlock(const uint8_t* q4b, int odd, const int8_t* q8b,
                        int scale, uint8x16_t qperm_a, uint8x16_t qperm_b,
                        int32x4_t* acc_lo, int32x4_t* acc_hi) {
  const uint8x16_t raw0 = vld1q_u8(q4b);
  const uint8x16_t raw1 = vld1q_u8(q4b + 16);
  const uint8x16x2_t container =
      odd ? uint8x16x2_t{vshrq_n_u8(raw0, 4), vshrq_n_u8(raw1, 4)}
          : uint8x16x2_t{vandq_u8(raw0, vdupq_n_u8(0x0F)),
                         vandq_u8(raw1, vdupq_n_u8(0x0F))};
  const uint8x16_t np0 = vqtbl2q_u8(container, qperm_a);
  const uint8x16_t np1 = vqtbl2q_u8(container, qperm_b);
  const uint8x16x2_t act = {
      vld1q_u8(reinterpret_cast<const uint8_t*>(q8b)),
      vld1q_u8(reinterpret_cast<const uint8_t*>(q8b) + 16)};
  const uint8x16_t qp0 = vqtbl2q_u8(act, qperm_a);
  const uint8x16_t qp1 = vqtbl2q_u8(act, qperm_b);
  const int8x16_t qa0 = vreinterpretq_s8_u8(qp0);
  const int8x16_t qb0 = vreinterpretq_s8_u8(np0);
  const int8x16_t qa1 = vreinterpretq_s8_u8(qp1);
  const int8x16_t qb1 = vreinterpretq_s8_u8(np1);
  // Half 0 carries strided groups 0..3, half 1 groups 4..7; they stay in
  // SEPARATE accumulators so each lane is exactly one scalar aux32[l] (the
  // float stage folds d * (float)aux32[l] per l — merging the halves would
  // turn that into two float mul-adds and shift the rounding).
  const int32x4_t scale_v = vdupq_n_s32(scale);
  const int32x4_t dot_lo = vdotq_s32(vdupq_n_s32(0), qa0, qb0);
  const int32x4_t dot_hi = vdotq_s32(vdupq_n_s32(0), qa1, qb1);
  *acc_lo = vaddq_s32(*acc_lo, vmulq_s32(scale_v, dot_lo));
  *acc_hi = vaddq_s32(*acc_hi, vmulq_s32(scale_v, dot_hi));
}

// quants.c:645 — ggml_vec_dot_q4_K_q8_K_generic, NEON DotProd arm. Bit-identical
// to the scalar reference by construction (see the header comment).
void VecDotQ4KNeon(int n, float* s, size_t bs, const void* vx, size_t bx,
                   const void* vy, size_t by, int nrc) {
  VT_CHECK(n % kQK_K == 0,
           "vec_dot_q4_K_q8_K neon: n must be a multiple of 256");
  VT_CHECK(nrc == 1, "vec_dot_q4_K_q8_K neon: supports nrc == 1 only");
  (void)nrc;
  (void)bx;
  (void)by;
  (void)bs;

  const BlockQ4_K* x = static_cast<const BlockQ4_K*>(vx);
  const BlockQ8_K* y = static_cast<const BlockQ8_K*>(vy);

  const int nb = n / kQK_K;

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];

  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);

  float sums[8];
  std::memset(sums, 0, 8 * sizeof(float));

  // The 8x4 transpose: permuted byte t pairs sub-block value (t/4) + 8*(t%4),
  // so vdotq lane L holds exactly the scalar's strided group L. Both the
  // nibble plane and the activation bytes are permuted with this one table.
  static const uint8_t kPerm32[32] = {0,  8,  16, 24, 1,  9,  17, 25,
                                      2,  10, 18, 26, 3,  11, 19, 27,
                                      4,  12, 20, 28, 5,  13, 21, 29,
                                      6,  14, 22, 30, 7,  15, 23, 31};
  const uint8x16_t qperm_a = vld1q_u8(kPerm32);
  const uint8x16_t qperm_b = vld1q_u8(kPerm32 + 16);

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t* q4 = x[i].qs;
    const int8_t* q8 = y[i].qs;
    std::memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    int sumi = 0;
    for (int j = 0; j < kQK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];

    // Sub-block s folds into acc0 (s < 4) or acc1 (s >= 4), each with its OWN
    // sub-block scale. Sub-block s reads q4 bytes [32*(s>>1), +32) and picks
    // the low/high nibble plane by parity; the activation side reads q8
    // values [32s, 32s+32).
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    for (int sb = 0; sb < kQK_K / 32; ++sb) {
      DotSubBlock(q4 + 16 * (sb & ~1), sb & 1, q8 + 32 * sb,
                  static_cast<int>(scales[sb]), qperm_a, qperm_b, &acc0,
                  &acc1);
    }

    // The scalar kernel's float stage in its exact order: eight separate
    // accumulators, non-fused mul-then-add, per super-block.
    const float d = F16ToF32(x[i].d) * y[i].d;
    const float32x4_t scaled0 = vmulq_n_f32(vcvtq_f32_s32(acc0), d);
    const float32x4_t scaled1 = vmulq_n_f32(vcvtq_f32_s32(acc1), d);
    vst1q_f32(sums, vaddq_f32(vld1q_f32(sums), scaled0));
    vst1q_f32(sums + 4, vaddq_f32(vld1q_f32(sums + 4), scaled1));

    const float dmin = F16ToF32(x[i].dmin) * y[i].d;
    sumf -= dmin * static_cast<float>(sumi);
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  *s = sumf;
}

}  // namespace

VecDotFn QuantQ4KNeonVecDot() {
  return CpuHasDotProd() ? &VecDotQ4KNeon : nullptr;
}

VecDotFn SelectQuantQ4KVecDot(VecDotFn portable) {
  const char* value = std::getenv("VT_CPU_Q4K_DOT");
  if (value == nullptr || std::strcmp(value, "auto") == 0) {
    const VecDotFn neon = QuantQ4KNeonVecDot();
    return neon != nullptr ? neon : portable;
  }
  if (std::strcmp(value, "portable") == 0) {
    return portable;
  }
  if (std::strcmp(value, "neon") == 0) {
    const VecDotFn selected = QuantQ4KNeonVecDot();
    VT_CHECK(selected != nullptr,
             "VT_CPU_Q4K_DOT=neon requires NEON DotProd OS capability");
    return selected;
  }
  VT_CHECK(false, "VT_CPU_Q4K_DOT must be auto, portable, or neon");
  return portable;
}

bool QuantQ4KNeonActive() { return QuantQ4KNeonVecDot() != nullptr; }

}  // namespace vt::cpu

#else

namespace vt::cpu {
VecDotFn QuantQ4KNeonVecDot() { return nullptr; }
VecDotFn SelectQuantQ4KVecDot(VecDotFn portable) { return portable; }
bool QuantQ4KNeonActive() { return false; }
}  // namespace vt::cpu

#endif

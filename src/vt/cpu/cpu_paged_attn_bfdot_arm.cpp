// Arm bf16 (FEAT_BF16, `vbfdotq_f32`) K-dot tier for the CPU paged-attention
// NEON lane — DRAFT row work on top of the FMA-contract draft (kolibri1).
//
// WHAT IT IS. One out-of-line kernel: the Q.K^T dot over bf16 keys, computed
// with native `vbfdotq_f32` — 8 bf16 products per instruction (4 f32 lanes,
// each the exact dot of a bf16 PAIR), accumulated in f32, combined with
// `vaddvq_f32` in the SAME lane-serial-in-vector, block-sequential order the
// widen+vfma NEON arm uses (cpu_paged_attn.cpp:346).
//
// WHAT IT IS NOT. It is NOT a drop-in arm. `vbfdotq_f32` consumes bf16 PAIRS,
// so the query must stay bf16 (no f32 widen), and the pair-dot regrouping
// changes the reduction DAG versus both the FMA-contracted scalar body and
// the widen+vfma NEON arm. The C draft's bit-identity (scalar == NEON) rests
// on that DAG; this arm therefore runs only behind its own knob,
// `VT_CPU_PAGED_ATTN_BFDOT` (default off), as a THIRD mode for measurement.
// The V accumulation is NOT touched: BFMLA/BFDOT have no per-element
// f32-accumulator form that matches the scalar `fma(pw, v_e, acc_e)` DAG
// (pair-dot regroups elements), so V stays on the existing widen+vfma path.
//
// COMPILE/RUNTIME GATING. The same shape as cpu_quant_dot_arm.cpp: this TU is
// compiled with `+bf16` (CMakeLists per-file COMPILE_OPTIONS) so
// `arm_neon.h` exposes `vbfdotq_f32`; the whole body is additionally
// `#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)` so a build
// without the feature (or any non-Arm build) links empty stubs. At RUNTIME
// the kernel is handed out only after `DetectArmBf16()` (the shared
// HWCAP2/sysctl detector in cpu_isa_arm.cpp) proves FEAT_BF16.

#include "vt/cpu/cpu_isa_arm.h"

#include <cstdint>

#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)
// `arm_neon.h` exposes `vbfdotq_f32`; <arm_neon.h> does not exist off-Arm, so
// the include lives inside the guard (the cpu_ops.cpp:24 shape on main).
#include <arm_neon.h>
#endif

namespace vt::cpu {

#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)

bool PagedAttnBfDotUsable() {
  if (DetectArmBf16() == false) return false;
  // Legacy per-arm opt-in keeps working as an explicit `=1` (deprecated).
  const char* legacy = std::getenv("VT_CPU_PAGED_ATTN_BFDOT");
  if (legacy != nullptr && legacy[0] == '1') return true;
  // DEFAULT dispatch (SPEC kolibri1-bf16-dot-cpu): feature-gated, no env
  // gate on the default path. `VT_CPU_KOLIBRI_BF16_DOT=0` is the single
  // escape hatch back to the f32 tier for every kolibri1 bf16-dot arm.
  const char* e = std::getenv("VT_CPU_KOLIBRI_BF16_DOT");
  return !(e != nullptr && e[0] == '0');
}

float PagedAttnKDotBfDot(const uint16_t* q, const uint16_t* k, int64_t d) {
  float32x4_t vacc = vdupq_n_f32(0.0f);
  for (int64_t e = 0; e < d; e += 8) {
    const uint16x8_t qb = vld1q_u16(q + e);
    const uint16x8_t kb = vld1q_u16(k + e);
    vacc = vbfdotq_f32(vacc, vreinterpretq_bf16_u16(qb), vreinterpretq_bf16_u16(kb));
  }
  return vaddvq_f32(vacc);  // same combine order as the widen+vfma arm
}

#else  // no bf16 in the build: link the stubs, never selected at runtime

bool PagedAttnBfDotUsable() { return false; }

float PagedAttnKDotBfDot(const uint16_t*, const uint16_t*, int64_t) { return 0.0f; }

#endif

}  // namespace vt::cpu

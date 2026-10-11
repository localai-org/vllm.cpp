// Kolibri-1 BFDOT bf16 GEMV arm (DRAFT, knob-gated, default OFF).
//
// Draft measurement arm for the t==1 CPU decode GEMV over the WARM-CACHE
// bf16 weights. The production linear path already holds every decode
// weight as bf16 [N, K] row-major (the dequant cache decodes fp8-block
// weights once and rounds to bf16 on store — kolibri1_forward.cpp
// DequantFp8Block / the cache lease), then widens both operands to f32 and
// runs the sequential-f32 MatmulBT. This arm instead feeds the bf16 bytes
// STRAIGHT into `vbfdotq_f32`: each instruction computes four exact
// bf16-pair dots into f32 lanes (8 elements/instruction), f32 accumulated.
//
// NUMERICS: this is a ROUNDING-POLICY change, not a reassociation of the
// same f32 DAG. The multiply rounds to bf16 per operand pair (the operands
// are already bf16 in memory, so the multiply itself is exact) and the
// pair-dot regroups the reduction (4 lane accumulators, sequential over K
// in 8-element blocks) versus the strict per-element f32 sequence of
// MatmulOneChunkRef. Expected envelope: bf16-store floor, ~2 bf16 ULP
// relative on typical magnitudes. The chain WILL move under the knob; this
// draft exists to measure tok/s, not to hold parity. No gate assertion is
// relaxed.
//
// GATING: compiled with per-file `-march=armv8.2-a+bf16` (CMakeLists, same
// shape as cpu_paged_attn_bfdot_arm.cpp) so <arm_neon.h> exposes
// vbfdotq_f32; the whole body is additionally
// `#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)` so a build
// without the feature links portable stubs. At runtime the arm is handed
// out only behind `VT_CPU_KOLIBRI_BFDOT_GEMV=1` (opt-in, default OFF) AND
// DetectArmBf16() (HWCAP2/sysctl FEAT_BF16 probe). The knob deliberately
// does NOT require dotprod (ASIMDDP): FEAT_BF16 is an independent
// extension, and requiring dotprod would refuse on hosts (like the
// measurement Neoverse V2) that have bf16 without needing SDOT at all —
// dotprod-free by construction, refusal names what is missing.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_BFDOT_GEMV_H_
#define VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_BFDOT_GEMV_H_

#include <cstdint>

namespace vllm {
namespace kolibri1_bfdot_gemv {

// The raw `VT_CPU_KOLIBRI_BFDOT_GEMV` env parse (house style: `=1` opts in).
bool KnobRequested();

// Requested AND FEAT_BF16-probed. `=1` without the hardware feature refuses
// BY NAME (VT_CHECK), so the knob can never silently stay off.
bool KnobActive();

// out[1, n] (bf16) = x[1, k] (bf16) @ w[1, n*k] (bf16 [n, k] row-major)^T —
// the t==1 decode GEMV. `w` is the warm-cache bf16 tensor reused DIRECTLY
// (no conversion, no repack). Threaded over output rows through the ONE
// pool (host_parallel.h), so the result is identical to the serial loop by
// the pool determinism contract.
void Gemv(const uint16_t* w_bf16, const uint16_t* x_bf16, uint16_t* out_bf16,
          int64_t n, int64_t k);

// The f32-out form for the two linear sites the t==1 LinearBT hook does not
// reach — the UNTIED lm_head (bf16 [vocab, H] weight, f32 logits) and the
// bf16 router gate (f32 logits). Same kernel, same numerics contract as
// `Gemv`, minus the final bf16 store round: the f32 accumulator IS the f32
// output, exactly as the widen-f32 MatmulBT these sites replace produces f32
// without an intermediate narrower round. The f32 store cannot move the
// envelope relative to `Gemv`'s (it REMOVES one rounding the bf16 form has).
void GemvF32Out(const uint16_t* w_bf16, const uint16_t* x_bf16, float* out_f32,
                int64_t n, int64_t k);

}  // namespace kolibri1_bfdot_gemv
}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_BFDOT_GEMV_H_

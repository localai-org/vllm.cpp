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
// DEPRECATED as a default-shaping knob: kept one release as an explicit `=1`
// opt-in; the default dispatch is `Bf16DotActive` below.
bool KnobActive();

// The DEFAULT dispatch (SPEC kolibri1-bf16-dot-cpu): feature-gated at
// runtime — true when `DetectArmBf16()` proves FEAT_BF16, independent of any
// environment variable. Escape hatch: `VT_CPU_KOLIBRI_BF16_DOT=0` forces the
// f32 tier (returns false) on every kolibri1 bf16-dot arm. The legacy
// `VT_CPU_KOLIBRI_BFDOT_GEMV=1` opt-in still selects the arm explicitly and
// keeps its refuse-by-name contract.
bool Bf16DotActive();

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

// out[t, n] (bf16) = x[t, k] (bf16) @ w[n, k]^T — the MULTI-ROW form. THE
// EXACTNESS CONTRACT (ISSUE-LOCAL-01M5H6AE8666C5SPQ3GGCMTB1M): every output
// element runs the IDENTICAL per-row instruction sequence `Gemv` runs at
// t == 1 — same bf16-pair dots, same 4-lane f32 accumulator, same K order —
// so row i of the result is bit-identical to a t == 1 call on x row i. The
// spec-decode verify step forwards its 1+k query rows in ONE call; a t-gated
// class there would verify every position in the f32-widen class while plain
// greedy decode sampled it in the bf16-dot class, and greedy spec decode
// stops being exactness-preserving on every near-tie. The class must not
// depend on the batch shape.
void GemvRows(const uint16_t* w_bf16, const uint16_t* x_bf16, uint16_t* out_bf16,
              int64_t t, int64_t n, int64_t k);

// The f32-out multi-row form (same contract as `GemvRows` / `GemvF32Out`).
void GemvF32OutRows(const uint16_t* w_bf16, const uint16_t* x_bf16,
                    float* out_f32, int64_t t, int64_t n, int64_t k);

}  // namespace kolibri1_bfdot_gemv
}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_BFDOT_GEMV_H_

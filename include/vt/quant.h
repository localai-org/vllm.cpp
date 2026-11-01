// vt:: block-quant type traits — 1:1 mirror of llama.cpp's `type_traits_cpu[]`
// table, ported from the local fork @ 237ad9b96
// `ggml/src/ggml-cpu/ggml-cpu.c:211-406` (the table itself; Q4_0 :230-239,
// Q8_0 :262-271, Q3_K :295-300, Q4_K :301-310, Q5_K :311-316, Q6_K :317-326).
//
// The table shape is kept identical to upstream's so future llama.cpp diffs
// port mechanically: one row per block dtype carrying `{from_float, vec_dot,
// vec_dot_type, nrows}`. `to_float` is upstream's `ggml_type_traits.to_float`
// (`ggml/src/ggml.c`), folded into this one CPU-side table because vt:: has no
// separate device-neutral trait registry — a recorded deviation, nothing else
// changes.
//
// Population status (G1 skeleton + G2 activation quant + G3 tier-0 vec_dot):
//   to_float      — POPULATED for all six executable weight types (G1).
//   vec_dot       — POPULATED for all six by the portable tier-0 kernels in
//                   cpu_quant_dot.cpp (G3). The x86 AVX2 (G5) and Arm
//                   NEON/i8mm (G6) tiers replace these entries per-ISA later;
//                   the generic tier is always built and is what CI runs.
//   from_float    — POPULATED for the two ACTIVATION encodings Q8_0 and Q8_K
//                   in cpu_quant_act.cpp (G2). It stays nullptr for the
//                   weight-only encodings: nothing in this project quantizes
//                   an activation INTO a k-quant, so upstream's k-quant
//                   encoders are deliberately not ported.
//   vec_dot_type  — POPULATED (the dispatch fact G1 owns).
//   nrows         — POPULATED (1 everywhere on the generic tier; the
//                   `__ARM_FEATURE_MATMUL_INT8` nrows==2 rows arrive with the
//                   i8mm mmla kernels in G6, together with the boundary guards
//                   at ggml-cpu.c:1426-1433).
#pragma once

#include <cstddef>
#include <cstdint>

#include "vt/dtype.h"

namespace vt::cpu {

// Upstream `ggml_from_float_t` (ggml-cpu-traits / ggml-impl.h): quantize `k`
// f32 activations into the block encoding at `y`.
using FromFloatFn = void (*)(const float* x, void* y, int64_t k);

// Upstream `ggml_to_float_t`: decode `k` elements of packed blocks into f32.
using ToFloatFn = void (*)(const void* x, float* y, int64_t k);

// Upstream `ggml_vec_dot_t` (ggml-impl.h): `nrc` row-dots of length `n` between
// weight blocks `x` (row stride `bx` BYTES) and activation blocks `y` (row
// stride `by` BYTES), written to `s` with row stride `bs` BYTES.
using VecDotFn = void (*)(int n, float* s, size_t bs, const void* x, size_t bx,
                          const void* y, size_t by, int nrc);

struct QuantTypeTraits {
  FromFloatFn from_float = nullptr;
  ToFloatFn to_float = nullptr;
  VecDotFn vec_dot = nullptr;
  // The activation encoding this weight type is dotted against: Q8_0 for the
  // legacy 32-element types, Q8_K for the K-quants (ggml-cpu.c:230-326).
  DType vec_dot_type = DType::kF32;
  // Rows the vec_dot kernel consumes per call (2 only on the i8mm mmla tier).
  int nrows = 1;
};

// The `to_float` decoder for a block dtype (nullptr for elementwise dtypes).
// Implemented in cpu_quant_dequant.cpp.
ToFloatFn BlockToFloat(DType dtype);

// The `from_float` activation quantizer for a block dtype — non-null only for
// the two `vec_dot_type` encodings Q8_0 and Q8_K. Implemented in
// cpu_quant_act.cpp (ports of quantize_row_q8_0_ref / quantize_row_q8_K_ref).
FromFloatFn BlockFromFloat(DType dtype);

// The tier-0 generic `vec_dot` for a block WEIGHT dtype (nullptr for Q8_K,
// which is activation-only, and for elementwise dtypes). Implemented in
// cpu_quant_dot.cpp.
VecDotFn BlockVecDot(DType dtype);

// The TRUE portable K-quant `vec_dot` for a dtype, never the runtime-selected
// tier. QUANT-GGUF-CIQ-GEMM G5: on AVX2 x86 `BlockVecDot` returns the wide
// kernel, so a comparison that used it as its portable reference would compare
// the selected kernel against itself (the same trap QuantQ8PortableVecDot
// exists to avoid). Returns nullptr for dtypes outside the three-encoding AVX2
// tier.
VecDotFn QuantKPortableVecDot(DType dtype);

// QUANT-GGUF-CIQ-GEMM G5 — the x86 AVX2 K-quant `vec_dot` tier for Q4_K,
// Q6_K and IQ3_S, implemented in cpu_quant_dot_avx2.cpp. CMake links that
// translation unit only on x86-64, so a non-x86 caller must not reference
// these two symbols; BlockVecDot guards the call at compile time.
//
// `portable` is returned unchanged when the host lacks AVX2 or when
// VT_CPU_QUANT_AVX2=portable. VT_CPU_QUANT_AVX2=avx2 forces the wide tier and
// fails closed when the CPU cannot run it; `auto` is the default.
VecDotFn SelectQuantKVecDot(DType dtype, VecDotFn portable);

// True when the AVX2 K-quant tier would be selected by `auto`: an AVX2 host
// and no VT_CPU_QUANT_AVX2=portable defeat.
bool QuantKAvx2Active();

// Q8_0 x Q8_0 DotProd variants for KERNEL-CPU-A76-Q8-DOT. The explicit
// getters are test/benchmark seams; SelectQuantQ8VecDot applies
// VT_CPU_Q8_DOT=auto|portable|sdot|a76-asm while retaining `portable` as the
// universal fallback. The assembly getter is ISA-safe on any DotProd core;
// QuantQ8A76AsmActive additionally reports whether the running CPU is A76.
// QuantQ8PortableVecDot is the TRUE portable reference (quants.c:400 order),
// never the runtime-selected kernel: on an A76 the QuantTraits vec_dot IS the
// assembly tier, so a byte-equality test that used it as its reference would
// compare the selected kernel against itself.
VecDotFn QuantQ8PortableVecDot();
VecDotFn QuantQ8SdotVecDot();
VecDotFn QuantQ8A76AsmVecDot();
VecDotFn SelectQuantQ8VecDot(VecDotFn portable);
bool QuantQ8SdotActive();
bool QuantQ8A76AsmActive();

// Q4_K x Q8_K NEON DotProd variant for FLINT-NEON-Q4K-DOT
// (cpu_quant_dot_q4k_neon.cpp). Same shape as the Q8_0 tier above:
// `SelectQuantQ4KVecDot` applies `VT_CPU_Q4K_DOT=auto|portable|neon` with the
// portable scalar kernel (quants.c:645 order) as the universal fallback.
// `QuantQ4KPortableVecDot` is the TRUE scalar reference — on a DotProd host
// the QuantTraits vec_dot IS the NEON tier, so a bit-identity test that used
// it as its reference would compare the kernel against itself.
VecDotFn QuantQ4KPortableVecDot();
VecDotFn QuantQ4KNeonVecDot();
VecDotFn SelectQuantQ4KVecDot(VecDotFn portable);
bool QuantQ4KNeonActive();

// The Arm i8mm (mmla) `nrc == 2` `vec_dot` for a block WEIGHT dtype — QUANT-
// GGUF-CIQ-GEMM work row G6 (cpu_quant_dot_arm.cpp). Non-null ONLY when the
// process runs on i8mm-capable aarch64 (compile-time `__ARM_FEATURE_MATMUL_INT8`
// AND runtime `HWCAP2_I8MM`) AND the dtype is one of the four encodings upstream
// gives an mmla path (Q8_0, Q4_0, Q4_K, Q6_K). Returns nullptr everywhere else —
// on any other CPU, when `VT_CPU_QUANT_MMLA=portable`, and for q3_K/q5_K (no upstream
// mmla) — so the caller falls back to the portable nrc==1 tier. A returned
// kernel produces a 2x2 output tile: it MUST be called with nrc==2, two
// consecutive weight rows (stride bx) and two consecutive activation rows
// (stride by), writing s[0]=(w0,a0), s[1]=(w1,a0), s[bs]=(w0,a1), s[bs+1]=(w1,a1).
VecDotFn QuantMmlaVecDot(DType dtype);

// True when the Arm i8mm mmla tier is live in this process. A forced unsupported
// tier fails closed; auto uses the shared Linux HWCAP/Darwin sysctl probe.
bool QuantMmlaActive();

// Bytes one quantized ACTIVATION row occupies for a given weight dtype, i.e.
// `ggml_row_size(vec_dot_type(weight_dtype), k)`. Throws when `k` is not a
// whole number of activation blocks (256 for the K-quants, 32 otherwise) —
// the fail-loud guard against a ragged-K GEMM silently mis-striding scratch.
size_t QuantActRowBytes(DType weight_dtype, int64_t k);

// Total scratch for quantizing `rows` activation rows of length `k` ahead of a
// `kMatmulBTQuant` against `weight_dtype`. Mirrors the mul_mat `wdata` sizing
// in `ggml_graph_plan` (ggml-cpu.c:2752-2980): contiguous rows, no padding.
size_t QuantActScratchBytes(DType weight_dtype, int64_t rows, int64_t k);

// Traits for a block-quantized dtype; throws for any other dtype and for a
// block dtype this table does not (yet) carry.
const QuantTypeTraits& QuantTraits(DType dtype);

// True when `dtype` has a traits row AND that row can currently execute a
// quantized dot: it needs BOTH its own `vec_dot` and a `from_float` on the
// activation encoding its `vec_dot_type` names. True for the six executable
// weight types since G2+G3; false for Q8_K (activation-only, no vec_dot),
// which is what keeps it on the generic dequant-composite fallback.
bool HasQuantDotKernel(DType dtype);

// --- CIQ G7: repack-at-load for the q8_0 quant GEMM -----------------------
//
// llama.cpp repacks a quantized WEIGHT once at load into a SIMD/cache-friendly
// interleave so the GEMM inner loop reads contiguous, pre-shuffled blocks with
// no in-register row shuffles (repack.cpp @ 237ad9b96). On GB10 (aarch64 NEON +
// i8mm) `ggml_repack_get_optimal_repack_type` selects `q8_0_4x8_q8_0`
// (block_q8_0x4, nrows_interleaved=4, interleave_block=8) for a q8_0 weight
// with ne[1] % 4 == 0 (repack.cpp:4683-4695). We mirror exactly that tier.
//
// The transform is a pure BYTE PERMUTATION (the quant values and fp16 deltas
// are untouched) and the gemm/gemv fold the per-block scale in the SAME order
// as the tier-0 / mmla path with a NON-FUSED multiply-add, so the repacked GEMM
// is BIT-IDENTICAL to `kMatmulBTQuant`'s non-repacked output (proven by the
// memcmp round-trip test). Only q8_0 is repacked in G7 (the profile's
// kMatmulBTQuant is q8_0); the k-quants keep the mmla tier.

// True when the i8mm repack tier is LIVE in this process: compiled for aarch64
// with i8mm, the exact HWCAP/sysctl bits probed present, and not forced to
// `VT_CPU_QUANT_REPACK=portable`. Always false off i8mm-capable aarch64, so
// the loader never repacks and every consumer keeps the portable/mmla path.
bool QuantRepackActive();

// True when a weight of (dtype, N=out, K=in) is repack-eligible on THIS process:
// QuantRepackActive() AND dtype == kQ8_0 AND N % 4 == 0 AND K % 32 == 0 (the
// exact `ne[1] % 4 == 0`, `ne[0] % 8 == 0` guard of repack_q8_0_to_q8_0_4_bl;
// K % 32 subsumes the K % 8 one). A weight that fails it stays plain and takes
// the normal path — correct, just unrepacked.
bool QuantRepackEligible(DType weight_dtype, int64_t n, int64_t k);

// --- ELEMENTWISE repack-at-load (KERNEL-GEMM-CPU-TILED lever 2) -------------
// The non-quant sibling of the block repack above, declared here for the same
// reason: the loader needs it and `src/vt/cpu/cpu_matmul_elem.h` is private.
//
// Transposes an ELEMENTWISE (f32/f16/bf16) [N,K] matmul weight into [K,N] so
// `vt::MatmulBT` reaches the transpose-free `nk`/`nkm` micro-kernels, measured
// 1.16x to 1.30x on dgx AT THE SHAPES THAT ROW MEASURED, and BYTE-IDENTICAL
// (both orientations accumulate each output over K in strict increasing order).
// Pure permutation: same bytes, same count, so no product and no sum can change.
//
// THE SPEEDUP IS SHAPE- AND ISA-DEPENDENT AND CAN INVERT. IT IS NOT A GENERAL
// WIN, and that is why it stays opt-in. Row LTX25-CONNECTOR-GEMM measured this
// lever at the LTX-2.5 text connector's f32 shapes (M=1024, N/K in
// {2048, 4096, 8192, 16384}) and found it SLOWER everywhere:
//
//   dgx / GB10 Cortex-X925  1.22x - 2.70x slower, 1.806x over the whole set
//   Thor aarch64            1.389x over the whole set
//   x86-64 AVX-512          1.78x - 2.06x over three load regimes
//
// -- byte-identical on every shape, with same-arm controls at 0.90x to 1.04x, so
// the regression is not noise and not a numerical difference. The plausible
// mechanism is that the [N,K] path's per-group register transpose costs less
// than the [K,N] path's K-strided weight walk once N*4 exceeds a page, which is
// exactly the trade that flips between shapes and between ISAs. So the 1.16x to
// 1.30x above is a measurement of ITS shapes, not a property of the transform,
// and a caller enabling VT_CPU_ELEM_KN_REPACK for a new weight set must measure
// that weight set. Evidence:
// /mnt/nas_share/rc/ltx25-connector-gemm/run/rc-worker-4b8lj-20260831T023318Z.
//
// The caller must set `Tensor.elem_kn_repacked` on the resulting weight. Only
// the CPU `MatmulBTKernel` honours that flag, so a repacked buffer handed to
// any other consumer would be read as [N,K] and be garbage; the loader keeps
// this opt-in (VT_CPU_ELEM_KN_REPACK=1) for exactly that reason.
bool ElemRepackEligible(DType weight_dtype, int64_t n, int64_t k);
void ElemRepackWeight(DType weight_dtype, uint8_t* bytes, int64_t n, int64_t k);

// Repack a [N,K] q8_0 weight buffer IN PLACE into the block_q8_0x4 interleave.
// `blocks` holds N*(K/32) plain BlockQ8_0 on entry and N/4 groups of (K/32)
// BlockQ8_0x4 on return (same total bytes). Requires QuantRepackEligible.
void QuantRepackWeight(DType weight_dtype, uint8_t* blocks, int64_t n,
                       int64_t k);

// Brick 4 (DeepSeek-V4 last-mile): CUDA coalesced-load repack. Eligible = Q8_0 +
// K a whole number of 32-blocks (any N). RepackQ8_0Cuda rewrites the [N,K] Q8_0
// buffer IN PLACE from block-interleaved `[d,qs]×nblk` into two contiguous
// sections `[all qs (32B/block) | all scales (2B/block)]` (same total bytes) so
// the CUDA Q8_0 GEMM reads qs via aligned int4 loads. Byte permutation only ->
// BIT-IDENTICAL integer dot. Host code (always compiled); the CONSUMER kernel is
// CUDA-only. See cpu_quant_repack.cpp.
bool RepackQ8_0CudaEligible(DType weight_dtype, int64_t n, int64_t k);
void RepackQ8_0Cuda(uint8_t* blocks, int64_t n, int64_t k);

}  // namespace vt::cpu

namespace vt {
struct Tensor;
namespace cpu {
// The repacked-weight GEMM dispatched by `kMatmulBTQuant` when `b.repacked`.
// out[M,N] = a[M,K] @ b[N,K]^T with `b` a q8_0 weight already repacked by
// QuantRepackWeight. Bit-identical to the non-repacked quant path: the i8mm
// gemm runs 4-row activation groups, the i8mm gemv the M=1 / leftover rows.
void QuantRepackMatmul(vt::Tensor& out, const vt::Tensor& a, const vt::Tensor& b);
}  // namespace cpu
}  // namespace vt

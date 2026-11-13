// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#define VT_CHECK(cond, msg)                                                       \
  do {                                                                            \
    if (!(cond)) {                                                                \
      throw std::runtime_error(std::string("vt: ") + (msg) + " at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));                   \
    }                                                                             \
  } while (0)

namespace vt {

// Storage dtypes. The first six are ELEMENTWISE (one scalar per element, a
// well-defined SizeOf); the trailing entries are BLOCK-QUANTIZED ggml
// encodings, where a fixed group of `BlockElems` scalars shares one packed
// `BlockBytes` record and there is NO per-element size.
//
// Block dtypes live in the enum (rather than a parallel `QuantTensor`) to
// mirror ggml's "the type carries the layout" model: a `Tensor` stays
// self-describing and quant dispatch keys on `b.dtype` instead of forking
// every op signature (spec `.agents/specs/gguf-compute-in-quant-gemm.md`
// § Risks/decisions). The consequence is enforced here: block dtypes are
// STORAGE-ONLY — `SizeOf` on one is a `VT_CHECK` failure, so any elementwise
// kernel that reaches one fails loudly instead of reading garbage.
//
// Ids/geometry mirror llama.cpp @ 237ad9b96:
//   ggml/include/ggml.h:390-432 (enum ggml_type)
//   ggml/src/ggml-common.h:288-299 (block_q2_K), :242-245 (block_q8_0),
//     :305-310 (block_q3_K), :317-327 (block_q4_K), :334-345 (block_q5_K),
//     :352-357 (block_q6_K), :361-365 (block_q8_K), :371-374 (block_iq2_xxs),
//     :385-400 (block_iq3_xxs)
// kQ8_K is ACTIVATION-ONLY as a DOT operand: it is the `vec_dot_type` of the
// K-quants and has no weight-side `vec_dot`, so no keep-quant DOT serves it.
// As a WEIGHT encoding it is keep-quant capable since TT-KEEPQUANT-Q8K-WEIGHTS
// (2026-11-13): a GGUF file can carry it (the unsloth/Qwen3.8-27B-GGUF
// UD-Q8_K_XL checkpoint does), the loader keeps its blocks COMPRESSED, and it
// serves through the DEQUANT-COMPOSITE arm — the tree-standard for Q8_K
// weights (cpu_quant_dot.cpp's selector returns nullptr by design; the CPU
// oracle path is cpu_quant_gemm.cpp's to_float fallthrough, staged at 80
// padded words per 292-B block_q8_K on the TT device path). The
// decode-only embedding gather also takes it packed, as before.
//
// kQ2_K, kIQ2_XXS and kIQ3_XXS are the ~2-3-bit storage encodings the `unsloth/
// DeepSeek-V4-Flash-GGUF UD-IQ2_XXS/UD-Q2_K_XL` checkpoints use (the real
// UD-IQ2_XXS routed experts are IQ2_XXS gate/up + IQ3_XXS down; Q2_K is the
// UD-Q2_K_XL sibling vehicle). As of DeepSeek-V4 W8 (CLAIM-DEEPSEEK-V4-W8) all
// three carry a keep-quant `vec_dot` against the Q8_K activation encoding
// (cpu_quant_dot.cpp), so `HasQuantDotKernel` is TRUE and the GGUF loader keeps
// their blocks COMPRESSED and dots them directly — the memory enabler that lets
// a 158 B DeepSeek-V4 stay ~91 GiB instead of OOM-expanding to bf16. See
// `.agents/specs/gguf-iquant-dsv4.md` and `.agents/specs/deepseek-v4-flash.md`.
//
// kIQ2_S (2.5625 bpw, Q8_K-activation) and kMXFP4 (OCP micro-scaling fp4, 32-elem
// blocks, Q8_0-activation) are the extra per-tensor "dynamic" encodings the
// `unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_M` checkpoint mixes into a handful of
// routed-expert slabs (IQ2_S ffn_gate/up + MXFP4 ffn_down). Both carry a
// keep-quant `vec_dot` (IQ2_S vs Q8_K, MXFP4 vs Q8_0) so they load COMPRESSED
// on the same memory-safe path — expanding them to bf16 would OOM the box.
//
// kQ5_0 (ggml id 6) and kIQ4_NL (ggml id 20) are the two remaining 32-ELEMENT
// legacy-family encodings, added for Qwen3.8-Flash-Next (`qwen4exp`). They exist
// because that model's `moe_intermediate_size` is 640 and its per-layer table
// row is 160: neither is a multiple of 256, so NO K-quant can encode them and
// llama.cpp's own `tensor_type_fallback` drops IQ4_XS -> IQ4_NL and Q4_K -> Q5_0.
// The shipped `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` carries 49 IQ4_NL
// tensors: all 48 `ffn_down_exps` and the 20M-entry n-gram table
// `per_layer_token_embd.weight`. Both dot against Q8_0, like every other
// 32-element block. IQ4_NL is Q4_0's shape with a NON-LINEAR 16-entry codebook
// (`kValuesIq4nl`) in place of the `nibble - 8` affine step.
// kIQ2_XS (ggml id 17, 2.3125 bpw) and kIQ4_XS (ggml id 23, 4.25 bpw) are the
// last two encodings the staged `unsloth/GLM-5.3-Flash-GGUF UD-Q2_K_XL` arm
// needs. "UD-Q2_K_XL" names a target average, not a format: of that artifact's
// 1412 tensors only TWO are Q2_K, while 82 are IQ2_XS and 3 are IQ4_XS, so the
// loader stopped on `blk.3.ffn_gate_exps.weight` (#2240).
//
// kIQ3_S (ggml id 21, 3.4375 bpw) is the one hole that was left in the reader's
// i-quant run, and it locked out the `unsloth` "UD" dynamic-quant family:
// `Qwen3.8-27B-UD-Q4_K_M.gguf` stores 4 of its 866 tensors in it and
// `UD-Q4_K_XL` stores 1 of 866, while every OTHER encoding in those files was
// already handled, so `GgufFile::Open` refused 16.4 GB of loadable weights over
// 146 MiB of them (#2510). It is NOT a variant of kIQ3_XXS: it reads the
// 512-entry `iq3s_grid` against IQ3_XXS's 256-entry `iq3xxs_grid` (both u32,
// both read four bytes at a time, so the wrong one still decodes), it splices
// the ninth index bit out of `qh`, and it carries DIRECT sign bytes like IQ2_S
// rather than IQ3_XXS's packed `ksigns_iq2xs` selector. It is the one FILE
// encoding in this tree that DECODES and does not DOT: `HasQuantDotKernel` is
// FALSE, so the loader expands it to bf16 on the GEMM arm and keeps it
// compressed only on the gather. The `vec_dot` is owed together with the CUDA
// `WType` arm, because landing the CPU half alone would send every CUDA IQ3_S
// GEMM down `IsCudaKeepQuantSupported`'s host fallback. See
// `.agents/specs/gguf-iq3s.md`.
//
// IQ2_XS is the middle member of the IQ2 family and shares neither table nor
// sign convention with its siblings: its 9-bit index addresses a 512-entry
// `iq2xs_grid` (against 256 for IQ2_XXS and 1024 for IQ2_S) and the 7-bit
// `ksigns_iq2xs` selector lives in the SAME u16 as that index. IQ4_XS is NOT a
// codebook delta from IQ4_NL — it reuses `kvalues_iq4nl` byte for byte — and
// differs only in the SUPER-BLOCK SCALE LAYOUT: 256-element blocks whose 6-bit
// per-32 scale is spliced from a `scales_l` nibble and a `scales_h` bit pair and
// then biased by -32, where IQ4_NL carries one unbiased f16 delta per 32.
//
// Both carry a keep-quant `vec_dot` against Q8_K as of QUANT-GGUF-IQ-VECDOT
// (#2247), so `HasQuantDotKernel` is TRUE and the loader keeps their blocks
// COMPRESSED. IQ4_XS pairs with Q8_K and NOT with the Q8_0 of its codebook
// sibling IQ4_NL, because its block is a 256-element super-block
// (ggml-cpu.c:385-390 against :379-384). Between #2245 and #2247 they were
// decode-only, which cost 325.58 GiB of residency on the staged artifact — 82
// IQ2_XS tensors expanding from 53.33 GiB to 369.00 GiB and 3 IQ4_XS tensors
// from 3.59 GiB to 13.50 GiB — and was the difference between fitting the
// ~119.63 GiB of `dgx:gpu0` and overflowing it 3.6x.
enum class DType : uint8_t {
  kF32,
  kF16,
  kBF16,
  kI8,
  kI32,
  kI64,
  // --- block-quantized (storage-only) ---
  kQ4_0,
  kQ5_0,
  kQ8_0,
  kQ2_K,
  kQ3_K,
  kQ4_K,
  kQ5_K,
  kQ6_K,
  kQ8_K,
  kIQ2_XXS,
  kIQ3_XXS,
  kIQ2_S,
  kIQ1_S,
  kIQ1_M,
  kIQ1_XXXS,
  kIQ4_NL,
  kMXFP4,
  kIQ2_XS,
  kIQ4_XS,
  kIQ3_S,
  // --- ternary keep-quant (Vulkan-native, Q8_K-activation) ---
  // TQ2_0 (llama.cpp PR #25850): 2-bit ternary codes, 256-elem blocks.
  // TQ1_0: packed base-3 trits, 256-elem blocks. Both are weight-only encodings
  // dotted against Q8_K activations; the Vulkan backend has native keep-quant
  // shaders for both, and the CPU dequant-composite path serves as the
  // reference oracle. No CPU vec_dot (the keep-quant dot is Vulkan-only).
  kTQ2_0,
  kTQ1_0,
};

const char* Name(DType dtype);

// The two refusals `SizeOf` below can produce. They live OUT OF LINE, and cold,
// so the inline `SizeOf` carries a switch over six integers and nothing else —
// no `std::string`, no `Name`, no throw machinery in the hot path.
[[noreturn]] void ThrowBlockQuantHasNoElementSize(DType dtype);
[[noreturn]] void ThrowUnknownDType();

// Bytes per ELEMENT. Throws for block-quantized dtypes (they have no
// per-element size) — see IsBlockQuant/BlockBytes/RowSizeBytes.
//
// INLINE, and that is load-bearing rather than cosmetic (row VT-CPU-ELEM-DISPATCH,
// .agents/specs/vt-cpu-elem-dispatch.md). Every per-element `LoadF32`/`StoreF32`
// helper in `src/vt/cpu` computes its byte offset with this call, inside loops
// whose body is one multiply and one add. The build enables no LTO — CMakeLists.txt
// sets no INTERPROCEDURAL_OPTIMIZATION and passes no -flto — so while this was
// defined in `src/vt/dtype.cpp` it was a cross-translation-unit call that no
// optimizer could remove, and a `perf` profile of `vt::AttentionCross` put
// `LoadF32` at 36.14% and `vt::SizeOf` at 28.41% of the kernel's own CPU time
// against 15.60% for its arithmetic. Inline, the switch is loop-invariant code
// the caller's optimizer hoists on its own.
//
// It changes no value this function returns, so it is bit-exact by construction.
// The enumeration below stays EXHAUSTIVE with no `default:` label, so adding a
// dtype to the enum is still a -Wswitch error here rather than a silent 0.
inline size_t SizeOf(DType dtype) {
  switch (dtype) {
    case DType::kF32: return 4;
    case DType::kF16: return 2;
    case DType::kBF16: return 2;
    case DType::kI8: return 1;
    case DType::kI32: return 4;
    case DType::kI64: return 8;
    // Block-quantized dtypes are storage-only: there is no per-element size,
    // so every elementwise path that reaches one fails loudly here rather than
    // silently mis-striding a packed block buffer.
    case DType::kQ4_0:
    case DType::kQ5_0:
    case DType::kQ8_0:
    case DType::kQ2_K:
    case DType::kQ3_K:
    case DType::kQ4_K:
    case DType::kQ5_K:
    case DType::kQ6_K:
    case DType::kQ8_K:
    case DType::kIQ2_XXS:
    case DType::kIQ3_XXS:
    case DType::kIQ2_S:
    case DType::kIQ1_S:
    case DType::kIQ1_XXXS:
    case DType::kIQ1_M:
    case DType::kIQ4_NL:
    case DType::kMXFP4:
    case DType::kIQ2_XS:
    case DType::kIQ4_XS:
    case DType::kIQ3_S:
    case DType::kTQ2_0:
    case DType::kTQ1_0:
      ThrowBlockQuantHasNoElementSize(dtype);
  }
  ThrowUnknownDType();
}

// True for the ggml block-quantized encodings above.
bool IsBlockQuant(DType dtype);

// Block geometry for a block-quantized dtype (throws for elementwise dtypes).
// `BlockElems` = ggml's blck_size, `BlockBytes` = ggml's type_size.
int64_t BlockElems(DType dtype);
int64_t BlockBytes(DType dtype);

// ggml_row_size (ggml/src/ggml.c): bytes occupied by `k` contiguous elements.
// `k` must be a whole number of blocks — rows are whole blocks, which is also
// the keep-quant eligibility rule for a GEMM weight (K % BlockElems == 0).
// Defined for elementwise dtypes too (k * SizeOf) so callers stay uniform.
size_t RowSizeBytes(DType dtype, int64_t k);

// The ggml type id (ggml.h:390-432) a block dtype corresponds to, so callers
// can cross-check against the GGUF reader's independent `GgmlTraits` table.
uint32_t GgmlTypeId(DType dtype);

// Inverse of GgmlTypeId for the block dtypes we execute; returns false when
// the id is not one of them (F32/F16/BF16 and every unported encoding).
bool BlockDTypeFromGgmlTypeId(uint32_t ggml_type, DType* out);

// The four reduced-width converters. `F16ToF32` and `BF16ToF32` are INLINE for
// the same reason `SizeOf` above is (row VT-CPU-ELEM-DISPATCH): they are called
// once per ELEMENT by every `LoadF32` in `src/vt/cpu`, the build enables no LTO,
// and out of line they were a cross-translation-unit call per element on the
// f16 and bf16 arms of every CPU kernel. The bodies are byte-for-byte the ones
// that were in `src/vt/dtype.cpp`, so every value they return is unchanged.
//
// The two f32 -> narrow directions stay OUT of line. They are STORE-side, called
// once per output element rather than once per operand element, and they carry
// the round-to-nearest-even logic that is the part of this file most worth
// keeping in one compiled place.
namespace detail {
inline float BitsToF32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
}  // namespace detail

inline float F16ToF32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0x1F) {  // inf/nan
    return detail::BitsToF32(sign | 0x7F800000 | (mant << 13));
  }
  if (exp == 0) {
    if (mant == 0) return detail::BitsToF32(sign);  // signed zero
    // subnormal: normalize
    int shift = 0;
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      ++shift;
    }
    mant &= 0x3FF;
    return detail::BitsToF32(sign | ((113 - shift) << 23) | (mant << 13));
  }
  return detail::BitsToF32(sign | ((exp + 112) << 23) | (mant << 13));
}

inline float BF16ToF32(uint16_t b) { return detail::BitsToF32(static_cast<uint32_t>(b) << 16); }

uint16_t F32ToF16(float f);
uint16_t F32ToBF16(float f);

}  // namespace vt

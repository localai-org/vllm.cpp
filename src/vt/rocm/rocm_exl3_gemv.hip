// EXL3 small-m GEMV — ROCm arm. Row BACKEND-ROCM, spec
// .agents/specs/rocm-exl3-gemv.md, issue
// .agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M3QJSFATGWXMMQKTVDG6BG5G.md.
//
// THE DONOR IS UPSTREAM'S OWN KERNEL, NOT THE CPU REFERENCE. rocm_exl3.hip
// documents why the m<=8 GEMM could not be a CUDA transcription; this file is
// the OTHER arm of the same op, where a CUDA-shaped port DOES exist on this
// hardware, proven in exllamav3-rocm @ 4a19d7a (which serves ~11 tok/s on
// gfx110x). Source, exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT):
//
//   exllamav3_ext/quant/exl3_gemv_kernel.cuh  the kernel (geometry, decode,
//                                              prefetch ring, MMA, reduction)
//   exllamav3_ext/quant/exl3_gemv.cu          the try-launch + shape heuristic
//   exllamav3_ext/ptx_hip.cuh + hip_compat_hip.cuh  exllamav3-rocm's MMA and
//                                              intrinsic shims (same tree,
//                                              f16-accumulate emu and WMMA)
//
// THREE ADAPTATIONS FROM THE UPSTREAM KERNEL, each named in the spec:
//
//   1. NO COOPERATIVE LAUNCH. Upstream wraps an input-Hadamard stage and an
//      output-Hadamard stage around the GEMM body inside ONE kernel and orders
//      them with grid.sync(), which is why its launch is cooperative and its
//      grid is co-residency capped. rocm_exl3.hip's Exl3GemmKernelRocm already
//      runs those transforms as separate LaunchHad launches on one stream —
//      kernel boundaries ARE the grid.sync. The kernel below is therefore the
//      GEMM body only, launched normally between the same two Had calls, and
//      `locks`/`suh`/`A_had`/`svh` are not kernel arguments. The strided
//      `group` loop is kept, so the upstream occupancy-derived grid survives
//      verbatim.
//   2. ACCUMULATE PATH SELECTABLE AT LAUNCH (WMMA template flag). The emulated
//      `mma_m16n8k16_f16` (shuffle-gather + packed __hfma2) is the DEFAULT — it
//      reproduces the CUDA arm's fp16-fragment accumulation order exactly, so
//      the CUDA arm's 6.0e-3 relative-RMS bound (tests/vt/test_exl3_gemv.cpp,
//      spec quant-exl3-perf tier 3c) transfers to this arm unchanged. The gfx11
//      WMMA path (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`) is instantiated
//      for the 4 bpw arms and selected by VT_ROCM_EXL3_WMMA=1; exllamav3-rocm
//      carries it behind its own EXL3_WMMA flag with a recorded NaN issue on
//      m>=3 paths, so it is evaluation-only until its own gate clears it.
//   3. STREAMING LOADS. `__ldcs` is `__builtin_nontemporal_load`, the
//      exllamav3-rocm shim: on AMD it emits .slc evict-first loads, which is
//      what a trellis you touch exactly once wants.
//
// THE REST IS VERBATIM: the (WK, WNT, PF, FOLD) config pair, the k-split
// across warps, the register prefetch ring, the per-lane extraction constants
// for bits 3, the two-word funnel windows for bits 4, the FragB decode order
// (w0..w7 as upstream decodes them), the ch -> acc0 fold cadence, and the
// sh_red cross-warp reduction. The epilogue writes f32 into the same `raw`
// scratch Exl3GemmK writes, so LaunchHad post-scale sees an identical operand
// on both arms.
#include <algorithm>
#include <hip/hip_runtime.h>
#include "rocm_env.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"
#include "vt/rocm/rocm_f16_codec.h"
#include "vt/rocm/rocm_exl3_decode.h"  // Exl3DecodePairCb2Dp4a
#include "vt/rocm/rocm_exl3_fold.h"

namespace vt::rocm {
namespace {

constexpr int kGemvMaxM = 8;  // EXL3_GEMV_MAX_M (exl3_gemv_kernel.cuh:31)

// 1/sqrt(128), the same literal rocm_exl3.hip's kInvSqrt128 spells (transcribed;
// recomputing 1/sqrt(128) can differ in the last f32 bit). Prefixed kGemv*
// because both TUs sit in vt::rocm's anonymous namespace scope when a bench
// translation unit includes them together.
constexpr float kGemvInvSqrt128 = 0.088388347648f;

inline void CheckG(hipError_t err, const char* what) {
  if (err != hipSuccess)
    throw std::runtime_error(std::string("vt rocm exl3 gemv: ") + what + ": " +
                             hipGetErrorString(err));
}

// ─── fragment types (ptx.cuh:16, util.cuh:83-90) ─────────────────────────────

template <typename T, int n>
struct GVec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
  __device__ const T& operator[](int i) const { return elems[i]; }
};
using FragA = GVec<half2, 4>;
using FragB = GVec<half2, 2>;
using FragC_h = GVec<half2, 2>;  // fp16-accumulate fragment — the GEMV arm
using FragC = GVec<float, 4>;    // f32 fragment — the WMMA converter needs it

union half2_uint32 {
  uint32_t as_uint32;
  half2 as_half2;
  __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
};
union half_uint16 {
  uint16_t as_uint16;
  half as_half;
  __device__ half_uint16(uint16_t val) : as_uint16(val) {}
};

// ─── the codebook decode (codebook.cuh:25-116) ───────────────────────────────
//
// decode_3inst_2 spelled as the donor's HIP file spells it — the lop3 LUT 0x6a
// `(a & b) ^ c` as plain ops, which is bit-identical (integer arithmetic) to
// cuda_exl3.cu's asm.


template <int cb>
__device__ __forceinline__ half2 decode_mcg_or_3inst_2(uint32_t x0, uint32_t x1) {
  if constexpr (cb == 0) {
    x0 *= 89226354u;
    x0 += 64248484u;
    x1 *= 89226354u;
    x1 += 64248484u;
  } else {
    x0 *= 0xCBAC1FEDu;
    x1 *= 0xCBAC1FEDu;
  }
  x0 = 0x3b603b60u ^ (x0 & 0x8fff8fffu);  // lop3 LUT 0x6a: c ^ (a & b)
  x1 = 0x3b603b60u ^ (x1 & 0x8fff8fffu);
  half2_uint32 xu0(x0);
  half2_uint32 xu1(x1);
  half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
  half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
  return __hadd2(d0, d1);
}


// ─── tail-biting window (exl3_dq.cuh) ────────────────────────────────────────

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift) {
  // __funnelshift_r lowers to v_alignbit_b32 (one VALU op) on gfx11; the
  // (a<<32|b)>>s idiom lowers to v_lshrrev_b64, a multi-cycle 64-bit shift.
  // Same bits either way — for shift < 32. For shift >= 32 v_alignbit wraps
  // mod 32 while the 64-bit shift yields a >> (s-32) =
  // funnelshift_r(a, 0, s & 31). Selecting the operands instead of the
  // result keeps a per-lane shift branchless (two v_cndmask + one
  // v_alignbit); a ternary on the result compiles to a divergent branch.
  // Constant shifts fold to the single v_alignbit.
  const bool hi = shift >= 32;
  return __funnelshift_r(hi ? a : b, hi ? 0u : a, shift & 31);
}

// Raw buffer resource over a tensor for the m=1 arms. A buffer load takes the
// wave-uniform byte offset in an SGPR (soffset) and the per-lane offset in a
// loop-invariant VGPR, so the stream issues no 64-bit VALU address adds.
// Word 3 is the gfx11 raw-buffer descriptor (DST_SEL_XYZW, FORMAT
// 32_FLOAT; CK's CK_BUFFER_RESOURCE_3RD_DWORD for gfx11). num_records is an
// int: the launcher keeps tensors >= 2 GiB off these arms.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t M1BufferRsrc(const void* p, size_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, static_cast<int>(bytes),
                                           0x31004000);
}

// Codewords are the LOW 16 bits of w0..w7; the extractors leave window spill
// in the high halves. The mul1 pair reads only the low half; the other
// codebooks multiply the whole word, so they mask here.
template <int cb>
__device__ __forceinline__ void decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
                                        uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7,
                                        FragB& f0, FragB& f1) {
  if constexpr (cb == 2) {
    f0[0] = Exl3DecodePairCb2Dp4a(w0, w1);
    f0[1] = Exl3DecodePairCb2Dp4a(w2, w3);
    f1[0] = Exl3DecodePairCb2Dp4a(w4, w5);
    f1[1] = Exl3DecodePairCb2Dp4a(w6, w7);
  } else {
    f0[0] = decode_mcg_or_3inst_2<cb>(w0 & 0xffffu, w1 & 0xffffu);
    f0[1] = decode_mcg_or_3inst_2<cb>(w2 & 0xffffu, w3 & 0xffffu);
    f1[0] = decode_mcg_or_3inst_2<cb>(w4 & 0xffffu, w5 & 0xffffu);
    f1[1] = decode_mcg_or_3inst_2<cb>(w6 & 0xffffu, w7 & 0xffffu);
  }
}

// Window extraction from two already-loaded words, same order as
// dq8_aligned_4bits (exl3_dq.cuh:164-180). VT_EXL3_FSHF/BFE16 spell it in
// cuda_exl3.cu:1316-1358; shifts and masks are the same arithmetic.
template <int cb>
__device__ __forceinline__ void dq8_regs_4bits(uint32_t a, uint32_t b, FragB& f0, FragB& f1) {
  uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
  s = fshift(b, a, 20);  // FSHF_IMM(s, b, a, 20) — funnelshift_r(b, a, 20), args named (lo,hi)
  w7 = b;
  w6 = (b >> 4);
  w5 = (b >> 8);
  w4 = (b >> 12);
  w3 = (b >> 16);
  w2 = s;
  w1 = (s >> 4);
  w0 = (s >> 8);
  decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// Register form of dq8_aligned_2bits (exl3_dq.cuh:186-203).
template <int cb>
__device__ __forceinline__ void dq8_regs_2bits(uint32_t a, uint32_t b, int t_offset,
                                               FragB& f0, FragB& f1) {
  uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
  b = fshift(b, a, ((~t_offset) & 8) << 1);
  w7 = b;
  w6 = (b >> 2);
  w5 = (b >> 4);
  w4 = (b >> 6);
  w3 = (b >> 8);
  w2 = (b >> 10);
  w1 = (b >> 12);
  w0 = (b >> 14);
  decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// Register form of dq8<3, cb, 4> (exl3_dq.cuh) with the per-lane funnel
// alignment precomputed by the caller.
template <int cb>
__device__ __forceinline__ void dq8_regs_3bits(uint32_t a, uint32_t b, int s2,
                                               FragB& f0, FragB& f1) {
  uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
  // s2 is in [0, 31] by construction ((i2+1)*32 - b2 with i2 = (b2-1)/32),
  // so the first window is one v_alignbit. The second is (a:b) >> (s2+12),
  // whose low word is the 64-bit value (a >> s2 : w7) shifted by 12 — the
  // same bits as the 64-bit shift without a per-lane >= 32 select.
  w7 = __funnelshift_r(b, a, s2);
  w6 = w7 >> 3;
  w5 = w6 >> 3;
  w4 = w5 >> 3;
  w3 = __funnelshift_r(w7, a >> s2, 12);
  w2 = w3 >> 3;
  w1 = w2 >> 3;
  w0 = w1 >> 3;
  decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// ─── the emulated fp16-accumulate MMA (hip_compat_hip.cuh:424-468, 487-522) ──
//
// mma.sync.aligned.m16n8k16.row.col.f16 has no AMD form; RDNA3.5's WMMA is
// 16x16x16 and its register layout is different. The emulation gathers each
// lane's two A rows and two B columns with in-warp shuffles (32 lanes, wave32
// — gfx10/11 default), then runs the 16 FMAs per output element as packed
// __hfma2. Accumulating each output element in a half2 (even/odd k in the two
// lanes) and folding once is the SAME operation order CUDA's
// mma.m16n8k16.f16.f16.f16.f16 performs internally, so the 6.0e-3 bound the
// CUDA arm's gate carries is this arm's bound too.

__device__ __forceinline__ half2 shfl_h2(half2 v, int src) {
  uint32_t u = __builtin_bit_cast(uint32_t, v);
  u = __shfl_sync(0xffffffffffffffffULL, u, src);
  return __builtin_bit_cast(half2, u);
}

__device__ __forceinline__ void mma_m16n8k16_gather(const FragA& frag_a, const FragB& frag_b,
                                                    half2 (&a_rows)[2][8],
                                                    half2 (&b_cols)[2][8]) {
  int lane = threadIdx.x & 31;
  int g = lane >> 2, t = lane & 3;

  // A row m lives in lanes 4*(m%8)..4*(m%8)+3. For row g: k pairs 0..7 come from
  // frag_a[0], k pairs 8..15 from frag_a[2]; for row g+8: frag_a[1] / frag_a[3].
  // Lane 4g+i holds k = 2i, 2i+1.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a_rows[0][i] = shfl_h2(frag_a[0], 4 * g + i);
    a_rows[0][4 + i] = shfl_h2(frag_a[2], 4 * g + i);
    a_rows[1][i] = shfl_h2(frag_a[1], 4 * g + i);
    a_rows[1][4 + i] = shfl_h2(frag_a[3], 4 * g + i);
  }
  // B column n lives in lanes 4n..4n+3: lane 4n+t holds k = 2t, 2t+1 in b[0]
  // and k = 2t+8, 2t+9 in b[1]. Columns 2t and 2t+1 are at lane bases 8t, 8t+4.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    b_cols[0][i] = shfl_h2(frag_b[0], 8 * t + i);
    b_cols[0][4 + i] = shfl_h2(frag_b[1], 8 * t + i);
    b_cols[1][i] = shfl_h2(frag_b[0], 8 * t + 4 + i);
    b_cols[1][4 + i] = shfl_h2(frag_b[1], 8 * t + 4 + i);
  }
}

__device__ __forceinline__ void mma_m16n8k16_f16_emu(const FragA& frag_a, const FragB& frag_b,
                                                     FragC_h& frag_c) {
  half2 a_rows[2][8], b_cols[2][8];
  mma_m16n8k16_gather(frag_a, frag_b, a_rows, b_cols);

  half2* c = reinterpret_cast<half2*>(&frag_c);
  half2 acc00 = __float2half2_rn(0.0f), acc01 = acc00, acc10 = acc00, acc11 = acc00;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc00 = __hfma2(a_rows[0][i], b_cols[0][i], acc00);
    acc01 = __hfma2(a_rows[0][i], b_cols[1][i], acc01);
    acc10 = __hfma2(a_rows[1][i], b_cols[0][i], acc10);
    acc11 = __hfma2(a_rows[1][i], b_cols[1][i], acc11);
  }
  c[0] = __hadd2(c[0], __halves2half2(__hadd(__low2half(acc00), __high2half(acc00)),
                                     __hadd(__low2half(acc01), __high2half(acc01))));
  c[1] = __hadd2(c[1], __halves2half2(__hadd(__low2half(acc10), __high2half(acc10)),
                                     __hadd(__low2half(acc11), __high2half(acc11))));
}

// MMODE==0 form of the emulated pair: rows 8..15 of the fragment are provably
// zero (a[1]/a[3] are hzero at the call site), so the second row half of the
// gather — and its 16 hfma2 — is dead work. Keeping the hot decode path off
// them is measured at ~30% of the MMA loop's issue slots.
__device__ __forceinline__ void mma_m16n8k16_f16_emu_m1(const FragA& frag_a,
                                                        const FragB& frag_b,
                                                        FragC_h& frag_c) {
  int lane = threadIdx.x & 31;
  int g = lane >> 2, t = lane & 3;

  // Only row g's eight k pairs: frag_a[0] (k pairs 0..3) and frag_a[2]
  // (4..7) live in lanes 4g+i — the a[1]/a[3] half never loads.
  half2 a_row[8], b_cols[2][8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a_row[i] = shfl_h2(frag_a[0], 4 * g + i);
    a_row[4 + i] = shfl_h2(frag_a[2], 4 * g + i);
    b_cols[0][i] = shfl_h2(frag_b[0], 8 * t + i);
    b_cols[0][4 + i] = shfl_h2(frag_b[1], 8 * t + i);
    b_cols[1][i] = shfl_h2(frag_b[0], 8 * t + 4 + i);
    b_cols[1][4 + i] = shfl_h2(frag_b[1], 8 * t + 4 + i);
  }

  half2* c = reinterpret_cast<half2*>(&frag_c);
  half2 acc00 = __float2half2_rn(0.0f), acc01 = acc00;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc00 = __hfma2(a_row[i], b_cols[0][i], acc00);
    acc01 = __hfma2(a_row[i], b_cols[1][i], acc01);
  }
  // Same fold order as the full form: acc00 folds into c[0], acc01's
  // contribution is what c[0] would carry — and the c[1] accumulate the full
  // form does adds only zeros under m1, so it is skipped outright.
  c[0] = __hadd2(c[0], __halves2half2(__hadd(__low2half(acc00), __high2half(acc00)),
                                     __hadd(__low2half(acc01), __high2half(acc01))));
}

template <bool WMMA>
__device__ __forceinline__ void gemv_mma_16x16_m1(const FragA& a, const FragB& f0,
                                                  const FragB& f1, FragC_h& c0,
                                                  FragC_h& c1) {
  // WMMA has no dead-work to cut: the instruction runs the same 16x16 tile
  // either way, so the WMMA arm keeps the shared path.
  if constexpr (WMMA) {
    gemv_mma_16x16<WMMA>(a, f0, f1, c0, c1);
    return;
  }
  mma_m16n8k16_f16_emu_m1(a, f0, c0);
  mma_m16n8k16_f16_emu_m1(a, f1, c1);
}

// ─── gfx11 WMMA converters (hip_compat_hip.cuh:540-780) ──────────────────────
// EVALUATION ONLY — see the file header and the spec. Instantiated for the
// 4 bpw arms; the default launch never selects it.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define VT_ROCM_EXL3_WMMA_INST 1

using WmmaA16 = _Float16 __attribute__((ext_vector_type(16)));
using WmmaB16 = _Float16 __attribute__((ext_vector_type(16)));
using WmmaC32 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ WmmaA16 ptx_to_wmma_a(const FragA& frag_a) {
  int lane = threadIdx.x & 31;
  int row = lane % 16;

  const half2* a = reinterpret_cast<const half2*>(&frag_a);
  WmmaA16 result;
  half2* r = reinterpret_cast<half2*>(&result);

  half2 va0 = a[0], va1 = a[1], va2 = a[2], va3 = a[3];
  int base = (row % 8) * 4;
  int e_lo = (row >= 8) ? 1 : 0;
  int e_hi = (row >= 8) ? 3 : 2;

  half2 s00 = shfl_h2(va0, base + 0);
  half2 s01 = shfl_h2(va1, base + 0);
  half2 s02 = shfl_h2(va2, base + 0);
  half2 s03 = shfl_h2(va3, base + 0);
  half2 s10 = shfl_h2(va0, base + 1);
  half2 s11 = shfl_h2(va1, base + 1);
  half2 s12 = shfl_h2(va2, base + 1);
  half2 s13 = shfl_h2(va3, base + 1);
  half2 s20 = shfl_h2(va0, base + 2);
  half2 s21 = shfl_h2(va1, base + 2);
  half2 s22 = shfl_h2(va2, base + 2);
  half2 s23 = shfl_h2(va3, base + 2);
  half2 s30 = shfl_h2(va0, base + 3);
  half2 s31 = shfl_h2(va1, base + 3);
  half2 s32 = shfl_h2(va2, base + 3);
  half2 s33 = shfl_h2(va3, base + 3);

  r[0] = (e_lo == 0) ? s00 : s01;
  r[1] = (e_lo == 0) ? s10 : s11;
  r[2] = (e_lo == 0) ? s20 : s21;
  r[3] = (e_lo == 0) ? s30 : s31;
  r[4] = (e_hi == 2) ? s02 : s03;
  r[5] = (e_hi == 2) ? s12 : s13;
  r[6] = (e_hi == 2) ? s22 : s23;
  r[7] = (e_hi == 2) ? s32 : s33;
  return result;
}

__device__ __forceinline__ WmmaB16 ptx_to_wmma_b(const FragB& frag_b0,
                                               const FragB& frag_b1) {
  int lane = threadIdx.x & 31;
  int col = lane % 16;

  const half2* b0 = reinterpret_cast<const half2*>(&frag_b0);
  const half2* b1 = reinterpret_cast<const half2*>(&frag_b1);
  WmmaB16 result;
  half2* r = reinterpret_cast<half2*>(&result);

  half2 vb00 = b0[0], vb01 = b0[1], vb10 = b1[0], vb11 = b1[1];
  int base = (col % 8) * 4;

  half2 shuffles[4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    shuffles[s][0] = shfl_h2(vb00, base + s);
    shuffles[s][1] = shfl_h2(vb01, base + s);
    shuffles[s][2] = shfl_h2(vb10, base + s);
    shuffles[s][3] = shfl_h2(vb11, base + s);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int be = j < 4 ? 0 : 1;
    int s = j % 4;
    int elem = be + (col >= 8 ? 2 : 0);
    r[j] = shuffles[s][elem];
  }
  return result;
}

// WMMA D: lane l holds D[2r + l/16][l%16] for r=0..7. PTX C: lane (g,t); c[i]
// holds D[g+8*(i/2)][2t+(i%2)]. D[i][j] is at WMMA lane (i%2)*16 + j, reg i/2.
__device__ __forceinline__ void wmma_to_ptx_c(const WmmaC32& wmma_c, FragC& frag_c0,
                                              FragC& frag_c1) {
  int lane = threadIdx.x & 31;
  int g = lane / 4;
  int t = lane % 4;
  const float* c = reinterpret_cast<const float*>(&wmma_c);
  float* d0 = reinterpret_cast<float*>(&frag_c0);
  float* d1 = reinterpret_cast<float*>(&frag_c1);

  int e0 = g / 2;
  int e1 = g / 2 + 4;
  int lane_base = (g % 2) * 16;

  float cv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = c[i];

  int src_lanes[4] = {lane_base + 2 * t, lane_base + 2 * t + 1, lane_base + 2 * t + 8,
                      lane_base + 2 * t + 9};

  float shuffles[4][8];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
#pragma unroll
    for (int e = 0; e < 8; ++e)
      shuffles[s][e] = __shfl_sync(0xffffffffffffffffULL, cv[e], src_lanes[s]);
  }

  d0[0] = shuffles[0][e0];
  d0[1] = shuffles[1][e0];
  d0[2] = shuffles[0][e1];
  d0[3] = shuffles[1][e1];
  d1[0] = shuffles[2][e0];
  d1[1] = shuffles[3][e0];
  d1[2] = shuffles[2][e1];
  d1[3] = shuffles[3][e1];
}

// WMMA 16x16x16 fp16→fp32 internally, f16 accumulation — replaces two emulated
// mma_m16n8k16_f16 calls. f32 WMMA + conversion, NOT the f16 WMMA builtin: the
// donor documents that choice as simpler and numerically more robust.
__device__ __forceinline__ void wmma_m16n16k16_f16(const FragA& frag_a, const FragB& frag_b0,
                                                   const FragB& frag_b1, FragC_h& frag_c0,
                                                   FragC_h& frag_c1) {
  WmmaA16 a = ptx_to_wmma_a(frag_a);
  WmmaB16 b = ptx_to_wmma_b(frag_b0, frag_b1);

  WmmaC32 c = {};
  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);

  FragC add_c0 = {}, add_c1 = {};
  wmma_to_ptx_c(c, add_c0, add_c1);

  float* a0 = reinterpret_cast<float*>(&add_c0);
  float* a1 = reinterpret_cast<float*>(&add_c1);
  half2* d0 = reinterpret_cast<half2*>(&frag_c0);
  half2* d1 = reinterpret_cast<half2*>(&frag_c1);
  d0[0] = __hadd2(d0[0], __floats2half2_rn(a0[0], a0[1]));
  d0[1] = __hadd2(d0[1], __floats2half2_rn(a0[2], a0[3]));
  d1[0] = __hadd2(d1[0], __floats2half2_rn(a1[0], a1[1]));
  d1[1] = __hadd2(d1[1], __floats2half2_rn(a1[2], a1[3]));
}
#else
#define VT_ROCM_EXL3_WMMA_INST 0
#endif

// The paired 16x16 MMA the kernel body calls: one FragA against two adjacent
// FragB tiles into two FragC_h accumulators. WMMA selects the hardware path;
// otherwise the two emulated 16x8 calls upstream's own m16n16 pair expands to.
template <bool WMMA>
__device__ __forceinline__ void gemv_mma_16x16(const FragA& a, const FragB& f0,
                                               const FragB& f1, FragC_h& c0, FragC_h& c1) {
#if VT_ROCM_EXL3_WMMA_INST
  if constexpr (WMMA) {
    wmma_m16n16k16_f16(a, f0, f1, c0, c1);
    return;
  }
#else
  static_assert(!WMMA, "exl3 gemv: WMMA instantiated off gfx11");
#endif
  mma_m16n8k16_f16_emu(a, f0, c0);
  mma_m16n8k16_f16_emu(a, f1, c1);
}

// ─── the kernel body (exl3_gemv_kernel.cuh:138-402) ─────────────────────────
//
// Hadamard stages REMOVED — see adaptation 1 in the header. A is the already-
// transformed a_had; C is the pre-post-scale f32 scratch. Everything between
// the removed stages is verbatim, including the unused-here guards that keep
// the codebook argument exercised identically.
template <int bits, int cb, int MMODE, int CFG, bool SMEM_STAGE, bool WMMA>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256) void Exl3GemvK(
    const half* __restrict__ A, const uint16_t* __restrict__ B, float* __restrict__ C,
    int size_m, int size_k, int size_n) {
  static_assert(bits == 2 || bits == 3 || bits == 4,
                "exl3_gemv_kernel supports 2, 3 and 4 bpw");
  constexpr int WK = CFG == 0 ? 16 : 8;   // k-split (warps per block)
  constexpr int WNT = CFG == 0 ? 2 : 4;   // adjacent n-tiles per warp
  constexpr int PF = CFG == 0 ? 4 : 2;    // prefetch ring depth
  constexpr int FOLD = CFG == 0 ? 4 : 2;  // fp16->fp32 fold cadence (divides PF)
  constexpr int THREADS = WK * 32;
  constexpr int ROWS = MMODE == 0 ? 1 : kGemvMaxM;
  constexpr int COLS = WNT * 16;

  constexpr int TWORDS = 8 * bits;                  // uint32 per 16x16 tile
  constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;  // warp loads per k-slice
  constexpr int LSTRIDE = bits == 3 ? 24 : 32;      // uint32 per load
  static_assert(bits != 2 || WNT % 2 == 0, "2 bpw packs two tiles per warp load");

  const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
  const int lane = threadIdx.x % 32;

  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;

  // Grid-level k-split: blockIdx.y owns slice blockIdx.y of gridDim.y k
  // slices, and the warps split THAT sub-range. The C writeout becomes an
  // atomicAdd whenever more than one block produces a partial per (m,col).
  const int ksplit = gridDim.y;
  const int chunk = (kslices + WK * ksplit - 1) / (WK * ksplit);
  const int ks0 = (blockIdx.y * WK + warp) * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  const size_t slice_stride = (size_t)ntiles * TWORDS;  // uint32 per k-slice row
  const half2* A2 = reinterpret_cast<const half2*>(A);
  const half2 hzero = __half2half2(__ushort_as_half(0));

  // A fragment row indices for this lane
  const int r0 = lane >> 2;
  const size_t a_row0 = (size_t)r0 * (size_k / 2);
  const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

  // Per-lane extraction constants (dq8_aligned_2bits / dq8<3, cb, 4> in
  // exl3_dq.cuh)
  [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
  if constexpr (bits == 2) {
    int i1 = lane >> 1;
    x_src_b = i1;
    x_src_a = (i1 + 15) & 15;
  }
  if constexpr (bits == 3) {
    int t_offset = lane << 3;
    int b1 = (t_offset + 257) * 3;
    int b2 = b1 + 21;
    int i0 = (b1 - 16) / 32;
    int i2 = (b2 - 1) / 32;
    x_s2 = (i2 + 1) * 32 - b2;
    x_src_a = i0 % 24;
    x_src_b = i2 % 24;
  }

  __shared__ float sh_red[WK][ROWS][COLS];
  [[maybe_unused]] __shared__ uint32_t
      sh_stage[SMEM_STAGE ? WK : 1][SMEM_STAGE ? LOADS * LSTRIDE : 1];

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + group * WNT * TWORDS + lane;

    // Prefetch ring (indices must be compile-time or pf lands in local memory)
    auto ld_b = [&](int i, int l) -> uint32_t {
      if constexpr (bits == 3)
        return lane < 24 ? __builtin_nontemporal_load(bp + (size_t)i * slice_stride +
                                                      l * LSTRIDE)
                         : 0;
      else
        return __builtin_nontemporal_load(bp + (size_t)i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
#pragma unroll
    for (int d = 0; d < PF; ++d)
      if (d < myn)
#pragma unroll
        for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;

        uint32_t bw[LOADS];
#pragma unroll
        for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];

        if (i + PF < myn) {
#pragma unroll
          for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
        }

        if constexpr (SMEM_STAGE) {
          __syncwarp();
#pragma unroll
          for (int l = 0; l < LOADS; ++l)
            if (bits != 3 || lane < 24) sh_stage[warp][l * LSTRIDE + lane] = bw[l];
          __syncwarp();
        }

        // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
        const size_t a_col = (size_t)(ks0 + i) * 8 + (lane & 3);
        FragB a01, a23;
        a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
        a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
        a01[1] = hzero;
        a23[1] = hzero;

#pragma unroll
        for (int t = 0; t < WNT; ++t) {
          FragB f0, f1;
          if constexpr (SMEM_STAGE) {
            const uint32_t* tp = &sh_stage[warp][t * TWORDS];
            if constexpr (bits == 4)
              dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
            else if constexpr (bits == 2)
              dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
            else
              dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);
          } else if constexpr (bits == 4) {
            uint32_t aw = __shfl_sync(0xffffffffffffffffULL, bw[t], (lane + 31) & 31);
            dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
          } else if constexpr (bits == 2) {
            // Two tiles per loaded word group: tile t lives in lanes (t&1)*16 .. +15
            const uint32_t w = bw[t >> 1];
            const int base = (t & 1) << 4;
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_b);
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_a);
            dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
          } else {  // bits == 3
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_a);
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_b);
            dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
          }

          FragA a;
          a[0] = a01[0];
          a[1] = a01[1];
          a[2] = a23[0];
          a[3] = a23[1];
          if constexpr (MMODE == 0)
            gemv_mma_16x16_m1<WMMA>(a, f0, f1, ch[t][0], ch[t][1]);
          else
            gemv_mma_16x16<WMMA>(a, f0, f1, ch[t][0], ch[t][1]);
        }

        if ((d + 1) % FOLD == 0 || i + 1 == myn) {
#pragma unroll
          for (int t = 0; t < WNT; ++t)
#pragma unroll
            for (int f = 0; f < 2; ++f) {
              acc0[t][f].x += __low2float(ch[t][f][0]);
              acc0[t][f].y += __high2float(ch[t][f][0]);
              ch[t][f][0] = hzero;
            }
        }
      }
    }

    // Cross-warp reduction over the k splits. Lane l holds row l/4, cols
    // tile*16 + frag*8 + 2*(l%4) (+1)
    {
      const int c0 = 2 * (lane & 3);
      const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
      const int sr0 = MMODE == 0 ? 0 : r0;
      if (store0) {
#pragma unroll
        for (int t = 0; t < WNT; ++t)
#pragma unroll
          for (int f = 0; f < 2; ++f) {
            const int col = t * 16 + f * 8 + c0;
            sh_red[warp][sr0][col + 0] = acc0[t][f].x;
            sh_red[warp][sr0][col + 1] = acc0[t][f].y;
          }
      }
    }
    __syncthreads();

    const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
    for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS) {
      const int r = idx / COLS;
      const int c = idx % COLS;
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][r][c];
      const int col = group * COLS + c;
      if (ksplit > 1)
        atomicAdd(C + (size_t)r * size_n + col, sum);
      else
        C[(size_t)r * size_n + col] = sum;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ float GemvNegBySignBit(float v) {
  return __int_as_float(static_cast<int>(__float_as_uint(v) ^ 0x80000000u));
}

// ─── the output-transform fold (rocm_exl3_fold.h) ───────────────────────────
//
// HadK<0, 2>'s per-block arithmetic for one 128-column block, run by one wave
// of the m=1 GEMV block that finishes the block's last column stripe: lane t
// holds columns 4t..4t+3, levels 1-2 in-lane, the five xor-partner levels over
// DPP row_xmask / v_permlanex16 with own-first adds and sign-bit flips, then
// res = x * r_scale, res * post in f32 and one RN bf16 round — HadXorLane,
// HadXorLevel and the OutD == 2 store of rocm_exl3.hip, op for op.
template <int I>
__device__ __forceinline__ float GemvHadXorLane(float v) {
  const int x = __float_as_int(v);
  if constexpr (I == 16)
    return __int_as_float(
        __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false));
  else
    return __int_as_float(__builtin_amdgcn_mov_dpp(x, 0x160 + I, 0xf, 0xf, false));
}

template <int I>
__device__ __forceinline__ void GemvHadXorLevel(float (&x)[4], int t) {
  const bool flip = (t & I) != 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float par = GemvHadXorLane<I>(x[j]);
    const float own = flip ? GemvNegBySignBit(x[j]) : x[j];
    x[j] = own + par;
  }
}

__device__ __forceinline__ void GemvHadOutBf16(const float* in, const uint16_t* post,
                                               uint16_t* out, float r_scale, int t) {
  const float4 f = *reinterpret_cast<const float4*>(in + 4 * t);
  const float v[4] = {f.x, f.y, f.z, f.w};
  const float s0 = v[0] + v[1];
  const float d0 = v[0] - v[1];
  const float s1 = v[2] + v[3];
  const float d1 = v[2] - v[3];
  float x[4] = {s0 + s1, d0 + d1, s0 - s1, d0 - d1};
  GemvHadXorLevel<1>(x, t);
  GemvHadXorLevel<2>(x, t);
  GemvHadXorLevel<4>(x, t);
  GemvHadXorLevel<8>(x, t);
  GemvHadXorLevel<16>(x, t);
  const uint2 pw = *reinterpret_cast<const uint2*>(post + 4 * t);
  const uint16_t qs[4] = {static_cast<uint16_t>(pw.x), static_cast<uint16_t>(pw.x >> 16),
                          static_cast<uint16_t>(pw.y), static_cast<uint16_t>(pw.y >> 16)};
  uint16_t o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float res = x[j] * r_scale;
    o[j] = __bfloat16_as_ushort(__float2bfloat16(res * DF16ToF32(qs[j])));
  }
  *reinterpret_cast<uint2*>(out + 4 * t) =
      make_uint2(o[0] | (static_cast<uint32_t>(o[1]) << 16),
                 o[2] | (static_cast<uint32_t>(o[3]) << 16));
}

// The m=1 arms' epilogue tail: called by every thread of the block after its
// stripe of C is stored. Each stripe's writers release (fence) before the
// block's one counter increment; the block that completes a 128-column block
// acquires (fence) and runs GemvHadOutBf16 on it, then re-zeroes the counter
// for the next launch (stream order is the fence there).
template <int COLS>
__device__ __forceinline__ void GemvFoldOutTail(const float* C, int group, void* out,
                                                const uint16_t* post, uint32_t* cnt,
                                                float r_scale, int warp, int lane) {
  static_assert(128 % COLS == 0, "a stripe must not straddle a 128-block");
  __shared__ uint32_t sh_last;
  __threadfence();
  __syncthreads();
  const int hb = (group * COLS) >> 7;
  if (threadIdx.x == 0)
    sh_last = atomicAdd(cnt + hb, 1u) == static_cast<uint32_t>(128 / COLS - 1);
  __syncthreads();
  if (sh_last && warp == 0) {
    __threadfence();
    GemvHadOutBf16(C + hb * 128, post + hb * 128, static_cast<uint16_t*>(out) + hb * 128,
                   r_scale, lane);
    if (lane == 0) cnt[hb] = 0;
  }
}

// ─── the m=1 direct form (RDNA-specific, spec rocm-exl3-gemv-residual) ──────
//
// dq8's output already lands on the layout a one-token decode wants: lane l
// decodes the four codewords of column l>>2 AND of column l>>2+8, at the k
// quarter t = l&3. Upstream's fragment machinery then gathers every lane's
// data to every other lane because the CUDA MMA's operand layout demands it —
// for m=1 that machinery reduces to a dot product per (column, k quarter),
// which v_dot2_f32_f16 does in one instruction per pair. Per warp-tile this
// arm issues ~60 instructions where the fragment form issues ~350 (measured
// SASS: 296 ds_bpermute + 256 pk_fma + 156 mov_b16 + 124 and per 4-slice),
// and accumulates in f32 instead of the fragment's f16 — strictly nearer the
// CPU reference, not looser.
template <int bits, int cb, bool FOLD, int WNT_ = 2, int PF_ = 2>
__global__ __launch_bounds__(512) void Exl3GemvM1K(
    const half* __restrict__ A, const uint16_t* __restrict__ B, float* __restrict__ C,
    int size_m, int size_k, int size_n, void* fold_out, const uint16_t* fold_post,
    uint32_t* fold_cnt, float fold_scale) {
  (void)size_m;  // m==1 is a launch-time fact; keeping Exl3GemvK's leading
                 // arg list so the launcher shares one kargs prefix.
  constexpr int WK = 16;   // k-split (warps per block)
  // Adjacent n-tiles per warp. The 3 bpw row takes WNT=4 where it measures
  // faster: eight independent v_dot2 accumulator chains per warp instead of
  // four, and the per-slice A half-pairs are reused across twice the trellis
  // words. Bench3/gemvbench on gfx1100: k=5120 n=17408 640 -> 696 GB/s
  // (+8.8%), k=17408 n=5120 423 -> 606 GB/s (+42%), occupancy 4 -> 3
  // blocks/CU (50 -> 82-86 VGPR, 2 -> 4 KiB smem). The 2 bpw row and the
  // 4 bpw wide-load arm keep their own widths.
  constexpr int WNT = WNT_;
  constexpr int PF = PF_;  // prefetch ring depth (2 beat 1/3/4/6/8 on gfx1100;
                           // 3 for the 3 bpw row on long per-warp k-chunks)
  constexpr int COLS = WNT * 16;

  constexpr int TWORDS = 8 * bits;                  // uint32 per 16x16 tile
  constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;  // warp loads per k-slice
  constexpr int LSTRIDE = bits == 3 ? 24 : 32;      // uint32 per load
  static_assert(bits != 2 || WNT % 2 == 0, "2 bpw packs two tiles per warp load");

  // readfirstlane: warp (and so ks0/myn) is wave-uniform, which the
  // compiler cannot prove from threadIdx.x / 32; without it every k-loop
  // bound is a VGPR compare + exec-mask branch.
  const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
  const int lane = threadIdx.x % 32;
  // 3 bpw: lanes 24..31 own no tile word (the window shuffles read lanes
  // 0..23 only), so they load word 23 again instead of branching around it.
  const int lane_ld = bits == 3 ? min(lane, 23) : lane;

  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;

  const int ksplit = gridDim.y;
  const int chunk = (kslices + WK * ksplit - 1) / (WK * ksplit);
  const int ks0 = (blockIdx.y * WK + warp) * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  const size_t slice_stride = (size_t)ntiles * TWORDS;

  // Per-lane extraction constants — the same formulas the fragment arm uses
  // (dq8_aligned_2bits / dq8<3, cb, 4> in exl3_dq.cuh).
  [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
  if constexpr (bits == 2) {
    int i1 = lane >> 1;
    x_src_b = i1;
    x_src_a = (i1 + 15) & 15;
  }
  if constexpr (bits == 3) {
    int t_offset = lane << 3;
    int b1 = (t_offset + 257) * 3;
    int b2 = b1 + 21;
    int i0 = (b1 - 16) / 32;
    int i2 = (b2 - 1) / 32;
    x_s2 = (i2 + 1) * 32 - b2;
    x_src_a = i0 % 24;
    x_src_b = i2 % 24;
  }

  __shared__ float sh_red[WK][COLS];

  const __amdgpu_buffer_rsrc_t b_rsrc = M1BufferRsrc(B, (size_t)kslices * slice_stride * 4);
  const __amdgpu_buffer_rsrc_t a_rsrc = M1BufferRsrc(A, (size_t)size_k * 2);
  const int b_lane = lane_ld * 4;
  const int a_lane = (lane & 3) * 4;
  const uint32_t sstride = static_cast<uint32_t>(slice_stride) * 4;

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t gbyte = static_cast<uint32_t>(ks0) * sstride + group * WNT * TWORDS * 4;

    auto ld_b = [&](int i, int l) -> uint32_t {
      return __builtin_amdgcn_raw_buffer_load_b32(b_rsrc, b_lane,
                                                  gbyte + i * sstride + l * LSTRIDE * 4, 6);
    };

    uint32_t pf[PF][LOADS];
#pragma unroll
    for (int d = 0; d < PF; ++d)
      if (d < myn)
#pragma unroll
        for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);

    // Two accumulators: acc0 for column l>>2, acc1 for column l>>2 + 8.
    float acc0[WNT] = {}, acc1[WNT] = {};

    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;

        uint32_t bw[LOADS];
#pragma unroll
        for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];

        if (i + PF < myn) {
#pragma unroll
          for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
        }

        // A row 0's two half2s at this lane's k quarter — the same pair the
        // fragment arm loads into a01/a23.
        const uint32_t a_s = static_cast<uint32_t>(ks0 + i) * 32;
        const half2 a01 = __builtin_bit_cast(
            half2, __builtin_amdgcn_raw_buffer_load_b32(a_rsrc, a_lane, a_s, 0));
        const half2 a23 = __builtin_bit_cast(
            half2, __builtin_amdgcn_raw_buffer_load_b32(a_rsrc, a_lane, a_s + 16, 0));

#pragma unroll
        for (int t = 0; t < WNT; ++t) {
          FragB f0, f1;
          if constexpr (bits == 4) {
            uint32_t aw = __shfl_sync(0xffffffffffffffffULL, bw[t], (lane + 31) & 31);
            dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
          } else if constexpr (bits == 2) {
            const uint32_t w = bw[t >> 1];
            const int base = (t & 1) << 4;
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_b);
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_a);
            dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
          } else {  // bits == 3
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_a);
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_b);
            dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
          }

          // f0 -> column l>>2 (k pairs 2t/2t+1 and 2t+8/2t+9), f1 -> column
          // l>>2+8, same k quarter. f32 accumulate via v_dot2_f32_f16.


          acc0[t] = amd_mixed_dot(f0[0], a01, acc0[t], false);
          acc0[t] = amd_mixed_dot(f0[1], a23, acc0[t], false);
          acc1[t] = amd_mixed_dot(f1[0], a01, acc1[t], false);
          acc1[t] = amd_mixed_dot(f1[1], a23, acc1[t], false);
        }
      }
    }

    // Fold the four lanes of each column (t bits of the lane id) and drop the
    // pair into the block-level k reduction, which is the same shared-memory
    // shape the fragment arm uses.
    const int c0 = lane >> 2;
#pragma unroll
    for (int t = 0; t < WNT; ++t) {
      float v0 = acc0[t], v1 = acc1[t];
      v0 += __shfl_xor_sync(0xffffffffffffffffULL, v0, 1);
      v0 += __shfl_xor_sync(0xffffffffffffffffULL, v0, 2);
      v1 += __shfl_xor_sync(0xffffffffffffffffULL, v1, 1);
      v1 += __shfl_xor_sync(0xffffffffffffffffULL, v1, 2);
      if ((lane & 3) == 0) {
        sh_red[warp][t * 16 + c0] = v0;
        sh_red[warp][t * 16 + c0 + 8] = v1;
      }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += WK * 32) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][idx];
      const int col = group * COLS + idx;
      if (ksplit > 1)
        atomicAdd(C + col, sum);
      else
        C[col] = sum;
    }
    if constexpr (FOLD)
      GemvFoldOutTail<COLS>(C, group, fold_out, fold_post, fold_cnt, fold_scale, warp, lane);
    __syncthreads();
  }
}

// Wide-load variant of the m=1 arm for 4 bpw. The upstream dq8 lane mapping
// (lane l owns codewords 8l..8l+7) forces one funnel-shift neighbour word per
// lane per tile and a 128-byte strided access pattern. Assigning ownership by
// LOAD POSITION instead — lane l decodes the codewords whose bits sit inside
// its own four words 4l..4l+3 of a four-tile group — removes the shuffle
// entirely: each lane's window set needs only its uint4 plus one boundary
// word (4l-1, the trailing lane's data, an L2 hit after the uint4 stream).
// RowMajorIndex maps t in [32l, 32l+31] to columns (l&7) and (l&7)+8 of tile
// l>>3 across ALL 16 rows, so each lane accumulates whole column partials and
// the epilogue is a block k-fold only — no per-tile xor-shuffles either.
// Per warp-tile-group: one 512-byte contiguous global_load_dwordx4 stream,
// one scalar boundary load, then 32 codeword extracts + 16 v_dot2_f32_f16.
template <int cb, bool FOLD>
__global__ __launch_bounds__(512) void Exl3GemvM1K4(
    const half* __restrict__ A, const uint16_t* __restrict__ B, float* __restrict__ C,
    int size_m, int size_k, int size_n, void* fold_out, const uint16_t* fold_post,
    uint32_t* fold_cnt, float fold_scale) {
  (void)size_m;
  constexpr int WK = 16;   // k-split (warps per block)
  constexpr int PF = 2;    // prefetch ring depth (2 beat 1/3/4/6/8 on gfx1100)
  constexpr int COLS = 64; // one warp covers 4 tiles = 64 columns

  const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
  const int lane = threadIdx.x % 32;
  const int l7 = lane & 7;   // column base within the lane's tile

  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;

  const int ksplit = gridDim.y;
  const int chunk = (kslices + WK * ksplit - 1) / (WK * ksplit);
  const int ks0 = (blockIdx.y * WK + warp) * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  // Tile-word index of this lane's four words inside the 128-word group.
  const int w0 = (lane >> 3) * 32 + (l7 << 2);
  const int wprev = (lane >> 3) * 32 + (((l7 << 2) - 1) & 31);
  const size_t slice_stride = (size_t)ntiles * 32;  // u32 per k-slice at 4bpw
  const half2* A2 = reinterpret_cast<const half2*>(A);

  __shared__ float sh_red[WK][COLS];

  const __amdgpu_buffer_rsrc_t b_rsrc = M1BufferRsrc(B, (size_t)kslices * slice_stride * 4);
  const uint32_t sstride = static_cast<uint32_t>(slice_stride) * 4;

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t gbyte = static_cast<uint32_t>(ks0) * sstride + group * 512;

    auto ld_b = [&](int i) -> uint4 {
      const auto r = __builtin_amdgcn_raw_buffer_load_b128(b_rsrc, w0 * 4, gbyte + i * sstride, 0);
      return uint4{r[0], r[1], r[2], r[3]};
    };

    uint4 pf[PF];
#pragma unroll
    for (int d = 0; d < PF; ++d)
      if (d < myn) pf[d] = ld_b(d);

    float accA = 0.0f, accB = 0.0f;  // columns l7 and l7+8 of tile lane>>3

    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;

        const uint4 v = pf[d];
        if (i + PF < myn) pf[d] = ld_b(i + PF);

        const uint32_t wb =
            __builtin_amdgcn_raw_buffer_load_b32(b_rsrc, wprev * 4, gbyte + i * sstride, 0);

        // Window for group-codeword s (tile codeword t = 32*l7 + s): from
        // Exl3TileCodeword (b0 = 4t + 1012, b1 = 4t + 1028), words (i0,i1) map
        // to regs [(4s+1012)>>5, (4s+1027)>>5] relative to word 4*l7-1, and
        // s0 = (28 - 4s) & 31.
        uint32_t w[5] = {wb, v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          FragB fA, fB;
          uint32_t cw[8];
#pragma unroll
          for (int s = 0; s < 8; ++s) {
            const int sc = j * 8 + s;
            // i0 = (4*sc + 1012) >> 5, i1 = (4*sc + 1027) >> 5, both
            // relative to word 4*l7-1 of the tile (index offset 31).
            // fshift(b, a) = (a<<32|b)>>s: the HIGH word is i0's, the LOW
            // word is i1's — argument order is (w[i1], w[i0]).
            cw[s] = fshift(w[((4 * sc + 1027) >> 5) - 31],
                           w[((4 * sc + 1012) >> 5) - 31],
                           (28 - 4 * sc) & 31);
          }
          // decode8 emits f=[w0..w3],f1=[w4..w7]; for s in [8j,8j+8) the f
          // half is column l7 (sub<4) and f1 is column l7+8.
          decode8<cb>(cw[0], cw[1], cw[2], cw[3], cw[4], cw[5], cw[6], cw[7],
                      fA, fB);
          const size_t a_col = (size_t)(ks0 + i) * 8;
          const half2 a01 = A2[a_col + j];
          const half2 a89 = A2[a_col + j + 4];
          accA = amd_mixed_dot(fA[0], a01, accA, false);
          accA = amd_mixed_dot(fA[1], a89, accA, false);
          accB = amd_mixed_dot(fB[0], a01, accB, false);
          accB = amd_mixed_dot(fB[1], a89, accB, false);
        }
      }
    }

    // Whole-column partials already: no intra-warp fold. Warp's tile is
    // (lane>>3), column l7 / l7+8.
    sh_red[warp][(lane >> 3) * 16 + l7] = accA;
    sh_red[warp][(lane >> 3) * 16 + l7 + 8] = accB;
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += WK * 32) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][idx];
      const int col = group * COLS + idx;
      if (ksplit > 1)
        atomicAdd(C + col, sum);
      else
        C[col] = sum;
    }
    if constexpr (FOLD)
      GemvFoldOutTail<COLS>(C, group, fold_out, fold_post, fold_cnt, fold_scale, warp, lane);
    __syncthreads();
  }
}
// ─── the fused input-Hadamard m=1 arm ────────────────────────────────────────
//
// For m=1 decode the three-kernel chain launches had_r_128 (A, f16→f16) as a
// separate kernel, then every GEMV block re-streams A_had from global memory
// once per column group. M1K4 v5 (bench3.hip, verified bit-identical against
// HadK<true,true> + the unfused kernel) instead computes a_had in LDS once per
// block: k/128 warp-jobs of the same 5-level xor-partner butterfly HadK runs,
// stores packed half2s, and the dot loop reads shA where the unfused arm read
// A2. One fewer launch and no a_had write+read round-trip per projection.
//
// The butterfly is HadK<true,true> op-for-op: MulF16 pre-scale (the fp16
// product rounded once from f32, same as the host __hmul2), in-lane levels
// 1-2 as (s0+s1, d0+d1, s0-s1, d0-s1), then the five xor-partner levels with
// own-first adds and NegBySignBit flips — the identical instruction sequence,
// LDS in HadK replaced by __shfl_xor here. DF16ToF32/DF32ToF16 come from
// rocm_f16_codec.h; NegBySignBit/MulF16 are transcribed 1:1 from
// rocm_exl3.hip (the two files share no TU).

__device__ __forceinline__ uint16_t GemvMulF16(uint16_t a, uint16_t b) {
  return DF32ToF16(DF16ToF32(a) * DF16ToF32(b));
}
// ABf16 selects the bf16-residual arm: the activation arrives bf16 and is
// narrowed in-register by the same DF32ToF16(bf16->f32) chain CastF16K<bf16>
// runs, so the staged a_had is bit-identical to f16-input + a separate cast.
template <bool ABf16>
__device__ __forceinline__ void HadBlock128ToH2(const void* __restrict__ a,
                                              const uint16_t* __restrict__ pre,
                                              int64_t base, half2* out4) {
  const int t = threadIdx.x & 31;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint16_t h =
        ABf16 ? DF32ToF16(__bfloat162float(
                    static_cast<const __hip_bfloat16*>(a)[base + 4 * t + j]))
              : static_cast<const uint16_t*>(a)[base + 4 * t + j];
    v[j] = DF16ToF32(GemvMulF16(h, pre[base + 4 * t + j]));
  }
  const float s0 = v[0] + v[1], d0 = v[0] - v[1];
  const float s1 = v[2] + v[3], d1 = v[2] - v[3];
  float x[4] = {s0 + s1, d0 + d1, s0 - s1, d0 - d1};
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    const bool flip = (t & i) != 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float par = __shfl_xor_sync(0xffffffffffffffffULL, x[j], i);
      const float own = flip ? GemvNegBySignBit(x[j]) : x[j];
      x[j] = own + par;  // own FIRST, matching the host adds
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
    // DF32ToF16, not __float2half: HadK rounds f32(res) then f32->f16 (double
    // rounding). __float2half(x * k) contracts to v_fma_mixlo_f16, a SINGLE
    // rounding, which diffs by 1 ulp on ~1/5K boundary inputs and silently
    // perturbs every GEMV column downstream. Write through uint16_t* —
    // DF32ToF16 returns the bit pattern, not a numeric half.
    reinterpret_cast<uint16_t*>(out4)[j] = DF32ToF16(x[j] * kGemvInvSqrt128);
}

// 4 bpw fused arm — Exl3GemvM1K4 with the input hadamard staged in LDS.
// `shA` is dynamic LDS sized size_k * 2 bytes; every block stages the same
// a_had (only its own group columns differ), which is the trade the launch
// count buys.
template <int cb, bool ABf16>
__global__ __launch_bounds__(512) void Exl3GemvM1K4Fused(
    const void* __restrict__ A, const uint16_t* __restrict__ suh,
    const uint16_t* __restrict__ B, float* __restrict__ C, int size_m,
    int size_k, int size_n) {
  (void)size_m;
  constexpr int WK = 16;
  constexpr int PF = 4;
  constexpr int COLS = 64;
  extern __shared__ half2 shA[];

  const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
  const int lane = threadIdx.x % 32;
  const int l7 = lane & 7;

  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;

  const int ksplit = gridDim.y;
  const int chunk = (kslices + WK * ksplit - 1) / (WK * ksplit);
  const int ks0 = (blockIdx.y * WK + warp) * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  // In-had phase: warp-jobs over k/128 blocks. Every warp iterates its stride;
  // each produces one 128-column block of a_had into LDS as packed half2s.
  for (int hb = warp; hb < size_k / 128; hb += WK) {
    half2 o2[2];
    HadBlock128ToH2<ABf16>(A, suh, hb * 128, o2);
    shA[hb * 64 + lane * 2] = o2[0];
    shA[hb * 64 + lane * 2 + 1] = o2[1];
  }
  __syncthreads();

  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  const int w0 = (lane >> 3) * 32 + (l7 << 2);
  const size_t slice_stride = (size_t)ntiles * 32;

  __shared__ float sh_red[WK][COLS];

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + (size_t)group * 128;
    // The trellis is touched once per decode step — the sub-4 fused arm's
    // nontemporal spelling kept for parity (evict-first .slc on gfx11).
    auto ld_b = [&](int i) -> uint4 {
      const uint32_t* p = bp + (size_t)i * slice_stride + w0;
      return make_uint4(__builtin_nontemporal_load(p),
                        __builtin_nontemporal_load(p + 1),
                        __builtin_nontemporal_load(p + 2),
                        __builtin_nontemporal_load(p + 3));
    };
    uint4 pf[PF];
#pragma unroll
    for (int d = 0; d < PF; ++d) if (d < myn) pf[d] = ld_b(d);

    float accA = 0.0f, accB = 0.0f;
    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;
        const uint4 v = pf[d];
        if (i + PF < myn) pf[d] = ld_b(i + PF);
        const int bsrc = (l7 == 0) ? (lane | 7) : (lane - 1);
        const uint32_t wb = __shfl_sync(0xffffffffffffffffULL, v.w, bsrc);
        uint32_t w[5] = {wb, v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          FragB fA, fB;
          uint32_t cw[8];
#pragma unroll
          for (int s2 = 0; s2 < 8; ++s2) {
            const int sc = j * 8 + s2;
            cw[s2] = fshift(w[((4 * sc + 1027) >> 5) - 31],
                            w[((4 * sc + 1012) >> 5) - 31],
                            (28 - 4 * sc) & 31);
          }
          decode8<cb>(cw[0], cw[1], cw[2], cw[3], cw[4], cw[5], cw[6], cw[7],
                      fA, fB);
          const int a_col = (ks0 + i) * 8;
          const half2 a01 = shA[a_col + j];
          const half2 a89 = shA[a_col + j + 4];
          accA = amd_mixed_dot(fA[0], a01, accA, false);
          accA = amd_mixed_dot(fA[1], a89, accA, false);
          accB = amd_mixed_dot(fB[0], a01, accB, false);
          accB = amd_mixed_dot(fB[1], a89, accB, false);
        }
      }
    }

    sh_red[warp][(lane >> 3) * 16 + l7] = accA;
    sh_red[warp][(lane >> 3) * 16 + l7 + 8] = accB;
    __syncthreads();
    for (int idx = threadIdx.x; idx < COLS; idx += WK * 32) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][idx];
      const int col = group * COLS + idx;
      if (ksplit > 1)
        atomicAdd(C + col, sum);
      else
        C[col] = sum;
    }
    __syncthreads();
  }
}

// Sub-4 bpw fused arm — Exl3GemvM1K with the same LDS input-hadamard phase.
// The A read it replaces is `A2[a_col]`/`A2[a_col+4]` (the lane's k quarter),
// which maps 1:1 onto shA since a_had's layout is column-major-contiguous
// half2s.
template <int bits, int cb, bool ABf16>
__global__ __launch_bounds__(512) void Exl3GemvM1KFused(
    const void* __restrict__ A, const uint16_t* __restrict__ suh,
    const uint16_t* __restrict__ B, float* __restrict__ C, int size_m,
    int size_k, int size_n) {
  (void)size_m;
  constexpr int WK = 16;
  constexpr int WNT = 2;
  constexpr int PF = 4;
  constexpr int COLS = WNT * 16;
  constexpr int TWORDS = 8 * bits;
  constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
  constexpr int LSTRIDE = bits == 3 ? 24 : 32;
  extern __shared__ half2 shA[];

  const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
  const int lane = threadIdx.x % 32;

  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;

  const int ksplit = gridDim.y;
  const int chunk = (kslices + WK * ksplit - 1) / (WK * ksplit);
  const int ks0 = (blockIdx.y * WK + warp) * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));

  for (int hb = warp; hb < size_k / 128; hb += WK) {
    half2 o2[2];
    HadBlock128ToH2<ABf16>(A, suh, hb * 128, o2);
    shA[hb * 64 + lane * 2] = o2[0];
    shA[hb * 64 + lane * 2 + 1] = o2[1];
  }
  __syncthreads();

  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  const size_t slice_stride = (size_t)ntiles * TWORDS;

  [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
  if constexpr (bits == 2) {
    int i1 = lane >> 1;
    x_src_b = i1;
    x_src_a = (i1 + 15) & 15;
  }
  if constexpr (bits == 3) {
    int t_offset = lane << 3;
    int b1 = (t_offset + 257) * 3;
    int b2 = b1 + 21;
    int i0 = (b1 - 16) / 32;
    int i2 = (b2 - 1) / 32;
    x_s2 = (i2 + 1) * 32 - b2;
    x_src_a = i0 % 24;
    x_src_b = i2 % 24;
  }

  __shared__ float sh_red[WK][COLS];

  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + group * WNT * TWORDS + lane;
    auto ld_b = [&](int i, int l) -> uint32_t {
      if constexpr (bits == 3)
        return lane < 24 ? __builtin_nontemporal_load(bp + (size_t)i * slice_stride +
                                                          l * LSTRIDE)
                         : 0;
      else
        return __builtin_nontemporal_load(bp + (size_t)i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
#pragma unroll
    for (int d = 0; d < PF; ++d)
      if (d < myn)
#pragma unroll
        for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);

    float acc0[WNT] = {}, acc1[WNT] = {};
    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;

        uint32_t bw[LOADS];
#pragma unroll
        for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];

        if (i + PF < myn) {
#pragma unroll
          for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
        }

        const size_t a_col = (size_t)(ks0 + i) * 8 + (lane & 3);
        const half2 a01 = shA[a_col];
        const half2 a23 = shA[a_col + 4];

#pragma unroll
        for (int t = 0; t < WNT; ++t) {
          FragB f0, f1;
          if constexpr (bits == 4) {
            uint32_t aw = __shfl_sync(0xffffffffffffffffULL, bw[t], (lane + 31) & 31);
            dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
          } else if constexpr (bits == 2) {
            const uint32_t w = bw[t >> 1];
            const int base = (t & 1) << 4;
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_b);
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, w, base + x_src_a);
            dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
          } else {
            uint32_t awv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_a);
            uint32_t bwv = __shfl_sync(0xffffffffffffffffULL, bw[t], x_src_b);
            dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
          }

          acc0[t] = amd_mixed_dot(f0[0], a01, acc0[t], false);
          acc0[t] = amd_mixed_dot(f0[1], a23, acc0[t], false);
          acc1[t] = amd_mixed_dot(f1[0], a01, acc1[t], false);
          acc1[t] = amd_mixed_dot(f1[1], a23, acc1[t], false);
        }
      }
    }

    const int c0 = lane >> 2;
#pragma unroll
    for (int t = 0; t < WNT; ++t) {
      float v0 = acc0[t], v1 = acc1[t];
      v0 += __shfl_xor_sync(0xffffffffffffffffULL, v0, 1);
      v0 += __shfl_xor_sync(0xffffffffffffffffULL, v0, 2);
      v1 += __shfl_xor_sync(0xffffffffffffffffULL, v1, 1);
      v1 += __shfl_xor_sync(0xffffffffffffffffULL, v1, 2);
      if ((lane & 3) == 0) {
        sh_red[warp][t * 16 + c0] = v0;
        sh_red[warp][t * 16 + c0 + 8] = v1;
      }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += WK * 32) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < WK; ++j) sum += sh_red[j][idx];
      const int col = group * COLS + idx;
      if (ksplit > 1)
        atomicAdd(C + col, sum);
      else
        C[col] = sum;
    }
    __syncthreads();
  }
}

// ─── the try-launch (exl3_gemv.cu:29-179) ────────────────────────────────────

constexpr bool GemvArmInstantiated(int bits, int cb) {
  // Upstream's selector table (exl3_gemv.cu SEL_GRID list), read as a
  // predicate: 4 bpw decodes all three codebooks, 2 and 3 bpw instantiate
  // cb 1/2 only (cb 0 at those widths is refused upstream, exl3_gemv.cu:113).
  // 6 bpw (this checkpoint's lm_head) has no arm here — the dot kernel in
  // rocm_exl3_dot.hip serves it.
  return (bits == 4 && (cb == 0 || cb == 1 || cb == 2)) ||
         ((bits == 2 || bits == 3) && (cb == 1 || cb == 2));
}

// 3 bpw m=1 arm shape policy (see the WNT note on Exl3GemvM1K). Both switches
// are measured: the deeper prefetch ring pays only where the per-warp k-chunk
// is long, and the wide (64-column) form is taken where the extra ILP beats
// the shorter grid it leaves. `VT_EXL3_M1K3_WNT=2` restores the narrow form
// as the same-binary A/B.
constexpr int kGemvM1LongChunk = 40;  // 16-row k-slices per warp
bool GemvM1K3Wide(int bits, int size_n) {
  if (bits != 3) return false;
  static const int mode = EnvIntStrict("VT_EXL3_M1K3_WNT", 0);
  if (mode == 2) return false;
  return size_n >= 8192;
}
bool GemvM1K3Pf3(int bits, int size_k, int ksplit) {
  if (bits != 3) return false;
  const int kslices = size_k / 16;
  const int chunk = (kslices + 16 * ksplit - 1) / (16 * ksplit);
  return chunk >= kGemvM1LongChunk;
}

const void* GemvM1KernelFor(int bits, int cb, bool fold, int size_k, int size_n,
                            int ksplit) {
  // The direct m=1 arm's instantiation set is the non-WMMA half of upstream's
  // selector table minus its smem rows (this kernel stages nothing). `fold`
  // selects the output-transform epilogue twin of each row.
#define VT_GEMV_M1N(bits_, cb_, wnt_, pf_)                                  \
  if (bits == bits_ && cb == cb_)                                           \
    return fold                                                            \
               ? reinterpret_cast<const void*>(                             \
                     &Exl3GemvM1K<bits_, cb_, true, wnt_, pf_>)              \
               : reinterpret_cast<const void*>(                             \
                     &Exl3GemvM1K<bits_, cb_, false, wnt_, pf_>);
#define VT_GEMV_M1W4(bits_, cb_)                                            \
  if (bits == bits_ && cb == cb_)                                           \
    return fold ? reinterpret_cast<const void*>(&Exl3GemvM1K4<cb_, true>)  \
                : reinterpret_cast<const void*>(&Exl3GemvM1K4<cb_, false>);
  // The wide-load ownership arm is instantiated for the 4 bpw rows only —
  // its codeword-span arithmetic is the 4 bpw window math; 2/3 bpw keep the
  // dq8-mapping arm.
  VT_GEMV_M1W4(4, 0) VT_GEMV_M1W4(4, 1) VT_GEMV_M1W4(4, 2)
  VT_GEMV_M1N(2, 1, 2, 2) VT_GEMV_M1N(2, 2, 2, 2)
  if (GemvM1K3Wide(bits, size_n)) {
    if (GemvM1K3Pf3(bits, size_k, ksplit)) {
      VT_GEMV_M1N(3, 1, 4, 3) VT_GEMV_M1N(3, 2, 4, 3)
    } else {
      VT_GEMV_M1N(3, 1, 4, 2) VT_GEMV_M1N(3, 2, 4, 2)
    }
  } else {
    VT_GEMV_M1N(3, 1, 2, 2) VT_GEMV_M1N(3, 2, 2, 2)
  }
#undef VT_GEMV_M1W4
#undef VT_GEMV_M1N
  return nullptr;
}
template <bool ABf16>
const void* GemvM1FusedKernelFor(int bits, int cb) {
  // The fused in-had arm's instantiation set mirrors GemvM1KernelFor's: same
  // (bits, cb) coverage, one kernel per row.
#define VT_GEMV_M1F(bits_, cb_)                                             \
  if (bits == bits_ && cb == cb_)                                           \
    return reinterpret_cast<const void*>(&Exl3GemvM1KFused<bits_, cb_, ABf16>);
#define VT_GEMV_M1F4(bits_, cb_)                                            \
  if (bits == bits_ && cb == cb_)                                           \
    return reinterpret_cast<const void*>(&Exl3GemvM1K4Fused<cb_, ABf16>);
  // Only 4 bpw rows instantiate: the call site gates the arm on bits == 4.
  VT_GEMV_M1F4(4, 0) VT_GEMV_M1F4(4, 1) VT_GEMV_M1F4(4, 2)
#undef VT_GEMV_M1F4
#undef VT_GEMV_M1F
  return nullptr;
}

const void* GemvKernelFor(int bits, int cb, int mmode, int cfg, bool smem, bool wmma) {
  // The selector spells every instantiation so a typo in the table is a
  // compile error, not a wrong kernel.
#define VT_GEMV_SEL(bits_, cb_, mm_, cfg_, sm_, wm_)                                    \
  if (bits == bits_ && cb == cb_ && mmode == mm_ && cfg == cfg_ && smem == sm_ &&       \
      wmma == wm_)                                                                      \
    return reinterpret_cast<const void*>(&Exl3GemvK<bits_, cb_, mm_, cfg_, sm_, wm_>);
#define VT_GEMV_GRID(bits_, cb_, sm_, wm_)                                              \
  VT_GEMV_SEL(bits_, cb_, 0, 0, sm_, wm_) VT_GEMV_SEL(bits_, cb_, 0, 1, sm_, wm_)       \
      VT_GEMV_SEL(bits_, cb_, 1, 0, sm_, wm_) VT_GEMV_SEL(bits_, cb_, 1, 1, sm_, wm_)

  if (!wmma) {
    // Upstream's table minus its c_fp32=false row: C here is always the f32
    // scratch, so only the fp32-write arm exists on this TU. smem staging is
    // instantiated where upstream instantiates it (2 and 3 bpw only).
    VT_GEMV_GRID(4, 0, false, false) VT_GEMV_GRID(4, 1, false, false)
        VT_GEMV_GRID(4, 2, false, false)
        VT_GEMV_GRID(2, 1, false, false) VT_GEMV_GRID(2, 2, false, false)
        VT_GEMV_GRID(2, 1, true, false) VT_GEMV_GRID(2, 2, true, false)
        VT_GEMV_GRID(3, 1, false, false) VT_GEMV_GRID(3, 2, false, false)
        VT_GEMV_GRID(3, 1, true, false) VT_GEMV_GRID(3, 2, true, false)
  } else {
#if VT_ROCM_EXL3_WMMA_INST
    // WMMA is evaluation-only (header, adaptation 2): the 4 bpw arms, which is
    // the decode shape this row serves.
    VT_GEMV_GRID(4, 0, false, true) VT_GEMV_GRID(4, 1, false, true)
        VT_GEMV_GRID(4, 2, false, true)
#endif
  }

#undef VT_GEMV_GRID
#undef VT_GEMV_SEL
  return nullptr;
}
// Batched-row 3bpw GEMV (bench3.hip M1Km3, verify_m at m=8 vs the scalar
// transcription). For 2 <= m <= 8 the production fragment arm re-streams the
// trellis per row and measured 114-138 GB/s; this arm keeps the M1K3v
// position-ownership mapping, loads and decodes each tile ONCE, and dots the
// decoded fragment against all M activation rows — the same per-row rounding
// order as the m=1 arm. Serving at concurrency 8 the m<=8 fragment path is
// the decode bottleneck; this arm reached ~250-305 GB/s on the qwen shapes.
// Instantiated per M so the row accumulators stay in registers unrolled.
template <int cb, int M>
__global__ __launch_bounds__(512) void Exl3GemvMK3(
    const half* __restrict__ A, const uint16_t* __restrict__ B, float* __restrict__ C,
    int size_m, int size_k, int size_n) {
  (void)size_m;  // dispatch contract: size_m == M
  constexpr int WK = 16;
  constexpr int PF = 4;
  constexpr int COLS = 64;
  const int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
  const int lane = threadIdx.x % 32;
  const int l7 = lane & 7;
  const int ntiles = size_n / 16;
  const int kslices = size_k / 16;
  const int num_groups = size_n / COLS;
  const int ksplit = gridDim.y;
  const int chunk = (kslices + WK * ksplit - 1) / (WK * ksplit);
  const int ks0 = (blockIdx.y * WK + warp) * chunk;
  const int myn = max(0, min(chunk, kslices - ks0));
  const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
  const int w0 = (lane >> 3) * 24 + l7 * 3;
  const size_t slice_stride = (size_t)ntiles * 24;
  const int Kh2 = size_k >> 1;
  const half2* A2 = reinterpret_cast<const half2*>(A);
  __shared__ float sh_red[WK][COLS];
  for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + (size_t)group * 96;
    auto ld_b = [&](int i, uint32_t& x, uint32_t& y, uint32_t& z) {
      const uint32_t* p = bp + (size_t)i * slice_stride + w0;
      x = p[0]; y = p[1]; z = p[2];
    };
    uint32_t pf[PF][3];
#pragma unroll
    for (int d = 0; d < PF; ++d)
      if (d < myn) ld_b(d, pf[d][0], pf[d][1], pf[d][2]);
    float accA[M], accB[M];
#pragma unroll
    for (int mi = 0; mi < M; ++mi) { accA[mi] = 0.0f; accB[mi] = 0.0f; }
    for (int ib = 0; ib < myn; ib += PF) {
#pragma unroll
      for (int d = 0; d < PF; ++d) {
        const int i = ib + d;
        if (i >= myn) break;
        const uint32_t v0 = pf[d][0], v1 = pf[d][1], v2 = pf[d][2];
        if (i + PF < myn) ld_b(i + PF, pf[d][0], pf[d][1], pf[d][2]);
        const int bsrc = (l7 == 0) ? (lane | 7) : (lane - 1);
        const uint32_t wb = __shfl_sync(0xffffffffffffffffULL, v2, bsrc);
        const uint32_t w[4] = {wb, v0, v1, v2};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          FragB fA, fB;
          uint32_t cw[8];
#pragma unroll
          for (int ss = 0; ss < 8; ++ss) {
            const int sc = j * 8 + ss;
            const int r0 = ((3 * sc + 755) >> 5) - 23;
            const int r1 = ((3 * sc + 770) >> 5) - 23;
            const int s0 = 32 * (((3 * sc + 770) >> 5) + 1) - (3 * sc + 771);
            cw[ss] = fshift(w[r1], w[r0], s0);
          }
          decode8<cb>(cw[0], cw[1], cw[2], cw[3], cw[4], cw[5], cw[6], cw[7],
                      fA, fB);
          const size_t a_col = (size_t)(ks0 + i) * 8 + j;
#pragma unroll
          for (int mi = 0; mi < M; ++mi) {
            const half2 a01 = A2[(size_t)mi * Kh2 + a_col];
            const half2 a89 = A2[(size_t)mi * Kh2 + a_col + 4];
            accA[mi] = amd_mixed_dot(fA[0], a01, accA[mi], false);
            accA[mi] = amd_mixed_dot(fA[1], a89, accA[mi], false);
            accB[mi] = amd_mixed_dot(fB[0], a01, accB[mi], false);
            accB[mi] = amd_mixed_dot(fB[1], a89, accB[mi], false);
          }
        }
      }
    }
#pragma unroll
    for (int mi = 0; mi < M; ++mi) {
      __syncthreads();
      sh_red[warp][(lane >> 3) * 16 + l7] = accA[mi];
      sh_red[warp][(lane >> 3) * 16 + l7 + 8] = accB[mi];
      __syncthreads();
      for (int idx = threadIdx.x; idx < COLS; idx += WK * 32) {
        float sum = 0.0f;
#pragma unroll
        for (int wj = 0; wj < WK; ++wj) sum += sh_red[wj][idx];
        const int col = group * COLS + idx;
        if (ksplit > 1) atomicAdd(C + (size_t)mi * size_n + col, sum);
        else C[(size_t)mi * size_n + col] = sum;
      }
    }
    __syncthreads();
  }
}

// M dispatch for the batched-row arm. M is a template parameter so the
// register-file accumulators unroll per instantiation; size_m 2..8 cover the
// mmode=1 range (kExl3GemvMaxM == 8).
const void* GemvMK3KernelFor(int m) {
  // cb=2 only (the mul1 pair decoder is the templated row): the caller's
  // `args.codebook == 2` gate is the contract — a future cb here serves the
  // cb=2 window constants through the wrong codebook with no compile error.
  switch (m) {
    case 2: return (const void*)&Exl3GemvMK3<2, 2>;
    case 3: return (const void*)&Exl3GemvMK3<2, 3>;
    case 4: return (const void*)&Exl3GemvMK3<2, 4>;
    case 5: return (const void*)&Exl3GemvMK3<2, 5>;
    case 6: return (const void*)&Exl3GemvMK3<2, 6>;
    case 7: return (const void*)&Exl3GemvMK3<2, 7>;
    case 8: return (const void*)&Exl3GemvMK3<2, 8>;
    default: return nullptr;
  }
}


int GemvOccupancyRocm(const void* kernel, int block_dim, size_t dyn_smem = 0) {
  // exl3_gemv.cu:120-134 + exllamav3-rocm's RDNA correction: the occupancy
  // query must see the kernel's real dynamic-LDS request — a fused arm that
  // stages size_k halves under-counts co-residency when queried at 0.
  static std::mutex mtx;
  static std::map<std::pair<const void*, size_t>, int> cache;
  std::lock_guard<std::mutex> lock(mtx);
  const auto key = std::make_pair(kernel, dyn_smem);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  int blocks_per_sm = 0;
  // A failed query is a decline (0), not a throw — this file's contract is
  // that false/fallback paths never surface as hard errors.
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel,
                                                   block_dim, dyn_smem) !=
      hipSuccess)
    blocks_per_sm = 0;
  if (blocks_per_sm > 1) blocks_per_sm -= 1;
  cache[key] = blocks_per_sm;
  return blocks_per_sm;
}

// VT_EXL3_GEMV_GRID — 1 launches one block per column group; 0 caps the grid
// at the occupancy-derived co-resident count (upstream's cooperative-launch
// sizing, minus the RDNA one-block margin). These are ordinary launches with
// grid-stride group loops, so co-residency is not required for correctness;
// the cap only matters for scheduling: with n/cols just above the cap, a few
// blocks run a second group alone while the rest of the GPU idles.
int GemvGridForGroups(int groups, int max_coresident) {
  static const int mode = EnvIntStrict("VT_EXL3_GEMV_GRID", 1);
  if (mode == 1) return std::min(groups, 65535);
  return std::min(groups, max_coresident);
}

bool GemvWmmaMode() {
  // VT_ROCM_EXL3_WMMA — documented in docs/ENVIRONMENT.md. Evaluation only:
  // the default (unset/0) is the emulated path the gates measure.
  static const int mode = EnvIntStrict("VT_ROCM_EXL3_WMMA", 0);
  return mode == 1;
}

}  // namespace

// Returns true when the GEMV arm launched. The same decline contract the CUDA
// arm documents: a false here is a fall-through to Exl3GemmK, never a failure.
// A non-null `fold` offers the output transform to the m=1 direct arms; it is
// taken only when fold->applied comes back true.
bool Exl3GemvTryLaunchRocm(hipStream_t st, const Tensor& a_had, const Tensor& trellis,
                           float* raw, int size_m, int size_k, int size_n, int device,
                           const Exl3GemmArgs& args, Exl3GemvOutFold* fold) {
  // exl3_gemv.cu:108-116: the free integer tests first, then the env read.
  if (args.force_gemv == 0) return false;
  // HALF-INTEGER rate (K+0.5, mul1): no frac GEMV arm exists (spec ## Owed).
  // The decline must come FIRST — bits = KA passes the integer tests below as
  // e.g. (3, 2), an instantiated arm, and would decode the frac bitstream
  // through the integer window arithmetic.
  if (args.half) return false;
  if (!GemvArmInstantiated(args.bits, args.codebook)) return false;
  if (!Exl3GemvHardEligible(size_m, size_k, size_n, args.bits, args.codebook,
                            /*has_su_sv=*/true, /*half=*/args.half))
    return false;
  const int mode = args.force_gemv > 0 ? 2 : Exl3GemvMode();
  if (mode == 0) return false;
  const int mmode = size_m == 1 ? 0 : 1;
  // smem staging is instantiated for 2 and 3 bpw only (GemvKernelFor); a
  // forced SMEM=1 on 4 bpw must not decline the whole arm.
  const bool smem = Exl3GemvSmemMode() == 1 &&
                    (args.bits == 2 || args.bits == 3);
  const bool wmma = GemvWmmaMode();

  // wave32 is structural: the A/B fragment gathers and the bits-3 24-lane
  // window are written for 32-lane warps. A wave64 build takes the scalar arm
  // rather than decode wrong.
  int warp_size = 0;
  CheckG(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device),
         "warp size query");
  if (warp_size != 32) return false;

  int num_sms = 0;
  CheckG(hipDeviceGetAttribute(&num_sms, hipDeviceAttributeMultiprocessorCount, device),
         "multiprocessor count");
  if (num_sms < 1) num_sms = 1;

  const void* narrow = GemvKernelFor(args.bits, args.codebook, mmode, 0, smem, wmma);
  if (narrow == nullptr) return false;

  // VT_EXL3_GEMV_CFG forces the upstream cfg choice — a tuning probe, not a
  // contract: 0 is the 512-thread narrow arm, 1 the 256-thread WNT=4 arm.
  static const int kCfgEnv = EnvIntStrict("VT_EXL3_GEMV_CFG", -1);

  int cfg = 0;
  if (narrow != nullptr) {
    const int narrow_coresident = GemvOccupancyRocm(narrow, 512) * num_sms;

    // exllamav3-rocm passes CC_RDNA3 into this heuristic, where every
    // RDNA3-reachable arm lands on the same decision as CC_ADA (K==3 gate)
    // or the fallthrough. Exl3Cc has no gfx bucket and gains none: passing
    // kAda is the same function evaluated, not an approximation of it.
    const int cfg0 =
        Exl3GemvSelectConfig(Exl3Cc::kAda, size_m, size_k, size_n, args.bits,
                             args.codebook, mode, narrow_coresident);
    // ROCm extension (spec rocm-exl3-gemv-residual §R1): a -1 here is
    // upstream's Ampere-tuned heuristic having no opinion. On this backend
    // the alternative is the byte-exact scalar transcription (~30 GB/s
    // effective), never another fast arm, so a hard-eligible call falls
    // back to upstream's own forced rule (exl3_gemv.cu:66-67).
    cfg = kCfgEnv >= 0 ? kCfgEnv
                       : (cfg0 >= 0 ? cfg0 : (size_n <= 8192 ? 0 : 1));
  }

  // m=1 decode takes the direct arm: same launch geometry as cfg 0 but the
  // fragment gather + fp16-fragment MMA replaced by four v_dot2_f32_f16 per
  // lane-pair — ~6x fewer issued instructions per tile (SASS-measured), and
  // f32 accumulation where the fragment arm used f16 pairs. The WMMA env
  // still selects the WMMA evaluation arm for A/B completeness.
  // Grid-level k-split (the kernel halves its per-warp serial k-chain per
  // extra blockIdx.y) is a bench lever first: VT_EXL3_GEMV_KSPLIT=N raises
  // grid.y to N and pays one f32 atomicAdd per (m, stripe col) per slice.
  // It stays OFF by default until serving measurements pick a value.
  static const int kKsplitEnv = EnvIntStrict("VT_EXL3_GEMV_KSPLIT", 1);
  const int ksplit = std::max(
      1, std::min(kKsplitEnv, (size_k / 16) / 16));

  // VT_EXL3_GEMV_FOLD_OUT=0 keeps the output transform as its own HadK
  // launch — the A/B and test lever for the epilogue fold.
  static const bool kFoldEnv = EnvBoolStrict("VT_EXL3_GEMV_FOLD_OUT", true);
  // A k-split C is complete only after every slice's atomicAdd, which no
  // per-stripe counter of 128/cols arrivals can see.
  const bool fold_ok = fold != nullptr && kFoldEnv && ksplit == 1 &&
                       fold->out != nullptr && fold->post != nullptr &&
                       fold->cnt != nullptr && size_n % 128 == 0;

  const void* kernel;
  int block_dim, cols;
  bool m1_direct = false;
  if (mmode == 1 && args.bits == 3 && args.codebook == 2 && !wmma &&
      !smem && kCfgEnv < 0) {
    // Batched-row arm: m=2..8 decode on 3bpw. The fragment arm re-streams the
    // trellis per row group; Exl3GemvMK3 decodes each tile once and amortizes
    // it across all m rows. Measured ~2.2x the fragment arm at m=8 on the
    // qwen shapes (bench3 own3-m8 vs prod3-m8 rows).
    kernel = GemvMK3KernelFor(size_m);
    block_dim = 512;
    cols = 64;
  } else if (mmode == 0 && !wmma && kCfgEnv < 0 &&
             (int64_t)size_k * size_n * args.bits / 8 <= INT32_MAX) {
    // The bound is M1BufferRsrc's int num_records.
    kernel = GemvM1KernelFor(args.bits, args.codebook, fold_ok, size_k, size_n,
                             ksplit);
    m1_direct = kernel != nullptr;
    block_dim = 512;
    // The wide-load arms' warp owns a four-tile group (64 columns); the narrow
    // dq8 arm covers two tiles (32). The 3 bpw row follows GemvM1K3Wide's
    // choice, so cols and the selected kernel can never disagree.
    cols = (args.bits == 4 || GemvM1K3Wide(args.bits, size_n)) ? 64 : 32;
  } else {
    kernel = cfg == 0 ? narrow
                      : GemvKernelFor(args.bits, args.codebook, mmode, cfg, smem,
                                      wmma);
    block_dim = cfg == 0 ? 512 : 256;
    cols = cfg == 0 ? 32 : 64;
  }

  // Lab diagnostic: VT_EXL3_GEMV_DUMP=1 prints one stderr line per launch with
  // the resolved shape. Off by default; launch sites only.
  static const bool kDump = [] {
    const char* e = std::getenv("VT_EXL3_GEMV_DUMP");
    return e != nullptr && e[0] == '1';
  }();
  if (kDump)
    std::fprintf(stderr,
                 "[exl3_gemv] m=%d k=%d n=%d bits=%d cb=%d cols=%d block=%d\n",
                 size_m, size_k, size_n, args.bits, args.codebook, cols,
                 block_dim);
  if (kernel == nullptr) return false;

  // Lab diagnostic: VT_EXL3_GEMV_OCC=1 prints the resolved occupancy once per
  // (kernel, block) pair alongside the launch line. Off by default.
  static const bool kOcc = [] {
    const char* e = std::getenv("VT_EXL3_GEMV_OCC");
    return e != nullptr && e[0] == '1';
  }();
  const int occ_blocks = GemvOccupancyRocm(kernel, block_dim);
  if (kOcc)
    std::fprintf(stderr,
                 "[exl3_gemv occ] m=%d k=%d n=%d bits=%d cb=%d block=%d occ=%d\n",
                 size_m, size_k, size_n, args.bits, args.codebook, block_dim,
                 occ_blocks);

  // 0 blocks/SM means the kernel's LDS footprint can never launch on this
  // part — decline to the transcription rather than fail in CheckG.
  if (occ_blocks < 1) return false;
  const int max_blocks = occ_blocks * num_sms;
  const int grid = GemvGridForGroups(size_n / cols, max_blocks);
  if (grid < 1) return false;

  if (ksplit > 1)
    CheckG(hipMemsetAsync(raw, 0, (size_t)size_m * size_n * sizeof(float), st),
           "exl3_gemv ksplit zero");
  const half* a_ptr = a_had.Ptr<half>();
  const uint16_t* b_ptr = trellis.Ptr<uint16_t>();
  // Normal launch, not hipLaunchCooperativeKernel — adaptation 1. The kernel
  // pointer table types it as void*, so dispatch by cfg size.
  // The m=1 direct arms take the four fold arguments after the shared six
  // (unused in their non-fold twins); the runtime packs only as many entries
  // as the selected kernel declares, so the six-argument arms ignore them.
  void* f_out = fold_ok ? fold->out : nullptr;
  const uint16_t* f_post = fold_ok ? fold->post : nullptr;
  uint32_t* f_cnt = fold_ok ? fold->cnt : nullptr;
  float f_scale = fold_ok ? fold->r_scale : 0.0f;
  void* kargs[] = {static_cast<void*>(&a_ptr),  static_cast<void*>(&b_ptr),
                   static_cast<void*>(&raw),    static_cast<void*>(&size_m),
                   static_cast<void*>(&size_k), static_cast<void*>(&size_n),
                   static_cast<void*>(&f_out),  static_cast<void*>(&f_post),
                   static_cast<void*>(&f_cnt),  static_cast<void*>(&f_scale)};
  CheckG(hipLaunchKernel(kernel,
                         dim3(static_cast<unsigned>(grid),
                              static_cast<unsigned>(ksplit)),
                         dim3(static_cast<unsigned>(block_dim)), kargs, 0, st),
         "exl3_gemv launch");
  CheckG(hipGetLastError(), "exl3_gemv launch");
  if (fold_ok && m1_direct) fold->applied = true;
  return true;
}
// Spec gfx1100-exl3-decode-60tps: the fused in-had m=1 arm. Runs the
// HadBlock128ToH2 staging phase (bit-identical to HadK<true,true>) inside the
// GEMV kernel, so the caller must NOT launch the separate input hadamard when
// this returns true. Declines on the same terms Exl3GemvTryLaunchRocm does,
// plus m != 1, an uninstantiated fused (bits, cb), a null suh (the kernel
// dereferences it), or size_k not a multiple of 128 (the had block size).
bool Exl3GemvM1FusedTryLaunchRocm(hipStream_t st, const void* a_raw,
                                  const uint16_t* suh_ptr,
                                  const Tensor& trellis, float* raw,
                                  int size_k, int size_n, int device,
                                  const Exl3GemmArgs& args, bool a_bf16) {
  if (args.force_gemv == 0) return false;
  // Same hard gates as the unfused arm — n%128 included. Without it a
  // shape like n%64==0 && n%128!=0 was accepted and the group loop left
  // trailing columns of C stale (reviewer finding on 1d04d72dc).
  if (!Exl3GemvHardEligible(1, size_k, size_n, args.bits, args.codebook,
                            /*has_su_sv=*/true, /*half=*/args.half))
    return false;
  // The m<=8 GEMV decline above names the reason; this fused arm refuses the
  // frac rates by the same token (no frac GEMV arm, spec ## Owed).
  if (args.half) return false;
  if (suh_ptr == nullptr || a_raw == nullptr) return false;
  const int mode = args.force_gemv > 0 ? 2 : Exl3GemvMode();
  if (mode == 0) return false;
  // The fused arm loses to unfused+separate-Had below ~12288 columns on every
  // serving shape measured on gfx1100 (bench3.hip: 4bpw down/attn-z/o-proj/
  // attn-gate run unfused at 570-670 GB/s vs fused 470-530 — the in-LDS had
  // staging amortizes only when the group loop is long), and at 3 bpw fused
  // loses everywhere (~330-390 vs ~440-520 GB/s). Only 4bpw at n>=12288 —
  // qkvo and gateup — keeps the fused arm; everything else falls through to
  // Exl3GemvTryLaunchRocm plus the separate Had.
  if (args.bits != 4 || size_n < 12288) return false;
  if (!GemvArmInstantiated(args.bits, args.codebook)) return false;
  // Default OFF since the unfused M1K4 took buffer loads, PF=2 and the
  // hardware fp16 codec: unfused + HadK now beats this arm on both shapes it
  // served (gemvbench attn_q 56.3 -> 50.5 us, gate/up_4 78.7 -> 70.1 us;
  // served tg@32 38.23 -> 39.02 tok/s, same binary, output byte-identical).
  // The arm is co-residency capped (one a_had staging per block), which
  // leaves a tail on 192/272-group shapes. VT_EXL3_FUSED_HAD=1 restores it.
  static const int kFuseHadEnv = EnvIntStrict("VT_EXL3_FUSED_HAD", 0);
  if (kFuseHadEnv != 1) return false;
  if (GemvWmmaMode()) return false;  // fused arm is the dot decode, not WMMA

  int warp_size = 0;
  CheckG(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device),
         "warp size query");
  if (warp_size != 32) return false;
  int num_sms = 0;
  CheckG(hipDeviceGetAttribute(&num_sms, hipDeviceAttributeMultiprocessorCount, device),
         "multiprocessor count");
  if (num_sms < 1) num_sms = 1;

  const void* kernel = a_bf16 ? GemvM1FusedKernelFor<true>(args.bits, args.codebook)
                              : GemvM1FusedKernelFor<false>(args.bits, args.codebook);
  if (kernel == nullptr) return false;
  const int cols = (args.bits == 4) ? 64 : 32;

  // Dynamic LDS is the whole a_had row: size_k halves. Above the 48 KiB
  // hipLaunchKernel limit the arm declines (bigger-k models keep the
  // separate-had path); hipFuncSetAttributeOpt-in is a later row.
  const size_t shmem = static_cast<size_t>(size_k) * sizeof(uint16_t);
  if (shmem > 48 * 1024) return false;

  int max_blocks = GemvOccupancyRocm(kernel, 512, shmem);
  if (max_blocks < 1) max_blocks = 1;
  // Capped at co-residency regardless of VT_EXL3_GEMV_GRID: every block
  // re-stages the whole a_had row in LDS, so extra blocks add staging work
  // (measured 65 -> 75 us/call on the 4 bpw gate/up shapes uncapped).
  const int groups = size_n / cols;
  const int grid = std::min(groups, max_blocks * num_sms);
  if (grid < 1) return false;

  static const int kKsplitEnv = EnvIntStrict("VT_EXL3_GEMV_KSPLIT", 1);
  const int ksplit = std::max(1, std::min(kKsplitEnv, (size_k / 16) / 16));
  if (ksplit > 1)
    CheckG(hipMemsetAsync(raw, 0, (size_t)size_n * sizeof(float), st),
           "exl3_gemv fused ksplit zero");


  static const bool kDumpF = [] {
    const char* e = std::getenv("VT_EXL3_GEMV_DUMP");
    return e != nullptr && e[0] == '1';
  }();
  if (kDumpF)
    std::fprintf(stderr,
                 "[exl3_gemv] fused m=1 k=%d n=%d bits=%d cb=%d grid=%d lds=%zu\n",
                 size_k, size_n, args.bits, args.codebook, grid, shmem);
  const void* a_ptr = a_raw;
  const uint16_t* b_ptr = trellis.Ptr<uint16_t>();
  int size_m = 1;
  void* kargs[] = {static_cast<void*>(&a_ptr),  static_cast<void*>(&suh_ptr),
                   static_cast<void*>(&b_ptr),  static_cast<void*>(&raw),
                   static_cast<void*>(&size_m), static_cast<void*>(&size_k),
                   static_cast<void*>(&size_n)};
  CheckG(hipLaunchKernel(kernel,
                         dim3(static_cast<unsigned>(grid),
                              static_cast<unsigned>(ksplit)),
                         dim3(512), kargs, shmem, st),
         "exl3_gemv fused-had launch");
  CheckG(hipGetLastError(), "exl3_gemv fused-had launch");
  return true;
}

}  // namespace vt::rocm

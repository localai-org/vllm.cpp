// CUDA keep-quant GGUF k-quant GEMM (MMVQ-style) — the kCUDA provider for
// `OpId::kMatmulBTQuant` (QUANT-GGUF-CIQ-GEMM-CUDA). Runs the DeepSeek-V4 routed
// experts / MLA projections ON THE GPU with the weights kept COMPRESSED in the
// unified pool (no bf16 expansion): quantize the activation tile to Q8_K on the
// GPU, then integer-dot it against the compressed k-quant weight blocks
// (dequant-in-kernel via the block scales / codebook), exactly as vLLM /
// llama.cpp / ds4 do.
//
// GROUNDING (AGENTS.md: mirror the source, cite file:line on both sides).
// This is NOT a copy of llama.cpp's CUDA MMQ/MMVQ path: that path quantizes the
// activation to Q8_1 (32-wide) and uses its own Q8_1-based `vec_dot`s
// (ggml-cuda/mmvq.cu + vecdotq.cuh), so it would NOT reproduce OUR landed CPU
// keep-quant reference, which follows ggml's CPU tier (Q8_K activation). The
// ORACLE this kernel must match is our CPU `kMatmulBTQuant`:
//   src/vt/cpu/cpu_quant_gemm.cpp        MatmulBTQuantKernel   (the GEMM wiring)
//   src/vt/cpu/cpu_quant_dot.cpp         VecDot{Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,
//                                        IQ2_XXS,IQ3_XXS}Q8_K  (the per-block dot)
//   src/vt/cpu/cpu_quant_act.cpp         QuantizeRowQ8_K       (the activation quant)
// which are themselves 1:1 ports of llama.cpp @ 237ad9b96
//   ggml/src/ggml-cpu/quants.c:514/:566/:645/:720/:800/:855/:999  (the vec_dots)
//   ggml/src/ggml-quants.c:2696                                    (quantize_row_q8_K)
//   ggml/src/ggml-cpu/ggml-cpu.c:1245-1443                         (mul_mat wiring)
// The device numeric helpers (DF16ToF32 / DBF16ToF32 / DF32ToBF16 / DNearestInt)
// are bit-exact ports of src/vt/dtype.cpp + cpu_quant_act.cpp so the Q8_K
// activation bytes — and therefore the whole INTEGER dot — are IDENTICAL to the
// CPU reference. Only the per-super-block float scale sum is reassociated (warp
// reduction vs the CPU's sequential add), so the gate is: INTEGER core bit-exact,
// final scale within the same NMSE band `test_ops_quant_dot` uses (5e-4).
//
// COVERAGE. The eight Q8_K-family encodings (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
// IQ2_XXS, IQ3_XXS, IQ2_S — all dot against a Q8_K activation) run natively on
// the GPU; DeepSeek-V4's experts are IQ2_XXS / IQ3_XXS / Q2_K (UD-IQ2_XXS) and
// IQ2_S (UD-IQ2_M gate/up). The Q8_0-ACTIVATION encodings dot a 32-element block
// against a BlockQ8_0, not the 256-element Q8_K super-block this templated GEMM
// quantizes to, so they run on a SECOND templated GEMM beside it: Q8_0 on its own
// tuned path (MatmulQ8_0Cuda), and IQ4_NL / Q5_0 / Q4_0 on the 32-block lane
// (IsCuda32BlockKeepQuantSupported). MXFP4 (the UD-IQ2_M ffn_down) is the one
// encoding still falling back to the CPU keep-quant kernel over the same tensors
// — correct, host-speed, and NOT graph-capturable. DotMXFP4 has its device math
// ready and the 32-block lane is the seam it needs; owed, not box-deferred.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_legacy_intrinsics.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/cpu/cpu_quant_blocks.h"        // vt::cpu::Block* struct mirror (single source)
#include "vt/cuda/cuda_embedding_quant.h"   // EmbeddingQuantErr + the KGATHER seam
#include "vt/cuda/cuda_iq_table_seal.h"      // IqTableSnapshot (the device-codebook drift seal)
#include "vt/cuda/cuda_quant_iq_tables.cuh"  // d_iq2xxs_grid / d_iq3xxs_grid / d_ksigns / d_kmask
#include "vt/cuda/graph_safe_scratch.h"      // RetireGraphScratch (cudagraph-safe grow-only)
#include "vt/ops.h"
#include "cuda_device_caps.h"
#include "vt/quant.h"

namespace vt::cuda {
namespace {

using vt::cpu::BlockIQ1_S;
using vt::cpu::BlockIQ1_XXXS;
using vt::cpu::BlockIQ2_S;
using vt::cpu::BlockIQ2_XS;
using vt::cpu::BlockIQ2_XXS;
using vt::cpu::BlockIQ3_S;
using vt::cpu::BlockIQ3_XXS;
using vt::cpu::BlockIQ4_NL;
using vt::cpu::BlockIQ4_XS;
using vt::cpu::BlockMXFP4;
using vt::cpu::BlockQ2_K;
using vt::cpu::BlockQ3_K;
using vt::cpu::BlockQ4_0;
using vt::cpu::BlockQ4_K;
using vt::cpu::BlockQ5_0;
using vt::cpu::BlockQ5_K;
using vt::cpu::BlockQ6_K;
using vt::cpu::BlockQ8_0;
using vt::cpu::BlockQ8_K;
using vt::cpu::kQK4_NL;    // 32
using vt::cpu::kQK8_0;     // 32
using vt::cpu::kQK_K;      // 256
using vt::cpu::kQK_MXFP4;  // 32

void CheckCuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: matmul_bt_quant: ") + what +
                             ": " + cudaGetErrorString(err));
  }
}

// --- device numeric helpers — bit-exact ports of src/vt/dtype.cpp -------------
__device__ inline float DF16ToF32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0x1F) return __int_as_float(sign | 0x7F800000 | (mant << 13));
  if (exp == 0) {
    if (mant == 0) return __int_as_float(sign);
    int shift = 0;
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      ++shift;
    }
    mant &= 0x3FF;
    return __int_as_float(sign | ((113 - shift) << 23) | (mant << 13));
  }
  return __int_as_float(sign | ((exp + 112) << 23) | (mant << 13));
}

__device__ inline float DBF16ToF32(uint16_t b) {
  return __int_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ inline uint16_t DF32ToBF16(float f) {
  uint32_t u = __float_as_int(f);
  if ((u & 0x7F800000) == 0x7F800000 && (u & 0x7FFFFF)) {
    return static_cast<uint16_t>((u >> 16) | 0x0040);
  }
  uint32_t rounding = 0x7FFF + ((u >> 16) & 1);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

// dtype.cpp F32ToF16 — bit-exact port (round-to-nearest-even, subnormals, inf/nan).
// Used only for the Q8_0-activation scale `y.d` (the round-trip F32ToF16→F16ToF32 the
// CPU Q8_0 vec_dot applies); the integer core is scale-independent, so the whole Q8_0
// INTEGER dot stays bit-identical to the CPU reference.
__device__ inline uint16_t DF32ToF16(float f) {
  uint32_t u = __float_as_uint(f);
  uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000);
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = u & 0x7FFFFF;
  if (((u >> 23) & 0xFF) == 0xFF)
    return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 | (mant >> 13) : 0));
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00);
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = static_cast<uint32_t>(exp << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
  return static_cast<uint16_t>(sign | half);
}

// cpu_quant_iq_tables.h E8M0ToF32Half (ggml-impl.h:477) — bit-exact port. Decodes
// an MXFP4 E8M0 block scale to 2^(byte-128); pairs with d_kvalues_mxfp4 (= 2*e2m1).
__device__ inline float DE8M0ToF32Half(uint8_t x) {
  const uint32_t bits =
      x < 2 ? (0x00200000u << x) : (static_cast<uint32_t>(x - 1) << 23);
  return __int_as_float(static_cast<int>(bits));
}

// cpu_quant_act.cpp NearestInt (ggml-quants.c:563) — magic-constant round-to-even.
__device__ inline int DNearestInt(float fval) {
  float val = fval + 12582912.0f;
  int i = __float_as_int(val);
  return (i & 0x007fffff) - 0x00400000;
}

// Load one activation element (dtype-decoded, exactly like cpu LoadActF32).
enum class ActDT : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

__device__ inline float DLoadAct(const void* base, ActDT dt, int64_t idx) {
  switch (dt) {
    case ActDT::kF32: return static_cast<const float*>(base)[idx];
    case ActDT::kF16: return DF16ToF32(static_cast<const uint16_t*>(base)[idx]);
    default: return DBF16ToF32(static_cast<const uint16_t*>(base)[idx]);
  }
}

// The dequantizing gather's device row decoders (KGATHER). Included HERE, inside
// `vt::cuda::{anonymous}`, because it uses the DF16ToF32/DF32ToBF16/DE8M0ToF32Half
// helpers above and the device codebooks below, and because those codebooks are
// DEFINED by cuda_quant_iq_tables.cuh -- a second including TU is a duplicate
// symbol, not a second copy. It opens no namespace of its own.
#include "vt/cuda/cuda_quant_dequant.cuh"

// FP16 gather output is distinct from the existing uint16_t BF16 carrier.
template <> struct DqStore<__half> {
  __device__ static void Set(__half* p, int64_t i, float v) { p[i] = __float2half_rn(v); }
};

// The block geometry the decoders hard-code, tied to the SINGLE source of block
// layout (cpu_quant_blocks.h, itself static_asserted against ggml-common.h). A
// stride typed by hand into a decoder above is a compile error here, not a row
// of plausible garbage at the first forward.
static_assert(DqQ4_0::kBytes == sizeof(vt::cpu::BlockQ4_0), "q4_0 gather stride");
static_assert(DqQ5_0::kBytes == sizeof(vt::cpu::BlockQ5_0), "q5_0 gather stride");
static_assert(DqQ8_0::kBytes == sizeof(vt::cpu::BlockQ8_0), "q8_0 gather stride");
static_assert(DqIQ4_NL::kBytes == sizeof(vt::cpu::BlockIQ4_NL), "iq4_nl gather stride");
static_assert(DqMXFP4::kBytes == sizeof(vt::cpu::BlockMXFP4), "mxfp4 gather stride");
static_assert(DqQ2_K::kBytes == sizeof(vt::cpu::BlockQ2_K), "q2_K gather stride");
static_assert(DqQ3_K::kBytes == sizeof(vt::cpu::BlockQ3_K), "q3_K gather stride");
static_assert(DqQ4_K::kBytes == sizeof(vt::cpu::BlockQ4_K), "q4_K gather stride");
static_assert(DqQ5_K::kBytes == sizeof(vt::cpu::BlockQ5_K), "q5_K gather stride");
static_assert(DqQ6_K::kBytes == sizeof(vt::cpu::BlockQ6_K), "q6_K gather stride");
static_assert(DqQ8_K::kBytes == sizeof(vt::cpu::BlockQ8_K), "q8_K gather stride");
static_assert(DqIQ2_XXS::kBytes == sizeof(vt::cpu::BlockIQ2_XXS), "iq2_xxs gather stride");
static_assert(DqIQ3_XXS::kBytes == sizeof(vt::cpu::BlockIQ3_XXS), "iq3_xxs gather stride");
static_assert(DqIQ2_XS::kBytes == sizeof(vt::cpu::BlockIQ2_XS), "iq2_xs gather stride");
static_assert(DqIQ2_S::kBytes == sizeof(vt::cpu::BlockIQ2_S), "iq2_s gather stride");
static_assert(DqIQ4_XS::kBytes == sizeof(vt::cpu::BlockIQ4_XS), "iq4_xs gather stride");
static_assert(DqIQ3_S::kBytes == sizeof(vt::cpu::BlockIQ3_S), "iq3_s gather stride");
static_assert(DqIQ1_S::kBytes == sizeof(vt::cpu::BlockIQ1_S), "iq1_s gather stride");
static_assert(DqIQ1_XXXS::kBytes == sizeof(vt::cpu::BlockIQ1_XXXS), "iq1_xxxs gather stride");

// One row of the dispatch: the vt DType, its codec, and nothing else. Every
// list below is generated from this macro so a dtype cannot be decodable in one
// place and refused in another -- the asymmetry that made the CPU-only gather a
// silent load-time expansion instead of a loud refusal.
#define VT_DQ_GATHER_TYPES(X)     \
  X(kQ4_0, DqQ4_0)                \
  X(kQ5_0, DqQ5_0)                \
  X(kQ8_0, DqQ8_0)                \
  X(kQ2_K, DqQ2_K)                \
  X(kQ3_K, DqQ3_K)                \
  X(kQ4_K, DqQ4_K)                \
  X(kQ5_K, DqQ5_K)                \
  X(kQ6_K, DqQ6_K)                \
  X(kQ8_K, DqQ8_K)                \
  X(kIQ2_XXS, DqIQ2_XXS)          \
  X(kIQ3_XXS, DqIQ3_XXS)          \
  X(kIQ2_S, DqIQ2_S)              \
  X(kIQ2_XS, DqIQ2_XS)            \
  X(kIQ1_S, DqIQ1_S)              \
  X(kIQ1_XXXS, DqIQ1_XXXS)        \
  X(kIQ4_NL, DqIQ4_NL)            \
  X(kIQ4_XS, DqIQ4_XS)            \
  X(kIQ3_S, DqIQ3_S)              \
  X(kMXFP4, DqMXFP4)

template <typename Tout, typename Tid>
cudaError_t LaunchEmbeddingQuantTyped(cudaStream_t s, Tout* out, const uint8_t* table,
                                      const Tid* ids, DType dt, int64_t t, int64_t nb,
                                      int64_t h, int64_t v, size_t row_bytes,
                                      EmbeddingQuantErr* err) {
  constexpr int kGatherBlock = 128;
  const int64_t total = t * nb;
  int64_t blocks = (total + kGatherBlock - 1) / kGatherBlock;
  if (blocks > 65535) blocks = 65535;  // grid-stride loop covers the remainder
  const unsigned grid = static_cast<unsigned>(blocks < 1 ? 1 : blocks);
  switch (dt) {
#define VT_DQ_LAUNCH(DT, CODEC)                                                   \
  case DType::DT:                                                                 \
    EmbeddingQuantGatherKernel<Tout, Tid, CODEC>                                   \
        <<<grid, kGatherBlock, 0, s>>>(out, table, ids, t, nb, h, v, row_bytes, err); \
    break;
    VT_DQ_GATHER_TYPES(VT_DQ_LAUNCH)
#undef VT_DQ_LAUNCH
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// GPU activation quantizer — one thread per Q8_K super-block (256 elements).
// Bit-exact port of QuantizeRowQ8_K (cpu_quant_act.cpp / ggml-quants.c:2696).
// Scratch layout: row i is `nsb` contiguous BlockQ8_K; block (i,sb) sits at
// scratch[(i*nsb + sb)]. The per-row activation stride `a_rs` is in ELEMENTS.
// ---------------------------------------------------------------------------
__global__ void QuantizeQ8KKernel(BlockQ8_K* __restrict__ scratch,
                                  const void* __restrict__ a, ActDT adt,
                                  int64_t a_rs, int64_t m, int64_t nsb) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = m * nsb;
  if (t >= total) return;
  const int64_t i = t / nsb;   // activation row
  const int64_t sb = t % nsb;  // super-block within the row
  const int64_t elem0 = i * a_rs + sb * kQK_K;

  float mx = 0.0f;
  float amax = 0.0f;
  for (int j = 0; j < kQK_K; ++j) {
    const float ax = fabsf(DLoadAct(a, adt, elem0 + j));
    if (ax > amax) {
      amax = ax;
      mx = DLoadAct(a, adt, elem0 + j);
    }
  }
  BlockQ8_K& y = scratch[t];
  if (amax == 0.0f) {
    y.d = 0.0f;
    for (int j = 0; j < kQK_K; ++j) y.qs[j] = 0;
    for (int g = 0; g < kQK_K / 16; ++g) y.bsums[g] = 0;
    return;
  }
  const float iscale = -127.0f / mx;
  for (int j = 0; j < kQK_K; ++j) {
    const int v = DNearestInt(iscale * DLoadAct(a, adt, elem0 + j));
    y.qs[j] = static_cast<int8_t>(v < 127 ? v : 127);
  }
  for (int g = 0; g < kQK_K / 16; ++g) {
    int sum = 0;
    for (int ii = 0; ii < 16; ++ii) sum += y.qs[g * 16 + ii];
    y.bsums[g] = static_cast<int16_t>(sum);
  }
  y.d = 1.0f / iscale;
}

// ---------------------------------------------------------------------------
// ds4-parity Q8_K activation quantizer — ONE BLOCK per (super-block, row), 256
// threads (one thread per element) with a shared-memory reduction. Port of the
// GRID GEOMETRY of ds4 `q8_K_quantize_kernel` (ds4_cuda.cu:16627): grid=(nsb,m).
// The legacy QuantizeQ8KKernel above maps one THREAD to a whole 256-element
// super-block, so a decode grouped-MoE activation quant (Pa∈{1,P}, nsb≈8–28)
// launches a SINGLE 128-thread block with ≤28 active threads on a >100-SM
// device — occupancy/latency-bound (nsys: ~166 ms whole-run of the ds4flash
// decode). This kernel launches `nsb·m` independent 256-thread blocks (one per
// super-block), spreading the tiny per-step quant across the SMs exactly like
// ds4 does. NOTE: this is NOT the Brick-8 GEMM-prologue fusion (fold the quant
// into the grouped dot) — that re-quantized the SAME row in every one of the
// thousands of GEMM blocks and regressed −22%; ds4 itself keeps the Q8_K quant
// a SEPARATE stage (ds4_cuda.cu:25951 "Stage 1: quantize x rows to q8_K"). The
// activation is still quantized exactly ONCE into scratch; only the quant
// kernel's thread→work map changes.
// BYTE-IDENTICAL to QuantizeQ8KKernel: the amax reduction carries the ORIGINAL
// element index and breaks ties by LOWEST index — reproducing the legacy
// sequential `if (ax > amax)` FIRST-occurrence rule, so the signed `mx` (hence
// iscale's sign) is identical even when two elements share |x| with opposite
// sign. iscale=-127/mx, qs=DNearestInt(iscale·x) upper-clamped to 127, bsums,
// and d=1/iscale are unchanged. (Asserted byte-exact in test_cuda_quant_dot.)
// ---------------------------------------------------------------------------
__global__ void QuantizeQ8KPreqKernel(BlockQ8_K* __restrict__ scratch,
                                      const void* __restrict__ a, ActDT adt,
                                      int64_t a_rs, int64_t m, int64_t nsb) {
  const int64_t b = static_cast<int64_t>(blockIdx.x);  // super-block within the row
  const int64_t i = static_cast<int64_t>(blockIdx.y);  // activation row
  if (b >= nsb || i >= m) return;
  const int tid = static_cast<int>(threadIdx.x);  // 0..255, one element per thread
  const int64_t elem0 = i * a_rs + b * kQK_K;
  const float v = DLoadAct(a, adt, elem0 + tid);

  __shared__ float sabs[kQK_K];
  __shared__ float sval[kQK_K];
  __shared__ int sidx[kQK_K];
  sabs[tid] = fabsf(v);
  sval[tid] = v;
  sidx[tid] = tid;
  __syncthreads();
  // Reduce for the argmax-|x|: take the larger |x|, and on an EXACT tie keep the
  // LOWEST original index (== legacy first-occurrence scan). sval[0] = signed mx.
#pragma unroll
  for (int stride = kQK_K >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float oa = sabs[tid + stride];
      if (oa > sabs[tid] || (oa == sabs[tid] && sidx[tid + stride] < sidx[tid])) {
        sabs[tid] = oa;
        sval[tid] = sval[tid + stride];
        sidx[tid] = sidx[tid + stride];
      }
    }
    __syncthreads();
  }

  BlockQ8_K& y = scratch[i * nsb + b];
  const float amax = sabs[0];
  if (amax == 0.0f) {
    if (tid == 0) y.d = 0.0f;
    y.qs[tid] = 0;
    if (tid < kQK_K / 16) y.bsums[tid] = 0;
    return;
  }
  const float iscale = -127.0f / sval[0];
  const int q = DNearestInt(iscale * v);
  y.qs[tid] = static_cast<int8_t>(q < 127 ? q : 127);
  __syncthreads();  // all 256 qs written (block-scoped global fence) before bsums read
  if (tid < kQK_K / 16) {
    int sum = 0;
#pragma unroll
    for (int ii = 0; ii < 16; ++ii) sum += y.qs[tid * 16 + ii];
    y.bsums[tid] = static_cast<int16_t>(sum);
  }
  if (tid == 0) y.d = 1.0f / iscale;
}

// ---------------------------------------------------------------------------
// Per-super-block dot kernels. Each returns the block's float contribution with
// the INTEGER core computed bit-identically to the CPU vec_dot; the type's final
// constant magnitude factor (0.125 iq2 / 0.25 iq3 / 1 otherwise) is folded once
// at the very end (matching the CPU `*s = factor * sumf`).
// ---------------------------------------------------------------------------

// cpu_quant_dot.cpp VecDotIQ2_XXSQ8_K (quants.c:855) — one super-block.
// Brick 1 (last-mile): __dp4a vectorized-dequant matvec, ported from llama.cpp
// ggml-cuda/vecdotq.cuh:920-928 (`vec_dot_iq2_xxs_q8_1`) + ds4 `dev_iq2_dp4a_8`
// (ds4_cuda.cu:16147). The per-element `g*q8[j]*sign` branch → SIMD sign-apply
// (__vcmpne4/__vsub4) + `__dp4a` (4 int8 products/instr). BIT-IDENTICAL integer core:
// __dp4a is an EXACT int32 accumulation of int8 products (order-independent), the
// grid bytes are ≤~43 so ±g fits int8, and d_kmask_iq2xs[j]==1<<j so the packed
// sign masks 0x08040201 / 0x80402010 select the same per-byte sign as the scalar.
// The per-block ls fold + final *0.125 (after the warp reduction) are unchanged.
__device__ inline float DotIQ2XXS(const BlockIQ2_XXS* xb, const BlockQ8_K* yb) {
  const float d = DF16ToF32(xb->d) * yb->d;
  const uint16_t* qs = xb->qs;
  const int8_t* q8 = yb->qs;
  int32_t bsum = 0;
  for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
    const uint32_t a0 = static_cast<uint32_t>(qs[4 * ib32 + 0]) |
                        (static_cast<uint32_t>(qs[4 * ib32 + 1]) << 16);
    const uint32_t a1 = static_cast<uint32_t>(qs[4 * ib32 + 2]) |
                        (static_cast<uint32_t>(qs[4 * ib32 + 3]) << 16);
    const uint32_t ls = 2 * (a1 >> 28) + 1;
    int32_t sumi = 0;
    for (int l = 0; l < 4; ++l) {
      const uint32_t* grid =
          reinterpret_cast<const uint32_t*>(&d_iq2xxs_grid[(a0 >> (8 * l)) & 0xff]);
      // Broadcast the 8-bit sign pattern to 4 bytes (UNSIGNED — signed *0x01010101
      // overflows int for signs≥128 = UB), then per-byte 0xff mask where bit b/b+4 set.
      const unsigned sbc =
          static_cast<unsigned>(d_ksigns_iq2xs[(a1 >> (7 * l)) & 127]) * 0x01010101u;
      const int slo = __vcmpne4(static_cast<int>(sbc & 0x08040201u), 0);  // bytes 0-3
      const int shi = __vcmpne4(static_cast<int>(sbc & 0x80402010u), 0);  // bytes 4-7
      const int glo = __vsub4(static_cast<int>(grid[0]) ^ slo, slo);  // ±grid bytes 0-3
      const int ghi = __vsub4(static_cast<int>(grid[1]) ^ shi, shi);  // ±grid bytes 4-7
      sumi = LegacyDp4a(glo, *reinterpret_cast<const int*>(q8 + 0), sumi);
      sumi = LegacyDp4a(ghi, *reinterpret_cast<const int*>(q8 + 4), sumi);
      q8 += 8;
    }
    bsum += sumi * static_cast<int32_t>(ls);
  }
  return d * bsum;  // final *0.125 applied after the warp reduction
}

// cpu_quant_dot.cpp VecDotIQ3_XXSQ8_K (quants.c:999) — one super-block.
__device__ inline float DotIQ3XXS(const BlockIQ3_XXS* xb, const BlockQ8_K* yb) {
  const float d = DF16ToF32(xb->d) * yb->d;
  const uint8_t* q3 = xb->qs;
  const uint8_t* gas = xb->qs + kQK_K / 4;
  const int8_t* q8 = yb->qs;
  int32_t bsum = 0;
  for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
    uint32_t a32;
    memcpy(&a32, gas, sizeof(uint32_t));
    gas += sizeof(uint32_t);
    const uint32_t ls = 2 * (a32 >> 28) + 1;
    int32_t sumi = 0;
    for (int l = 0; l < 4; ++l) {
      const uint32_t g1 = d_iq3xxs_grid[q3[2 * l + 0]];
      const uint32_t g2 = d_iq3xxs_grid[q3[2 * l + 1]];
      const uint8_t signs = d_ksigns_iq2xs[(a32 >> (7 * l)) & 127];
      for (int j = 0; j < 4; ++j) {
        const int b1 = static_cast<int>((g1 >> (8 * j)) & 0xff);
        const int b2 = static_cast<int>((g2 >> (8 * j)) & 0xff);
        sumi += b1 * q8[j + 0] * ((signs & d_kmask_iq2xs[j + 0]) ? -1 : 1);
        sumi += b2 * q8[j + 4] * ((signs & d_kmask_iq2xs[j + 4]) ? -1 : 1);
      }
      q8 += 8;
    }
    q3 += 8;
    bsum += sumi * static_cast<int32_t>(ls);
  }
  return d * bsum;  // final *0.25 applied after the warp reduction
}

// cpu_quant_dot.cpp VecDotIQ2_SQ8_K (quants.c:947) — one super-block. Scalar
// port (mirrors DotIQ3XXS's structure) — a __dp4a last-mile pass can follow the
// IQ2_XXS treatment later. 10-bit grid index = qs[l] | qh high 2 bits into
// d_iq2s_grid; the DIRECT sign byte signs[l] (= qs + QK_K/8, NO ksigns lookup)
// flips lanes; per-32 ls = 1 + 2*ls_nibble fold in; final *0.125 after the warp
// reduction. BIT-IDENTICAL integer core to the CPU reference.
// IQ1_S (quants.c:1099) and IQ1_XXXS (fork quants.c:1281), transcribed from the
// CPU kernels they must agree with bit for bit.
//
// Both split into an integer grid dot PLUS a delta term over the activation's
// per-16 sums, which is why they read `bsums` where no other device dot here
// does. The grids are ternary, so there is no sign table and no FinalFactor
// scaling: the whole value is already in the returned sum.
__device__ inline float DotIQ1S(const BlockIQ1_S* xb, const BlockQ8_K* yb) {
  const int8_t* q8 = yb->qs;
  const uint8_t* qs = xb->qs;
  const uint16_t* qh = xb->qh;
  int32_t sumi = 0;
  int32_t sumi1 = 0;
  for (int ib = 0; ib < kQK_K / 32; ++ib) {
    const int ls = 2 * ((qh[ib] >> 12) & 7) + 1;
    const int delta = (qh[ib] & 0x8000) ? -1 : 1;
    int lsum = 0;
    for (int l = 0; l < 4; ++l) {
      const int8_t* grid = reinterpret_cast<const int8_t*>(
          &d_iq1s_grid[qs[l] | (((qh[ib] >> (3 * l)) & 7) << 8)]);
      for (int j = 0; j < 8; ++j) lsum += q8[j] * grid[j];
      q8 += 8;
    }
    sumi += ls * lsum;
    sumi1 += ls * delta * (yb->bsums[2 * ib + 0] + yb->bsums[2 * ib + 1]);
    qs += 4;
  }
  return DF16ToF32(xb->d) * yb->d *
         (static_cast<float>(sumi) + 0.125f * static_cast<float>(sumi1));
}

__device__ inline float DotIQ1XXXS(const BlockIQ1_XXXS* xb, const BlockQ8_K* yb) {
  const int8_t* q8 = yb->qs;
  const uint8_t* qs = xb->qs;
  const uint8_t* sc = xb->sc;
  int32_t sumi = 0;
  int32_t sumi1 = 0;
  for (int ib = 0; ib < kQK_K / 32; ++ib) {
    const int nib = (sc[ib / 2] >> (4 * (ib & 1))) & 0xf;
    const int ls = 2 * (nib & 7) + 1;
    const int delta = (nib & 8) ? -1 : 1;
    int lsum = 0;
    for (int l = 0; l < 4; ++l) {
      const int8_t* grid =
          reinterpret_cast<const int8_t*>(&d_iq1xxxs_grid[qs[l]]);
      for (int j = 0; j < 8; ++j) lsum += q8[j] * grid[j];
      q8 += 8;
    }
    sumi += ls * lsum;
    sumi1 += ls * delta * (yb->bsums[2 * ib + 0] + yb->bsums[2 * ib + 1]);
    qs += 4;
  }
  return DF16ToF32(xb->d) * yb->d *
         (static_cast<float>(sumi) + 0.125f * static_cast<float>(sumi1));
}

__device__ inline float DotIQ2S(const BlockIQ2_S* xb, const BlockQ8_K* yb) {
  const float d = DF16ToF32(xb->d) * yb->d;
  const int8_t* q8 = yb->qs;
  const uint8_t* qs = xb->qs;
  const uint8_t* qh = xb->qh;
  const uint8_t* signs = qs + kQK_K / 8;
  int32_t bsum = 0;
  for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
    const int ls1 = 1 + 2 * (xb->scales[ib32] & 0xf);
    const int ls2 = 1 + 2 * (xb->scales[ib32] >> 4);
    int sumi1 = 0;
    int sumi2 = 0;
    for (int l = 0; l < 2; ++l) {
      const uint8_t* grid = reinterpret_cast<const uint8_t*>(
          &d_iq2s_grid[qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)]);
      for (int j = 0; j < 8; ++j)
        sumi1 += q8[j] * grid[j] * ((signs[l] & d_kmask_iq2xs[j]) ? -1 : 1);
      q8 += 8;
    }
    for (int l = 2; l < 4; ++l) {
      const uint8_t* grid = reinterpret_cast<const uint8_t*>(
          &d_iq2s_grid[qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)]);
      for (int j = 0; j < 8; ++j)
        sumi2 += q8[j] * grid[j] * ((signs[l] & d_kmask_iq2xs[j]) ? -1 : 1);
      q8 += 8;
    }
    bsum += ls1 * sumi1 + ls2 * sumi2;
    qs += 4;
    signs += 4;
  }
  return d * bsum;  // final *0.125 applied after the warp reduction
}

// cpu_quant_dot.cpp VecDotIQ2_XSQ8_K (llama.cpp @ b10451 quants.c:948,
// `ggml_vec_dot_iq2_xs_q8_K_generic`) — one super-block, and the THIRD sign
// convention in the IQ2 family. Each `qs` u16 does two jobs at once: its low 9
// bits index the 512-entry d_iq2xs_grid (`& 511`) and its high 7 bits select a
// d_ksigns_iq2xs byte (`>> 9`). IQ2_XXS keeps the signs in a separate u32 and
// IQ2_S in a direct sign byte, so no part of either sibling's decode transfers.
//
// The two halves of a sub-block take DIFFERENT scales — `ls1` from the low
// nibble of `scales[ib32]`, `ls2` from the high one — so upstream folds `sumi`
// into `bsum` TWICE per sub-block rather than once. That split is the
// accumulation ORDER and is kept verbatim. The final 0.125 folds the grid's
// fixed 8x magnitude and is applied ONCE after the warp reduction, via
// FinalFactor<kIQ2_XS>, exactly as for IQ2_XXS and IQ2_S.
//
// Alignment: `qs` is a u16 array at offset 2 of a 74-byte block, and every
// block address in the GEMM is a 256-B-aligned base plus a multiple of 74, so
// each u16 read is 2-byte aligned. No 4-byte __dp4a read is taken here.
__device__ inline float DotIQ2XS(const BlockIQ2_XS* xb, const BlockQ8_K* yb) {
  const float d = DF16ToF32(xb->d) * yb->d;
  const uint16_t* q2 = xb->qs;
  const uint8_t* sc = xb->scales;
  const int8_t* q8 = yb->qs;
  int32_t bsum = 0;
  for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
    const int ls1 = 2 * (sc[ib32] & 0xf) + 1;
    const int ls2 = 2 * (sc[ib32] >> 4) + 1;
    int32_t sumi = 0;
    for (int l = 0; l < 2; ++l) {
      const uint8_t* grid =
          reinterpret_cast<const uint8_t*>(&d_iq2xs_grid[q2[l] & 511]);
      const uint8_t signs = d_ksigns_iq2xs[q2[l] >> 9];
      for (int j = 0; j < 8; ++j)
        sumi += grid[j] * q8[j] * ((signs & d_kmask_iq2xs[j]) ? -1 : 1);
      q8 += 8;
    }
    bsum += sumi * ls1;
    sumi = 0;
    for (int l = 2; l < 4; ++l) {
      const uint8_t* grid =
          reinterpret_cast<const uint8_t*>(&d_iq2xs_grid[q2[l] & 511]);
      const uint8_t signs = d_ksigns_iq2xs[q2[l] >> 9];
      for (int j = 0; j < 8; ++j)
        sumi += grid[j] * q8[j] * ((signs & d_kmask_iq2xs[j]) ? -1 : 1);
      q8 += 8;
    }
    bsum += sumi * ls2;
    q2 += 4;
  }
  return d * bsum;  // final *0.125 applied after the warp reduction
}

// cpu_quant_dot.cpp VecDotIQ4_XSQ8_K (llama.cpp @ b10451 quants.c:1283,
// `ggml_vec_dot_iq4_xs_q8_K_generic`). IQ4_XS shares IQ4_NL's 16-entry
// d_kvalues_iq4nl codebook and NOT its activation pairing: its block is a
// 256-element SUPER-block, so it dots Q8_K where IQ4_NL dots Q8_0
// (type_traits_cpu at ggml-cpu.c:385-390 against :379-384). That is read off the
// oracle, not inferred from the shared codebook, and getting it wrong is how
// this encoding has been mis-ported before.
//
// The `ib` loop steps by TWO because one `scales_l` byte serves two sub-blocks
// and `h` is consumed 4 bits in the same step. UNLIKE every other dot in this
// file there is no single integer core: `d1`/`d2` are formed as f32 BEFORE the
// integer sums fold in, and each sub-block contributes its own `sumf +=`, so a
// super-block is EIGHT f32 accumulation steps. That association is upstream's,
// it is what the pinned oracle's golden numbers were produced with, and it is
// kept rather than hoisted into one integer accumulator — which would be a
// faster kernel computing a different number. FinalFactor<kIQ4_XS> is 1.
//
// ─── WHY __fmul_rn/__fadd_rn AND NOT `sumf += d1 * x` ───────────────────────
// Because nvcc CONTRACTS that expression into an FMA (-fmad=true is the
// default), and an FMA rounds ONCE where upstream rounds twice. That is not a
// tolerance question, it is the association above being silently replaced: the
// paragraph promising upstream's eight accumulation steps was true of the
// source and false of the SASS.
//
// MEASURED, not feared. The first sm_121a run of this kernel disagreed with the
// pinned oracle on exactly two of eight real checkpoint super-blocks, by 1 and
// 4 ULP, while IQ2_XS was bit-exact everywhere — the split an FMA predicts,
// because IQ2_XS's core is integer with a single final multiply and has nothing
// to contract. Recomputing those same blocks on the host with `std::fma` at
// this line reproduced the device's bits EXACTLY (0xBFC39E59 and 0x3E173708)
// and left the other two blocks equal to the oracle, which identifies the cause
// with no free parameters left over.
//
// `__fmul_rn` and `__fadd_rn` are the round-to-nearest primitives nvcc is not
// permitted to fuse, so they restore upstream's rounding exactly. Scoped to
// THIS kernel rather than fixed with `-fmad=false` on the translation unit,
// which would change the numerics of ten other encodings that no one measured.
// Q4_K and Q5_K end in `d * isum - dmin * sumi`, which is contractable the same
// way; they are gated against the CPU at NMSE and not bit-exactly, so nothing
// here has ever been able to see it. Recorded in the row's spec, not fixed
// here.
//
// Alignment: `d` and `scales_h` are u16 at offsets 0 and 2 of a 136-byte block,
// so both are 2-byte aligned wherever the GEMM addresses a block.
__device__ inline float DotIQ4XS(const BlockIQ4_XS* xb, const BlockQ8_K* yb) {
  const float d4d8 = DF16ToF32(xb->d) * yb->d;
  uint16_t h = xb->scales_h;
  const uint8_t* qs = xb->qs;
  const int8_t* q8 = yb->qs;
  float sumf = 0.0f;
  for (int ib = 0; ib < kQK_K / 32; ib += 2) {
    const uint8_t ls1 =
        static_cast<uint8_t>((xb->scales_l[ib / 2] & 0xf) | ((h << 4) & 0x30));
    const uint8_t ls2 =
        static_cast<uint8_t>((xb->scales_l[ib / 2] >> 4) | ((h << 2) & 0x30));
    h = static_cast<uint16_t>(h >> 4);
    const float d1 = d4d8 * (ls1 - 32);
    const float d2 = d4d8 * (ls2 - 32);
    int sumi1 = 0;
    int sumi2 = 0;
    for (int j = 0; j < 16; ++j) {
      sumi1 += q8[j + 0] * d_kvalues_iq4nl[qs[j] & 0xf];
      sumi2 += q8[j + 16] * d_kvalues_iq4nl[qs[j] >> 4];
    }
    sumf = __fadd_rn(sumf, __fmul_rn(d1, static_cast<float>(sumi1 + sumi2)));
    qs += 16;
    q8 += 32;
    sumi1 = 0;
    sumi2 = 0;
    for (int j = 0; j < 16; ++j) {
      sumi1 += q8[j + 0] * d_kvalues_iq4nl[qs[j] & 0xf];
      sumi2 += q8[j + 16] * d_kvalues_iq4nl[qs[j] >> 4];
    }
    sumf = __fadd_rn(sumf, __fmul_rn(d2, static_cast<float>(sumi1 + sumi2)));
    qs += 16;
    q8 += 32;
  }
  return sumf;
}

// cpu_quant_dot.cpp VecDotMXFP4Q8_0 (quants.c:247) — one MXFP4 (32-elem) block
// dotted against ONE Q8_0 activation block. BOX-DEFERRED / NOT YET WIRED into a
// GEMM: the templated GEMM below is Q8_K-activation-only (its DotSuperblock takes
// a BlockQ8_K and the K super-block is 256 elements), whereas MXFP4 dots against
// a 32-element Q8_0 activation. Reaching the GPU therefore needs a separate
// Q8_0-activation grouped-MoE GEMM variant (the Q8_0 prologue + a 32-elem block
// walk); until that lands, MXFP4 correctly CPU-fallbacks like Q4_0/Q8_0
// (IsCudaKeepQuantSupported returns false for it). The math here is ready for
// that variant and is BIT-IDENTICAL to the CPU reference's integer core.
// [[maybe_unused]]: intentionally not yet referenced (awaits the Q8_0-activation
// GEMM variant above) — keeps the ready device math without tripping nvcc #177-D
// under -Werror.
[[maybe_unused]] __device__ inline float DotMXFP4(const BlockMXFP4* xb,
                                                  const BlockQ8_0* yb) {
  const float d = DF16ToF32(yb->d) * DE8M0ToF32Half(xb->e);
  int sumi1 = 0;
  int sumi2 = 0;
  for (int j = 0; j < kQK_MXFP4 / 2; ++j) {
    sumi1 += yb->qs[j + 0] * d_kvalues_mxfp4[xb->qs[j] & 0xf];
    sumi2 += yb->qs[j + kQK_MXFP4 / 2] * d_kvalues_mxfp4[xb->qs[j] >> 4];
  }
  return d * (sumi1 + sumi2);
}

// cpu_quant_dot.cpp VecDotQ2_KQ8_K (quants.c:514) — one super-block.
// Brick 1 (last-mile): __dp4a vectorized-dequant, ported from llama.cpp
// ggml-cuda/vecdotq.cuh:329-354 (`vec_dot_q2_K_q8_1`) + ds4 `dev_dot_q2_16`
// (ds4_cuda.cu:16158). The scalar per-element `q8 * ((q2>>shift)&3)` → `__dp4a`
// on the 0x03030303-masked 2-bit packs. BIT-IDENTICAL: `(word>>shift)&0x03030303`
// per byte == `(byte>>shift)&3` (the cross-byte bits land in bits 6-7, masked off),
// __dp4a is exact int32; the summs (min) term + dall/dmin fold are unchanged. All
// int reads are 4-aligned (Q2_K block=84 B ÷4, qs@16 ÷4; the Q8_K activation ÷4).
__device__ inline float DotQ2K(const BlockQ2_K* xb, const BlockQ8_K* yb) {
  const uint8_t* q2 = xb->qs;
  const int8_t* q8 = yb->qs;
  const uint8_t* sc = xb->scales;
  int summs = 0;
  for (int j = 0; j < 16; ++j) summs += yb->bsums[j] * (sc[j] >> 4);
  const float dall = yb->d * DF16ToF32(xb->d);
  const float dmin = yb->d * DF16ToF32(xb->dmin);
  int isum = 0;
  int is = 0;
  for (int k = 0; k < kQK_K / 128; ++k) {
    const uint8_t* q2k = q2 + k * 32;
    const int8_t* q8k = q8 + k * 128;
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      int sl = 0, sh = 0;
      for (int l = 0; l < 16; l += 4) {
        const int v = (*reinterpret_cast<const int*>(q2k + l) >> shift) & 0x03030303;
        sl = LegacyDp4a(v, *reinterpret_cast<const int*>(q8k + j * 32 + l), sl);
      }
      isum += (sc[is++] & 0xF) * sl;
      for (int l = 16; l < 32; l += 4) {
        const int v = (*reinterpret_cast<const int*>(q2k + l) >> shift) & 0x03030303;
        sh = LegacyDp4a(v, *reinterpret_cast<const int*>(q8k + j * 32 + l), sh);
      }
      isum += (sc[is++] & 0xF) * sh;
      shift += 2;
    }
  }
  return dall * isum - dmin * summs;
}

// cpu_quant_dot.cpp VecDotQ3_KQ8_K (quants.c:566) — one super-block.
__device__ inline float DotQ3K(const BlockQ3_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint8_t* hm = xb->hmask;
  const int8_t* q8 = yb->qs;
  int8_t aux8[kQK_K];
  int8_t* a = aux8;
  const uint8_t* q3 = xb->qs;
  uint8_t m = 1;
  for (int jj = 0; jj < kQK_K; jj += 128) {
    for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    q3 += 32;
  }
  uint32_t auxs[4];
  memcpy(auxs, xb->scales, 12);
  const int8_t* scales = reinterpret_cast<const int8_t*>(auxs);
  uint32_t tmp = auxs[2];
  auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  a = aux8;
  const int8_t* q8p = q8;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 16; ++j) {
    for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * (q8p[l] * a[l]);
    q8p += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * (q8p[l] * a[l]);
    q8p += 8; a += 8;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  return d * isum;
}

// cpu_quant_dot.cpp VecDotQ4_KQ8_K (quants.c:645) — one super-block.
__device__ inline float DotQ4K(const BlockQ4_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x3f3f3f3f;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint32_t kmask3 = 0x03030303;
  const uint8_t* q4 = xb->qs;
  const int8_t* q8 = yb->qs;
  uint32_t utmp[4];
  memcpy(utmp, xb->scales, 12);
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
  int sumi = 0;
  for (int j = 0; j < kQK_K / 16; ++j) sumi += yb->bsums[j] * mins[j / 2];
  // __dp4a-VECTORIZED (mirrors DotQ2K): the 8 sub-blocks of 32 each carry ONE scale;
  // sub-block sb reads the low (sb even) / high (sb odd) nibble of q4 group (sb/2)*32
  // against q8[sb*32..], and isum = Σ_sb scale_sb·Σ(q4·q8). BYTE-IDENTICAL to the
  // scalar aux8 form (int32 dp4a == the scalar int accumulation), but kills the
  // 256-B/lane aux8 local-mem spill + scalar MAC that throttled the dominant
  // routed-expert (Q4_K) decode GEMM. Ref: llama.cpp vec_dot_q4_K_q8_1_impl_vmmq.
  int isum = 0;
  for (int sb = 0; sb < kQK_K / 32; ++sb) {
    const int scale = scales[sb];
    const uint8_t* q4b = q4 + (sb / 2) * 32;
    const int8_t* q8b = q8 + sb * 32;
    const int shift = (sb & 1) ? 4 : 0;
    int sub = 0;
    for (int l = 0; l < 32; l += 4) {
      const int v = (*reinterpret_cast<const int*>(q4b + l) >> shift) & 0x0F0F0F0F;
      sub = LegacyDp4a(v, *reinterpret_cast<const int*>(q8b + l), sub);
    }
    isum += scale * sub;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  const float dmin = DF16ToF32(xb->dmin) * yb->d;
  return d * isum - dmin * sumi;
}

// cpu_quant_dot.cpp VecDotQ5_KQ8_K (quants.c:720) — one super-block.
__device__ inline float DotQ5K(const BlockQ5_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x3f3f3f3f;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint32_t kmask3 = 0x03030303;
  const uint8_t* q4 = xb->qs;
  const uint8_t* hm = xb->qh;
  const int8_t* q8 = yb->qs;
  uint32_t utmp[4];
  memcpy(utmp, xb->scales, 12);
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
  int sumi = 0;
  for (int j = 0; j < kQK_K / 16; ++j) sumi += yb->bsums[j] * mins[j / 2];
  // __dp4a-VECTORIZED (see DotQ4K): 8 sub-blocks of 32, one scale each. The Q5_K
  // 5-bit value = 4-bit nibble | (qh bit << 4); sub-block sb uses qh mask (1<<sb) over
  // hm[0..31]. isum = Σ_sb scale_sb·Σ((nibble|hi)·q8) — BYTE-IDENTICAL to the scalar
  // aux8 form, no 256-B local spill. Values 0..31 are non-negative so signed dp4a matches.
  int isum = 0;
  for (int sb = 0; sb < kQK_K / 32; ++sb) {
    const int scale = scales[sb];
    const uint8_t* q4b = q4 + (sb / 2) * 32;
    const int8_t* q8b = q8 + sb * 32;
    const int shift = (sb & 1) ? 4 : 0;
    int sub = 0;
    for (int l = 0; l < 32; l += 4) {
      const int lo = (*reinterpret_cast<const int*>(q4b + l) >> shift) & 0x0F0F0F0F;
      const int hi = ((*reinterpret_cast<const int*>(hm + l) >> sb) & 0x01010101) << 4;
      sub = LegacyDp4a(lo | hi, *reinterpret_cast<const int*>(q8b + l), sub);
    }
    isum += scale * sub;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  const float dmin = DF16ToF32(xb->dmin) * yb->d;
  return d * isum - dmin * sumi;
}

// cpu_quant_dot.cpp VecDotQ6_KQ8_K (quants.c:800) — one super-block.
__device__ inline float DotQ6K(const BlockQ6_K* xb, const BlockQ8_K* yb) {
  const uint8_t* q4 = xb->ql;
  const uint8_t* qh = xb->qh;
  const int8_t* q8 = yb->qs;
  int8_t aux8[kQK_K];
  int8_t* a = aux8;
  for (int j = 0; j < kQK_K; j += 128) {
    for (int l = 0; l < 32; ++l) {
      a[l + 0] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
      a[l + 32] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
      a[l + 64] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
      a[l + 96] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
    }
    a += 128; q4 += 64; qh += 32;
  }
  a = aux8;
  const int8_t* q8p = q8;
  int is = 0;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 16; ++j) {
    const int scale = xb->scales[is++];
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
    q8p += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
    q8p += 8; a += 8;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  return d * isum;
}

// The supported Q8_K-family encodings, as small integer tags for the templated
// GEMM. Kept in sync with `IsCudaKeepQuantSupported` below.
enum class WType : int {
  kIQ2_XXS = 0,
  kIQ3_XXS = 1,
  kQ2_K = 2,
  kQ3_K = 3,
  kQ4_K = 4,
  kQ5_K = 5,
  kQ6_K = 6,
  kIQ2_S = 7,  // UD-IQ2_M ffn_gate/up (Q8_K activation, fits this GEMM)
  kIQ1_S = 8,     // Qwen3.8-2.4T UD-IQ1_S routed experts (96.92 % of it)
  kIQ1_XXXS = 9,  // Qwen3.8-2.4T UD-Q1_0 routed experts (96.92 % of it)
  kIQ2_XS = 10,   // GLM-5.3-Flash UD-Q2_K_XL ffn_gate/up_exps (82 of 1412 tensors)
  kIQ4_XS = 11,   // GLM-5.3-Flash ffn_down_exps + the GLM-5.3 UD-IQ1_S arm's last
                  // uncovered encoding. Shares IQ4_NL's codebook, NOT its Q8_0
                  // activation: a 256-element super-block dots Q8_K.
  // NOTE: MXFP4 is intentionally ABSENT — it dots against a 32-element Q8_0
  // activation, not Q8_K, so it cannot slot into this Q8_K super-block GEMM
  // (see DotMXFP4). It CPU-fallbacks until a Q8_0-activation GEMM variant lands.
};

template <WType W>
__device__ inline float DotSuperblock(const void* w_sb, const BlockQ8_K* a_sb);
template <>
__device__ inline float DotSuperblock<WType::kIQ2_XXS>(const void* w, const BlockQ8_K* a) {
  return DotIQ2XXS(static_cast<const BlockIQ2_XXS*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kIQ3_XXS>(const void* w, const BlockQ8_K* a) {
  return DotIQ3XXS(static_cast<const BlockIQ3_XXS*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ2_K>(const void* w, const BlockQ8_K* a) {
  return DotQ2K(static_cast<const BlockQ2_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ3_K>(const void* w, const BlockQ8_K* a) {
  return DotQ3K(static_cast<const BlockQ3_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ4_K>(const void* w, const BlockQ8_K* a) {
  return DotQ4K(static_cast<const BlockQ4_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ5_K>(const void* w, const BlockQ8_K* a) {
  return DotQ5K(static_cast<const BlockQ5_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ6_K>(const void* w, const BlockQ8_K* a) {
  return DotQ6K(static_cast<const BlockQ6_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kIQ2_S>(const void* w, const BlockQ8_K* a) {
  return DotIQ2S(static_cast<const BlockIQ2_S*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kIQ1_S>(const void* w, const BlockQ8_K* a) {
  return DotIQ1S(static_cast<const BlockIQ1_S*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kIQ1_XXXS>(const void* w, const BlockQ8_K* a) {
  return DotIQ1XXXS(static_cast<const BlockIQ1_XXXS*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kIQ2_XS>(const void* w, const BlockQ8_K* a) {
  return DotIQ2XS(static_cast<const BlockIQ2_XS*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kIQ4_XS>(const void* w, const BlockQ8_K* a) {
  return DotIQ4XS(static_cast<const BlockIQ4_XS*>(w), a);
}

template <WType W>
__device__ constexpr float FinalFactor() {
  // IQ2_XS joins its two siblings at 0.125: all three codebooks store lanes at
  // a fixed 8x magnitude. IQ4_XS does NOT — its per-sub-block delta is already
  // folded in as `d * (ls - 32)` inside the dot — so it takes the 1.0 default.
  return (W == WType::kIQ2_XXS || W == WType::kIQ2_S || W == WType::kIQ2_XS)
             ? 0.125f
             : (W == WType::kIQ3_XXS ? 0.25f : 1.0f);
}

// ---------------------------------------------------------------------------
// The MMVQ-style GEMM: one WARP per output element (i,j). The 32 lanes split the
// K super-blocks (lane `w` handles sb = w, w+32, ...), each computing the exact
// integer core + per-block float scale, then a warp reduction sums the partials.
// out[i,j] = FinalFactor * sum_sb DotSuperblock(weight_row_j[sb], act_row_i[sb]).
// Determinism note: the integer core is order-independent (exact); only the
// float scale sum is reassociated (warp tree vs CPU sequential) — within NMSE.
// ---------------------------------------------------------------------------
template <WType W, typename OutT>
__global__ void QuantDotGemmKernel(OutT* __restrict__ out,
                                   const uint8_t* __restrict__ weight,
                                   const BlockQ8_K* __restrict__ act, int64_t m,
                                   int64_t n, int64_t nsb, size_t w_row_bytes,
                                   size_t w_block_bytes) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= m * n) return;  // uniform across the whole warp (idx independent of lane)
  // Map consecutive warps to the SAME output row j and consecutive activation
  // rows i, so the four warps of a block share one weight row through L1
  // instead of streaming four different rows. m == 1 (decode) is unchanged.
  const int64_t j = warp / m;
  const int64_t i = warp % m;
  const int lane = threadIdx.x;

  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_K* a_row = act + i * nsb;

  float partial = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += 32) {
    const void* w_sb = w_row + static_cast<size_t>(sb) * w_block_bytes;
    partial += DotSuperblock<W>(w_sb, a_row + sb);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, off);

  if (lane == 0) {
    const float v = FinalFactor<W>() * partial;
    if constexpr (sizeof(OutT) == 4) {
      out[i * n + j] = v;
    } else {
      out[i * n + j] = DF32ToBF16(v);  // OutT == uint16_t (bf16)
    }
  }
}

// GROUPED variant (kMatmulBTQuantGrouped): warp per (p, n); the weight row is
// selected by the per-group expert index — row (expert_ids[p]*N + n) of the
// stacked [E*N,K] block weight. Same integer-dot core as QuantDotGemmKernel; the
// ONLY difference is the weight-row index, so it is numerically identical to the
// per-expert kMatmulBTQuant. Collapses the DeepSeek-V4 MoE's per-expert matvecs
// into one launch with P*N warps of parallelism (higher GB10 occupancy at T=1).
template <WType W, typename OutT>
__global__ void QuantDotGemmGroupedKernel(OutT* __restrict__ out,
                                          const uint8_t* __restrict__ weight,
                                          const BlockQ8_K* __restrict__ act,
                                          const int32_t* __restrict__ expert_ids,
                                          int64_t P, int64_t n, int64_t nsb,
                                          size_t w_row_bytes, size_t w_block_bytes,
                                          bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;

  const int64_t e = expert_ids[p];
  const uint8_t* w_row = weight + static_cast<size_t>(e * n + j) * w_row_bytes;
  // Broadcast activation: the routed gate/up share ONE quantized hidden (all P
  // experts see the SAME x), so a 1-row Q8_K feeds every p — bit-identical to the
  // per-row path (identical input ⇒ identical Q8_K ⇒ identical integer dot).
  const BlockQ8_K* a_row = act + (bcast ? 0 : p) * nsb;

  float partial = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += 32) {
    const void* w_sb = w_row + static_cast<size_t>(sb) * w_block_bytes;
    partial += DotSuperblock<W>(w_sb, a_row + sb);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, off);

  if (lane == 0) {
    const float v = FinalFactor<W>() * partial;
    if constexpr (sizeof(OutT) == 4) {
      out[p * n + j] = v;
    } else {
      out[p * n + j] = DF32ToBF16(v);
    }
  }
}

// FUSED gate+up+silu grouped kernel — the ds4 `moe_gate_up_mid` epilogue
// (ds4_cuda.cu moe_gate_up_mid_decode_lut_qwarp32_kernel:17127). ONE warp per
// (p,j) computes BOTH the gate dot (gate_w[e,j]·xq) AND the up dot (up_w[e,j]·xq)
// against the SAME broadcast Q8_K activation, then writes the clamped-SwiGLU
// product adown[p*n+j] = silu(min(gate,limit)) · clamp(up,±limit). This collapses
// the resident-decode routed-MoE's {gate grouped-GEMM + up grouped-GEMM + topk×2
// AsyncCopyF + topk ClampedSwiGLU} into ONE launch and NEVER writes the gate/up
// intermediates to HBM (they stay in registers). BIT-IDENTICAL to that chain: the
// SAME DotSuperblock integer core, the SAME 32-lane warp-tree reduce, the SAME
// FinalFactor, and the SAME ClampedSwiGLUKernel formula with alpha=1,beta=0
// (cuda_deepseek_v4.cu:612-619, Sig(x)=1/(1+e^-x)). The route weight is NOT folded
// here — it stays in moe_combine (post-down), preserving the down-GEMM's exact
// input bytes → the whole change is a pure launch-count + HBM-traffic fusion.
template <WType W>
__global__ void QuantDotGemmGroupedFusedSwiGLUKernel(float* __restrict__ out,
                                                     const uint8_t* __restrict__ gate_w,
                                                     const uint8_t* __restrict__ up_w,
                                                     const BlockQ8_K* __restrict__ act,
                                                     const int32_t* __restrict__ expert_ids,
                                                     int64_t P, int64_t n, int64_t nsb,
                                                     size_t w_row_bytes, size_t w_block_bytes,
                                                     float limit, bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;

  const int64_t e = expert_ids[p];
  const uint8_t* g_row = gate_w + static_cast<size_t>(e * n + j) * w_row_bytes;
  const uint8_t* u_row = up_w + static_cast<size_t>(e * n + j) * w_row_bytes;
  const BlockQ8_K* a_row = act + (bcast ? 0 : p) * nsb;

  float pg = 0.0f, pu = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += 32) {
    const void* gw_sb = g_row + static_cast<size_t>(sb) * w_block_bytes;
    const void* uw_sb = u_row + static_cast<size_t>(sb) * w_block_bytes;
    pg += DotSuperblock<W>(gw_sb, a_row + sb);
    pu += DotSuperblock<W>(uw_sb, a_row + sb);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    pg += __shfl_down_sync(0xffffffffu, pg, off);
    pu += __shfl_down_sync(0xffffffffu, pu, off);
  }
  if (lane == 0) {
    // == ClampedSwiGLUKernel(gate_up, mi, limit, alpha=1, beta=0): out[i] =
    //    gate·Sig(gate)·up, gate=min(g,limit), up=clamp(u,±limit). Bit-identical.
    const float gate = fminf(FinalFactor<W>() * pg, limit);
    const float up = fminf(fmaxf(FinalFactor<W>() * pu, -limit), limit);
    out[p * n + j] = gate * (1.0f / (1.0f + expf(-gate))) * up;
  }
}

template <WType W>
void LaunchGroupedFusedSwiGLU(Tensor& out, const uint8_t* gate_w, const uint8_t* up_w,
                              const BlockQ8_K* act, const int32_t* expert_ids, int64_t P,
                              int64_t n, int64_t nsb, size_t w_row_bytes, size_t w_block_bytes,
                              float limit, bool bcast, cudaStream_t s) {
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t warps = P * n;
  const int64_t grid = (warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  QuantDotGemmGroupedFusedSwiGLUKernel<W><<<static_cast<unsigned>(grid), block, 0, s>>>(
      static_cast<float*>(out.data), gate_w, up_w, act, expert_ids, P, n, nsb, w_row_bytes,
      w_block_bytes, limit, bcast);
}

// ===========================================================================
// Q8_0 keep-quant GEMM — the DeepSeek-V4 MLA projections / o-LoRA / shared
// experts / lm_head (the "AProjQ8/SExpQ8/OutQ8" weights) run ON THE GPU instead
// of the CPU keep-quant fallback (which drained the stream + made the decode
// step uncapturable). Q8_0 is a LEGACY (32-element, single-fp16-scale) encoding
// whose CPU vec_dot pairs it with a Q8_0 ACTIVATION (not the K-quants' Q8_K), so
// this is a self-contained path: quantize the activation to Q8_0 on the GPU, then
// the Q8_0×Q8_0 integer dot. ORACLE = our CPU reference:
//   cpu_quant_act.cpp QuantizeRowQ8_0 (ggml-quants.c quantize_row_q8_0) — the quant
//   cpu_quant_dot.cpp VecDotQ8_0Q8_0  (quants.c:400)                    — the dot
// The INTEGER core (Σ x.qs·y.qs per 32-block) is bit-identical; only the per-block
// float scale sum is reassociated (warp tree vs CPU sequential) — the same near-tie
// band the K-quant path is gated at (NMSE 5e-4).
// ---------------------------------------------------------------------------
// GPU Q8_0 activation quantizer — one thread per 32-element block. Bit-exact port
// of QuantizeRowQ8_0 (ternary amax, d=amax/127, y.d=F32ToF16(d), qs=roundf(x·id)).
__global__ void QuantizeQ8_0Kernel(BlockQ8_0* __restrict__ scratch,
                                   const void* __restrict__ a, ActDT adt, int64_t a_rs,
                                   int64_t m, int64_t nb) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= m * nb) return;
  const int64_t i = t / nb;   // activation row
  const int64_t b = t % nb;   // 32-block within the row
  const int64_t elem0 = i * a_rs + b * kQK8_0;
  float amax = 0.0f;
  for (int j = 0; j < kQK8_0; ++j) {
    const float av = fabsf(DLoadAct(a, adt, elem0 + j));
    amax = amax > av ? amax : av;  // ternary MAX (matches CPU, NaN-propagating)
  }
  BlockQ8_0& y = scratch[t];
  const float d = amax / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  y.d = DF32ToF16(d);
  for (int j = 0; j < kQK8_0; ++j) {
    const float x0 = DLoadAct(a, adt, elem0 + j) * id;
    y.qs[j] = static_cast<int8_t>(roundf(x0));  // round half away from zero (== std::roundf)
  }
}

// Lever 1 (ds4-gap "preq"): quantize the activation with ONE warp per 32-block
// (grid = {nb, m}, 32 threads/block), mirroring ds4 `quantize_q8_0_f32_kernel`
// (ds4_cuda.cu:4228). The classic QuantizeQ8_0Kernel above maps one THREAD to a
// whole 32-block, so a single-row [1,K] decode activation launches only
// ceil(nb/128) blocks (2 for K=7168) — far too few to fill the SMs (measured
// 6.97 us/launch × 646 = 4.5 ms/step). One warp per block gives `nb` resident
// blocks (ds4 hits ~1.2 us for the same work). This is NOT the Brick-8 per-block
// prologue re-quant (which re-quantized the SAME activation in every one of
// thousands of GEMM blocks behind a __syncthreads and regressed -22%): here the
// activation is still quantized exactly ONCE into scratch, then the GEMM reads
// the pre-quantized buffer — only the quant kernel's thread->work mapping changes.
// BIT-IDENTICAL to QuantizeQ8_0Kernel: amax is a warp MAX-reduction (associative
// + exact for floats, NaN-propagating ternary preserved), d = amax/127, and the
// round-half-away-from-zero (roundf) + DF32ToF16 are unchanged.
__global__ void QuantizeQ8_0PreqKernel(BlockQ8_0* __restrict__ scratch,
                                       const void* __restrict__ a, ActDT adt, int64_t a_rs,
                                       int64_t m, int64_t nb) {
  const int64_t b = static_cast<int64_t>(blockIdx.x);  // 32-block within the row
  const int64_t i = static_cast<int64_t>(blockIdx.y);  // activation row
  if (b >= nb || i >= m) return;
  const int lane = static_cast<int>(threadIdx.x);      // 0..31, one per element
  const int64_t elem0 = i * a_rs + b * kQK8_0;
  const float xv = DLoadAct(a, adt, elem0 + lane);
  float amax = fabsf(xv);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float o = __shfl_down_sync(0xffffffffu, amax, off);
    amax = amax > o ? amax : o;  // ternary MAX (matches CPU, NaN-propagating)
  }
  amax = __shfl_sync(0xffffffffu, amax, 0);  // broadcast lane0's reduced amax
  const float d = amax / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  BlockQ8_0& y = scratch[i * nb + b];
  if (lane == 0) y.d = DF32ToF16(d);
  y.qs[lane] = static_cast<int8_t>(roundf(xv * id));  // round half away from zero
}

// A/B flag for the ds4-preq activation-quant grid. Default ON (parity enabler
// ships as default); VT_V4_Q8_PREQ_QUANT=0 forces the legacy one-thread-per-block
// QuantizeQ8_0Kernel for baseline measurement. Read per call so in-process CUDA
// tests can flip it. BIT-IDENTICAL either way (asserted in test_cuda_quant_dot).
inline bool Q8PreqQuantOn(const char* v) { return !(v && v[0] == '0' && v[1] == '\0'); }

// Lever 2 / Brick 11 (ds4-gap): A/B flag for the sub-warp Q8_0 GEMV tiling.
// Default OFF (=speculative near-tie lever, opt-in `VT_V4_Q8_SUBWARP=1`); the plain
// 32-lane-per-output kernel is the baseline. Read per call so in-process CUDA tests
// and the captured decode graph pick it up at launch/capture time.
inline bool Q8SubwarpOn(const char* v) { return v && v[0] == '1' && v[1] == '\0'; }

// Brick 13 (ds4-gap ILP lever): A/B flag for the N-output-rows-per-warp Q8_0 GEMV.
// Default OFF (=1 → the plain one-row-per-warp kernel is the baseline). VT_V4_Q8_ILP=2
// or =4 selects the multi-row kernel (N independent weight-load streams per warp → more
// in-flight loads → lower long-scoreboard). Read per call so in-process CUDA tests and
// the captured decode graph pick it up at launch/capture time. BIT-IDENTICAL either way
// (asserted in test_cuda_quant_dot: multi-row == N separate plain outputs, byte-exact).
inline int Q8IlpRows(const char* v) {
  if (!v) return 1;
  if (v[0] == '2' && v[1] == '\0') return 2;
  if (v[0] == '4' && v[1] == '\0') return 4;
  return 1;
}

// Brick 14 (ds4 raw-mechanism lever): A/B flag for the INTRA-ROW multi-BLOCK register
// PREFETCH Q8_0 GEMV. Default OFF (=1 → the plain one-row-per-warp kernel). VT_V4_Q8_PREFETCH
// =2 or =4 selects the software-pipelined kernel: for a SINGLE output row, PF Q8_0 super-block
// loads (int8 qs + f16 scale) are hoisted into registers BEFORE the dependent __dp4a chains,
// so PF independent weight+act load streams are outstanding per row. This is the ds4 mechanism
// (register-resident memory-level parallelism — the ncu DIFF in ds4-q8-raw-mechanism-2026-07-30.md
// showed ds4 longSB 17.2 @ 56 regs vs OURS 54.4 @ 39 regs, longSB INVERSELY tracking register
// count). DISTINCT from the Brick-13 multi-ROW ILP (which changed the row→warp map, raised longSB
// to 85-127 → measured-negative): here the warp→output map is UNCHANGED (one row per warp), only
// the per-row block loop is software-pipelined. Read per call so in-process CUDA tests and the
// captured decode graph pick it up at launch/capture time. BIT-IDENTICAL (same integer __dp4a
// order, same f16-scale fold, same per-row accumulation order — asserted in test_cuda_quant_dot).
inline int Q8Prefetch(const char* v) {
  if (!v) return 1;
  if (v[0] == '2' && v[1] == '\0') return 2;
  if (v[0] == '4' && v[1] == '\0') return 4;
  return 1;
}

// Probe (measurement only): print each DISTINCT Q8_0 dense projection shape (nb,n)
// once so the nsys per-grid.x breakdown can be attributed to K. Guarded (default off).
inline void Q8ProbeShape(int64_t m, int64_t n, int64_t nb, bool grouped) {
  if (!std::getenv("VT_V4_Q8_PROBE")) return;
  static std::mutex pmu;
  static std::set<int64_t> seen;
  const int64_t key = (nb << 44) ^ (n << 12) ^ (m << 2) ^ (grouped ? 1 : 0);
  std::lock_guard<std::mutex> lk(pmu);
  if (seen.insert(key).second)
    std::fprintf(stderr, "[Q8PROBE] %s nb=%lld n=%lld m=%lld K=%lld grid.x=%lld\n",
                 grouped ? "grouped" : "dense", (long long)nb, (long long)n, (long long)m,
                 (long long)(nb * kQK8_0), (long long)((m * n + 7) / 8));
}

// Brick 3 (last-mile): read a 4-byte int from a 2-BYTE-aligned int8 stream. The Q8_0
// block `qs` starts at offset 2 in the 34-byte block (uint16 d + 32×int8), so — unlike
// the 4-byte-aligned Q8_K qs the Brick-1 IQ2/Q2_K path int-loads directly — a naked
// int32 load here would be MIS-ALIGNED (UB / fault on half the blocks). Bit-exact port
// of llama.cpp `ggml-cuda/common.cuh:get_int_b2` (two uint16 loads, little-endian) — the
// reconstructed byte pattern is identical to a valid int32 load, so __dp4a extracts the
// same signed int8 lanes as the scalar `(int)qs[p]`.
__device__ __forceinline__ int GetIntB2(const int8_t* qs, int i32) {
  const uint16_t* x16 = reinterpret_cast<const uint16_t*>(qs);
  return static_cast<int>(x16[2 * i32 + 0]) | (static_cast<int>(x16[2 * i32 + 1]) << 16);
}

// Q8_0×Q8_0 GEMM: one warp per output (i,j); lane `w` handles blocks b=w,w+32,…,
// each a 32-element integer dot scaled by f16(wd)·f16(ad); warp-reduce the partials.
// Brick 3: the per-block dot reads the 32 int8 as 8 int32 (`GetIntB2`, coalesced 2-byte
// loads) + 8 `__dp4a` (mirrors llama.cpp `ggml-cuda/vecdotq.cuh:vec_dot_q8_0_q8_1_impl`
// + `VDR_Q8_0_Q8_1_MMVQ`) instead of 32 scattered int8 loads + scalar MACs. BIT-IDENTICAL
// (__dp4a = exact int32 accumulation; integer sumi unchanged; the f16-scale fold unchanged).
template <typename OutT>
__global__ void QuantDotGemmQ8_0Kernel(OutT* __restrict__ out,
                                       const uint8_t* __restrict__ weight,
                                       const BlockQ8_0* __restrict__ act, int64_t m, int64_t n,
                                       int64_t nb, size_t w_row_bytes) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= m * n) return;
  const int64_t i = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_0* a_row = act + i * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w_row + static_cast<size_t>(b) *
                                                                          sizeof(BlockQ8_0));
    const BlockQ8_0* ab = a_row + b;
    int sumi = 0;
#pragma unroll
    for (int k = 0; k < kQK8_0 / 4; ++k) sumi = LegacyDp4a(GetIntB2(wb->qs, k), GetIntB2(ab->qs, k), sumi);
    partial += sumi * (DF16ToF32(wb->d) * DF16ToF32(ab->d));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[i * n + j] = partial;
    else out[i * n + j] = DF32ToBF16(partial);
  }
}

// Brick 13 (ds4-gap ILP lever): N-OUTPUT-ROWS-PER-WARP Q8_0 GEMV. The plain kernel above
// maps ONE output row per warp → each lane runs a DEPENDENT load→unpack→__dp4a chain with
// nothing to hide the load latency (measured ncu 2026-07-30: long-scoreboard 54.4 at 71.9%
// occupancy, L1 hit 96.7% → memory-LATENCY-bound, not bandwidth/occupancy/alignment-bound).
// This kernel gives each warp NROWS CONSECUTIVE output columns of the SAME activation row i
// (j0..j0+NROWS-1): the activation block is read ONCE and __dp4a-dotted against NROWS
// INDEPENDENT weight rows, so NROWS separate weight-load streams are issued per block —
// while row r's load is in flight the compiler issues row r+1's load (higher memory-level
// parallelism → lower long-scoreboard). This is the ILP axis none of Bricks 4/11/12 touched.
// BIT-IDENTICAL to NROWS separate QuantDotGemmQ8_0Kernel outputs: each row's integer __dp4a
// order (same GetIntB2 8×dp4a over the same blocks), the 32-wide warp reduce, and the
// f16-scale fold `sumi*(DF16ToF32(wd)*DF16ToF32(ad))` are ALL UNCHANGED — only co-issued.
// Grid maps warp → (i, jg) with jg over ceil(n/NROWS) column-groups; tail rows j>=n skipped.
template <typename OutT, int NROWS>
__global__ void QuantDotGemmQ8_0MultiRowKernel(OutT* __restrict__ out,
                                               const uint8_t* __restrict__ weight,
                                               const BlockQ8_0* __restrict__ act, int64_t m,
                                               int64_t n, int64_t nb, size_t w_row_bytes) {
  const int64_t njg = (n + NROWS - 1) / NROWS;  // column-groups per activation row
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= m * njg) return;
  const int64_t i = warp / njg;
  const int64_t jg = warp % njg;
  const int64_t j0 = jg * NROWS;
  const int lane = threadIdx.x;
  const BlockQ8_0* a_row = act + i * nb;
  float partial[NROWS];
#pragma unroll
  for (int r = 0; r < NROWS; ++r) partial[r] = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const BlockQ8_0* ab = a_row + b;
    const float ad = DF16ToF32(ab->d);
#pragma unroll
    for (int r = 0; r < NROWS; ++r) {
      const int64_t j = j0 + r;
      if (j >= n) continue;
      const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
      const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w_row + static_cast<size_t>(b) *
                                                                            sizeof(BlockQ8_0));
      int sumi = 0;
#pragma unroll
      for (int k = 0; k < kQK8_0 / 4; ++k) sumi = LegacyDp4a(GetIntB2(wb->qs, k), GetIntB2(ab->qs, k), sumi);
      partial[r] += sumi * (DF16ToF32(wb->d) * ad);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int r = 0; r < NROWS; ++r) partial[r] += __shfl_down_sync(0xffffffffu, partial[r], off);
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < NROWS; ++r) {
      const int64_t j = j0 + r;
      if (j >= n) continue;
      if constexpr (sizeof(OutT) == 4) out[i * n + j] = partial[r];
      else out[i * n + j] = DF32ToBF16(partial[r]);
    }
  }
}

// Brick 14 (ds4 raw-mechanism lever): INTRA-ROW multi-BLOCK register-PREFETCH Q8_0 GEMV.
// The plain kernel (QuantDotGemmQ8_0Kernel) maps ONE output row per warp and runs a
// DEPENDENT load→unpack→__dp4a chain per lane with a SHALLOW in-flight-load pipeline (ncu
// 2026-07-31: long-scoreboard 54.4 @ 39 regs, 71.9% occ). ds4's byte-identical `preq` kernel
// hits long-scoreboard 17.2 @ 56 regs, 60% occ — the ncu DIFF (ds4-q8-raw-mechanism-2026-07-30.md)
// showed long-scoreboard INVERSELY tracks register count (56→17, 48→36, 39→54): ds4 spends the
// extra registers on a DEEPER in-flight weight-block load pipeline (register-resident memory-level
// parallelism) that hides LPDDR5X latency so DRAM stays saturated at LOWER occupancy. L2 and
// coalescing were REFUTED (ds4 WORSE on both). This kernel reproduces that mechanism on the SAME
// single-row-per-warp structure (NOT the failed Brick-13 multi-ROW axis): the per-lane block loop
// is UNROLL-AND-JAMmed by PF — each group of PF blocks issues ALL of its PF weight+act int32 loads
// (+f16 scales) into registers FIRST, THEN consumes them with PF independent __dp4a chains, so PF
// block loads are outstanding per row before the first dependent MAC. More live registers ⇒ nvcc
// keeps a deeper load pipeline ⇒ lower long-scoreboard.
// BIT-IDENTICAL to QuantDotGemmQ8_0Kernel: each lane visits blocks in the SAME ascending order
// (lane, lane+32, lane+64, …: group g's PF blocks are base_g+{0,32,…,32(PF-1)} with base_g=lane+g*32*PF,
// accumulated p-ascending then g-ascending), the 8×__dp4a per block, the 32-wide warp reduce, and
// the f16-scale fold `sumi*(DF16ToF32(wd)*DF16ToF32(ad))` are ALL UNCHANGED — only the loads are
// hoisted ahead of the MACs. So `partial +=` runs the identical float-add sequence ⇒ same bits.
template <typename OutT, int PF>
__global__ void QuantDotGemmQ8_0PrefetchKernel(OutT* __restrict__ out,
                                               const uint8_t* __restrict__ weight,
                                               const BlockQ8_0* __restrict__ act, int64_t m,
                                               int64_t n, int64_t nb, size_t w_row_bytes) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= m * n) return;
  const int64_t i = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_0* a_row = act + i * nb;
  float partial = 0.0f;
  constexpr int kI32 = kQK8_0 / 4;  // 8 int32 per 32-int8 block
  for (int64_t base = lane; base < nb; base += static_cast<int64_t>(32) * PF) {
    // Prefetch phase: hoist ALL PF blocks' weight+act int32 words + folded scales into
    // registers so PF independent load streams are issued before any dependent __dp4a.
    int wq[PF][kI32];
    int aq[PF][kI32];
    float sc[PF];
    bool ok[PF];
#pragma unroll
    for (int p = 0; p < PF; ++p) {
      const int64_t b = base + static_cast<int64_t>(p) * 32;
      ok[p] = (b < nb);
      if (ok[p]) {
        const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(
            w_row + static_cast<size_t>(b) * sizeof(BlockQ8_0));
        const BlockQ8_0* ab = a_row + b;
#pragma unroll
        for (int k = 0; k < kI32; ++k) {
          wq[p][k] = GetIntB2(wb->qs, k);
          aq[p][k] = GetIntB2(ab->qs, k);
        }
        sc[p] = DF16ToF32(wb->d) * DF16ToF32(ab->d);
      }
    }
    // Compute phase: p-ascending, identical per-block __dp4a order + scale fold as the plain kernel.
#pragma unroll
    for (int p = 0; p < PF; ++p) {
      if (!ok[p]) continue;
      int sumi = 0;
#pragma unroll
      for (int k = 0; k < kI32; ++k) sumi = LegacyDp4a(wq[p][k], aq[p][k], sumi);
      partial += sumi * sc[p];
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[i * n + j] = partial;
    else out[i * n + j] = DF32ToBF16(partial);
  }
}

// Brick 12 (ds4-gap "launch consolidation"): PAIRED Q8_0 GEMV. Computes TWO weight
// matrices w0,w1 against the SAME pre-quantized activation `act` (m==1 decode) in ONE
// launch — the port of ds4 `matmul_q8_0_pair_preq_warp8_kernel` (ds4_cuda.cu:4485).
// One warp per output row `j`; the warp reads its activation block once and dp4a-dots
// it against BOTH w0[j] and w1[j] (when j < the respective out-dim), amortizing the
// activation load and — the measured lever — HALVING the launch count for the two
// A-projections that share the layer hidden (MLA q_a+kv_a; shared-expert gate+up).
// BIT-IDENTICAL to two QuantDotGemmQ8_0Kernel launches: each output's integer __dp4a
// order is UNCHANGED (same GetIntB2 8×dp4a over the same blocks), the 32-wide warp
// reduce is UNCHANGED, and the f16-scale fold is UNCHANGED — only co-scheduled.
template <typename OutT>
__global__ void QuantDotGemmQ8_0PairKernel(OutT* __restrict__ out0, OutT* __restrict__ out1,
                                           const uint8_t* __restrict__ w0,
                                           const uint8_t* __restrict__ w1,
                                           const BlockQ8_0* __restrict__ act, int64_t n0,
                                           int64_t n1, int64_t nb, size_t w0_row_bytes,
                                           size_t w1_row_bytes) {
  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const int64_t nmax = n0 > n1 ? n0 : n1;
  if (j >= nmax) return;
  const int lane = threadIdx.x;
  const bool has0 = j < n0;
  const bool has1 = j < n1;
  const uint8_t* w0_row = has0 ? w0 + static_cast<size_t>(j) * w0_row_bytes : nullptr;
  const uint8_t* w1_row = has1 ? w1 + static_cast<size_t>(j) * w1_row_bytes : nullptr;
  const BlockQ8_0* a_row = act;  // m == 1: single shared activation row
  float p0 = 0.0f;
  float p1 = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const BlockQ8_0* ab = a_row + b;
    const float ad = DF16ToF32(ab->d);
    if (has0) {
      const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w0_row + static_cast<size_t>(b) *
                                                                            sizeof(BlockQ8_0));
      int sumi = 0;
#pragma unroll
      for (int k = 0; k < kQK8_0 / 4; ++k) sumi = LegacyDp4a(GetIntB2(wb->qs, k), GetIntB2(ab->qs, k), sumi);
      p0 += sumi * (DF16ToF32(wb->d) * ad);
    }
    if (has1) {
      const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w1_row + static_cast<size_t>(b) *
                                                                            sizeof(BlockQ8_0));
      int sumi = 0;
#pragma unroll
      for (int k = 0; k < kQK8_0 / 4; ++k) sumi = LegacyDp4a(GetIntB2(wb->qs, k), GetIntB2(ab->qs, k), sumi);
      p1 += sumi * (DF16ToF32(wb->d) * ad);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    p0 += __shfl_down_sync(0xffffffffu, p0, off);
    p1 += __shfl_down_sync(0xffffffffu, p1, off);
  }
  if (lane == 0) {
    if (has0) {
      if constexpr (sizeof(OutT) == 4) out0[j] = p0;
      else out0[j] = DF32ToBF16(p0);
    }
    if (has1) {
      if constexpr (sizeof(OutT) == 4) out1[j] = p1;
      else out1[j] = DF32ToBF16(p1);
    }
  }
}

// Brick 12 (ds4-gap "row-split consolidation"): BLOCK-DIAGONAL grouped Q8_0 GEMV — the
// resident grouped OUTPUT-LoRA `wo_a` (o_proj.py:58-73). The host path launches ONE
// GEMV per group (ng=8 → 344 launches/step, 53% of the 646), each over a DISJOINT
// ipg-wide slice of the attention output against the group's olr weight rows. This
// kernel does all ng groups in ONE launch: output row `rr` (0..ng*rpg) belongs to group
// gp=rr/rpg and dots weight row `rr` (nb_g=ipg/32 blocks) against the activation blocks
// [gp*nb_g, (gp+1)*nb_g) of the ONCE-quantized full [nh*hd] activation. BIT-IDENTICAL to
// the ng separate GemmRowSliceInto launches: since ipg is a multiple of 32, each group's
// slice is a whole set of 32-blocks whose per-block amax/quant is byte-identical whether
// quantized as a slice or as part of the full row; the per-row __dp4a + warp reduce +
// scale fold are the plain-kernel math with a group base offset. Mirrors ds4
// `grouped_q8_0_a_preq_warp8_kernel` (ds4_cuda.cu:5509).
template <typename OutT>
__global__ void QuantDotGemmQ8_0GroupDiagKernel(OutT* __restrict__ out,
                                                const uint8_t* __restrict__ weight,
                                                const BlockQ8_0* __restrict__ act, int64_t rpg,
                                                int64_t ng, int64_t nb_g, size_t w_row_bytes) {
  const int64_t rr = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (rr >= ng * rpg) return;
  const int64_t gp = rr / rpg;
  const int lane = threadIdx.x;
  const uint8_t* w_row = weight + static_cast<size_t>(rr) * w_row_bytes;
  const BlockQ8_0* a_row = act + gp * nb_g;  // this group's activation slice (block-aligned)
  float partial = 0.0f;
  for (int64_t b = lane; b < nb_g; b += 32) {
    const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w_row + static_cast<size_t>(b) *
                                                                          sizeof(BlockQ8_0));
    const BlockQ8_0* ab = a_row + b;
    int sumi = 0;
#pragma unroll
    for (int k = 0; k < kQK8_0 / 4; ++k) sumi = LegacyDp4a(GetIntB2(wb->qs, k), GetIntB2(ab->qs, k), sumi);
    partial += sumi * (DF16ToF32(wb->d) * DF16ToF32(ab->d));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[rr] = partial;
    else out[rr] = DF32ToBF16(partial);
  }
}

// Lever 2 / Brick 11 (ds4-gap): sub-warp Q8_0 GEMV. The plain kernel above maps ONE
// full 32-lane warp to each output. For SHORT-K projections (MLA/LoRA/kv, K∈{512,1536}
// → nb∈{16,48}) a 32-lane warp leaves lanes b∈[nb,32) idle (nb=16 wastes 50% of the
// warp) AND issues only nb concurrent block-loads → too little memory-level parallelism
// to hide the M=1 GEMV's DRAM latency (Bricks 3/4 established this matvec is
// latency/occupancy-bound, NOT ALU/alignment-bound). This kernel splits the 32-lane
// warp into LANES-wide subgroups, ONE output per subgroup, so ALL 32 lanes of a warp
// issue loads (LANES=16 → 2 outputs/warp; LANES=8 → 4 outputs/warp), raising in-flight
// loads/warp. Ground: ds4's sub-warp reduction quarter/half_warp_sum_f32
// (ds4_cuda.cu:16609-16625) + moe_gate_up_mid decode sub-warp kernels (:17073-17119).
// CORRECTNESS: the integer __dp4a accumulation is order-independent → `sumi` is
// bit-exact for any LANES; only the final float scale-sum re-associates across fewer
// lanes → a characterized NEAR-TIE (NMSE≤5e-4, test_cuda_quant_dot). LANES=32 is
// byte-identical to QuantDotGemmQ8_0Kernel (same lane→block map + same 32-wide reduce).
template <int LANES>
__device__ __forceinline__ unsigned SubwarpMask(unsigned tx) {
  if constexpr (LANES == 32) {
    (void)tx;
    return 0xffffffffu;
  } else {
    return ((1u << LANES) - 1u) << (tx & (32u - LANES));  // the subgroup's 8/16 lanes
  }
}

template <typename OutT, int LANES>
__global__ void QuantDotGemmQ8_0SubwarpKernel(OutT* __restrict__ out,
                                              const uint8_t* __restrict__ weight,
                                              const BlockQ8_0* __restrict__ act, int64_t m,
                                              int64_t n, int64_t nb, size_t w_row_bytes) {
  constexpr int kSub = 256 / LANES;  // subgroups (=outputs) per 256-thread block
  const int sg = threadIdx.x / LANES;
  const int lane = threadIdx.x & (LANES - 1);
  const int64_t o = static_cast<int64_t>(blockIdx.x) * kSub + sg;
  if (o >= m * n) return;
  const int64_t i = o / n;
  const int64_t j = o % n;
  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_0* a_row = act + i * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += LANES) {
    const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w_row + static_cast<size_t>(b) *
                                                                          sizeof(BlockQ8_0));
    const BlockQ8_0* ab = a_row + b;
    int sumi = 0;
#pragma unroll
    for (int k = 0; k < kQK8_0 / 4; ++k) sumi = LegacyDp4a(GetIntB2(wb->qs, k), GetIntB2(ab->qs, k), sumi);
    partial += sumi * (DF16ToF32(wb->d) * DF16ToF32(ab->d));
  }
  const unsigned mask = SubwarpMask<LANES>(threadIdx.x);
#pragma unroll
  for (int off = LANES / 2; off > 0; off >>= 1) partial += __shfl_down_sync(mask, partial, off, LANES);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[i * n + j] = partial;
    else out[i * n + j] = DF32ToBF16(partial);
  }
}

// nb-dispatch: nb≤16 → 8 lanes/output (4 outs/warp), nb≤48 → 16 (2 outs/warp), else 32
// (the big-K GEMMs — lm_head K=7168/nb=224 — are already lane-saturated: keep 32-lane).
template <typename OutT>
void LaunchQ8_0Subwarp(OutT* out, const uint8_t* w, const BlockQ8_0* act, int64_t m, int64_t n,
                       int64_t nb, size_t w_row_bytes, cudaStream_t s) {
  const int64_t outs = m * n;
  if (nb <= 16) {
    const int64_t grid = (outs + 32 - 1) / 32;  // 256/8 = 32 outputs/block
    QuantDotGemmQ8_0SubwarpKernel<OutT, 8><<<static_cast<unsigned>(grid), 256, 0, s>>>(
        out, w, act, m, n, nb, w_row_bytes);
  } else if (nb <= 48) {
    const int64_t grid = (outs + 16 - 1) / 16;  // 256/16 = 16 outputs/block
    QuantDotGemmQ8_0SubwarpKernel<OutT, 16><<<static_cast<unsigned>(grid), 256, 0, s>>>(
        out, w, act, m, n, nb, w_row_bytes);
  } else {
    const int64_t grid = (outs + 8 - 1) / 8;  // 256/32 = 8 outputs/block
    QuantDotGemmQ8_0SubwarpKernel<OutT, 32><<<static_cast<unsigned>(grid), 256, 0, s>>>(
        out, w, act, m, n, nb, w_row_bytes);
  }
}

// Brick 4 (last-mile): Q8_0 GEMM over the CUDA COALESCED-LOAD layout (RepackQ8_0Cuda).
// The weight tensor is deinterleaved into two contiguous sections — qs `[nblk*32]`
// (16-byte-aligned per block) then scales `[nblk]` uint16 (nblk = n*nb). Global block
// index for output row j, block b = j*nb + b, so a warp lane reads its 32 int8 via TWO
// aligned `int4` (128-bit) loads instead of the in-place block's 2-byte reads — the
// coalesced-load lever. BIT-IDENTICAL: same int8 + f16 scale values (byte permutation),
// same 8×__dp4a integer dot as QuantDotGemmQ8_0Kernel. The activation stays the plain
// Q8_0 scratch (small, reused — no repack needed).
template <typename OutT>
__global__ void QuantDotGemmQ8_0AlignedKernel(OutT* __restrict__ out,
                                              const uint8_t* __restrict__ weight,
                                              const BlockQ8_0* __restrict__ act, int64_t m,
                                              int64_t n, int64_t nb) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= m * n) return;
  const int64_t i = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const int8_t* qs_base = reinterpret_cast<const int8_t*>(weight);
  const uint16_t* d_base =
      reinterpret_cast<const uint16_t*>(weight + static_cast<size_t>(n) * nb * kQK8_0);
  const BlockQ8_0* a_row = act + i * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const int64_t gi = j * nb + b;  // global block index in the deinterleaved weight
    const int4* wq = reinterpret_cast<const int4*>(qs_base + static_cast<size_t>(gi) * kQK8_0);
    const int4 w0 = wq[0];  // aligned 128-bit loads (qs at gi*32, 16-byte aligned)
    const int4 w1 = wq[1];
    const int8_t* aq = a_row[b].qs;
    int sumi = 0;
    sumi = LegacyDp4a(w0.x, GetIntB2(aq, 0), sumi);
    sumi = LegacyDp4a(w0.y, GetIntB2(aq, 1), sumi);
    sumi = LegacyDp4a(w0.z, GetIntB2(aq, 2), sumi);
    sumi = LegacyDp4a(w0.w, GetIntB2(aq, 3), sumi);
    sumi = LegacyDp4a(w1.x, GetIntB2(aq, 4), sumi);
    sumi = LegacyDp4a(w1.y, GetIntB2(aq, 5), sumi);
    sumi = LegacyDp4a(w1.z, GetIntB2(aq, 6), sumi);
    sumi = LegacyDp4a(w1.w, GetIntB2(aq, 7), sumi);
    partial += sumi * (DF16ToF32(d_base[gi]) * DF16ToF32(a_row[b].d));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[i * n + j] = partial;
    else out[i * n + j] = DF32ToBF16(partial);
  }
}

// GROUPED Q8_0 variant (weight row = expert_ids[p]*n + j). Same dot core.
template <typename OutT>
__global__ void QuantDotGemmGroupedQ8_0Kernel(OutT* __restrict__ out,
                                              const uint8_t* __restrict__ weight,
                                              const BlockQ8_0* __restrict__ act,
                                              const int32_t* __restrict__ expert_ids, int64_t P,
                                              int64_t n, int64_t nb, size_t w_row_bytes,
                                              bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const int64_t e = expert_ids[p];
  const uint8_t* w_row = weight + static_cast<size_t>(e * n + j) * w_row_bytes;
  const BlockQ8_0* a_row = act + (bcast ? 0 : p) * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w_row + static_cast<size_t>(b) *
                                                                          sizeof(BlockQ8_0));
    const BlockQ8_0* ab = a_row + b;
    int sumi = 0;  // Brick 3: 8×__dp4a over GetIntB2 (see QuantDotGemmQ8_0Kernel) — bit-identical
#pragma unroll
    for (int k = 0; k < kQK8_0 / 4; ++k) sumi = LegacyDp4a(GetIntB2(wb->qs, k), GetIntB2(ab->qs, k), sumi);
    partial += sumi * (DF16ToF32(wb->d) * DF16ToF32(ab->d));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[p * n + j] = partial;
    else out[p * n + j] = DF32ToBF16(partial);
  }
}

// Lever 2 / Brick 11: GROUPED sub-warp Q8_0 GEMV (shared-expert path). Same sub-warp
// mapping as QuantDotGemmQ8_0SubwarpKernel but the weight row is expert_ids[p]*n+j and
// the activation row is broadcast(bcast?0:p). Near-tie identical to the grouped plain
// kernel (int core bit-exact; float scale-sum re-associates across LANES).
template <typename OutT, int LANES>
__global__ void QuantDotGemmGroupedQ8_0SubwarpKernel(OutT* __restrict__ out,
                                                     const uint8_t* __restrict__ weight,
                                                     const BlockQ8_0* __restrict__ act,
                                                     const int32_t* __restrict__ expert_ids,
                                                     int64_t P, int64_t n, int64_t nb,
                                                     size_t w_row_bytes, bool bcast) {
  constexpr int kSub = 256 / LANES;
  const int sg = threadIdx.x / LANES;
  const int lane = threadIdx.x & (LANES - 1);
  const int64_t o = static_cast<int64_t>(blockIdx.x) * kSub + sg;
  if (o >= P * n) return;
  const int64_t p = o / n;
  const int64_t j = o % n;
  const int64_t e = expert_ids[p];
  const uint8_t* w_row = weight + static_cast<size_t>(e * n + j) * w_row_bytes;
  const BlockQ8_0* a_row = act + (bcast ? 0 : p) * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += LANES) {
    const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w_row + static_cast<size_t>(b) *
                                                                          sizeof(BlockQ8_0));
    const BlockQ8_0* ab = a_row + b;
    int sumi = 0;
#pragma unroll
    for (int k = 0; k < kQK8_0 / 4; ++k) sumi = LegacyDp4a(GetIntB2(wb->qs, k), GetIntB2(ab->qs, k), sumi);
    partial += sumi * (DF16ToF32(wb->d) * DF16ToF32(ab->d));
  }
  const unsigned mask = SubwarpMask<LANES>(threadIdx.x);
#pragma unroll
  for (int off = LANES / 2; off > 0; off >>= 1) partial += __shfl_down_sync(mask, partial, off, LANES);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[p * n + j] = partial;
    else out[p * n + j] = DF32ToBF16(partial);
  }
}

template <typename OutT>
void LaunchGroupedQ8_0Subwarp(OutT* out, const uint8_t* w, const BlockQ8_0* act,
                              const int32_t* eids, int64_t P, int64_t n, int64_t nb,
                              size_t w_row_bytes, bool bcast, cudaStream_t s) {
  const int64_t outs = P * n;
  if (nb <= 16) {
    const int64_t grid = (outs + 32 - 1) / 32;
    QuantDotGemmGroupedQ8_0SubwarpKernel<OutT, 8><<<static_cast<unsigned>(grid), 256, 0, s>>>(
        out, w, act, eids, P, n, nb, w_row_bytes, bcast);
  } else if (nb <= 48) {
    const int64_t grid = (outs + 16 - 1) / 16;
    QuantDotGemmGroupedQ8_0SubwarpKernel<OutT, 16><<<static_cast<unsigned>(grid), 256, 0, s>>>(
        out, w, act, eids, P, n, nb, w_row_bytes, bcast);
  } else {
    const int64_t grid = (outs + 8 - 1) / 8;
    QuantDotGemmGroupedQ8_0SubwarpKernel<OutT, 32><<<static_cast<unsigned>(grid), 256, 0, s>>>(
        out, w, act, eids, P, n, nb, w_row_bytes, bcast);
  }
}

// --- per-stream grow-only Q8_K activation scratch (cudagraph-safe) -----------
struct StreamScratch {
  void* buf = nullptr;
  size_t bytes = 0;
};
StreamScratch& ScratchFor(cudaStream_t s) {
  static std::mutex mu;
  static std::unordered_map<cudaStream_t, StreamScratch> m;
  std::lock_guard<std::mutex> lk(mu);
  return m[s];
}
void* EnsureScratch(size_t need, cudaStream_t s) {
  StreamScratch& sc = ScratchFor(s);
  if (need > sc.bytes) {
    // Retire (never free) the old block: a captured decode graph may have baked
    // this pointer — freeing it would dangle on replay. See graph_safe_scratch.h.
    RetireGraphScratch(sc.buf);
    CheckCuda(cudaMallocAsync(&sc.buf, need, s), "cudaMallocAsync q8_K act scratch");
    sc.bytes = need;
  }
  return sc.buf;
}

// ===========================================================================
// The 32-ELEMENT keep-quant lane: IQ4_NL, Q5_0 and Q4_0.
//
// Everything above this point is the Q8_K lane: 256-element super-blocks dotted
// against a BlockQ8_K activation, dispatched by IsCudaKeepQuantSupported below.
// These three encodings cannot join it, and the reason is structural rather than
// a missing `case`. Their vec_dot pairs a 32-element weight block with a
// BlockQ8_0 activation, so `DotSuperblock<W>(const void*, const BlockQ8_K*)`
// cannot take them: the activation TYPE differs, and 640 (the released
// Qwen3.8-Flash-Next expert_feed_forward_length) is not a whole number of
// 256-element super-blocks in the first place.
//
// So this is a SECOND templated GEMM beside the first, sharing the Q8_0
// activation prologue (QuantizeQ8_0Kernel + EnsureScratch) that the Q8_0 path
// above already owns and already gates.
//
// WHY IT IS HERE. Without these arms the seams do not refuse. The dense and
// grouped GEMMs `cudaStreamSynchronize` and run the CPU kernel over the same
// tensors, which emits CORRECT tokens at host speed -- invisible to every value
// comparison in this file -- and cannot be graph-captured at all, because a sync
// on a capturing stream fails. The fused MoE seam simply THROWS. The released
// `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S stores its routed experts in IQ4_NL
// and Q5_0, so that is the shipped artifact's every expert GEMM.
//
// Q8_0 is deliberately NOT routed through this template. It has its own tuned
// path above (aligned layout, sub-warp tiling, ILP, prefetch, pair and
// block-diagonal levers, all separately gated), and replacing it with this
// generic kernel would trade measured DeepSeek-V4 decode work for uniformity.
//
// ORACLE = this tree's CPU arm, which is the same kernel the host drain runs
// today, ported from llama.cpp b10451:
//   cpu_quant_dot.cpp:50  VecDotQ4_0Q8_0    (quants.c:174)
//   cpu_quant_dot.cpp:92  VecDotQ5_0Q8_0    (quants.c:365)
//   cpu_quant_dot.cpp:140 VecDotIQ4_NLQ8_0  (quants.c:1254)
//
// THE THREE FOLD THEIR SCALES IN THREE DIFFERENT ORDERS, and that is upstream's,
// not an accident to be normalised:
//   q4_0   : sumf += sumi * f16(x.d) * f16(y.d)      -- left-assoc, sum first
//   q5_0   : sumf += (f16(x.d) * f16(y.d)) * (float)sumi -- scale product first
//   iq4_nl : d = f16(y.d) * f16(x.d); sumf += d * (float)(s1+s2)
//            -- scale product first AND the operands the other way round
// Reproduced literally below. See the __fmul_rn/__fadd_rn note at the top of
// DotIQ4XS: -ffp-contract=off is CXX-only (CMakeLists.txt:55) and never reaches
// a .cu, so nvcc would contract each of these into an FMA that rounds ONCE where
// upstream rounds twice. That was MEASURED here, not feared.
enum class W32 { kQ4_0, kQ5_0, kIQ4_NL };

// One 32-element weight block against ONE BlockQ8_0 activation block.
template <W32 W>
__device__ inline float Dot32(const void* vwb, const BlockQ8_0* ab);

// quants.c:174 -- ggml_vec_dot_q4_0_q8_0_generic, one block.
template <>
__device__ inline float Dot32<W32::kQ4_0>(const void* vwb, const BlockQ8_0* ab) {
  const BlockQ4_0* xb = static_cast<const BlockQ4_0*>(vwb);
  int sumi0 = 0;
  int sumi1 = 0;
#pragma unroll
  for (int j = 0; j < kQK8_0 / 2; ++j) {
    const int v0 = (xb->qs[j] & 0x0F) - 8;
    const int v1 = (xb->qs[j] >> 4) - 8;
    sumi0 += v0 * ab->qs[j];
    sumi1 += v1 * ab->qs[j + kQK8_0 / 2];
  }
  const int sumi = sumi0 + sumi1;
  // `sumi * f16(x.d) * f16(y.d)` associates LEFT: the integer sum meets the
  // weight scale first. Two explicit rounds, matching upstream's two.
  return __fmul_rn(__fmul_rn(static_cast<float>(sumi), DF16ToF32(xb->d)), DF16ToF32(ab->d));
}

// quants.c:365 -- ggml_vec_dot_q5_0_q8_0_generic, one block. The `(int8_t)` casts
// upstream applies are kept for the same reason cpu_quant_dot.cpp keeps them: the
// value range is [-16,15] so the cast is a no-op, but it is what fixes the type
// of the multiply.
template <>
__device__ inline float Dot32<W32::kQ5_0>(const void* vwb, const BlockQ8_0* ab) {
  const BlockQ5_0* xb = static_cast<const BlockQ5_0*>(vwb);
  // `qh` is at offset 2 of a 22-byte block, so it is 2-byte aligned and a
  // `uint32_t` load through it is MISALIGNED -- undefined on the host and a
  // fault or a silently wrong read on the device. Assembled from bytes, which is
  // exactly what the CPU arm's `memcpy` compiles to. Little-endian, as upstream.
  const uint32_t qh = static_cast<uint32_t>(xb->qh[0]) |
                      (static_cast<uint32_t>(xb->qh[1]) << 8) |
                      (static_cast<uint32_t>(xb->qh[2]) << 16) |
                      (static_cast<uint32_t>(xb->qh[3]) << 24);
  int sumi0 = 0;
  int sumi1 = 0;
#pragma unroll
  for (int j = 0; j < kQK8_0 / 2; ++j) {
    const uint8_t xh_0 = static_cast<uint8_t>(((qh & (1u << (j + 0))) >> (j + 0)) << 4);
    const uint8_t xh_1 = static_cast<uint8_t>((qh & (1u << (j + 16))) >> (j + 12));
    const int32_t x0 = static_cast<int8_t>(((xb->qs[j] & 0x0F) | xh_0) - 16);
    const int32_t x1 = static_cast<int8_t>(((xb->qs[j] >> 4) | xh_1) - 16);
    sumi0 += x0 * ab->qs[j];
    sumi1 += x1 * ab->qs[j + kQK8_0 / 2];
  }
  const int sumi = sumi0 + sumi1;
  // `(f16(x.d) * f16(y.d)) * (float)sumi` -- the scale PRODUCT is formed first.
  return __fmul_rn(__fmul_rn(DF16ToF32(xb->d), DF16ToF32(ab->d)), static_cast<float>(sumi));
}

// quants.c:1254 -- ggml_vec_dot_iq4_nl_q8_0_generic, one block. Shares q4_0's
// byte layout; the nibble is a CODEBOOK INDEX into d_kvalues_iq4nl (the same
// device table IQ4_XS already reads) rather than an affine quant.
template <>
__device__ inline float Dot32<W32::kIQ4_NL>(const void* vwb, const BlockQ8_0* ab) {
  const BlockIQ4_NL* xb = static_cast<const BlockIQ4_NL*>(vwb);
  // `f16(y.d) * f16(x.d)` -- ACTIVATION scale first, and formed before the
  // integer sum is folded in. Both differences from q4_0 above are upstream's.
  const float d = __fmul_rn(DF16ToF32(ab->d), DF16ToF32(xb->d));
  int sumi1 = 0;
  int sumi2 = 0;
#pragma unroll
  for (int j = 0; j < kQK4_NL / 2; ++j) {
    sumi1 += ab->qs[j + 0] * d_kvalues_iq4nl[xb->qs[j] & 0x0F];
    sumi2 += ab->qs[j + kQK4_NL / 2] * d_kvalues_iq4nl[xb->qs[j] >> 4];
  }
  return __fmul_rn(d, static_cast<float>(sumi1 + sumi2));
}

// Block size in bytes, restated per encoding rather than read from BlockBytes()
// so a decoder change cannot silently re-stride this GEMM. ggml-common.h @ b10451:
// q4_0 :213 (18 B), q5_0 :229 (22 B), iq4_nl :447 (18 B).
template <W32 W>
__device__ __host__ inline constexpr size_t W32Bytes() {
  return W == W32::kQ4_0 ? sizeof(BlockQ4_0)
         : W == W32::kQ5_0 ? sizeof(BlockQ5_0)
                           : sizeof(BlockIQ4_NL);
}

// DENSE 32-block GEMM: one warp per output (i,j); lane `w` walks blocks
// b = w, w+32, ...; warp-tree reduce. Mirrors QuantDotGemmQ8_0Kernel exactly,
// with Dot32<W> in place of the inline q8_0 dot. The accumulator is f32, the
// same width as the CPU arm's `float sumf` and as the Q8_K lane's.
template <W32 W, typename OutT>
__global__ void QuantDotGemm32Kernel(OutT* __restrict__ out,
                                     const uint8_t* __restrict__ weight,
                                     const BlockQ8_0* __restrict__ act, int64_t m, int64_t n,
                                     int64_t nb, size_t w_row_bytes) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= m * n) return;
  const int64_t i = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_0* a_row = act + i * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    partial = __fadd_rn(partial,
                        Dot32<W>(w_row + static_cast<size_t>(b) * W32Bytes<W>(), a_row + b));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[i * n + j] = partial;
    else out[i * n + j] = DF32ToBF16(partial);
  }
}

// GROUPED 32-block variant (weight row = expert_ids[p]*n + j, activation row
// broadcast when Pa==1). Mirrors QuantDotGemmGroupedQ8_0Kernel.
template <W32 W, typename OutT>
__global__ void QuantDotGemmGrouped32Kernel(OutT* __restrict__ out,
                                            const uint8_t* __restrict__ weight,
                                            const BlockQ8_0* __restrict__ act,
                                            const int32_t* __restrict__ expert_ids, int64_t P,
                                            int64_t n, int64_t nb, size_t w_row_bytes,
                                            bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const int64_t e = expert_ids[p];
  const uint8_t* w_row = weight + static_cast<size_t>(e * n + j) * w_row_bytes;
  const BlockQ8_0* a_row = act + (bcast ? 0 : p) * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    partial = __fadd_rn(partial,
                        Dot32<W>(w_row + static_cast<size_t>(b) * W32Bytes<W>(), a_row + b));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[p * n + j] = partial;
    else out[p * n + j] = DF32ToBF16(partial);
  }
}

// MULTI-COLUMN grouped 32-block variant. One warp computes C CONSECUTIVE output
// columns (p, j0 .. j0+C-1) against ONE broadcast activation row, keeping C
// separate accumulators and paying the 5-step warp reduction once for C results.
//
// WHY: at the released qwen4_exp shape the one-warp-per-output kernel above runs
// `for (b = lane; b < 80; b += 32)` -- 2.50 iterations per lane -- then five
// __shfl_down_sync steps, then 31 of 32 lanes idle while lane 0 writes one float.
// It measured 41.6 GB/s, 15.2% of GB10's peak, the lowest row in the decode
// bandwidth table (ISSUE-LOCAL-01M2TZ9AE416FW4GH23V09783Q). Widening the warp's
// output tile amortises the reduction over C results and reads the broadcast
// activation block ONCE per warp instead of once per output element.
//
// BIT-IDENTICAL to QuantDotGemmGrouped32Kernel, and this is not a tolerance: each
// output element keeps its OWN accumulator and its OWN ascending block order
// (b = lane, lane+32, ...), and the reduction tree is the same five steps in the
// same order. Only the assignment of work to warps changes -- there is no
// re-association anywhere. Asserted byte-for-byte in test_cuda_quant_dot.
//
// This generalises what QuantDotGemmGroupedFusedSwiGLU32Kernel below already does
// at width 2 across the gate and up towers; here the C rows are consecutive
// columns of ONE tower, so the weight rows the warp touches are contiguous.
template <W32 W, typename OutT, int C>
__global__ void QuantDotGemmGrouped32MultiColKernel(OutT* __restrict__ out,
                                                    const uint8_t* __restrict__ weight,
                                                    const BlockQ8_0* __restrict__ act,
                                                    const int32_t* __restrict__ expert_ids,
                                                    int64_t P, int64_t n, int64_t nb,
                                                    size_t w_row_bytes, bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const int64_t njg = (n + C - 1) / C;
  if (warp >= P * njg) return;
  const int64_t p = warp / njg;
  const int64_t j0 = (warp % njg) * C;
  const int lane = threadIdx.x;
  const int64_t e = expert_ids[p];
  const uint8_t* w_base = weight + static_cast<size_t>(e * n + j0) * w_row_bytes;
  const BlockQ8_0* a_row = act + (bcast ? 0 : p) * nb;
  // The TAIL. n is not required to divide by C -- the released gate/up width is
  // 640 and a fused-MoE mid width need not be either -- so the last group of the
  // row runs short. Held in a register mask rather than recomputed per block.
  const int cols = static_cast<int>(min(static_cast<int64_t>(C), n - j0));
  float partial[C];
#pragma unroll
  for (int c = 0; c < C; ++c) partial[c] = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const size_t off = static_cast<size_t>(b) * W32Bytes<W>();
    const BlockQ8_0* a_blk = a_row + b;  // read ONCE for all C columns
#pragma unroll
    for (int c = 0; c < C; ++c) {
      if (c < cols)
        partial[c] = __fadd_rn(
            partial[c], Dot32<W>(w_base + static_cast<size_t>(c) * w_row_bytes + off, a_blk));
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int c = 0; c < C; ++c) partial[c] += __shfl_down_sync(0xffffffffu, partial[c], off);
  }
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < C; ++c) {
      if (c >= cols) continue;
      if constexpr (sizeof(OutT) == 4) out[p * n + j0 + c] = partial[c];
      else out[p * n + j0 + c] = DF32ToBF16(partial[c]);
    }
  }
}

// FUSED grouped gate+up+SwiGLU over the 32-block lane. One warp computes BOTH
// towers' output (p,j) against the same broadcast Q8_0 activation and writes the
// clamped-SwiGLU product, never spilling the gate/up intermediates to HBM. The
// epilogue is the SAME ClampedSwiGLUKernel formula the Q8_K fused kernel uses
// (cuda_deepseek_v4.cu:612-619, alpha=1, beta=0); there is no FinalFactor on this
// lane because the 32-block encodings fold their scale per BLOCK, inside Dot32.
template <W32 W>
__global__ void QuantDotGemmGroupedFusedSwiGLU32Kernel(
    float* __restrict__ out, const uint8_t* __restrict__ gate_w,
    const uint8_t* __restrict__ up_w, const BlockQ8_0* __restrict__ act,
    const int32_t* __restrict__ expert_ids, int64_t P, int64_t n, int64_t nb,
    size_t w_row_bytes, float limit, bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const int64_t e = expert_ids[p];
  const uint8_t* g_row = gate_w + static_cast<size_t>(e * n + j) * w_row_bytes;
  const uint8_t* u_row = up_w + static_cast<size_t>(e * n + j) * w_row_bytes;
  const BlockQ8_0* a_row = act + (bcast ? 0 : p) * nb;
  float pg = 0.0f, pu = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const size_t off = static_cast<size_t>(b) * W32Bytes<W>();
    pg = __fadd_rn(pg, Dot32<W>(g_row + off, a_row + b));
    pu = __fadd_rn(pu, Dot32<W>(u_row + off, a_row + b));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    pg += __shfl_down_sync(0xffffffffu, pg, off);
    pu += __shfl_down_sync(0xffffffffu, pu, off);
  }
  if (lane == 0) {
    const float gate = fminf(pg, limit);
    const float up = fminf(fmaxf(pu, -limit), limit);
    out[p * n + j] = gate * (1.0f / (1.0f + expf(-gate))) * up;
  }
}

// The 32-block sibling of IsCudaKeepQuantSupported below. Structured the same way
// on purpose: the two predicates answer the same question for the two activation
// lanes, and a reader who finds one must find the other.
//
// MXFP4 is the fourth 32-element encoding and is deliberately NOT here. Its
// device math (DotMXFP4, above) is written and still unreferenced; this template
// is the seam it needs, but no artifact this row gates uses it, and an ungated
// arm is a dead arm. Owed in .agents/specs/cuda-keepquant-32block.md.
bool IsCuda32BlockKeepQuantSupported(DType dt, W32* out) {
  switch (dt) {
    // The two encodings the released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S
    // stores its routed-expert towers in. Not a free choice by the quantizer:
    // expert_feed_forward_length is 640, so every routed expert row is
    // indivisible by 256 and no K-quant can encode it at all
    // (qwen4_exp_gguf_weights.cpp:117-119 says so in the tree).
    case DType::kIQ4_NL: *out = W32::kIQ4_NL; return true;
    case DType::kQ5_0: *out = W32::kQ5_0; return true;
    // Q4_0 shares IQ4_NL's block layout and costs one more `case` plus one
    // template instantiation; leaving it out would have meant a second row for
    // a dot this one already had to write.
    case DType::kQ4_0: *out = W32::kQ4_0; return true;
    default: return false;  // Q8_0 (own path above) / MXFP4 (owed) / K-quants
  }
}

template <W32 W>
void LaunchGemm32(Tensor& out, const uint8_t* weight, const BlockQ8_0* act, int64_t m, int64_t n,
                  int64_t nb, cudaStream_t s) {
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t grid = (m * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const size_t w_row_bytes = static_cast<size_t>(nb) * W32Bytes<W>();
  if (out.dtype == DType::kF32)
    QuantDotGemm32Kernel<W, float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), weight, act, m, n, nb, w_row_bytes);
  else
    QuantDotGemm32Kernel<W, uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), weight, act, m, n, nb, w_row_bytes);
}

// The grouped 32-block lane's output COLUMNS per warp. `VT_V4_W32_COLS` selects
// 1, 2, 4 or 8; anything else -- including an unset variable -- takes
// kW32ColsDefault. Read PER CALL, like the Q8_0 ILP and prefetch levers above, so
// an in-process CUDA test and a captured decode graph both pick it up at
// launch/capture time.
//
// kW32ColsDefault is MEASURED, not guessed. See the sweep recorded in
// .agents/specs/qwen4-exp-flash-next.md.
constexpr int kW32ColsDefault = 1;  // PROVISIONAL until the sweep runs; see below

inline int W32Cols(const char* v) {
  if (!v) return kW32ColsDefault;
  if (v[1] != '\0') return kW32ColsDefault;
  switch (v[0]) {
    case '1': return 1;
    case '2': return 2;
    case '4': return 4;
    case '8': return 8;
    default: return kW32ColsDefault;
  }
}

// The grouped 32-block lane's WARPS PER BLOCK. `VT_V4_W32_WARPS` selects 1, 2, 4
// or 8; anything else takes kW32WarpsDefault, which is the 4 this lane has always
// launched. Separate from the column width because the two answer different
// questions: the width is how much work one warp does, this is how many warps a
// block carries, and the second one is an OCCUPANCY question that the same sweep
// can answer for free. It cannot move a float -- a warp's work does not depend on
// which block it sits in -- and the byte-identity gate asserts that rather than
// assuming it.
constexpr int kW32WarpsDefault = 4;

inline int W32Warps(const char* v) {
  if (!v) return kW32WarpsDefault;
  if (v[1] != '\0') return kW32WarpsDefault;
  switch (v[0]) {
    case '1': return 1;
    case '2': return 2;
    case '4': return 4;
    case '8': return 8;
    default: return kW32WarpsDefault;
  }
}

template <W32 W>
void LaunchGrouped32(Tensor& out, const uint8_t* weight, const BlockQ8_0* act,
                     const int32_t* expert_ids, int64_t P, int64_t n, int64_t nb, bool bcast,
                     cudaStream_t s) {
  const int kWarpsPerBlock = W32Warps(std::getenv("VT_V4_W32_WARPS"));
  dim3 block(32, static_cast<unsigned>(kWarpsPerBlock));
  const size_t w_row_bytes = static_cast<size_t>(nb) * W32Bytes<W>();
  const int cols = W32Cols(std::getenv("VT_V4_W32_COLS"));
  if (cols > 1) {
    const int64_t njg = (n + cols - 1) / cols;
    const int64_t grid = (P * njg + kWarpsPerBlock - 1) / kWarpsPerBlock;
#define VT_LAUNCH_W32_MULTICOL(OUT_T, C)                                                       \
  QuantDotGemmGrouped32MultiColKernel<W, OUT_T, C><<<static_cast<unsigned>(grid), block, 0, s>>>( \
      static_cast<OUT_T*>(out.data), weight, act, expert_ids, P, n, nb, w_row_bytes, bcast)
    if (out.dtype == DType::kF32) {
      if (cols == 2) VT_LAUNCH_W32_MULTICOL(float, 2);
      else if (cols == 4) VT_LAUNCH_W32_MULTICOL(float, 4);
      else VT_LAUNCH_W32_MULTICOL(float, 8);
    } else {
      if (cols == 2) VT_LAUNCH_W32_MULTICOL(uint16_t, 2);
      else if (cols == 4) VT_LAUNCH_W32_MULTICOL(uint16_t, 4);
      else VT_LAUNCH_W32_MULTICOL(uint16_t, 8);
    }
#undef VT_LAUNCH_W32_MULTICOL
    return;
  }
  const int64_t grid = (P * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (out.dtype == DType::kF32)
    QuantDotGemmGrouped32Kernel<W, float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), weight, act, expert_ids, P, n, nb, w_row_bytes, bcast);
  else
    QuantDotGemmGrouped32Kernel<W, uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), weight, act, expert_ids, P, n, nb, w_row_bytes, bcast);
}

template <W32 W>
void LaunchGroupedFusedSwiGLU32(Tensor& out, const uint8_t* gate_w, const uint8_t* up_w,
                                const BlockQ8_0* act, const int32_t* expert_ids, int64_t P,
                                int64_t n, int64_t nb, float limit, bool bcast, cudaStream_t s) {
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t grid = (P * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const size_t w_row_bytes = static_cast<size_t>(nb) * W32Bytes<W>();
  QuantDotGemmGroupedFusedSwiGLU32Kernel<W><<<static_cast<unsigned>(grid), block, 0, s>>>(
      static_cast<float*>(out.data), gate_w, up_w, act, expert_ids, P, n, nb, w_row_bytes, limit,
      bcast);
}

// Dispatch helper: `W32` is a runtime value at the seam and a template parameter
// in the kernel. One switch per seam, and no `default:` that silently launches
// nothing -- past IsCuda32BlockKeepQuantSupported there is no CPU fallback left,
// so a missing arm would leave `out` untouched with CheckCuda still reporting
// success. That is the exact failure #967 shipped on the Q8_K lane.
#define VT_DISPATCH_W32(w, CALL)                                                 \
  switch (w) {                                                                   \
    case W32::kQ4_0: CALL(W32::kQ4_0); break;                                    \
    case W32::kQ5_0: CALL(W32::kQ5_0); break;                                    \
    case W32::kIQ4_NL: CALL(W32::kIQ4_NL); break;                                \
  }

bool IsCudaKeepQuantSupported(DType dt, WType* out) {
  switch (dt) {
    case DType::kIQ2_XXS: *out = WType::kIQ2_XXS; return true;
    case DType::kIQ3_XXS: *out = WType::kIQ3_XXS; return true;
    case DType::kQ2_K: *out = WType::kQ2_K; return true;
    case DType::kQ3_K: *out = WType::kQ3_K; return true;
    case DType::kQ4_K: *out = WType::kQ4_K; return true;
    case DType::kQ5_K: *out = WType::kQ5_K; return true;
    case DType::kQ6_K: *out = WType::kQ6_K; return true;
    case DType::kIQ2_S: *out = WType::kIQ2_S; return true;
    // Without these two the 2.4 T Qwen3.8 experts fall to the CPU arm below
    // and still emit CORRECT tokens, just at CPU speed, which no token gate
    // can see. That is why they are here and not left owed.
    case DType::kIQ1_S: *out = WType::kIQ1_S; return true;
    case DType::kIQ1_XXXS: *out = WType::kIQ1_XXXS; return true;
    // QUANT-CUDA-IQ4XS-IQ2XS (#2260). Without these two the GLM-5.3-Flash
    // UD-Q2_K_XL artifact's 82 IQ2_XS + 3 IQ4_XS tensors drained the stream and
    // ran their expert GEMM on the host, and the fused MoE seam below THREW
    // outright, which is why that model shipped as `--device cpu`.
    //
    // The fallback below is worse than "slow but correct", and this is MEASURED
    // rather than read off its own comment: `Backend::Alloc` on CUDA is a plain
    // `cudaMalloc`, so its pointers are not host-addressable, and the CPU kernel
    // dereferencing them SEGFAULTS. A GB10 run of this file's gate at the commit
    // before these two lines died with SIGSEGV in the first case. Unified
    // PHYSICAL memory is not the same thing as a host-mapped pointer. IQ4_XS alone also completes the
    // GLM-5.3 non-flash UD-IQ1_S arm, whose other five encodings are all above.
    case DType::kIQ2_XS: *out = WType::kIQ2_XS; return true;
    case DType::kIQ4_XS: *out = WType::kIQ4_XS; return true;
    // The 32-element Q8_0-activation encodings are NOT handled by this Q8_K GEMM
    // and never can be. IQ4_NL, Q5_0 and Q4_0 are answered by
    // IsCuda32BlockKeepQuantSupported above; Q8_0 has its own path; MXFP4 alone
    // still falls through to the CPU keep-quant kernel.
    default: return false;  // 32-block lane (IQ4_NL/Q5_0/Q4_0), Q8_0, or MXFP4
  }
}

template <WType W>
void LaunchGemm(Tensor& out, const uint8_t* weight, const BlockQ8_K* act,
                int64_t m, int64_t n, int64_t nsb, size_t w_row_bytes,
                size_t w_block_bytes, cudaStream_t s) {
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t warps = m * n;
  const int64_t grid = (warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (out.dtype == DType::kF32) {
    QuantDotGemmKernel<W, float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), weight, act, m, n, nsb, w_row_bytes,
        w_block_bytes);
  } else {
    QuantDotGemmKernel<W, uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), weight, act, m, n, nsb, w_row_bytes,
        w_block_bytes);
  }
}

inline ActDT ActDtOf(DType dt) {
  return dt == DType::kF32 ? ActDT::kF32 : dt == DType::kF16 ? ActDT::kF16 : ActDT::kBF16;
}

// ds4-parity Q8_K activation-quant dispatch (Brick 15): when VT_V4_PREQ_FUSED is
// on, route the decode grouped-MoE / dense keep-quant activation quant through the
// block-per-super-block QuantizeQ8KPreqKernel (ds4 q8_K_quantize grid geometry,
// spreads the tiny per-step quant across the SMs); else the legacy 128-thread
// one-thread-per-super-block QuantizeQ8KKernel. BYTE-IDENTICAL either way. Read per
// call so in-process CUDA tests and the captured decode graph pick it up at
// launch/capture time. Default ON (parity enabler ships as default);
// VT_V4_PREQ_FUSED=0 forces the legacy quantizer for A/B measurement.
inline bool Q8KPreqOn(const char* v) { return !(v && v[0] == '0' && v[1] == '\0'); }

void LaunchQuantizeQ8K(BlockQ8_K* qact, const void* data, ActDT adt, int64_t a_rs,
                       int64_t rows, int64_t nsb, cudaStream_t s) {
  if (Q8KPreqOn(std::getenv("VT_V4_PREQ_FUSED"))) {
    dim3 qgrid(static_cast<unsigned>(nsb), static_cast<unsigned>(rows), 1);
    QuantizeQ8KPreqKernel<<<qgrid, kQK_K, 0, s>>>(qact, data, adt, a_rs, rows, nsb);
  } else {
    constexpr int kQBlock = 128;
    const int64_t total_sb = rows * nsb;
    const int64_t grid = (total_sb + kQBlock - 1) / kQBlock;
    QuantizeQ8KKernel<<<static_cast<unsigned>(grid), kQBlock, 0, s>>>(qact, data, adt, a_rs, rows,
                                                                      nsb);
  }
}

// Quantize `rows` activation rows to Q8_0 into the per-stream grow-only scratch.
// Shared verbatim with the Q8_0 path above -- same kernels, same flag, same
// scratch -- so this lane inherits its graph-safety and its gates rather than
// re-deriving them.
BlockQ8_0* Ensure32BlockAct(const Tensor& a, int64_t rows, int64_t nb, cudaStream_t s) {
  const size_t act_bytes = static_cast<size_t>(rows) * static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  BlockQ8_0* qact = static_cast<BlockQ8_0*>(EnsureScratch(act_bytes, s));
  if (Q8PreqQuantOn(std::getenv("VT_V4_Q8_PREQ_QUANT"))) {
    dim3 qgrid(static_cast<unsigned>(nb), static_cast<unsigned>(rows), 1);
    QuantizeQ8_0PreqKernel<<<qgrid, 32, 0, s>>>(qact, a.data, ActDtOf(a.dtype), a.stride[0], rows,
                                                nb);
  } else {
    constexpr int kQBlock = 128;
    const int64_t grid = (rows * nb + kQBlock - 1) / kQBlock;
    QuantizeQ8_0Kernel<<<static_cast<unsigned>(grid), kQBlock, 0, s>>>(
        qact, a.data, ActDtOf(a.dtype), a.stride[0], rows, nb);
  }
  return qact;
}

// Q8_0 keep-quant GEMM (single). Quantize the m activation rows to Q8_0 on the GPU
// (per-stream grow-only scratch, shared with the Q8_K path — sequential GEMMs), then
// one warp-per-output integer dot. NO CPU fallback, NO stream sync ⇒ capturable.
void MatmulQ8_0Cuda(Tensor& out, const Tensor& a, const Tensor& b, cudaStream_t s) {
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[0];
  if (m == 0 || n == 0) return;
  if (k % kQK8_0 != 0)
    throw std::runtime_error("vt cuda: matmul_bt_quant Q8_0: K must be a multiple of 32");
  const int64_t nb = k / kQK8_0;
  const size_t w_row_bytes = static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  const size_t act_bytes = static_cast<size_t>(m) * static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  BlockQ8_0* act = static_cast<BlockQ8_0*>(EnsureScratch(act_bytes, s));
  if (Q8PreqQuantOn(std::getenv("VT_V4_Q8_PREQ_QUANT"))) {
    // Lever 1: ds4-preq grid — one warp per 32-block, {nb, m} blocks.
    dim3 qgrid(static_cast<unsigned>(nb), static_cast<unsigned>(m), 1);
    QuantizeQ8_0PreqKernel<<<qgrid, 32, 0, s>>>(act, a.data, ActDtOf(a.dtype), a.stride[0], m,
                                                nb);
  } else {
    constexpr int kQBlock = 128;
    const int64_t grid = (m * nb + kQBlock - 1) / kQBlock;
    QuantizeQ8_0Kernel<<<static_cast<unsigned>(grid), kQBlock, 0, s>>>(
        act, a.data, ActDtOf(a.dtype), a.stride[0], m, nb);
  }
  // Step 0 (ds4-gap): 8 warps/block (8 output rows/block) matching ds4's
  // matmul_q8_0_*_warp8 layout (ds4_cuda.cu:4343, 8 rows/block) — the current
  // full-warp Q8_0 GEMV ran at ~16% occupancy ("grid too small to fill the
  // device", ds4_cuda.cu:17073). More warps/block = more resident blocks per SM.
  // BIT-IDENTICAL: each warp still computes one independent output (i,j) dot.
  constexpr int kWarpsPerBlock = 8;
  dim3 block(32, kWarpsPerBlock);
  const int64_t grid = (m * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const uint8_t* w = static_cast<const uint8_t*>(b.data);
  Q8ProbeShape(m, n, nb, /*grouped=*/false);
  // Lever 2 / Brick 11: sub-warp GEMV tiling (opt-in). Only on the plain in-place
  // layout (the Brick-4 aligned layout is default-OFF + a measured-negative).
  if (!b.q8_0_aligned && Q8SubwarpOn(std::getenv("VT_V4_Q8_SUBWARP"))) {
    if (out.dtype == DType::kF32)
      LaunchQ8_0Subwarp<float>(static_cast<float*>(out.data), w, act, m, n, nb, w_row_bytes, s);
    else
      LaunchQ8_0Subwarp<uint16_t>(static_cast<uint16_t*>(out.data), w, act, m, n, nb, w_row_bytes,
                                  s);
    CheckCuda(cudaGetLastError(), "matmul_bt_quant Q8_0 subwarp launch");
    return;
  }
  // Brick 13 (ds4-gap ILP lever): N-output-rows-per-warp GEMV (opt-in, plain layout only).
  // Bit-identical to the plain kernel; raises in-flight weight loads to hide L1 latency.
  const int ilp = b.q8_0_aligned ? 1 : Q8IlpRows(std::getenv("VT_V4_Q8_ILP"));
  if (ilp >= 2) {
    const int64_t njg = (n + ilp - 1) / ilp;
    const int64_t ilp_grid = (m * njg + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (out.dtype == DType::kF32) {
      if (ilp == 2)
        QuantDotGemmQ8_0MultiRowKernel<float, 2><<<static_cast<unsigned>(ilp_grid), block, 0, s>>>(
            static_cast<float*>(out.data), w, act, m, n, nb, w_row_bytes);
      else
        QuantDotGemmQ8_0MultiRowKernel<float, 4><<<static_cast<unsigned>(ilp_grid), block, 0, s>>>(
            static_cast<float*>(out.data), w, act, m, n, nb, w_row_bytes);
    } else if (ilp == 2) {
      QuantDotGemmQ8_0MultiRowKernel<uint16_t, 2><<<static_cast<unsigned>(ilp_grid), block, 0, s>>>(
          static_cast<uint16_t*>(out.data), w, act, m, n, nb, w_row_bytes);
    } else {
      QuantDotGemmQ8_0MultiRowKernel<uint16_t, 4><<<static_cast<unsigned>(ilp_grid), block, 0, s>>>(
          static_cast<uint16_t*>(out.data), w, act, m, n, nb, w_row_bytes);
    }
    CheckCuda(cudaGetLastError(), "matmul_bt_quant Q8_0 ILP launch");
    return;
  }
  // Brick 14 (ds4 raw-mechanism lever): INTRA-ROW multi-block register PREFETCH GEMV (opt-in,
  // plain layout only). Bit-identical to the plain kernel; hoists PF block loads into registers
  // ahead of the __dp4a chains to deepen the in-flight-load pipeline (register-resident MLP).
  const int pf = b.q8_0_aligned ? 1 : Q8Prefetch(std::getenv("VT_V4_Q8_PREFETCH"));
  if (pf >= 2) {
    if (out.dtype == DType::kF32) {
      if (pf == 2)
        QuantDotGemmQ8_0PrefetchKernel<float, 2><<<static_cast<unsigned>(grid), block, 0, s>>>(
            static_cast<float*>(out.data), w, act, m, n, nb, w_row_bytes);
      else
        QuantDotGemmQ8_0PrefetchKernel<float, 4><<<static_cast<unsigned>(grid), block, 0, s>>>(
            static_cast<float*>(out.data), w, act, m, n, nb, w_row_bytes);
    } else if (pf == 2) {
      QuantDotGemmQ8_0PrefetchKernel<uint16_t, 2><<<static_cast<unsigned>(grid), block, 0, s>>>(
          static_cast<uint16_t*>(out.data), w, act, m, n, nb, w_row_bytes);
    } else {
      QuantDotGemmQ8_0PrefetchKernel<uint16_t, 4><<<static_cast<unsigned>(grid), block, 0, s>>>(
          static_cast<uint16_t*>(out.data), w, act, m, n, nb, w_row_bytes);
    }
    CheckCuda(cudaGetLastError(), "matmul_bt_quant Q8_0 prefetch launch");
    return;
  }
  if (b.q8_0_aligned) {  // Brick 4: coalesced-load layout (RepackQ8_0Cuda) — aligned int4 loads
    if (out.dtype == DType::kF32)
      QuantDotGemmQ8_0AlignedKernel<float><<<static_cast<unsigned>(grid), block, 0, s>>>(
          static_cast<float*>(out.data), w, act, m, n, nb);
    else
      QuantDotGemmQ8_0AlignedKernel<uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
          static_cast<uint16_t*>(out.data), w, act, m, n, nb);
  } else if (out.dtype == DType::kF32)
    QuantDotGemmQ8_0Kernel<float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), w, act, m, n, nb, w_row_bytes);
  else
    QuantDotGemmQ8_0Kernel<uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), w, act, m, n, nb, w_row_bytes);
  CheckCuda(cudaGetLastError(), "matmul_bt_quant Q8_0 launch");
}

// Q8_0 keep-quant GEMM (grouped: weight row = expert_ids[p]*n + j).
void MatmulQ8_0GroupedCuda(Tensor& out, const Tensor& act_t, const Tensor& weight,
                           const Tensor& expert_ids, cudaStream_t s) {
  const int64_t P = out.shape[0], n = out.shape[1], k = act_t.shape[1];
  if (P == 0 || n == 0) return;
  if (k % kQK8_0 != 0)
    throw std::runtime_error("vt cuda: matmul_bt_quant_grouped Q8_0: K must be a multiple of 32");
  const int64_t nb = k / kQK8_0;
  const int64_t Pa = act_t.shape[0];  // broadcast when 1 row feeds P>1 experts (preq-reuse)
  const bool bcast = (Pa == 1 && P > 1);
  const size_t w_row_bytes = static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  const size_t act_bytes = static_cast<size_t>(Pa) * static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  BlockQ8_0* qact = static_cast<BlockQ8_0*>(EnsureScratch(act_bytes, s));
  if (Q8PreqQuantOn(std::getenv("VT_V4_Q8_PREQ_QUANT"))) {
    // Lever 1: ds4-preq grid — one warp per 32-block, {nb, Pa} blocks.
    dim3 qgrid(static_cast<unsigned>(nb), static_cast<unsigned>(Pa), 1);
    QuantizeQ8_0PreqKernel<<<qgrid, 32, 0, s>>>(qact, act_t.data, ActDtOf(act_t.dtype),
                                                act_t.stride[0], Pa, nb);
  } else {
    constexpr int kQBlock = 128;
    const int64_t grid = (Pa * nb + kQBlock - 1) / kQBlock;
    QuantizeQ8_0Kernel<<<static_cast<unsigned>(grid), kQBlock, 0, s>>>(
        qact, act_t.data, ActDtOf(act_t.dtype), act_t.stride[0], Pa, nb);
  }
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t grid = (P * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const uint8_t* w = static_cast<const uint8_t*>(weight.data);
  const int32_t* eids = static_cast<const int32_t*>(expert_ids.data);
  Q8ProbeShape(P, n, nb, /*grouped=*/true);
  // Lever 2 / Brick 11: sub-warp GEMV tiling (opt-in) — grouped shared-expert path.
  if (Q8SubwarpOn(std::getenv("VT_V4_Q8_SUBWARP"))) {
    if (out.dtype == DType::kF32)
      LaunchGroupedQ8_0Subwarp<float>(static_cast<float*>(out.data), w, qact, eids, P, n, nb,
                                      w_row_bytes, bcast, s);
    else
      LaunchGroupedQ8_0Subwarp<uint16_t>(static_cast<uint16_t*>(out.data), w, qact, eids, P, n, nb,
                                         w_row_bytes, bcast, s);
    CheckCuda(cudaGetLastError(), "matmul_bt_quant_grouped Q8_0 subwarp launch");
    return;
  }
  if (out.dtype == DType::kF32)
    QuantDotGemmGroupedQ8_0Kernel<float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), w, qact, eids, P, n, nb, w_row_bytes, bcast);
  else
    QuantDotGemmGroupedQ8_0Kernel<uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), w, qact, eids, P, n, nb, w_row_bytes, bcast);
  CheckCuda(cudaGetLastError(), "matmul_bt_quant_grouped Q8_0 launch");
}

#include "cuda_pascal_q5_decode.cuh"
#include "cuda_pascal_q6_decode.cuh"
#include "cuda_pascal_q4k_decode.cuh"
#include "cuda_pascal_mmq_d4.cuh"

// The kCUDA provider for OpId::kMatmulBTQuant. Validation already done by
// vt::MatmulBTQuant (ops.cpp) before dispatch; this mirrors the CPU kernel's
// contract: b is [N,K] block-quant, a is [M,K] f32/bf16 (row-packed), out [M,N].
void MatmulBTQuantKernelCuda(Queue& q, Tensor& out, const Tensor& a,
                             const Tensor& b) {
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = b.shape[0];
  if (m == 0 || n == 0) return;

  if ((b.dtype == DType::kQ5_0 || b.dtype == DType::kQ8_0) &&
      IsPascalMmqD4Eligible(out, a, b)) {
    LaunchPascalMmqD4(out, a, b, s);
    CheckCuda(cudaGetLastError(), "Pascal MMQ D4 launch");
    return;
  }

  // Q8_0 (32-block, Q8_0-activation) runs its own on-GPU path — the DeepSeek-V4
  // MLA/o-LoRA/shared-expert/lm_head weights. No CPU fallback, no stream sync.
  if (b.dtype == DType::kQ8_0) {
    MatmulQ8_0Cuda(out, a, b, s);
    return;
  }

  // The 32-element Q8_0-activation lane (IQ4_NL / Q5_0 / Q4_0), asked BEFORE the
  // Q8_K predicate because these dtypes can never satisfy it: their vec_dot pairs
  // a 32-element block with a BlockQ8_0, and the released Qwen3.8-Flash-Next
  // expert rows are 640 elements, which is not a whole number of 256-element
  // super-blocks either. Falling through would drain the stream instead.
  W32 w32{};
  if (IsCuda32BlockKeepQuantSupported(b.dtype, &w32)) {
    if (k % kQK8_0 != 0)
      throw std::runtime_error(
          "vt cuda: matmul_bt_quant: K must be a multiple of 32 on the 32-block keep-quant lane");
    const int64_t nb32 = k / kQK8_0;
    if (b.dtype == DType::kQ5_0 && m == 1 && IsPascalQ5Decode()) {
      LaunchPascalQ5Decode(out, a, b, n, nb32, s);
      CheckCuda(cudaGetLastError(), "Pascal Q5_0 decode launch");
      return;
    }
    const BlockQ8_0* qact = Ensure32BlockAct(a, m, nb32, s);
    const uint8_t* w32b = static_cast<const uint8_t*>(b.data);
#define VT_KQ32_DENSE(WV) LaunchGemm32<WV>(out, w32b, qact, m, n, nb32, s)
    VT_DISPATCH_W32(w32, VT_KQ32_DENSE);
#undef VT_KQ32_DENSE
    CheckCuda(cudaGetLastError(), "matmul_bt_quant 32-block launch");
    return;
  }

  WType w{};
  if (!IsCudaKeepQuantSupported(b.dtype, &w)) {
    // MXFP4 is what still reaches here (see IsCuda32BlockKeepQuantSupported).
    // Run the CPU keep-quant kernel over the SAME unified-memory tensors: drain
    // the stream first so any GPU-produced activation is visible to the host.
    // This drain is NOT capturable, which is what the capture gate in
    // tests/vt/test_cuda_quant_dot.cpp measures.
    CheckCuda(cudaStreamSynchronize(s), "keepquant CPU-fallback drain");
    reinterpret_cast<MatmulFn>(GetOp(OpId::kMatmulBTQuant, DeviceType::kCPU))(q, out, a, b);
    return;
  }

  // K must be a whole number of Q8_K super-blocks (256). vt::MatmulBTQuant has
  // already checked K % BlockElems(weight) == 0; the K-quants ARE 256-blocked,
  // so this is the same fact — assert defensively.
  if (k % kQK_K != 0) {
    throw std::runtime_error(
        "vt cuda: matmul_bt_quant: K must be a whole number of 256-element "
        "Q8_K super-blocks");
  }
  const int64_t nsb = k / kQK_K;
  if (b.dtype == DType::kQ6_K && m == 1 && IsPascalQ5Decode()) {
    LaunchPascalQ6Decode(out, a, b, n, nsb, s);
    CheckCuda(cudaGetLastError(), "Pascal Q6_K decode launch");
    return;
  }
  if (b.dtype == DType::kQ4_K && m == 1 && IsPascalQ5Decode() &&
      PascalQ4KDecodeEnabled()) {
    LaunchPascalQ4KDecode(out, a, b, n, nsb, s);
    CheckCuda(cudaGetLastError(), "Pascal Q4_K decode launch");
    return;
  }

  const size_t w_block_bytes = static_cast<size_t>(BlockBytes(b.dtype));
  const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;

  // 1. Quantize the M activation rows to Q8_K on the GPU (ggml-cpu.c:1313-1349
  //    "src1 -> wdata", done once per GEMM). Scratch is per-stream, grow-only.
  const size_t act_bytes = static_cast<size_t>(m) * static_cast<size_t>(nsb) *
                           sizeof(BlockQ8_K);
  BlockQ8_K* act = static_cast<BlockQ8_K*>(EnsureScratch(act_bytes, s));

  ActDT adt = a.dtype == DType::kF32 ? ActDT::kF32
              : a.dtype == DType::kF16 ? ActDT::kF16
                                       : ActDT::kBF16;
  LaunchQuantizeQ8K(act, a.data, adt, a.stride[0], m, nsb, s);

  // 2. The integer dot GEMM (one warp per output), dequant-in-kernel.
  const uint8_t* weight = static_cast<const uint8_t*>(b.data);
  switch (w) {
    case WType::kIQ2_XXS: LaunchGemm<WType::kIQ2_XXS>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ3_XXS: LaunchGemm<WType::kIQ3_XXS>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ2_K: LaunchGemm<WType::kQ2_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ3_K: LaunchGemm<WType::kQ3_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ4_K: LaunchGemm<WType::kQ4_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ5_K: LaunchGemm<WType::kQ5_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ6_K: LaunchGemm<WType::kQ6_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ2_S: LaunchGemm<WType::kIQ2_S>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ1_S: LaunchGemm<WType::kIQ1_S>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ1_XXXS: LaunchGemm<WType::kIQ1_XXXS>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ2_XS: LaunchGemm<WType::kIQ2_XS>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ4_XS: LaunchGemm<WType::kIQ4_XS>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    // See the identical arm in MatmulBTQuantGroupedKernelCuda below: past the
    // IsCudaKeepQuantSupported gate there is no CPU fallback left, so a missing
    // case launches nothing, leaves `out` untouched, and CheckCuda still reports
    // success. Throw instead of returning a stale buffer.
    default:
      throw std::runtime_error(
          std::string("vt cuda: matmul_bt_quant: no keep-quant kernel for dtype ") +
          Name(b.dtype));
  }
  CheckCuda(cudaGetLastError(), "matmul_bt_quant launch");
}

template <WType W>
void LaunchGroupedGemm(Tensor& out, const uint8_t* weight, const BlockQ8_K* act,
                       const int32_t* expert_ids, int64_t P, int64_t n, int64_t nsb,
                       size_t w_row_bytes, size_t w_block_bytes, bool bcast, cudaStream_t s) {
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t warps = P * n;
  const int64_t grid = (warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (out.dtype == DType::kF32) {
    QuantDotGemmGroupedKernel<W, float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), weight, act, expert_ids, P, n, nsb, w_row_bytes,
        w_block_bytes, bcast);
  } else {
    QuantDotGemmGroupedKernel<W, uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), weight, act, expert_ids, P, n, nsb, w_row_bytes,
        w_block_bytes, bcast);
  }
}

// The kCUDA provider for OpId::kMatmulBTQuantGrouped. out[P,N], act[P,K],
// weight[E*N,K] block-quant, expert_ids[P] i32 (unified memory). Validation done
// by vt::MatmulBTQuantGrouped. Quantizes the P activation rows to Q8_K once, then
// one grouped-kernel launch computes every (p,n) output — the expert-batched
// analog of MatmulBTQuantKernelCuda.
void MatmulBTQuantGroupedKernelCuda(Queue& q, Tensor& out, const Tensor& act,
                                    const Tensor& weight, const Tensor& expert_ids) {
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  const int64_t P = out.shape[0];
  const int64_t n = out.shape[1];
  const int64_t k = act.shape[1];
  if (P == 0 || n == 0) return;

  if (weight.dtype == DType::kQ8_0) {  // on-GPU Q8_0 grouped path (no CPU sync)
    MatmulQ8_0GroupedCuda(out, act, weight, expert_ids, s);
    return;
  }

  // The 32-element lane, for the same reason as the dense seam above. This is the
  // one the shipped qwen4_exp MoE actually takes: its three expert towers are
  // IQ4_NL / Q5_0 and reach `vt::MatmulBTQuantGrouped` stacked as [E*N, K].
  W32 w32{};
  if (IsCuda32BlockKeepQuantSupported(weight.dtype, &w32)) {
    if (k % kQK8_0 != 0)
      throw std::runtime_error(
          "vt cuda: matmul_bt_quant_grouped: K must be a multiple of 32 on the 32-block lane");
    const int64_t nb32 = k / kQK8_0;
    const int64_t Pa32 = act.shape[0];
    const bool bcast32 = (Pa32 == 1 && P > 1);
    const BlockQ8_0* qact = Ensure32BlockAct(act, Pa32, nb32, s);
    const uint8_t* w32b = static_cast<const uint8_t*>(weight.data);
    const int32_t* eids32 = static_cast<const int32_t*>(expert_ids.data);
#define VT_KQ32_GROUPED(WV) \
  LaunchGrouped32<WV>(out, w32b, qact, eids32, P, n, nb32, bcast32, s)
    VT_DISPATCH_W32(w32, VT_KQ32_GROUPED);
#undef VT_KQ32_GROUPED
    CheckCuda(cudaGetLastError(), "matmul_bt_quant_grouped 32-block launch");
    return;
  }

  WType w{};
  if (!IsCudaKeepQuantSupported(weight.dtype, &w)) {
    CheckCuda(cudaStreamSynchronize(s), "keepquant-grouped CPU-fallback drain");
    reinterpret_cast<MatmulBTQuantGroupedFn>(
        GetOp(OpId::kMatmulBTQuantGrouped, DeviceType::kCPU))(q, out, act, weight, expert_ids);
    return;
  }
  if (k % kQK_K != 0) {
    throw std::runtime_error(
        "vt cuda: matmul_bt_quant_grouped: K must be a whole number of 256-element "
        "Q8_K super-blocks");
  }
  const int64_t nsb = k / kQK_K;
  const size_t w_block_bytes = static_cast<size_t>(BlockBytes(weight.dtype));
  const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;

  // Broadcast activation (preq-reuse): when act has ONE row but P>1 outputs, the
  // routed experts all share the SAME hidden — quantize it ONCE and let every p read
  // Q8_K row 0. Eliminates the topk-fold redundant re-quant (was the bulk of the
  // QuantizeQ8K time) and the caller's xrep copy. Bit-identical (§Brick 2).
  const int64_t Pa = act.shape[0];
  const bool bcast = (Pa == 1 && P > 1);

  // Quantize the Pa activation rows to Q8_K (per-stream grow-only scratch).
  const size_t act_bytes = static_cast<size_t>(Pa) * static_cast<size_t>(nsb) * sizeof(BlockQ8_K);
  BlockQ8_K* qact = static_cast<BlockQ8_K*>(EnsureScratch(act_bytes, s));
  ActDT adt = act.dtype == DType::kF32 ? ActDT::kF32
              : act.dtype == DType::kF16 ? ActDT::kF16
                                         : ActDT::kBF16;
  LaunchQuantizeQ8K(qact, act.data, adt, act.stride[0], Pa, nsb, s);

  const uint8_t* wt = static_cast<const uint8_t*>(weight.data);
  const int32_t* eids = static_cast<const int32_t*>(expert_ids.data);
  switch (w) {
    case WType::kIQ2_XXS: LaunchGroupedGemm<WType::kIQ2_XXS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kIQ3_XXS: LaunchGroupedGemm<WType::kIQ3_XXS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ2_K: LaunchGroupedGemm<WType::kQ2_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ3_K: LaunchGroupedGemm<WType::kQ3_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ4_K: LaunchGroupedGemm<WType::kQ4_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ5_K: LaunchGroupedGemm<WType::kQ5_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ6_K: LaunchGroupedGemm<WType::kQ6_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kIQ2_S: LaunchGroupedGemm<WType::kIQ2_S>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kIQ1_S: LaunchGroupedGemm<WType::kIQ1_S>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kIQ1_XXXS: LaunchGroupedGemm<WType::kIQ1_XXXS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kIQ2_XS: LaunchGroupedGemm<WType::kIQ2_XS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kIQ4_XS: LaunchGroupedGemm<WType::kIQ4_XS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    // IsCudaKeepQuantSupported already answered yes, so the CPU fallback above is
    // skipped. A `w` with no case here launches NOTHING, leaves `out` exactly as
    // the caller left it, and the CheckCuda below reports SUCCESS because there
    // was no launch to fail. That is how #967 shipped silently wrong IQ1_S /
    // IQ1_XXXS routed experts: the predicate grew two dtypes and only the dense
    // switch grew with it. Throw, so the next omission is loud.
    default:
      throw std::runtime_error(
          std::string("vt cuda: matmul_bt_quant_grouped: no grouped kernel for keep-quant "
                      "dtype ") +
          Name(weight.dtype));
  }
  CheckCuda(cudaGetLastError(), "matmul_bt_quant_grouped launch");
}

// Registers the CUDA keep-quant GEMM during static init (table fill only, no
// CUDA calls — same rationale as cuda_matmul.cu Registrar). This makes
// vt::OpRegistered(kMatmulBTQuant, kCUDA) TRUE, which flips the GGUF loader's
// keep-quant default ON on a CUDA device (gguf_keep_quant.cpp
// GgufQuantComputeAvailable) so DeepSeek-V4's experts/MLA GEMMs dispatch here
// instead of the unified-memory CPU reference tier.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulBTQuant, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MatmulFn>(&MatmulBTQuantKernelCuda)));
    RegisterOp(OpId::kMatmulBTQuantGrouped, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MatmulBTQuantGroupedFn>(&MatmulBTQuantGroupedKernelCuda)));
  }
} registrar;

}  // namespace

// Device-codebook drift seal (see cuda_iq_table_seal.h). `cudaMemcpyFromSymbol`
// needs the host-side shadow of a `__device__` array, which only this TU has, so
// the copy lives here and the memcmp against the CPU tables lives in
// tests/vt/test_cuda_quant_dot.cpp. The static_asserts pin the snapshot extents
// to the device arrays: a table that changes length breaks the build rather than
// sealing a prefix of itself.
void SnapshotIqTablesFromDevice(IqTableSnapshot* out) {
  static_assert(sizeof(out->kmask_iq2xs) == sizeof(d_kmask_iq2xs), "kmask extent");
  static_assert(sizeof(out->ksigns_iq2xs) == sizeof(d_ksigns_iq2xs), "ksigns extent");
  static_assert(sizeof(out->iq1s_grid) == sizeof(d_iq1s_grid), "iq1s grid extent");
  static_assert(sizeof(out->iq1xxxs_grid) == sizeof(d_iq1xxxs_grid), "iq1xxxs grid extent");
  static_assert(sizeof(out->iq2xxs_grid) == sizeof(d_iq2xxs_grid), "iq2xxs grid extent");
  static_assert(sizeof(out->iq3xxs_grid) == sizeof(d_iq3xxs_grid), "iq3xxs grid extent");
  static_assert(sizeof(out->iq2s_grid) == sizeof(d_iq2s_grid), "iq2s grid extent");
  static_assert(sizeof(out->iq2xs_grid) == sizeof(d_iq2xs_grid), "iq2xs grid extent");
  static_assert(sizeof(out->kvalues_iq4nl) == sizeof(d_kvalues_iq4nl), "iq4nl values extent");
  static_assert(sizeof(out->kvalues_mxfp4) == sizeof(d_kvalues_mxfp4), "mxfp4 values extent");
  CheckCuda(cudaMemcpyFromSymbol(out->kmask_iq2xs, d_kmask_iq2xs, sizeof(d_kmask_iq2xs)),
            "snapshot d_kmask_iq2xs");
  CheckCuda(cudaMemcpyFromSymbol(out->ksigns_iq2xs, d_ksigns_iq2xs, sizeof(d_ksigns_iq2xs)),
            "snapshot d_ksigns_iq2xs");
  CheckCuda(cudaMemcpyFromSymbol(out->iq1s_grid, d_iq1s_grid, sizeof(d_iq1s_grid)),
            "snapshot d_iq1s_grid");
  CheckCuda(cudaMemcpyFromSymbol(out->iq1xxxs_grid, d_iq1xxxs_grid, sizeof(d_iq1xxxs_grid)),
            "snapshot d_iq1xxxs_grid");
  CheckCuda(cudaMemcpyFromSymbol(out->iq2xxs_grid, d_iq2xxs_grid, sizeof(d_iq2xxs_grid)),
            "snapshot d_iq2xxs_grid");
  CheckCuda(cudaMemcpyFromSymbol(out->iq3xxs_grid, d_iq3xxs_grid, sizeof(d_iq3xxs_grid)),
            "snapshot d_iq3xxs_grid");
  CheckCuda(cudaMemcpyFromSymbol(out->iq2s_grid, d_iq2s_grid, sizeof(d_iq2s_grid)),
            "snapshot d_iq2s_grid");
  CheckCuda(cudaMemcpyFromSymbol(out->iq2xs_grid, d_iq2xs_grid, sizeof(d_iq2xs_grid)),
            "snapshot d_iq2xs_grid");
  CheckCuda(cudaMemcpyFromSymbol(out->kvalues_iq4nl, d_kvalues_iq4nl, sizeof(d_kvalues_iq4nl)),
            "snapshot d_kvalues_iq4nl");
  CheckCuda(cudaMemcpyFromSymbol(out->kvalues_mxfp4, d_kvalues_mxfp4, sizeof(d_kvalues_mxfp4)),
            "snapshot d_kvalues_mxfp4");
}

// Brick 12 (ds4-gap "launch consolidation"): PAIRED Q8_0 decode GEMV (external linkage,
// called from cuda_deepseek_v4.cu's DsaDeviceKernels wrapper — same CUDA library). One
// launch computes out0=b0·a and out1=b1·a over the SAME activation `a` (m==1 decode),
// quantizing `a` to Q8_0 ONCE (grow-only per-stream scratch, identical grid to
// MatmulQ8_0Cuda). BIT-IDENTICAL to two MatmulQ8_0Cuda calls (see kernel comment):
// same preq quantization, same 8×__dp4a integer dot, same 32-wide warp reduce + scale
// fold. b0/b1 are the plain in-place Q8_0 layout ([N,K] blocks); not q8_0_aligned.
void MatmulQ8_0PairCuda(Tensor& out0, Tensor& out1, const Tensor& a, const Tensor& b0,
                        const Tensor& b1, cudaStream_t s) {
  const int64_t m = a.shape[0], k = a.shape[1], n0 = b0.shape[0], n1 = b1.shape[0];
  if (n0 == 0 && n1 == 0) return;
  if (m != 1)
    throw std::runtime_error("vt cuda: matmul_bt_quant Q8_0 pair: decode-only (m must be 1)");
  if (b0.shape[1] != k || b1.shape[1] != k)
    throw std::runtime_error("vt cuda: matmul_bt_quant Q8_0 pair: K mismatch");
  if (b0.q8_0_aligned || b1.q8_0_aligned)
    throw std::runtime_error("vt cuda: matmul_bt_quant Q8_0 pair: aligned layout unsupported");
  if (k % kQK8_0 != 0)
    throw std::runtime_error("vt cuda: matmul_bt_quant Q8_0 pair: K must be a multiple of 32");
  const int64_t nb = k / kQK8_0;
  const size_t w0_row_bytes = static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  const size_t w1_row_bytes = w0_row_bytes;
  const size_t act_bytes = static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  BlockQ8_0* act = static_cast<BlockQ8_0*>(EnsureScratch(act_bytes, s));
  if (Q8PreqQuantOn(std::getenv("VT_V4_Q8_PREQ_QUANT"))) {
    dim3 qgrid(static_cast<unsigned>(nb), 1, 1);
    QuantizeQ8_0PreqKernel<<<qgrid, 32, 0, s>>>(act, a.data, ActDtOf(a.dtype), a.stride[0], 1, nb);
  } else {
    constexpr int kQBlock = 128;
    const int64_t qgrid = (nb + kQBlock - 1) / kQBlock;
    QuantizeQ8_0Kernel<<<static_cast<unsigned>(qgrid), kQBlock, 0, s>>>(
        act, a.data, ActDtOf(a.dtype), a.stride[0], 1, nb);
  }
  constexpr int kWarpsPerBlock = 8;
  dim3 block(32, kWarpsPerBlock);
  const int64_t nmax = n0 > n1 ? n0 : n1;
  const int64_t grid = (nmax + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const uint8_t* w0 = static_cast<const uint8_t*>(b0.data);
  const uint8_t* w1 = static_cast<const uint8_t*>(b1.data);
  const bool f32 = out0.dtype == DType::kF32;
  if (f32)
    QuantDotGemmQ8_0PairKernel<float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out0.data), static_cast<float*>(out1.data), w0, w1, act, n0, n1, nb,
        w0_row_bytes, w1_row_bytes);
  else
    QuantDotGemmQ8_0PairKernel<uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out0.data), static_cast<uint16_t*>(out1.data), w0, w1, act, n0, n1,
        nb, w0_row_bytes, w1_row_bytes);
  CheckCuda(cudaGetLastError(), "matmul_bt_quant Q8_0 pair launch");
}

// Brick 12 (ds4-gap "row-split consolidation"): BLOCK-DIAGONAL grouped Q8_0 decode GEMV
// (external linkage). Consolidates the ng per-group `wo_a` output-LoRA GEMVs into ONE
// launch. `b` is the stacked [ng*rpg, ipg] weight (row rr → group rr/rpg); `a` is the
// full [1, ng*ipg] activation, quantized ONCE. BIT-IDENTICAL to the ng separate
// GemmRowSliceInto launches (see kernel comment). out is [1, ng*rpg] (contiguous).
void MatmulQ8_0GroupDiagCuda(Tensor& out, const Tensor& a, const Tensor& b, int64_t ng,
                             cudaStream_t s) {
  const int64_t total_rows = b.shape[0], ipg = b.shape[1];
  if (total_rows == 0) return;
  if (ng <= 0 || total_rows % ng != 0)
    throw std::runtime_error("vt cuda: matmul_bt_quant Q8_0 groupdiag: rows not a multiple of ng");
  if (b.q8_0_aligned)
    throw std::runtime_error("vt cuda: matmul_bt_quant Q8_0 groupdiag: aligned layout unsupported");
  if (ipg % kQK8_0 != 0)
    throw std::runtime_error("vt cuda: matmul_bt_quant Q8_0 groupdiag: ipg must be a multiple of 32");
  if (a.shape[0] != 1 || a.shape[1] != ng * ipg)
    throw std::runtime_error("vt cuda: matmul_bt_quant Q8_0 groupdiag: activation must be [1, ng*ipg]");
  const int64_t rpg = total_rows / ng;
  const int64_t nb_g = ipg / kQK8_0;
  const int64_t nb_total = ng * nb_g;
  const size_t w_row_bytes = static_cast<size_t>(nb_g) * sizeof(BlockQ8_0);
  const size_t act_bytes = static_cast<size_t>(nb_total) * sizeof(BlockQ8_0);
  BlockQ8_0* act = static_cast<BlockQ8_0*>(EnsureScratch(act_bytes, s));
  if (Q8PreqQuantOn(std::getenv("VT_V4_Q8_PREQ_QUANT"))) {
    dim3 qgrid(static_cast<unsigned>(nb_total), 1, 1);
    QuantizeQ8_0PreqKernel<<<qgrid, 32, 0, s>>>(act, a.data, ActDtOf(a.dtype), a.stride[0], 1,
                                                nb_total);
  } else {
    constexpr int kQBlock = 128;
    const int64_t qgrid = (nb_total + kQBlock - 1) / kQBlock;
    QuantizeQ8_0Kernel<<<static_cast<unsigned>(qgrid), kQBlock, 0, s>>>(
        act, a.data, ActDtOf(a.dtype), a.stride[0], 1, nb_total);
  }
  constexpr int kWarpsPerBlock = 8;
  dim3 block(32, kWarpsPerBlock);
  const int64_t grid = (total_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const uint8_t* w = static_cast<const uint8_t*>(b.data);
  if (out.dtype == DType::kF32)
    QuantDotGemmQ8_0GroupDiagKernel<float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), w, act, rpg, ng, nb_g, w_row_bytes);
  else
    QuantDotGemmQ8_0GroupDiagKernel<uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), w, act, rpg, ng, nb_g, w_row_bytes);
  CheckCuda(cudaGetLastError(), "matmul_bt_quant Q8_0 groupdiag launch");
}

// DeepSeek-V4 resident-decode fused routed-MoE gate+up+SwiGLU (external linkage,
// called from cuda_deepseek_v4.cu's MoeDeviceKernels wrapper — same CUDA library).
// out[P,n] adown, act[Pa,K] (Pa==1 broadcast), gate_w/up_w[E*n,K] block-quant (same
// dtype), expert_ids[P] i32. Quantizes act to Q8_K ONCE (grow-only per-stream
// scratch, identical to MatmulBTQuantGroupedKernelCuda), then one fused launch.
// Bit-identical to the two grouped GEMMs + ClampedSwiGLU it replaces.
void MoeGateUpSwiGLUGroupedCuda(Queue& q, Tensor& out, const Tensor& act, const Tensor& gate_w,
                                const Tensor& up_w, const Tensor& expert_ids, float limit) {
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  const int64_t P = out.shape[0];
  const int64_t n = out.shape[1];
  const int64_t k = act.shape[1];
  if (P == 0 || n == 0) return;

  // The 32-element lane. This seam has NO fallback behind it -- before this arm it
  // THREW by name for IQ4_NL and Q5_0, which is the released qwen4_exp MoE's own
  // encodings, so the fused path was unreachable for the shipped checkpoint.
  W32 wg32{}, wu32{};
  if (IsCuda32BlockKeepQuantSupported(gate_w.dtype, &wg32) &&
      IsCuda32BlockKeepQuantSupported(up_w.dtype, &wu32) && wg32 == wu32) {
    if (k % kQK8_0 != 0)
      throw std::runtime_error(
          "vt cuda: moe_gate_up_swiglu: K must be a multiple of 32 on the 32-block lane");
    const int64_t nb32 = k / kQK8_0;
    const int64_t Pa32 = act.shape[0];
    const bool bcast32 = (Pa32 == 1 && P > 1);
    const BlockQ8_0* qact32 = Ensure32BlockAct(act, Pa32, nb32, s);
    const uint8_t* gw32 = static_cast<const uint8_t*>(gate_w.data);
    const uint8_t* uw32 = static_cast<const uint8_t*>(up_w.data);
    const int32_t* eids32 = static_cast<const int32_t*>(expert_ids.data);
#define VT_KQ32_FUSED(WV)                                                          \
  LaunchGroupedFusedSwiGLU32<WV>(out, gw32, uw32, qact32, eids32, P, n, nb32, limit, \
                                 bcast32, s)
    VT_DISPATCH_W32(wg32, VT_KQ32_FUSED);
#undef VT_KQ32_FUSED
    CheckCuda(cudaGetLastError(), "moe_gate_up_swiglu 32-block launch");
    return;
  }

  WType w{}, wu{};
  if (!IsCudaKeepQuantSupported(gate_w.dtype, &w) || !IsCudaKeepQuantSupported(up_w.dtype, &wu) ||
      w != wu) {
    throw std::runtime_error(
        "vt cuda: moe_gate_up_swiglu: gate/up must be the SAME CUDA keep-quant dtype");
  }
  if (k % kQK_K != 0) {
    throw std::runtime_error(
        "vt cuda: moe_gate_up_swiglu: K must be a whole number of 256-element Q8_K super-blocks");
  }
  const int64_t nsb = k / kQK_K;
  const size_t w_block_bytes = static_cast<size_t>(BlockBytes(gate_w.dtype));
  const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;

  const int64_t Pa = act.shape[0];
  const bool bcast = (Pa == 1 && P > 1);
  const size_t act_bytes = static_cast<size_t>(Pa) * static_cast<size_t>(nsb) * sizeof(BlockQ8_K);
  BlockQ8_K* qact = static_cast<BlockQ8_K*>(EnsureScratch(act_bytes, s));
  ActDT adt = act.dtype == DType::kF32 ? ActDT::kF32
              : act.dtype == DType::kF16 ? ActDT::kF16
                                         : ActDT::kBF16;
  LaunchQuantizeQ8K(qact, act.data, adt, act.stride[0], Pa, nsb, s);

  const uint8_t* gw = static_cast<const uint8_t*>(gate_w.data);
  const uint8_t* uw = static_cast<const uint8_t*>(up_w.data);
  const int32_t* eids = static_cast<const int32_t*>(expert_ids.data);
  switch (w) {
    case WType::kIQ2_XXS: LaunchGroupedFusedSwiGLU<WType::kIQ2_XXS>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kIQ3_XXS: LaunchGroupedFusedSwiGLU<WType::kIQ3_XXS>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kQ2_K: LaunchGroupedFusedSwiGLU<WType::kQ2_K>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kQ3_K: LaunchGroupedFusedSwiGLU<WType::kQ3_K>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kQ4_K: LaunchGroupedFusedSwiGLU<WType::kQ4_K>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kQ5_K: LaunchGroupedFusedSwiGLU<WType::kQ5_K>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kQ6_K: LaunchGroupedFusedSwiGLU<WType::kQ6_K>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kIQ2_S: LaunchGroupedFusedSwiGLU<WType::kIQ2_S>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kIQ1_S: LaunchGroupedFusedSwiGLU<WType::kIQ1_S>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kIQ1_XXXS: LaunchGroupedFusedSwiGLU<WType::kIQ1_XXXS>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kIQ2_XS: LaunchGroupedFusedSwiGLU<WType::kIQ2_XS>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kIQ4_XS: LaunchGroupedFusedSwiGLU<WType::kIQ4_XS>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    // Before #967 the guard above THREW for these two dtypes, because
    // IsCudaKeepQuantSupported returned false and the named "same CUDA
    // keep-quant dtype" error fired. #967 made the predicate say yes without
    // adding the arms, which turned that named error into a silent no-op:
    // nothing launches, `out` keeps whatever it held, CheckCuda sees no launch
    // to fail. Same failure mode as the grouped GEMM above.
    default:
      throw std::runtime_error(
          std::string("vt cuda: moe_gate_up_swiglu: no fused kernel for keep-quant dtype ") +
          Name(gate_w.dtype));
  }
  CheckCuda(cudaGetLastError(), "moe_gate_up_swiglu launch");
}

// SHARED-OP registration (kMoeGateUpSwiGLUGrouped). Promotes the DeepSeek-private
// fused MoE gate+up+SwiGLU kernel above into a first-class vt:: op so any keep-quant
// MoE arch inherits it via vt::MoeGateUpSwiGLUGrouped. BIT-IDENTICAL: the registered
// entry IS MoeGateUpSwiGLUGroupedCuda (same kernel DeepSeek's MoeDeviceKernels wrapper
// calls) — a shared-op wrapper, not a rewrite. Kept a SEPARATE registrar (append-only)
// so the primary keep-quant Registrar above is untouched.
struct FusedMoeSharedRegistrar {
  FusedMoeSharedRegistrar() {
    RegisterOp(OpId::kMoeGateUpSwiGLUGrouped, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MoeGateUpSwiGLUGroupedFn>(&MoeGateUpSwiGLUGroupedCuda)));
  }
} fused_moe_shared_registrar;


// --- the KGATHER seam (see vt/cuda/cuda_embedding_quant.h) -------------------

bool EmbeddingQuantSupported(DType dt) {
  switch (dt) {
#define VT_DQ_SUPPORTED(DT, CODEC) \
  case DType::DT:                  \
    return true;
    VT_DQ_GATHER_TYPES(VT_DQ_SUPPORTED)
#undef VT_DQ_SUPPORTED
    default:
      return false;
  }
}

cudaError_t LaunchEmbeddingQuant(cudaStream_t s, Tensor& out, const Tensor& table,
                                 const Tensor& ids, EmbeddingQuantErr* err) {
  const int64_t t = ids.shape[0];
  const int64_t h = table.shape[1];
  const int64_t v = table.shape[0];
  const int64_t be = BlockElems(table.dtype);
  // `vt::Embedding` already refused a ragged K; restated because this function
  // computes a BYTE stride from it and a wrong stride is silent.
  if (be <= 0 || h % be != 0) return cudaErrorInvalidValue;
  const int64_t nb = h / be;
  const size_t row_bytes = RowSizeBytes(table.dtype, h);
  if (row_bytes != static_cast<size_t>(nb) * static_cast<size_t>(BlockBytes(table.dtype))) {
    return cudaErrorInvalidValue;
  }
  const uint8_t* tbl = static_cast<const uint8_t*>(table.data);
  if (out.dtype == DType::kF32) {
    if (ids.dtype == DType::kI32) {
      return LaunchEmbeddingQuantTyped<float, int32_t>(
          s, static_cast<float*>(out.data), tbl, static_cast<const int32_t*>(ids.data),
          table.dtype, t, nb, h, v, row_bytes, err);
    }
    return LaunchEmbeddingQuantTyped<float, int64_t>(
        s, static_cast<float*>(out.data), tbl, static_cast<const int64_t*>(ids.data),
        table.dtype, t, nb, h, v, row_bytes, err);
  }
  if (out.dtype == DType::kF16) {
    if (ids.dtype == DType::kI32) return LaunchEmbeddingQuantTyped<__half, int32_t>(
        s, out.Ptr<__half>(), tbl, ids.Ptr<int32_t>(), table.dtype, t, nb, h, v, row_bytes, err);
    return LaunchEmbeddingQuantTyped<__half, int64_t>(
        s, out.Ptr<__half>(), tbl, ids.Ptr<int64_t>(), table.dtype, t, nb, h, v, row_bytes, err);
  }
  if (ids.dtype == DType::kI32) {
    return LaunchEmbeddingQuantTyped<uint16_t, int32_t>(
        s, static_cast<uint16_t*>(out.data), tbl, static_cast<const int32_t*>(ids.data),
        table.dtype, t, nb, h, v, row_bytes, err);
  }
  return LaunchEmbeddingQuantTyped<uint16_t, int64_t>(
      s, static_cast<uint16_t*>(out.data), tbl, static_cast<const int64_t*>(ids.data),
      table.dtype, t, nb, h, v, row_bytes, err);
}

}  // namespace vt::cuda

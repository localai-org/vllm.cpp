// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// NVFP4 W4A16 dequant-GEMM for the CUDA backend (M2.2a, correctness-grade).
//
// out[M,N] = act[M,K] (f32/bf16) @ dequant(w).T, where w is a modelopt
// W4A16_NVFP4 weight [N=out_features, K=in_features] read DIRECTLY from device
// memory and dequantized on the fly (no host bf16 weight tensor). The decode is
// bit-for-bit vllm::DequantNvfp4ToBf16 (the authoritative reference):
//   * two 4-bit E2M1 codes per packed byte, low nibble = input elem 2i, high
//     nibble = 2i+1; nibble bit 3 = sign, bits 0..2 index the magnitude LUT
//     {0, .5, 1, 1.5, 2, 3, 4, 6};
//   * one IEEE fp8-e4m3fn scale per 16-element input group;
//   * group scale = f32(fp8_scale) * weight_scale_2 (per-tensor, multiplied);
//   * the dequanted value is ROUNDED TO BF16 before use — exactly the byte
//     DequantNvfp4ToBf16 stores — so this GEMM and Matmul(act, dequant(w).T)
//     differ only in K-reduction order, not in the per-element product.
// These are standard IEEE fp8-e4m3fn scales: the GGUF killgate fork's UE4M3
// x0.5 LUT trap (.agents/specs/gguf-nvfp4-notes.md) does NOT apply to modelopt NVFP4.
//
// Layout (M2.4-tile): SHARED-MEMORY TILED GEMM with register accumulation. A
// thread-block computes a [BM x BN] output tile; per K-tile it cooperatively
// (a) loads the activation tile [BM x BK] into shared memory (coalesced, act is
// K-contiguous) and (b) decodes the fp4 weight tile [BN x BK] into shared memory
// with COALESCED packed-byte reads — consecutive threads read consecutive packed
// bytes of a weight row (the naive one-thread-per-output kernel had each thread
// stride a full weight row = fully uncoalesced, the GB10 bottleneck). Each thread
// then accumulates a [TM x TN] register sub-tile over the shared tiles. The
// per-element decode is byte-for-byte the naive kernel's (same nibble order,
// group-16 fp8-e4m3 scale, scale2, bf16-round) so the GEMM stays token-for-token
// correct; only the K-reduction order changes (tiled), within matmul tolerance.
//
// Layout (M2.6-wmma): the PREFILL (large-M) bf16 path runs on Blackwell TENSOR
// CORES. The fp4 weight tile is decoded into shared memory AS BF16 (CUDA-core
// decode, same bf16-rounded value) and the bf16 activation tile is staged into
// shared; each warp then runs nvcuda::wmma m16n16k16 bf16xbf16 -> f32 over the
// tile (mirrors a cutlass W4A16 dequant-GEMM). For the MoE grouped GEMM the P
// token-major pair-rows are first GROUPED BY EXPERT on device (counting sort +
// ragged per-BM-tile map) so each expert's weight is decoded once per row tile
// and fed to the tensor cores — buying both the ~BM-x weight-bandwidth reduction
// the per-pair kernel lacked and the tensor-core MMA. VT_NVFP4_WMMA=0 falls back
// to the CUDA-core tiled kernels (A/B toggle). Small-M decode keeps the naive
// kernel (graph-safe). Measured GB10: 1024-token prefill TTFT 14.4s -> 6.1s.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "vt/cuda/cuda_arch_tactics.h"
#include "vt/cuda/cuda_device_caps.h"
#include "vt/cuda/cuda_nvfp4_sm12x.h"
#include "vt/cuda/fp4_quant_fast.h"
#include "vt/cuda/graph_safe_scratch.h"
#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: matmul_nvfp4: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Tensor-core (bf16 WMMA) NVFP4 GEMM path toggle (A/B; default ON). Set
// VT_NVFP4_WMMA=0 to fall back to the CUDA-core tiled kernels (baseline).
bool WmmaEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_NVFP4_WMMA");
    return e == nullptr || (e[0] != '0');
  }();
  // ARCH TERM (required, not an optimisation). Every WMMA body this predicate
  // selects is compiled `#if __CUDA_ARCH__ >= 800` with an `#else __trap()`
  // (bf16 WMMA fragments are Ampere+), while all six call sites are otherwise
  // host-side shape/env tests — so on a pre-Ampere board they would select a
  // trap. Folded in HERE rather than at each call site because every one of the
  // six means the same thing ("take the bf16 tensor-core path"), and each already
  // has a CUDA-core fallthrough (naive / tiled / split-K) that is
  // correctness-grade. Queried live rather than cached in the static above: the
  // device context need not exist at static-init time, and a wrong value latched
  // there would be unrecoverable. Fails safe: caps invalid -> CUDA-core path. On
  // sm_80+ this is always true, so gate-model selection is unchanged.
  // See .agents/specs/cuda-arch-breadth-fp16.md §V0-b / W1c.
  const DeviceCaps& caps = GetDeviceCaps();
  return on && caps.valid && caps.sm_major >= 8;
}

// M=1/decode-path 128-bit vectorized fp4 weight loads (A/B; default ON). Set
// VT_FP4_VEC=0 to force the byte-wise scalar reduction (bit-identical, slower).
bool Fp4VecEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP4_VEC");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

// W3-I1 packed one-input SiLU->NVFP4 producer (A/B; default OFF). The
// specialization mirrors vLLM's executed sm1xxa body only for the production
// BF16 + direct-swizzled route. All other dtypes/layouts/alignment cases retain
// the scalar producer below. Cache the environment decision: this launcher is
// reached 64 times per dense forward and must never call getenv per kernel.
//
// These two helpers are consumed ONLY by the native fp4-MMA fused-producer path
// (below, under `#if defined(VT_FP4_MMA_SM120A)`), so they are gated on the same
// define. Without this, a cross-family single-arch build that compiles this TU
// with the native path OFF (e.g. -DVLLM_CPP_CUDA_ARCHITECTURES=90a, where the
// `fp4-mma` FEATURE-TABLE row resolves EMPTY) leaves them referenced nowhere and
// -Werror rejects the TU (#177-D "declared but never referenced"). On sm_12x
// (VT_FP4_MMA_SM120A defined) they are compiled exactly as before — byte
// identical. See .agents/specs/cuda-arch-additivity.md §W9 (BACKEND-CUDA-SM090).
// Decode-specialized MoE grouped GEMM path toggle (M2.9; A/B, default ON). Set
// VT_MOE_DECODE=0 to force the prefill-tuned BM=64 WMMA tile for small-M decode
// (the pre-M2.9 behavior) for same-binary A/B. See LaunchGrouped for the gate.
bool MoeDecodeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_DECODE");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

// f32 load/store overloads: bf16 converts on the way in/out, math is f32.
__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, same as host F32ToBF16
}

// IEEE fp8-e4m3fn -> f32, mirroring vllm::F8E4M3ToF32 (1 sign, 4 exp, 3 mant,
// bias 7, no inf, NaN only at 0x7F/0xFF).
__device__ inline float F8E4M3ToF32Dev(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1u;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFu;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7u;
  const float sign_mul = sign ? -1.0f : 1.0f;
  if (exp == 0xFu && mant == 0x7u) return CUDART_NAN_F;
  if (exp == 0u) return sign_mul * (static_cast<float>(mant) * (1.0f / 512.0f));
  const float mantissa = 1.0f + static_cast<float>(mant) * (1.0f / 8.0f);
  return sign_mul * ldexpf(mantissa, static_cast<int>(exp) - 7);
}

// E2M1 (fp4) magnitude LUT, indexed by the low 3 bits of the nibble; bit 3 is
// the sign. 1x-scaled — modelopt NVFP4, NOT the 2x GGUF kvalues_mxfp4 LUT.
__constant__ float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Decode one packed fp4 byte (two E2M1 codes) into its two dequanted bf16-rounded
// weight values, given the already-computed group scale gs = f8(scale)*scale2.
// Bit-for-bit vllm::DequantNvfp4ToBf16: low nibble = elem 2j, high = 2j+1; bit 3
// = sign; magnitude via the LUT; product rounded through bf16.
__device__ __forceinline__ void DecodeFp4Byte(uint8_t b, float gs, float& w_lo, float& w_hi) {
  const uint8_t low = b & 0x0Fu;
  const uint8_t high = b >> 4;
  const float lo_mag = kE2M1[low & 0x7u] * ((low & 0x8u) ? -1.0f : 1.0f);
  const float hi_mag = kE2M1[high & 0x7u] * ((high & 0x8u) ? -1.0f : 1.0f);
  w_lo = __bfloat162float(__float2bfloat16(lo_mag * gs));
  w_hi = __bfloat162float(__float2bfloat16(hi_mag * gs));
}

// Decode + accumulate the 4 packed bytes (8 fp4 elements) held in one 32-bit
// word, at activation offset `base`. Byte b (little-endian: (word>>8b)&0xff)
// carries elements base+2b (low nibble) and base+2b+1 (high nibble) -- the exact
// element order of the scalar loop, so this stays bit-identical.
template <typename Tin>
__device__ __forceinline__ void AccumFp4Word(float& acc, uint32_t word, float gs, const Tin* arow,
                                             int64_t base) {
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const uint8_t byte = static_cast<uint8_t>((word >> (b * 8)) & 0xffu);
    float w_lo, w_hi;
    DecodeFp4Byte(byte, gs, w_lo, w_hi);
    acc += Load(arow, base + 2 * b) * w_lo;
    acc += Load(arow, base + 2 * b + 1) * w_hi;
  }
}

// One output column's fp4 W4A16 dot product: acc = sum_k act[k] * dequant(w[k]).
// Bandwidth-optimized for the M=1/decode path -- the packed weight row is read
// with 128-bit (uint4 = 16 byte = 2 group-16) loads instead of byte-at-a-time,
// cutting the weight LDG count 16x and issuing full-width memory transactions.
// The accumulation order is unchanged (group ascending, byte ascending, low then
// high nibble) so the result is BIT-IDENTICAL to the scalar loop below; only the
// load width differs. Falls back to the scalar path when the row is not 16-byte
// aligned (k_dim not a multiple of 32) or has an odd group tail.
template <typename Tin>
__device__ __forceinline__ float Fp4ColDot(const uint8_t* prow, const uint8_t* srow,
                                            const Tin* arow, int64_t k_dim, float scale2,
                                            bool use_vec) {
  const int64_t groups = k_dim / 16;
  float acc = 0.0f;
  const bool aligned16 = use_vec && ((k_dim / 2) % 16 == 0) &&
                         ((reinterpret_cast<uintptr_t>(prow) & 0xf) == 0);
  if (aligned16) {
    const uint4* prow4 = reinterpret_cast<const uint4*>(prow);
    const int64_t cpairs = groups / 2;  // each uint4 spans two group-16 blocks
    for (int64_t c = 0; c < cpairs; ++c) {
      const uint4 pk = prow4[c];
      const float gs0 = F8E4M3ToF32Dev(srow[2 * c]) * scale2;      // group 2c   (bytes 0..7)
      const float gs1 = F8E4M3ToF32Dev(srow[2 * c + 1]) * scale2;  // group 2c+1 (bytes 8..15)
      const int64_t base0 = (2 * c) * 16;
      const int64_t base1 = (2 * c + 1) * 16;
      AccumFp4Word(acc, pk.x, gs0, arow, base0);      // group 2c   elems 0..7
      AccumFp4Word(acc, pk.y, gs0, arow, base0 + 8);  // group 2c   elems 8..15
      AccumFp4Word(acc, pk.z, gs1, arow, base1);      // group 2c+1 elems 0..7
      AccumFp4Word(acc, pk.w, gs1, arow, base1 + 8);  // group 2c+1 elems 8..15
    }
    if (groups & 1) {  // odd trailing group: scalar tail (same order)
      const int64_t g = groups - 1;
      const float gs = F8E4M3ToF32Dev(srow[g]) * scale2;
      const int64_t base = g * 16;
      for (int j = 0; j < 8; ++j) {
        float w_lo, w_hi;
        DecodeFp4Byte(prow[g * 8 + j], gs, w_lo, w_hi);
        acc += Load(arow, base + 2 * j) * w_lo;
        acc += Load(arow, base + 2 * j + 1) * w_hi;
      }
    }
    return acc;
  }
  for (int64_t g = 0; g < groups; ++g) {
    const float gs = F8E4M3ToF32Dev(srow[g]) * scale2;
    const int64_t base = g * 16;
    for (int j = 0; j < 8; ++j) {
      float w_lo, w_hi;
      DecodeFp4Byte(prow[g * 8 + j], gs, w_lo, w_hi);
      acc += Load(arow, base + 2 * j) * w_lo;
      acc += Load(arow, base + 2 * j + 1) * w_hi;
    }
  }
  return acc;
}

// Naive one-thread-per-output NVFP4 dequant-GEMM (grid.x over N, grid.y over M).
// Retained for SMALL m (decode): there the weight fits L2 so coalescing buys
// little, and this kernel avoids the tiled kernel's shared-mem/sync overhead —
// measured faster than the tiled path below for small m/N (GB10 nsys, 2026-07-04).
// The weight row is read 128-bit-wide via Fp4ColDot (bit-identical to the scalar
// reduction); this is the M2.8 decode-path bandwidth optimization.
template <typename Tin, typename Tout>
__global__ void MatmulNvfp4KernelNaive(Tout* out, const Tin* act, const uint8_t* packed,
                                       const uint8_t* scale, float scale2, int64_t m_rows,
                                       int64_t n_cols, int64_t k_dim, bool use_vec) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t m = blockIdx.y;
  if (n >= n_cols || m >= m_rows) return;

  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;
  const uint8_t* prow = packed + n * packed_cols;
  const uint8_t* srow = scale + n * groups;
  const Tin* arow = act + m * k_dim;
  Store(out, m * n_cols + n, Fp4ColDot(prow, srow, arow, k_dim, scale2, use_vec));
}

// Shared-memory tiled NVFP4 dequant-GEMM. out[m,n] = sum_k act[m,k] *
// bf16(e2m1(w[n,k]) * group_scale[n,k/16]). A block owns output tile
// [row0..row0+BM, col0..col0+BN]; each thread owns a [TM x TN] register sub-tile.
// BK must be a multiple of 16 (group-16 scale alignment). blockDim.x must equal
// (BM/TM)*(BN/TN).
template <typename Tin, typename Tout, int BM, int BN, int BK, int TM, int TN>
__global__ void MatmulNvfp4Tiled(Tout* out, const Tin* act, const uint8_t* packed,
                                 const uint8_t* scale, float scale2, int64_t m_rows,
                                 int64_t n_cols, int64_t k_dim) {
  constexpr int kThreads = (BM / TM) * (BN / TN);
  __shared__ float As[BM * BK];  // activation tile [BM][BK]
  __shared__ float Ws[BN * BK];  // decoded weight tile [BN][BK]

  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;

  // This thread's output sub-tile origin within the block tile.
  const int t_row = (tid / (BN / TN)) * TM;
  const int t_col = (tid % (BN / TN)) * TN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  for (int64_t kt = 0; kt < k_dim; kt += BK) {
    // Cooperative coalesced load of the activation tile [BM][BK]. Guard both the
    // M edge (row0+r >= M) and the K edge (kt+c >= K, partial last K-tile) with 0.
    for (int idx = tid; idx < BM * BK; idx += kThreads) {
      const int r = idx / BK;
      const int c = idx % BK;
      const int64_t gr = row0 + r;
      As[idx] = (gr < m_rows && kt + c < k_dim) ? Load(act, gr * k_dim + kt + c) : 0.0f;
    }
    // Cooperative coalesced load + on-the-fly decode of the weight tile [BN][BK].
    // Iterate over PACKED BYTES (one byte -> two k elements); consecutive threads
    // read consecutive packed bytes of a weight row => coalesced fp4 weight reads.
    for (int idx = tid; idx < BN * (BK / 2); idx += kThreads) {
      const int nl = idx / (BK / 2);
      const int bc = idx % (BK / 2);  // packed-byte column within the K-tile
      const int kl = 2 * bc;          // element column within the K-tile
      const int64_t gn = col0 + nl;
      // K is a multiple of 16 (even), so a byte's two elements (kt+kl, kt+kl+1)
      // are both in-range iff kt+kl < K — one guard covers the whole byte.
      if (gn < n_cols && kt + kl < k_dim) {
        const uint8_t b = packed[gn * packed_cols + kt / 2 + bc];
        const float gs = F8E4M3ToF32Dev(scale[gn * groups + (kt + kl) / 16]) * scale2;
        float w_lo, w_hi;
        DecodeFp4Byte(b, gs, w_lo, w_hi);
        Ws[nl * BK + kl] = w_lo;
        Ws[nl * BK + kl + 1] = w_hi;
      } else {
        Ws[nl * BK + kl] = 0.0f;
        Ws[nl * BK + kl + 1] = 0.0f;
      }
    }
    __syncthreads();

    // Register-blocked accumulate over the K-tile.
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float ar[TM], br[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) ar[i] = As[(t_row + i) * BK + k];
#pragma unroll
      for (int j = 0; j < TN; ++j) br[j] = Ws[(t_col + j) * BK + k];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += ar[i] * br[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int64_t gr = row0 + t_row + i;
    if (gr >= m_rows) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int64_t gc = col0 + t_col + j;
      if (gc < n_cols) Store(out, gr * n_cols + gc, acc[i][j]);
    }
  }
}

// --- Tensor-core (bf16 WMMA) NVFP4 W4A16 GEMM (M2.6 prefill unlock) ----------
// Same math as MatmulNvfp4Tiled (out[m,n] = sum_k act[m,k] * bf16(e2m1(w[n,k]) *
// group_scale[n,k/16])) but the K-reduction runs on Blackwell tensor cores. The
// fp4 weight tile is dequantized INTO SHARED as bf16 (CUDA-core dequant, exactly
// DecodeFp4Byte's bf16-rounded value) and the bf16 activation tile is staged into
// shared; each warp then runs wmma m16n16k16 bf16xbf16 -> f32 over the tile. This
// mirrors a cutlass W4A16 dequant-GEMM (block tile [BM x BN], warp tiling
// [WARPS_M x WARPS_N], K-loop over BK; dequant on CUDA cores, MMA on tensor
// cores). BF16 activations only — the f32-act large-M case keeps the CUDA-core
// tiled path (no test exercises f32-act large-M, and the model feeds bf16).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
namespace wmma = nvcuda::wmma;
#endif

template <typename Tout, int BM, int BN, int BK, int WARPS_M, int WARPS_N>
__global__ void MatmulNvfp4Wmma(Tout* out, const __nv_bfloat16* act, const uint8_t* packed,
                                const uint8_t* scale, float scale2, int64_t m_rows,
                                int64_t n_cols, int64_t k_dim) {
// bf16 WMMA tensor-core body is Ampere+ (bf16 fragments are complete only for
// __CUDA_ARCH__ >= 800). Guard so Turing/Volta/Pascal device passes compile
// this TU; sm_80+ keeps the body verbatim (preprocessor identity ->
// byte-identical). This TU is compiled for EVERY arch (CMakeLists.txt:903)
// because it also carries the generic bf16 MoE GEMMs, so it cannot simply be
// gated behind its fp4-mma cell. See cuda-arch-breadth-fp16.md §V0-b.
#if __CUDA_ARCH__ >= 800
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  constexpr int WMPER = BM / WARPS_M;  // rows a warp owns
  constexpr int WNPER = BN / WARPS_N;  // cols a warp owns
  constexpr int MF = WMPER / 16;       // wmma M-fragments per warp
  constexpr int NF = WNPER / 16;       // wmma N-fragments per warp
  __shared__ __nv_bfloat16 As[BM * BK];
  __shared__ __nv_bfloat16 Ws[BN * BK];
  __shared__ float Cs[BM * BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int wm = warp / WARPS_N;
  const int wn = warp % WARPS_N;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NF];
#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni) wmma::fill_fragment(acc[mi][ni], 0.0f);

  for (int64_t kt = 0; kt < k_dim; kt += BK) {
    // Stage the bf16 activation tile [BM][BK] (coalesced, act is K-contiguous).
    for (int idx = tid; idx < BM * BK; idx += kThreads) {
      const int r = idx / BK, c = idx % BK;
      const int64_t gr = row0 + r;
      As[idx] = (gr < m_rows && kt + c < k_dim) ? act[gr * k_dim + kt + c]
                                                : __float2bfloat16(0.0f);
    }
    // Decode the fp4 weight tile [BN][BK] into shared bf16 (coalesced packed-byte
    // reads; one byte -> two consecutive K elements), stored [BN][BK] so a
    // col_major wmma B-load with ldm=BK yields B[k,n] = weight[n,k].
    for (int idx = tid; idx < BN * (BK / 2); idx += kThreads) {
      const int nl = idx / (BK / 2), bc = idx % (BK / 2), kl = 2 * bc;
      const int64_t gn = col0 + nl;
      if (gn < n_cols && kt + kl < k_dim) {
        const uint8_t b = packed[gn * packed_cols + kt / 2 + bc];
        const float gs = F8E4M3ToF32Dev(scale[gn * groups + (kt + kl) / 16]) * scale2;
        float w_lo, w_hi;
        DecodeFp4Byte(b, gs, w_lo, w_hi);
        Ws[nl * BK + kl] = __float2bfloat16(w_lo);
        Ws[nl * BK + kl + 1] = __float2bfloat16(w_hi);
      } else {
        Ws[nl * BK + kl] = __float2bfloat16(0.0f);
        Ws[nl * BK + kl + 1] = __float2bfloat16(0.0f);
      }
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[MF];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[NF];
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
        wmma::load_matrix_sync(af[mi], &As[(wm * WMPER + mi * 16) * BK + kk * 16], BK);
#pragma unroll
      for (int ni = 0; ni < NF; ++ni)
        wmma::load_matrix_sync(bf[ni], &Ws[(wn * WNPER + ni * 16) * BK + kk * 16], BK);
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
#pragma unroll
        for (int ni = 0; ni < NF; ++ni) wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni)
      wmma::store_matrix_sync(&Cs[(wm * WMPER + mi * 16) * BN + (wn * WNPER + ni * 16)],
                              acc[mi][ni], BN, wmma::mem_row_major);
  __syncthreads();

  for (int idx = tid; idx < BM * BN; idx += kThreads) {
    const int r = idx / BN, c = idx % BN;
    const int64_t gr = row0 + r, gc = col0 + c;
    if (gr < m_rows && gc < n_cols) Store(out, gr * n_cols + gc, Cs[idx]);
  }
#else
  __trap();
#endif
}

// Below this many activation rows the naive one-thread-per-output kernel wins
// (weight L2-resident, tiling overhead not amortized); at/above it the tiled
// register-blocked kernel wins (weight streamed, coalescing + reuse pay off).
// Measured on GB10 (nsys, 2026-07-04): decode m=1..2 favors naive by ~2-3x,
// prefill m>=128 favors tiled by ~2x (and kills the multi-100ms lm_head spikes).
constexpr int64_t kTileMinRows = 32;

template <typename Tin, typename Tout>
void Launch(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& packed,
            const Tensor& scale, float scale2, int64_t m, int64_t n, int64_t k) {
  if (m < kTileMinRows) {
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(m));
    MatmulNvfp4KernelNaive<Tin, Tout><<<grid, kBlock, 0, s>>>(
        out.Ptr<Tout>(), act.Ptr<Tin>(), packed.Ptr<uint8_t>(), scale.Ptr<uint8_t>(), scale2, m, n,
        k, Fp4VecEnabled());
    Check(cudaGetLastError(), "kernel launch (naive)");
    return;
  }
  // BF16 activations at large M run on tensor cores (wmma bf16 -> f32). Tile
  // [BM x BN] = 64x64, BK=32, warps 2x2 (128 threads); each warp owns 32x32 =
  // 2x2 wmma m16n16k16 fragments. Dequant into shared bf16 stays on CUDA cores.
  if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
    if (WmmaEnabled()) {
    constexpr int BM = 64, BN = 64, BK = 32, WARPS_M = 2, WARPS_N = 2;
    constexpr int kThreads = WARPS_M * WARPS_N * 32;
    const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN),
                    static_cast<unsigned>((m + BM - 1) / BM));
    MatmulNvfp4Wmma<Tout, BM, BN, BK, WARPS_M, WARPS_N><<<grid, kThreads, 0, s>>>(
        out.Ptr<Tout>(), act.Ptr<__nv_bfloat16>(), packed.Ptr<uint8_t>(), scale.Ptr<uint8_t>(),
        scale2, m, n, k);
    Check(cudaGetLastError(), "kernel launch (wmma)");
    return;
    }
  }
  // GB10-tuned tile: [BM x BN] = 64x64, BK=32, [TM x TN] = 4x4 => 256
  // threads/block. Weight tile is loaded once per block and reused across all BM
  // rows; coalesced packed-byte + scale reads feed the register-blocked accumulate.
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
  constexpr int kThreads = (BM / TM) * (BN / TN);
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>((m + BM - 1) / BM));
  MatmulNvfp4Tiled<Tin, Tout, BM, BN, BK, TM, TN><<<grid, kThreads, 0, s>>>(
      out.Ptr<Tout>(), act.Ptr<Tin>(), packed.Ptr<uint8_t>(), scale.Ptr<uint8_t>(), scale2, m, n,
      k);
  Check(cudaGetLastError(), "kernel launch (tiled)");
}

template <typename Tin>
void LaunchByOut(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& packed,
                 const Tensor& scale, float scale2, int64_t m, int64_t n, int64_t k) {
  switch (out.dtype) {
    case DType::kF32:
      Launch<Tin, float>(s, out, act, packed, scale, scale2, m, n, k);
      break;
    case DType::kBF16:
      Launch<Tin, __nv_bfloat16>(s, out, act, packed, scale, scale2, m, n, k);
      break;
    default: VT_CHECK(false, "cuda matmul_nvfp4: unsupported out dtype (f32/bf16 only)");
  }
}

// --- Fused MoE grouped NVFP4 GEMM (M2.4) --------------------------------------
// out[p, n] = sum_k act[row(p), k] * bf16(e2m1(W_e[n,k]) * gscale_e[n,k/16]),
// where e = expert_ids[p], W_e = packed_ptrs[e]/scale_ptrs[e]/scale2s[e], and
// row(p) = row_map ? row_map[p] : p. One block over (p rows, N tiles) replaces
// the per-expert loop of MatmulNvfp4 launches — the launch-count reduction is
// the win (GB10 is launch-overhead bound). The per-element decode is byte-for-
// byte MatmulNvfp4Kernel's, so each output row is bit-identical to running the
// single-expert kernel for that pair's expert.
// Naive grouped GEMM (one thread per output, grid-strided over pair rows).
// Retained for SMALL P (decode): weight L2-resident, coalescing buys little.
template <typename Tin, typename Tout>
__global__ void MoeGroupedGemmNvfp4KernelNaive(Tout* out, const Tin* act, const int32_t* expert_ids,
                                               const int32_t* row_map, const int64_t* packed_ptrs,
                                               const int64_t* scale_ptrs, const float* scale2s,
                                               int64_t p_rows, int64_t n_cols, int64_t k_dim,
                                               bool use_vec) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= n_cols) return;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;
  for (int64_t p = blockIdx.y; p < p_rows; p += gridDim.y) {
    const int64_t e = expert_ids[p];
    const int64_t r = row_map != nullptr ? row_map[p] : p;
    const auto* packed = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(packed_ptrs[e]));
    const auto* scale = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(scale_ptrs[e]));
    const float scale2 = scale2s[e];
    const uint8_t* prow = packed + n * packed_cols;
    const uint8_t* srow = scale + n * groups;
    const Tin* arow = act + r * k_dim;
    Store(out, p * n_cols + n, Fp4ColDot(prow, srow, arow, k_dim, scale2, use_vec));
  }
}

// Tiled grouped GEMM: one block computes a [1 x BN] slice of a single pair row p
// (MoE pairs are token-major, NOT expert-sorted, so consecutive rows use
// different expert weights — no cross-row weight reuse to exploit). Within the
// block the activation row is loaded once per K-tile into shared and REUSED
// across the BN columns, and the fp4 weight tile [BN x BK] is decoded into shared
// with COALESCED packed-byte reads (the naive kernel had thread n stride a full
// weight row = uncoalesced). blockDim.x == BN (one thread per output column).
template <typename Tin, typename Tout, int BN, int BK>
__global__ void MoeGroupedGemmNvfp4Tiled(Tout* out, const Tin* act, const int32_t* expert_ids,
                                         const int32_t* row_map, const int64_t* packed_ptrs,
                                         const int64_t* scale_ptrs, const float* scale2s,
                                         int64_t p_rows, int64_t n_cols, int64_t k_dim) {
  __shared__ float As[BK];       // activation row slice [BK]
  __shared__ float Ws[BN * BK];  // decoded weight tile [BN][BK]

  const int tid = static_cast<int>(threadIdx.x);  // 0..BN-1 => output column
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;
  const int64_t gc = col0 + tid;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;

  // Grid-stride over the pair rows: P = T*top_k can exceed the gridDim.y max.
  for (int64_t p = blockIdx.y; p < p_rows; p += gridDim.y) {
    const int64_t e = expert_ids[p];
    const int64_t r = row_map != nullptr ? row_map[p] : p;
    const auto* packed = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(packed_ptrs[e]));
    const auto* scale = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(scale_ptrs[e]));
    const float scale2 = scale2s[e];
    const Tin* arow = act + r * k_dim;

    float acc = 0.0f;
    for (int64_t kt = 0; kt < k_dim; kt += BK) {
      // Guard the K edge (kt+idx >= K, partial last K-tile) with 0.
      for (int idx = tid; idx < BK; idx += BN)
        As[idx] = (kt + idx < k_dim) ? Load(arow, kt + idx) : 0.0f;
      for (int idx = tid; idx < BN * (BK / 2); idx += BN) {
        const int nl = idx / (BK / 2);
        const int bc = idx % (BK / 2);
        const int kl = 2 * bc;
        const int64_t gn = col0 + nl;
        if (gn < n_cols && kt + kl < k_dim) {
          const uint8_t b = packed[gn * packed_cols + kt / 2 + bc];
          const float gs = F8E4M3ToF32Dev(scale[gn * groups + (kt + kl) / 16]) * scale2;
          float w_lo, w_hi;
          DecodeFp4Byte(b, gs, w_lo, w_hi);
          Ws[nl * BK + kl] = w_lo;
          Ws[nl * BK + kl + 1] = w_hi;
        } else {
          Ws[nl * BK + kl] = 0.0f;
          Ws[nl * BK + kl + 1] = 0.0f;
        }
      }
      __syncthreads();
      if (gc < n_cols) {
#pragma unroll
        for (int k = 0; k < BK; ++k) acc += As[k] * Ws[tid * BK + k];
      }
      __syncthreads();
    }
    if (gc < n_cols) Store(out, p * n_cols + gc, acc);
  }
}

// --- Expert-grouped tensor-core (bf16 WMMA) MoE grouped GEMM (M2.6) ----------
// The MoE pairs are token-major (row p uses expert eids[p], NOT expert-sorted),
// so the per-pair tiled kernel above re-reads each expert's whole weight once per
// row (no cross-row reuse) AND cannot feed tensor cores (M=1 per weight). This
// path first GROUPS the P pair-rows by expert on device (counting sort), then
// runs a dense bf16 WMMA GEMM per expert over its contiguous row block — the
// weight is decoded once per BM-row tile (mirrors cutlass/Marlin grouped MoE:
// token-sorted activations + per-tile expert map, ragged tail masked). Grouping
// buys BOTH the ~BM-x weight-bandwidth reduction and the tensor-core MMA. Runs
// only for large P (prefill), which is never CUDA-graph-captured, so the
// stream-ordered scratch alloc below is safe; small-P decode keeps the naive
// kernel. Row bookkeeping over sorted positions:
//   sp[pos] = original pair index p (output row = out[sp[pos], :])
//   sr[pos] = source activation row = row_map ? row_map[p] : p
constexpr int kGroupBM = 64;

__global__ void MoeHistKernel(const int32_t* eids, int32_t* count, int64_t p_rows) {
  const int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p < p_rows) atomicAdd(&count[eids[p]], 1);
}

// Exclusive prefix sum of count[E] -> offset[E] (E small; single thread serial).
__global__ void MoeOffsetsKernel(const int32_t* count, int32_t* offset, int e_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int32_t acc = 0;
  for (int e = 0; e < e_count; ++e) {
    offset[e] = acc;
    acc += count[e];
  }
}

__global__ void MoeScatterKernel(const int32_t* eids, const int32_t* row_map,
                                 const int32_t* offset, int32_t* cursor, int32_t* sp, int32_t* sr,
                                 int64_t p_rows) {
  const int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= p_rows) return;
  const int e = eids[p];
  const int pos = offset[e] + atomicAdd(&cursor[e], 1);
  sp[pos] = static_cast<int32_t>(p);
  sr[pos] = row_map != nullptr ? row_map[p] : static_cast<int32_t>(p);
}

// Build the ragged per-BM-tile expert map (no tile crosses an expert boundary);
// pad the unused tail tiles with tile_rows=0 (grid.y is an upper bound). Single
// thread serial over E (small).
__global__ void MoeTileMapKernel(const int32_t* count, const int32_t* offset, int32_t* tile_expert,
                                 int32_t* tile_row0, int32_t* tile_rows, int e_count, int bm,
                                 int max_tiles) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int t = 0;
  for (int e = 0; e < e_count; ++e) {
    const int c = count[e], o = offset[e];
    for (int off = 0; off < c; off += bm) {
      tile_expert[t] = e;
      tile_row0[t] = o + off;
      tile_rows[t] = (c - off < bm) ? (c - off) : bm;
      ++t;
    }
  }
  for (; t < max_tiles; ++t) tile_rows[t] = 0;
}

template <typename Tin, typename Tout, int BM, int BN, int BK, int WARPS_M, int WARPS_N>
__global__ void MoeGroupedGemmNvfp4Wmma(Tout* out, const Tin* act, const int32_t* sp,
                                        const int32_t* sr, const int64_t* packed_ptrs,
                                        const int64_t* scale_ptrs, const float* scale2s,
                                        const int32_t* tile_expert, const int32_t* tile_row0,
                                        const int32_t* tile_rows, int64_t n_cols, int64_t k_dim) {
// bf16 WMMA tensor-core body is Ampere+ (bf16 fragments are complete only for
// __CUDA_ARCH__ >= 800). Guard so Turing/Volta/Pascal device passes compile
// this TU; sm_80+ keeps the body verbatim (preprocessor identity ->
// byte-identical). This TU is compiled for EVERY arch (CMakeLists.txt:903)
// because it also carries the generic bf16 MoE GEMMs, so it cannot simply be
// gated behind its fp4-mma cell. See cuda-arch-breadth-fp16.md §V0-b.
#if __CUDA_ARCH__ >= 800
  const int rcount = tile_rows[blockIdx.y];
  if (rcount == 0) return;
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  constexpr int WMPER = BM / WARPS_M, WNPER = BN / WARPS_N;
  constexpr int MF = WMPER / 16, NF = WNPER / 16;
  __shared__ __nv_bfloat16 As[BM * BK];
  __shared__ __nv_bfloat16 Ws[BN * BK];
  __shared__ float Cs[BM * BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int e = tile_expert[blockIdx.y];
  const int row0 = tile_row0[blockIdx.y];
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;
  const int64_t packed_cols = k_dim / 2, groups = k_dim / 16;
  const auto* packed = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(packed_ptrs[e]));
  const auto* scale = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(scale_ptrs[e]));
  const float scale2 = scale2s[e];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NF];
#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni) wmma::fill_fragment(acc[mi][ni], 0.0f);

  for (int64_t kt = 0; kt < k_dim; kt += BK) {
    for (int idx = tid; idx < BM * BK; idx += kThreads) {
      const int r = idx / BK, c = idx % BK;
      __nv_bfloat16 v = __float2bfloat16(0.0f);
      if (r < rcount && kt + c < k_dim) {
        const int64_t srow = sr[row0 + r];
        v = static_cast<__nv_bfloat16>(act[srow * k_dim + kt + c]);
      }
      As[idx] = v;
    }
    for (int idx = tid; idx < BN * (BK / 2); idx += kThreads) {
      const int nl = idx / (BK / 2), bc = idx % (BK / 2), kl = 2 * bc;
      const int64_t gn = col0 + nl;
      if (gn < n_cols && kt + kl < k_dim) {
        const uint8_t b = packed[gn * packed_cols + kt / 2 + bc];
        const float gs = F8E4M3ToF32Dev(scale[gn * groups + (kt + kl) / 16]) * scale2;
        float w_lo, w_hi;
        DecodeFp4Byte(b, gs, w_lo, w_hi);
        Ws[nl * BK + kl] = __float2bfloat16(w_lo);
        Ws[nl * BK + kl + 1] = __float2bfloat16(w_hi);
      } else {
        Ws[nl * BK + kl] = __float2bfloat16(0.0f);
        Ws[nl * BK + kl + 1] = __float2bfloat16(0.0f);
      }
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[MF];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[NF];
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
        wmma::load_matrix_sync(af[mi], &As[(wm * WMPER + mi * 16) * BK + kk * 16], BK);
#pragma unroll
      for (int ni = 0; ni < NF; ++ni)
        wmma::load_matrix_sync(bf[ni], &Ws[(wn * WNPER + ni * 16) * BK + kk * 16], BK);
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
#pragma unroll
        for (int ni = 0; ni < NF; ++ni) wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni)
      wmma::store_matrix_sync(&Cs[(wm * WMPER + mi * 16) * BN + (wn * WNPER + ni * 16)],
                              acc[mi][ni], BN, wmma::mem_row_major);
  __syncthreads();

  for (int idx = tid; idx < BM * BN; idx += kThreads) {
    const int r = idx / BN, c = idx % BN;
    const int64_t gc = col0 + c;
    if (r < rcount && gc < n_cols) Store(out, static_cast<int64_t>(sp[row0 + r]) * n_cols + gc,
                                         Cs[idx]);
  }
#else
  __trap();
#endif
}

// Persistent scratch for the MoE expert-grouping counting-sort + ragged tile map
// (grown on demand, kept alive) so the decode path never pays the per-call
// cudaMallocAsync/cudaFreeAsync — at conc-8 the grouped GEMM runs ~80x/step, and
// the stream-ordered alloc/free serialize behind the (occupancy-bound) kernel.
// Mirrors the argmax persistent-scratch discipline (cuda_sample.cu). The model
// forward drives ONE CUDA stream, so consecutive grouped-GEMM calls are already
// serialized on the stream => reusing one buffer is race-free (the hist/scatter/
// tilemap kernels rewrite it every call). cudaMalloc (not async) fires once at
// warmup / on growth, never on the steady-state decode path.
int32_t* g_moe_scratch = nullptr;
size_t g_moe_scratch_cap = 0;  // capacity in int32 elements
int32_t* EnsureMoeScratch(int64_t elems) {
  if (static_cast<size_t>(elems) > g_moe_scratch_cap) {
    // RETIRE the old block instead of freeing it: the MoE grouped-GEMM index
    // scratch pointer is baked into the captured pure-decode CUDA graph (the
    // WMMA-scatter memset/hist/scatter/tilemap nodes). A later, larger forward
    // (a bigger co-scheduled prefill or a larger decode batch — only at
    // concurrency > 1) grows this buffer; freeing the old block here would dangle
    // the captured graph's pointer → illegal memory access on the next replay.
    // See graph_safe_scratch.h. (Growth is O(log); retired memory is negligible.)
    RetireGraphScratch(g_moe_scratch);
    Check(cudaMalloc(&g_moe_scratch, static_cast<size_t>(elems) * sizeof(int32_t)),
          "moe grouped persistent scratch");
    g_moe_scratch_cap = static_cast<size_t>(elems);
  }
  return g_moe_scratch;
}

// Small-M decode gate for the grouped WMMA path: at/below this many (token,expert)
// pairs we run the decode-specialized BM=16 tile (mirrors vLLM fused_moe's
// BLOCK_SIZE_M=16 for small M — moe_align_block_size pads each expert's ragged
// token run to block_size, one block per (expert-segment x N-tile), NOT a full
// prefill tile per expert). conc-8 decode = 8 tok x top_k 8 = 64 pairs; conc-64
// still fits. Above it (prefill), the BM=64 tile stays byte-identical.
constexpr int64_t kMoeDecodeMaxP = 512;
constexpr int kMoeDecodeBM = 16;

// Groups P pairs by expert on device (the moe_align_block_size analog) then runs
// the WMMA grouped GEMM over the ragged per-tile expert map. Templated on the
// M-tile so the decode path can size the tile to the ~1-row-per-expert reality
// (BM=16, one warp of M) while prefill keeps the large-M BM=64 tile. Persistent
// scratch (no per-call malloc/free). Not graph-captured on either path.
template <typename Tin, typename Tout, int BM, int WARPS_M, int WARPS_N>
void LaunchGroupedWmma(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                       const Tensor* row_map, const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                       const Tensor& scale2s, int64_t p, int64_t n, int64_t k, int64_t e_count) {
  constexpr int BN = 64, BK = 32;
  const int max_tiles = static_cast<int>((p + BM - 1) / BM + e_count);
  // Single scratch block, int32 sub-slices: count,offset,cursor [E]; sp,sr [P];
  // tile_expert,tile_row0,tile_rows [max_tiles].
  const int64_t n_i32 = 3 * e_count + 2 * p + 3 * max_tiles;
  int32_t* scratch = EnsureMoeScratch(n_i32);
  int32_t* count = scratch;
  int32_t* offset = count + e_count;
  int32_t* cursor = offset + e_count;
  int32_t* sp = cursor + e_count;
  int32_t* sr = sp + p;
  int32_t* tile_expert = sr + p;
  int32_t* tile_row0 = tile_expert + max_tiles;
  int32_t* tile_rows = tile_row0 + max_tiles;

  Check(cudaMemsetAsync(count, 0, static_cast<size_t>(e_count) * sizeof(int32_t), s), "memset count");
  Check(cudaMemsetAsync(cursor, 0, static_cast<size_t>(e_count) * sizeof(int32_t), s),
        "memset cursor");
  const int32_t* eids = expert_ids.Ptr<int32_t>();
  const int32_t* rmap = row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr;
  const int hb = 256;
  MoeHistKernel<<<static_cast<unsigned>((p + hb - 1) / hb), hb, 0, s>>>(eids, count, p);
  MoeOffsetsKernel<<<1, 1, 0, s>>>(count, offset, static_cast<int>(e_count));
  MoeScatterKernel<<<static_cast<unsigned>((p + hb - 1) / hb), hb, 0, s>>>(eids, rmap, offset,
                                                                          cursor, sp, sr, p);
  MoeTileMapKernel<<<1, 1, 0, s>>>(count, offset, tile_expert, tile_row0, tile_rows,
                                   static_cast<int>(e_count), BM, max_tiles);

  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>(max_tiles));
  MoeGroupedGemmNvfp4Wmma<Tin, Tout, BM, BN, BK, WARPS_M, WARPS_N><<<grid, kThreads, 0, s>>>(
      out.Ptr<Tout>(), act.Ptr<Tin>(), sp, sr, packed_ptrs.Ptr<int64_t>(),
      scale_ptrs.Ptr<int64_t>(), scale2s.Ptr<float>(), tile_expert, tile_row0, tile_rows, n, k);
  Check(cudaGetLastError(), "moe_grouped_gemm_nvfp4 kernel launch (wmma)");
}

template <typename Tin, typename Tout>
void LaunchGrouped(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                   const Tensor* row_map, const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                   const Tensor& scale2s, int64_t p, int64_t n, int64_t k, int64_t e_count) {
  const int64_t y = p < 65535 ? p : 65535;  // grid.y max; kernel strides over p
  if (p < kTileMinRows) {
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(y));
    MoeGroupedGemmNvfp4KernelNaive<Tin, Tout><<<grid, kBlock, 0, s>>>(
        out.Ptr<Tout>(), act.Ptr<Tin>(), expert_ids.Ptr<int32_t>(),
        row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr, packed_ptrs.Ptr<int64_t>(),
        scale_ptrs.Ptr<int64_t>(), scale2s.Ptr<float>(), p, n, k, Fp4VecEnabled());
    Check(cudaGetLastError(), "moe_grouped_gemm_nvfp4 kernel launch (naive)");
    return;
  }
  // bf16: expert-grouped tensor-core (WMMA) path. Decode/small-M (p small) runs
  // the BM=16 decode tile (vLLM BLOCK_SIZE_M=16 for small M); prefill/large-M
  // keeps the byte-identical BM=64 tile.
  if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
    if (WmmaEnabled()) {
      if (MoeDecodeEnabled() && p <= kMoeDecodeMaxP) {
        LaunchGroupedWmma<Tin, Tout, kMoeDecodeBM, 1, 2>(s, out, act, expert_ids, row_map,
                                                         packed_ptrs, scale_ptrs, scale2s, p, n, k,
                                                         e_count);
      } else {
        LaunchGroupedWmma<Tin, Tout, kGroupBM, 2, 2>(s, out, act, expert_ids, row_map, packed_ptrs,
                                                     scale_ptrs, scale2s, p, n, k, e_count);
      }
      return;
    }
  }
  constexpr int BN = 128, BK = 32;  // 128 threads/block; weight tile 128x32
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>(y));
  MoeGroupedGemmNvfp4Tiled<Tin, Tout, BN, BK><<<grid, BN, 0, s>>>(
      out.Ptr<Tout>(), act.Ptr<Tin>(), expert_ids.Ptr<int32_t>(),
      row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr, packed_ptrs.Ptr<int64_t>(),
      scale_ptrs.Ptr<int64_t>(), scale2s.Ptr<float>(), p, n, k);
  Check(cudaGetLastError(), "moe_grouped_gemm_nvfp4 kernel launch (tiled)");
}

template <typename Tin>
void LaunchGroupedByOut(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                        const Tensor* row_map, const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                        const Tensor& scale2s, int64_t p, int64_t n, int64_t k, int64_t e_count) {
  switch (out.dtype) {
    case DType::kF32:
      LaunchGrouped<Tin, float>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs, scale2s,
                                p, n, k, e_count);
      break;
    case DType::kBF16:
      LaunchGrouped<Tin, __nv_bfloat16>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs,
                                        scale2s, p, n, k, e_count);
      break;
    default: VT_CHECK(false, "cuda moe_grouped_gemm_nvfp4: unsupported out dtype (f32/bf16 only)");
  }
}

// --- BF16 grouped-MoE GEMM (Qwen3-Coder Qwen3MoeForCausalLM fast path) ---------
// The dtype-native analog of the NVFP4 grouped MoE GEMM above: identical pair
// scheduling (naive one-thread-per-output for small/decode P; expert counting-
// sort + bf16 WMMA tensor-core tiles for large/prefill P, reusing the SAME
// MoeHist/MoeOffsets/MoeScatter/MoeTileMap + EnsureMoeScratch scaffolding) with
// the on-the-fly fp4 decode replaced by a direct bf16 weight read. The per-expert
// weight is bf16 [K, N] (loader-transposed Matmul-B layout, element (k,n) at
// k*N+n). f32 accumulation matches the reference ExpertMlp's cuBLASLt MatmulF32/
// MatmulBf16 (both f32-accumulate tensor-core GEMMs), so each output row stays
// within the near-tie band of the per-expert reference path it replaces.

// Naive grouped GEMM (one thread per output column, grid-strided over pair rows).
// Retained for SMALL P (decode): weight L2-resident, sequential-k f32 dot mirrors
// the reference GEMM accumulation shape.
template <typename Tout>
__global__ void MoeGroupedGemmBf16Naive(Tout* out, const __nv_bfloat16* act,
                                        const int32_t* expert_ids, const int32_t* row_map,
                                        const int64_t* weight_ptrs, int64_t p_rows, int64_t n_cols,
                                        int64_t k_dim) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= n_cols) return;
  for (int64_t p = blockIdx.y; p < p_rows; p += gridDim.y) {
    const int64_t e = expert_ids[p];
    const int64_t r = row_map != nullptr ? row_map[p] : p;
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(weight_ptrs[e]));
    const __nv_bfloat16* arow = act + r * k_dim;
    float acc = 0.0f;
    for (int64_t kk = 0; kk < k_dim; ++kk)
      acc += __bfloat162float(arow[kk]) * __bfloat162float(w[kk * n_cols + n]);
    Store(out, p * n_cols + n, acc);
  }
}

// --- W6: DETERMINISTIC split-K for the small-P (decode) grouped GEMM ----------
// At Qwen3-Coder decode the naive kernel is launched with grid (ceil(N/256), P).
// For c=1 that is P = 1 token x top_k 8 = 8 pairs and N = 768 (gate/up), i.e.
// THREE x EIGHT = 24 thread blocks — a handful of SMs, against a kernel that is
// pure weight streaming (8 experts x [2048,768] bf16 = 25.2 MB per launch).
// MEASURED at W5: 151 GB/s = 55% of GB10's ~273 GB/s for gate/up (N=768, 24
// blocks) vs 211 GB/s = 77% for down (N=2048, 64 blocks) — the SAME bytes, so
// the deficit is block-parallelism starvation, not the memory system.
//
// Fix: partition the K reduction across SPLITS blocks, mirroring vLLM's
// `SPLIT_K` knob on the same kernel (`vllm/model_executor/layers/fused_moe/
// fused_moe.py:338` `SPLIT_K: tl.constexpr`, threaded through `get_default_config`
// at :1254/:1300/:1350). We deliberately do NOT mirror an atomicAdd accumulation:
// float atomics reduce in nondeterministic ORDER, which would make greedy decode
// non-reproducible run-to-run and break the token-exact gate. Instead each split
// writes an f32 PARTIAL and a second pass sums the partials in FIXED ascending
// split order — bit-reproducible, and (unlike a serial-k dot) the partial sums
// are a strict re-association, so only the near-tie band can move.
template <typename Tacc>
__global__ void MoeGroupedGemmBf16NaiveSplitK(Tacc* partials, const __nv_bfloat16* act,
                                              const int32_t* expert_ids, const int32_t* row_map,
                                              const int64_t* weight_ptrs, int64_t p_rows,
                                              int64_t n_cols, int64_t k_dim, int64_t k_chunk) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= n_cols) return;
  const int64_t split = blockIdx.z;
  const int64_t k0 = split * k_chunk;
  if (k0 >= k_dim) return;
  const int64_t k1 = (k0 + k_chunk < k_dim) ? (k0 + k_chunk) : k_dim;
  for (int64_t p = blockIdx.y; p < p_rows; p += gridDim.y) {
    const int64_t e = expert_ids[p];
    const int64_t r = row_map != nullptr ? row_map[p] : p;
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(weight_ptrs[e]));
    const __nv_bfloat16* arow = act + r * k_dim;
    float acc = 0.0f;
    for (int64_t kk = k0; kk < k1; ++kk)
      acc += __bfloat162float(arow[kk]) * __bfloat162float(w[kk * n_cols + n]);
    partials[(split * p_rows + p) * n_cols + n] = acc;
  }
}

// Fixed ascending-split reduction of the split-K partials (deterministic).
template <typename Tout>
__global__ void MoeGroupedGemmBf16SplitKReduce(Tout* out, const float* partials, int64_t p_rows,
                                               int64_t n_cols, int splits) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = p_rows * n_cols;
  if (idx >= total) return;
  float acc = 0.0f;
  for (int s = 0; s < splits; ++s) acc += partials[static_cast<int64_t>(s) * total + idx];
  Store(out, idx, acc);
}

// Graph-safe persistent f32 partial buffer for the split-K decode path — same
// retire-don't-free contract as EnsureMoeScratch (the pointer is baked into the
// captured pure-decode graph).
float* g_moe_partials = nullptr;
size_t g_moe_partials_cap = 0;
float* EnsureMoePartials(int64_t elems) {
  if (static_cast<size_t>(elems) > g_moe_partials_cap) {
    RetireGraphScratch(g_moe_partials);
    Check(cudaMalloc(&g_moe_partials, static_cast<size_t>(elems) * sizeof(float)),
          "moe split-k persistent partials");
    g_moe_partials_cap = static_cast<size_t>(elems);
  }
  return g_moe_partials;
}

// Split count: purely a function of the (deterministic) launch SHAPE, so the
// reduction tree — and therefore the emitted token — is reproducible. Targets
// ~4 blocks per SM on GB10 while keeping each split's K run long enough
// (>= kMoeSplitKMinChunk) that the per-split launch/reduce overhead stays small.
constexpr int kMoeSplitKTargetBlocks = 256;
constexpr int64_t kMoeSplitKMinChunk = 256;
int MoeSplitKCount(int64_t p, int64_t n, int64_t k, int block) {
  const int64_t base = ((n + block - 1) / block) * (p < 65535 ? p : 65535);
  if (base <= 0) return 1;
  int64_t want = (kMoeSplitKTargetBlocks + base - 1) / base;
  const int64_t cap = k / kMoeSplitKMinChunk;
  if (want > cap) want = cap;
  if (want < 1) want = 1;
  if (want > 16) want = 16;
  return static_cast<int>(want);
}

// VT_MOE_SPLIT_K=0 rolls back to the single-pass naive kernel (same-binary A/B).
bool Bf16SplitKEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_SPLIT_K");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// Expert-grouped tensor-core (bf16 WMMA) grouped GEMM: byte-for-byte the same
// tiling/MMA as MoeGroupedGemmNvfp4Wmma (counting-sorted pair rows, ragged
// per-BM-tile expert map), but the weight tile is read directly from the bf16
// [K, N] expert buffer instead of decoded from fp4. sp/sr/tile_* are the sort
// outputs (identical semantics to the NVFP4 path).
template <typename Tout, int BM, int BN, int BK, int WARPS_M, int WARPS_N>
__global__ void MoeGroupedGemmBf16Wmma(Tout* out, const __nv_bfloat16* act, const int32_t* sp,
                                       const int32_t* sr, const int64_t* weight_ptrs,
                                       const int32_t* tile_expert, const int32_t* tile_row0,
                                       const int32_t* tile_rows, int64_t n_cols, int64_t k_dim) {
// bf16 WMMA tensor-core body is Ampere+ (bf16 fragments are complete only for
// __CUDA_ARCH__ >= 800). Guard so Turing/Volta/Pascal device passes compile
// this TU; sm_80+ keeps the body verbatim (preprocessor identity ->
// byte-identical). This TU is compiled for EVERY arch (CMakeLists.txt:903)
// because it also carries the generic bf16 MoE GEMMs, so it cannot simply be
// gated behind its fp4-mma cell. See cuda-arch-breadth-fp16.md §V0-b.
#if __CUDA_ARCH__ >= 800
  const int rcount = tile_rows[blockIdx.y];
  if (rcount == 0) return;
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  constexpr int WMPER = BM / WARPS_M, WNPER = BN / WARPS_N;
  constexpr int MF = WMPER / 16, NF = WNPER / 16;
  __shared__ __nv_bfloat16 As[BM * BK];
  __shared__ __nv_bfloat16 Ws[BN * BK];
  __shared__ float Cs[BM * BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int e = tile_expert[blockIdx.y];
  const int row0 = tile_row0[blockIdx.y];
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;
  const auto* w = reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(weight_ptrs[e]));

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NF];
#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni) wmma::fill_fragment(acc[mi][ni], 0.0f);

  for (int64_t kt = 0; kt < k_dim; kt += BK) {
    for (int idx = tid; idx < BM * BK; idx += kThreads) {
      const int r = idx / BK, c = idx % BK;
      __nv_bfloat16 v = __float2bfloat16(0.0f);
      if (r < rcount && kt + c < k_dim) {
        const int64_t srow = sr[row0 + r];
        v = act[srow * k_dim + kt + c];
      }
      As[idx] = v;
    }
    for (int idx = tid; idx < BN * BK; idx += kThreads) {
      const int nl = idx / BK, kl = idx % BK;
      const int64_t gn = col0 + nl, gk = kt + kl;
      Ws[nl * BK + kl] =
          (gn < n_cols && gk < k_dim) ? w[gk * n_cols + gn] : __float2bfloat16(0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[MF];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[NF];
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
        wmma::load_matrix_sync(af[mi], &As[(wm * WMPER + mi * 16) * BK + kk * 16], BK);
#pragma unroll
      for (int ni = 0; ni < NF; ++ni)
        wmma::load_matrix_sync(bf[ni], &Ws[(wn * WNPER + ni * 16) * BK + kk * 16], BK);
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
#pragma unroll
        for (int ni = 0; ni < NF; ++ni) wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni)
      wmma::store_matrix_sync(&Cs[(wm * WMPER + mi * 16) * BN + (wn * WNPER + ni * 16)],
                              acc[mi][ni], BN, wmma::mem_row_major);
  __syncthreads();

  for (int idx = tid; idx < BM * BN; idx += kThreads) {
    const int r = idx / BN, c = idx % BN;
    const int64_t gc = col0 + c;
    if (r < rcount && gc < n_cols)
      Store(out, static_cast<int64_t>(sp[row0 + r]) * n_cols + gc, Cs[idx]);
  }
#else
  __trap();
#endif
}

// --- W6: PIPELINED bf16 grouped-MoE WMMA tile (the tuned prefill/decode GEMM) --
//
// The W5 tile above is a first-cut GEMM: BN=64, BK=32, ONE stage, and — the
// dominant defect — its weight stage reads `w[gk * n_cols + gn]` with the
// FASTEST-VARYING loop index on `gk`, i.e. consecutive lanes of a warp touch
// addresses `n_cols` elements apart. That is a fully UNCOALESCED global read of
// the [K,N] expert weight (32 distinct 32-byte sectors per warp, 2 useful bytes
// each). Since a 1024-token Qwen3-Coder prefill streams ~1.2 GB of expert weight
// per layer against GB10's ~273 GB/s, the MoE GEMM is WEIGHT-BANDWIDTH bound
// (arithmetic intensity ~64 flop/byte vs the machine's ~900), so the uncoalesced
// weight stage — not the MMA — is what pinned the W5 kernel at ~4.3 TFLOP/s.
//
// This kernel keeps the SAME per-output K-reduction ORDER (ascending k, one
// m16n16k16 wmma accumulate per 16 k, accumulator carried across K-tiles), so it
// is BIT-IDENTICAL to the W5 tile — only the staging is rewritten:
//   (1) COALESCED + VECTORIZED weight stage — shared `Ws` is [BK][BN] (k-major)
//       and the fragment is loaded as `wmma::matrix_b` in ROW_MAJOR (which for
//       matrix_b means K-major, element (k,n) at ptr[k*ldm+n]), so the global
//       read runs along N, which is contiguous in the [K,N] weight. Each lane
//       moves 16 bytes (8 bf16).
//   (2) 16-byte VECTORIZED activation stage, likewise.
//   (3) MULTI-STAGE `cp.async` PIPELINE (STAGES deep) so the DRAM latency of
//       k-tile t+STAGES-1 overlaps the MMA of tile t, replacing the W5 kernel's
//       full `__syncthreads` stall per k-tile.
//   (4) BN 64 -> 128 (wider weight tile per block => fewer redundant weight
//       streams) and 4 warps.
//   (5) SHARED-MEMORY PADDING (+8 halves per row, 16-byte aligned) so the wmma
//       fragment loads are at most 2-way bank conflicted.
//
// Upstream grounding: the tile/pipeline SHAPE mirrors vLLM's Triton grouped-MoE
// GEMM — `vllm/model_executor/layers/fused_moe/fused_moe.py:294` (`fused_moe_kernel`:
// the `for k in range(cdiv(K, BLOCK_SIZE_K))` accumulate-into-f32 mainloop over
// [BLOCK_SIZE_M, BLOCK_SIZE_K] x [BLOCK_SIZE_K, BLOCK_SIZE_N] tiles) with the tile
// constants from `fused_moe.py:1238` (`get_default_config`, the bf16 branch:
// `block_n = 64 if M <= 64 else 128`, `block_k = 128 if fp8 or M <= 64 else 64`,
// `num_stages = 4 if M <= 32 else 3`, `num_warps = 4 if M <= 128 else 8`).
// Triton's `num_stages` lowers to exactly this `cp.async` multistage mainloop,
// whose C++ form is CUTLASS `include/cutlass/gemm/threadblock/mma_multistage.h`
// (prologue of STAGES-1 `cp.async` groups, then `cp_async_wait<STAGES-2>` +
// issue-next-inside-the-loop). We keep BK=32 rather than vLLM's 64 because at
// BN=128/STAGES=3 that is what fits the 48 KB static shared-memory budget while
// still holding a 3-deep pipeline (a 2-stage BK=64 variant needs 52 KB and is
// strictly shallower).
//
// `sp`/`sr`/`tile_*` are the counting-sort outputs, identical to the W5 tile.
template <typename Tout, int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES>
__global__ __launch_bounds__(WARPS_M* WARPS_N * 32) void MoeGroupedGemmBf16WmmaPipe(
    Tout* out, const __nv_bfloat16* act, const int32_t* sp, const int32_t* sr,
    const int64_t* weight_ptrs, const int32_t* tile_expert, const int32_t* tile_row0,
    const int32_t* tile_rows, int64_t n_cols, int64_t k_dim) {
// bf16 WMMA tensor-core body is Ampere+ (bf16 fragments are complete only for
// __CUDA_ARCH__ >= 800). Guard so Turing/Volta/Pascal device passes compile
// this TU; sm_80+ keeps the body verbatim (preprocessor identity ->
// byte-identical). This TU is compiled for EVERY arch (CMakeLists.txt:903)
// because it also carries the generic bf16 MoE GEMMs, so it cannot simply be
// gated behind its fp4-mma cell. See cuda-arch-breadth-fp16.md §V0-b.
#if __CUDA_ARCH__ >= 800
  const int rcount = tile_rows[blockIdx.y];
  if (rcount == 0) return;
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  constexpr int WMPER = BM / WARPS_M, WNPER = BN / WARPS_N;
  constexpr int MF = WMPER / 16, NF = WNPER / 16;
  constexpr int kVec = 8;              // 8 bf16 = the 16-byte cp.async granule
  constexpr int kPad = kVec;           // row padding, keeps every row 16B-aligned
  constexpr int ALD = BK + kPad;       // As row stride, halves (m-major)
  constexpr int WLD = BN + kPad;       // Ws row stride, halves (k-major)
  constexpr int kASize = BM * ALD, kWSize = BK * WLD;
  constexpr int kAChunks = BM * (BK / kVec), kWChunks = BK * (BN / kVec);
  // One block-wide arena: the staging buffers during the mainloop, reinterpreted
  // as the f32 output tile in the epilogue (the epilogue runs after the last
  // consumer sync, so the two lifetimes never overlap).
  constexpr int kStageBytes = STAGES * (kASize + kWSize) * static_cast<int>(sizeof(__nv_bfloat16));
  constexpr int kOutBytes = BM * BN * static_cast<int>(sizeof(float));
  constexpr int kSmemBytes = kStageBytes > kOutBytes ? kStageBytes : kOutBytes;
  __shared__ __align__(16) char smem[kSmemBytes];
  auto* As = reinterpret_cast<__nv_bfloat16*>(smem);
  auto* Ws = As + STAGES * kASize;
  auto* Cs = reinterpret_cast<float*>(smem);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int e = tile_expert[blockIdx.y];
  const int row0 = tile_row0[blockIdx.y];
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;
  const auto* w = reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(weight_ptrs[e]));

  // Stage k-tile `kt` into buffer `st`. Every lane issues 16-byte `cp.async`
  // copies; out-of-range TAIL elements use the primitive's `zfill` (which zeroes
  // the trailing bytes of the granule) so ragged K/N/row counts stay exact
  // without a masked scalar path. `src` for a fully-invalid granule points at the
  // buffer base (always readable; nothing is copied when zfill == 16).
  auto stage_in = [&](int st, int64_t kt) {
    __nv_bfloat16* Ad = As + st * kASize;
    __nv_bfloat16* Wd = Ws + st * kWSize;
    for (int idx = tid; idx < kAChunks; idx += kThreads) {
      const int r = idx / (BK / kVec), c0 = (idx % (BK / kVec)) * kVec;
      const int64_t gk = kt + c0;
      const __nv_bfloat16* src = act;
      int valid = 0;
      if (r < rcount && gk < k_dim) {
        src = act + static_cast<int64_t>(sr[row0 + r]) * k_dim + gk;
        const int64_t rem = k_dim - gk;
        valid = static_cast<int>(rem < kVec ? rem : kVec);
      }
      __pipeline_memcpy_async(Ad + r * ALD + c0, src, 16,
                              static_cast<size_t>((kVec - valid) * 2));
    }
    for (int idx = tid; idx < kWChunks; idx += kThreads) {
      const int kl = idx / (BN / kVec), n0 = (idx % (BN / kVec)) * kVec;
      const int64_t gk = kt + kl, gn = col0 + n0;
      const __nv_bfloat16* src = w;
      int valid = 0;
      if (gk < k_dim && gn < n_cols) {
        src = w + gk * n_cols + gn;
        const int64_t rem = n_cols - gn;
        valid = static_cast<int>(rem < kVec ? rem : kVec);
      }
      __pipeline_memcpy_async(Wd + kl * WLD + n0, src, 16,
                              static_cast<size_t>((kVec - valid) * 2));
    }
    __pipeline_commit();
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NF];
#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni) wmma::fill_fragment(acc[mi][ni], 0.0f);

  const int64_t ktiles = (k_dim + BK - 1) / BK;
  // Prologue: always commit STAGES-1 groups (empty ones past the last k-tile) so
  // the `__pipeline_wait_prior(STAGES-2)` accounting below is uniform.
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ktiles)
      stage_in(s, static_cast<int64_t>(s) * BK);
    else
      __pipeline_commit();
  }

  for (int64_t t = 0; t < ktiles; ++t) {
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();  // stage t's copies visible; stage (t-1)'s readers all done
    const int64_t nt = t + STAGES - 1;
    // Past the last k-tile we still COMMIT an empty group: `__pipeline_wait_prior`
    // counts committed groups, so the commit count must stay 1:1 with the loop
    // index or the final tiles would be consumed before their copies land
    // (CUTLASS mma_multistage.h does the same — it issues the tail iterations
    // with all predicates false rather than skipping the commit).
    if (nt < ktiles)
      stage_in(static_cast<int>(nt % STAGES), nt * BK);
    else
      __pipeline_commit();

    const __nv_bfloat16* Ac = As + static_cast<int>(t % STAGES) * kASize;
    const __nv_bfloat16* Wc = Ws + static_cast<int>(t % STAGES) * kWSize;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[MF];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf[NF];
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
        wmma::load_matrix_sync(af[mi], Ac + (wm * WMPER + mi * 16) * ALD + kk * 16, ALD);
#pragma unroll
      for (int ni = 0; ni < NF; ++ni)
        wmma::load_matrix_sync(bf[ni], Wc + (kk * 16) * WLD + (wn * WNPER + ni * 16), WLD);
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
#pragma unroll
        for (int ni = 0; ni < NF; ++ni) wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
    }
  }

  __syncthreads();  // mainloop readers done before the arena becomes the C tile
#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni)
      wmma::store_matrix_sync(&Cs[(wm * WMPER + mi * 16) * BN + (wn * WNPER + ni * 16)],
                              acc[mi][ni], BN, wmma::mem_row_major);
  __syncthreads();

  for (int idx = tid; idx < BM * BN; idx += kThreads) {
    const int r = idx / BN, c = idx % BN;
    const int64_t gc = col0 + c;
    if (r < rcount && gc < n_cols)
      Store(out, static_cast<int64_t>(sp[row0 + r]) * n_cols + gc, Cs[idx]);
  }
#else
  __trap();
#endif
}

// The W6 pipelined tile stages 16-byte granules, so it requires the activation
// row pitch and the weight row pitch to be 8-bf16 multiples (every granule base
// is then 16-byte aligned). Production Qwen3-Coder shapes always are (K/N in
// {2048, 768}); ragged test/other shapes fall back to the W5 tile, which stays
// live as the layout-agnostic reference.
inline bool Bf16PipeShapeOk(int64_t n, int64_t k) { return (n % 8) == 0 && (k % 8) == 0; }

// VT_MOE_BF16_PIPE=0 rolls back to the W5 tile (same-binary A/B; the two are
// bit-identical, so this is a pure performance toggle).
bool Bf16PipeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_BF16_PIPE");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// PIPE_WN: the N-warp split of the W6 pipelined tile (BN=128); the W5 fallback
// tile keeps its own (BN=64, WARPS_M x WARPS_N) shape.
template <typename Tout, int BM, int WARPS_M, int WARPS_N, int PIPE_WM, int PIPE_WN>
void LaunchGroupedBf16Wmma(cudaStream_t s, Tensor& out, const Tensor& act,
                           const Tensor& expert_ids, const Tensor* row_map,
                           const Tensor& weight_ptrs, int64_t p, int64_t n, int64_t k,
                           int64_t e_count) {
  constexpr int BN = 64, BK = 32;
  constexpr int kPipeBN = 128, kPipeBK = 32, kPipeStages = 3;
  const int max_tiles = static_cast<int>((p + BM - 1) / BM + e_count);
  const int64_t n_i32 = 3 * e_count + 2 * p + 3 * max_tiles;
  int32_t* scratch = EnsureMoeScratch(n_i32);
  int32_t* count = scratch;
  int32_t* offset = count + e_count;
  int32_t* cursor = offset + e_count;
  int32_t* sp = cursor + e_count;
  int32_t* sr = sp + p;
  int32_t* tile_expert = sr + p;
  int32_t* tile_row0 = tile_expert + max_tiles;
  int32_t* tile_rows = tile_row0 + max_tiles;

  Check(cudaMemsetAsync(count, 0, static_cast<size_t>(e_count) * sizeof(int32_t), s),
        "memset count");
  Check(cudaMemsetAsync(cursor, 0, static_cast<size_t>(e_count) * sizeof(int32_t), s),
        "memset cursor");
  const int32_t* eids = expert_ids.Ptr<int32_t>();
  const int32_t* rmap = row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr;
  const int hb = 256;
  MoeHistKernel<<<static_cast<unsigned>((p + hb - 1) / hb), hb, 0, s>>>(eids, count, p);
  MoeOffsetsKernel<<<1, 1, 0, s>>>(count, offset, static_cast<int>(e_count));
  MoeScatterKernel<<<static_cast<unsigned>((p + hb - 1) / hb), hb, 0, s>>>(eids, rmap, offset,
                                                                          cursor, sp, sr, p);
  MoeTileMapKernel<<<1, 1, 0, s>>>(count, offset, tile_expert, tile_row0, tile_rows,
                                   static_cast<int>(e_count), BM, max_tiles);

  if (Bf16PipeEnabled() && Bf16PipeShapeOk(n, k)) {
    constexpr int kPipeThreads = PIPE_WM * PIPE_WN * 32;
    const dim3 pgrid(static_cast<unsigned>((n + kPipeBN - 1) / kPipeBN),
                     static_cast<unsigned>(max_tiles));
    MoeGroupedGemmBf16WmmaPipe<Tout, BM, kPipeBN, kPipeBK, PIPE_WM, PIPE_WN, kPipeStages>
        <<<pgrid, kPipeThreads, 0, s>>>(out.Ptr<Tout>(), act.Ptr<__nv_bfloat16>(), sp, sr,
                                        weight_ptrs.Ptr<int64_t>(), tile_expert, tile_row0,
                                        tile_rows, n, k);
    Check(cudaGetLastError(), "moe_grouped_gemm_bf16 kernel launch (wmma pipe)");
    return;
  }
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>(max_tiles));
  MoeGroupedGemmBf16Wmma<Tout, BM, BN, BK, WARPS_M, WARPS_N><<<grid, kThreads, 0, s>>>(
      out.Ptr<Tout>(), act.Ptr<__nv_bfloat16>(), sp, sr, weight_ptrs.Ptr<int64_t>(), tile_expert,
      tile_row0, tile_rows, n, k);
  Check(cudaGetLastError(), "moe_grouped_gemm_bf16 kernel launch (wmma)");
}

template <typename Tout>
void LaunchGroupedBf16(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                       const Tensor* row_map, const Tensor& weight_ptrs, int64_t p, int64_t n,
                       int64_t k, int64_t e_count) {
  // The <sm_80 arch term lives inside WmmaEnabled(), so this branch (and the
  // other five call sites) route to the WMMA-free naive / split-K path on a
  // pre-Ampere board without a local check.
  if (p < kTileMinRows || !WmmaEnabled()) {
    const int64_t y = p < 65535 ? p : 65535;
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(y));
    const int splits = Bf16SplitKEnabled() ? MoeSplitKCount(p, n, k, kBlock) : 1;
    if (splits > 1) {
      const int64_t total = p * n;
      float* partials = EnsureMoePartials(total * splits);
      const int64_t k_chunk = (k + splits - 1) / splits;
      const dim3 sgrid(grid.x, grid.y, static_cast<unsigned>(splits));
      MoeGroupedGemmBf16NaiveSplitK<float><<<sgrid, kBlock, 0, s>>>(
          partials, act.Ptr<__nv_bfloat16>(), expert_ids.Ptr<int32_t>(),
          row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr, weight_ptrs.Ptr<int64_t>(), p, n,
          k, k_chunk);
      Check(cudaGetLastError(), "moe_grouped_gemm_bf16 kernel launch (naive split-k)");
      constexpr int kRB = 256;
      MoeGroupedGemmBf16SplitKReduce<Tout>
          <<<static_cast<unsigned>((total + kRB - 1) / kRB), kRB, 0, s>>>(out.Ptr<Tout>(), partials,
                                                                         p, n, splits);
      Check(cudaGetLastError(), "moe_grouped_gemm_bf16 split-k reduce launch");
      return;
    }
    MoeGroupedGemmBf16Naive<Tout><<<grid, kBlock, 0, s>>>(
        out.Ptr<Tout>(), act.Ptr<__nv_bfloat16>(), expert_ids.Ptr<int32_t>(),
        row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr, weight_ptrs.Ptr<int64_t>(), p, n, k);
    Check(cudaGetLastError(), "moe_grouped_gemm_bf16 kernel launch (naive)");
    return;
  }
  // Large/prefill P: expert-grouped tensor-core (WMMA). Decode/small-M (p small)
  // runs the BM=16 decode tile (vLLM BLOCK_SIZE_M=16 for small M); prefill/large-M
  // keeps the byte-identical BM=64 tile.
  if (MoeDecodeEnabled() && p <= kMoeDecodeMaxP) {
    LaunchGroupedBf16Wmma<Tout, kMoeDecodeBM, 1, 2, 1, 4>(s, out, act, expert_ids, row_map,
                                                          weight_ptrs, p, n, k, e_count);
  } else {
    LaunchGroupedBf16Wmma<Tout, kGroupBM, 2, 2, 2, 2>(s, out, act, expert_ids, row_map, weight_ptrs,
                                                      p, n, k, e_count);
  }
}

void MoeGroupedGemmBf16KernelCuda(Queue& q, Tensor& out, const Tensor& act,
                                  const Tensor& expert_ids, const Tensor* row_map,
                                  const Tensor& weight_ptrs) {
  const int64_t p = out.shape[0], n = out.shape[1], k = act.shape[1];
  if (p == 0 || n == 0) return;
  const int64_t e_count = weight_ptrs.shape[0];
  cudaStream_t s = AsStream(q);
  switch (out.dtype) {
    case DType::kF32:
      LaunchGroupedBf16<float>(s, out, act, expert_ids, row_map, weight_ptrs, p, n, k, e_count);
      break;
    case DType::kBF16:
      LaunchGroupedBf16<__nv_bfloat16>(s, out, act, expert_ids, row_map, weight_ptrs, p, n, k,
                                       e_count);
      break;
    default: VT_CHECK(false, "cuda moe_grouped_gemm_bf16: unsupported out dtype (f32/bf16 only)");
  }
}

// --- Tier-A4: SHARED fused BF16 grouped-MoE gate+up+SwiGLU (kMoeGroupedGemmBf16-
// GateUpSilu) ------------------------------------------------------------------
// ONE vt entry replaces the {MoeGroupedGemmBf16(gate); MoeGroupedGemmBf16(up);
// MoeSiluMul} triplet the bf16 grouped-MoE archs ran (Qwen3-Coder, DeepSeek-V2,
// kimi). BIT-IDENTICAL to that composite in EVERY launch regime by construction:
//   * decode / non-WMMA (p < kTileMinRows || !WmmaEnabled): reuse the EXISTING
//     split-K partial kernel (MoeGroupedGemmBf16NaiveSplitK) once per weight into
//     two f32 partial buffers, then ONE fused reduce+SwiGLU launch that sums each
//     tower's partials in the SAME fixed ascending split order as
//     MoeGroupedGemmBf16SplitKReduce and applies the SAME silu(gate)*up math as
//     MoeSiluMulKernel. 5 launches -> 3; the two f32 [P,N] reduce outputs never
//     touch HBM (the reduce reads partials and writes only the bf16 act).
//   * prefill / WMMA (else): reuse LaunchGroupedBf16<float> twice into persistent
//     f32 gate/up scratch + the identical silu-mul — byte-for-byte the caller's
//     old three ops (no fused WMMA epilogue; prefill is compute-bound, launch
//     count is irrelevant there and a fused tensor-core epilogue would risk the
//     tuned path). Consistency win: ONE call site for all regimes.
// Both scratch buffers are persistent (retire-don't-free, the g_moe_partials
// contract) so a pointer baked into the captured pure-decode graph never dangles.

// Persistent f32 partials for the fused decode path (gate ++ up, so 2x the
// single-tower g_moe_partials capacity). Separate from g_moe_partials so the
// down-projection's own split-K (which reuses g_moe_partials) can never alias
// the gate/up partials within one captured graph.
float* g_moe_fused_partials = nullptr;
size_t g_moe_fused_partials_cap = 0;
float* EnsureMoeFusedPartials(int64_t elems) {
  if (static_cast<size_t>(elems) > g_moe_fused_partials_cap) {
    RetireGraphScratch(g_moe_fused_partials);
    Check(cudaMalloc(&g_moe_fused_partials, static_cast<size_t>(elems) * sizeof(float)),
          "moe fused gate/up persistent partials");
    g_moe_fused_partials_cap = static_cast<size_t>(elems);
  }
  return g_moe_fused_partials;
}

// Persistent f32 gate/up scratch for the WMMA composite branch (2 x [P,N]).
float* g_moe_fused_gateup = nullptr;
size_t g_moe_fused_gateup_cap = 0;
float* EnsureMoeFusedGateUp(int64_t elems) {
  if (static_cast<size_t>(elems) > g_moe_fused_gateup_cap) {
    RetireGraphScratch(g_moe_fused_gateup);
    Check(cudaMalloc(&g_moe_fused_gateup, static_cast<size_t>(elems) * sizeof(float)),
          "moe fused gate/up persistent f32 scratch");
    g_moe_fused_gateup_cap = static_cast<size_t>(elems);
  }
  return g_moe_fused_gateup;
}

// Fused reduce+SwiGLU: out[idx] = silu(sum_s gate_partials[s,idx]) *
// sum_s up_partials[s,idx], each sum in fixed ASCENDING split order. Combines the
// two MoeGroupedGemmBf16SplitKReduce passes and MoeSiluMulKernel into one launch;
// bit-identical to running all three (the f32 HBM round-trip the composite takes
// between reduce and silu is lossless).
template <typename Tout>
__global__ void MoeGroupedGemmBf16FusedGateUpReduceSilu(Tout* out, const float* gate_partials,
                                                        const float* up_partials, int64_t p_rows,
                                                        int64_t n_cols, int splits) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = p_rows * n_cols;
  if (idx >= total) return;
  float g = 0.0f, u = 0.0f;
  for (int s = 0; s < splits; ++s) {
    g += gate_partials[static_cast<int64_t>(s) * total + idx];
    u += up_partials[static_cast<int64_t>(s) * total + idx];
  }
  const float sv = g / (1.0f + expf(-g));  // == MoeSiluMulKernel (cuda_moe.cu)
  Store(out, idx, sv * u);
}

// Plain silu-mul over f32 gate/up -> bf16 (WMMA composite branch epilogue).
// Identical math to MoeSiluMulKernel; kept local so this TU need not reach into
// the cuda_moe.cu anonymous namespace.
template <typename Tout>
__global__ void MoeFusedSiluMulF32(Tout* out, const float* gate, const float* up, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step) {
    const float g = gate[i];
    const float sv = g / (1.0f + expf(-g));
    Store(out, i, sv * up[i]);
  }
}

// VT_MOE_BF16_FUSED_GATEUP=0 forces the composite (2x grouped GEMM + silu-mul)
// path in EVERY regime — the same-binary A/B rollback of the decode fusion. The
// fused decode path is bit-identical to the composite, so this only trades launch
// count, never numerics.
bool MoeBf16FusedGateUpEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_BF16_FUSED_GATEUP");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

void MoeGroupedGemmBf16GateUpSiluCuda(Queue& q, Tensor& out, const Tensor& act,
                                      const Tensor& expert_ids, const Tensor* row_map,
                                      const Tensor& gate_ptrs, const Tensor& up_ptrs) {
  const int64_t p = out.shape[0], n = out.shape[1], k = act.shape[1];
  if (p == 0 || n == 0) return;
  const int64_t e_count = gate_ptrs.shape[0];
  cudaStream_t s = AsStream(q);
  const int64_t total = p * n;
  __nv_bfloat16* out_ptr = out.Ptr<__nv_bfloat16>();

  if (MoeBf16FusedGateUpEnabled() && (p < kTileMinRows || !WmmaEnabled())) {
    // Decode / small-P: fuse via the split-K partial kernel + reduce+SwiGLU.
    const int64_t y = p < 65535 ? p : 65535;
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(y));
    const int splits = Bf16SplitKEnabled() ? MoeSplitKCount(p, n, k, kBlock) : 1;
    const int64_t part_elems = total * splits;
    float* base = EnsureMoeFusedPartials(part_elems * 2);
    float* gate_partials = base;
    float* up_partials = base + part_elems;
    const int64_t k_chunk = (k + splits - 1) / splits;
    const dim3 sgrid(grid.x, grid.y, static_cast<unsigned>(splits));
    const int32_t* rm = row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr;
    MoeGroupedGemmBf16NaiveSplitK<float><<<sgrid, kBlock, 0, s>>>(
        gate_partials, act.Ptr<__nv_bfloat16>(), expert_ids.Ptr<int32_t>(), rm,
        gate_ptrs.Ptr<int64_t>(), p, n, k, k_chunk);
    MoeGroupedGemmBf16NaiveSplitK<float><<<sgrid, kBlock, 0, s>>>(
        up_partials, act.Ptr<__nv_bfloat16>(), expert_ids.Ptr<int32_t>(), rm,
        up_ptrs.Ptr<int64_t>(), p, n, k, k_chunk);
    Check(cudaGetLastError(), "moe_grouped_gemm_bf16_gate_up_silu partials launch");
    constexpr int kRB = 256;
    MoeGroupedGemmBf16FusedGateUpReduceSilu<__nv_bfloat16>
        <<<static_cast<unsigned>((total + kRB - 1) / kRB), kRB, 0, s>>>(out_ptr, gate_partials,
                                                                        up_partials, p, n, splits);
    Check(cudaGetLastError(), "moe_grouped_gemm_bf16_gate_up_silu reduce+silu launch");
    return;
  }

  // Prefill / WMMA: reuse the tuned grouped GEMM twice into persistent f32 scratch
  // + the identical silu-mul (byte-for-byte the caller's old three ops).
  float* base = EnsureMoeFusedGateUp(total * 2);
  Tensor gate_f32 = Tensor::Contiguous(base, DType::kF32, act.device, {p, n});
  Tensor up_f32 = Tensor::Contiguous(base + total, DType::kF32, act.device, {p, n});
  LaunchGroupedBf16<float>(s, gate_f32, act, expert_ids, row_map, gate_ptrs, p, n, k, e_count);
  LaunchGroupedBf16<float>(s, up_f32, act, expert_ids, row_map, up_ptrs, p, n, k, e_count);
  constexpr int kSB = 256;
  MoeFusedSiluMulF32<__nv_bfloat16>
      <<<static_cast<unsigned>((total + kSB - 1) / kSB), kSB, 0, s>>>(out_ptr, gate_f32.Ptr<float>(),
                                                                      up_f32.Ptr<float>(), total);
  Check(cudaGetLastError(), "moe_grouped_gemm_bf16_gate_up_silu wmma silu-mul launch");
}

void MoeGroupedGemmNvfp4KernelCuda(Queue& q, Tensor& out, const Tensor& act,
                                   const Tensor& expert_ids, const Tensor* row_map,
                                   const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                                   const Tensor& scale2s) {
  const int64_t p = out.shape[0], n = out.shape[1], k = act.shape[1];
  if (p == 0 || n == 0) return;
  const int64_t e_count = packed_ptrs.shape[0];  // number of experts
  cudaStream_t s = AsStream(q);
  switch (act.dtype) {
    case DType::kF32:
      LaunchGroupedByOut<float>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs, scale2s,
                                p, n, k, e_count);
      break;
    case DType::kBF16:
      LaunchGroupedByOut<__nv_bfloat16>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs,
                                        scale2s, p, n, k, e_count);
      break;
    default: VT_CHECK(false, "cuda moe_grouped_gemm_nvfp4: unsupported act dtype (f32/bf16 only)");
  }
}

void MatmulNvfp4KernelCuda(Queue& q, Tensor& out, const Tensor& act, const Tensor& weight_packed,
                           const Tensor& weight_scale, float weight_scale_2) {
  const int64_t m = act.shape[0], k = act.shape[1], n = weight_packed.shape[0];
  if (m == 0 || n == 0) return;
  cudaStream_t s = AsStream(q);
  switch (act.dtype) {
    case DType::kF32:
      LaunchByOut<float>(s, out, act, weight_packed, weight_scale, weight_scale_2, m, n, k);
      break;
    case DType::kBF16:
      LaunchByOut<__nv_bfloat16>(s, out, act, weight_packed, weight_scale, weight_scale_2, m, n, k);
      break;
    default: VT_CHECK(false, "cuda matmul_nvfp4: unsupported act dtype (f32/bf16 only)");
  }
}

// --- TRUE W4A4 (fp4 activations x fp4 weights) — the 27B path (notes §7) --------
// vt::ScaledFp4Quant (mirror vllm scaled_fp4_quant / cvt_warp_fp16_to_fp4) +
// vt::MatmulNvfp4Fp4 (mirror cutlass_scaled_fp4_mm_sm120a). The GEMM here is the
// CORRECTNESS-FIRST compute: both fp4 operands are dequantized to bf16 in shared
// and the K-reduction runs on the bf16 tensor cores (numerically the exact
// vllm::RunNvfp4Emulation result — what the true-W4A4 oracle golden should match),
// then the accumulator is scaled by the folded `alpha`. The native block-scaled
// fp4xfp4 MMA (Blackwell mxf4 tensor cores) is the throughput follow-up (notes
// §7.6 step 6); the bf16-dequant path is the token-exact fallback.

// IEEE fp8-e4m3fn RNE-saturating cast (hardware conversion == vllm F32ToF8E4M3).
__device__ __forceinline__ uint8_t F32ToFp8Dev(float f) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3));
}

// f32 -> E2M1 nibble (bit-matches vllm CastToFp4 + Fp4ToNibble). Input pre-scaled.
__device__ __forceinline__ uint8_t CastToFp4NibbleDev(float x) {
  const float a = fabsf(x);
  uint8_t idx;
  if (a <= 0.25f) idx = 0;
  else if (a < 0.75f) idx = 1;
  else if (a <= 1.25f) idx = 2;
  else if (a < 1.75f) idx = 3;
  else if (a <= 2.5f) idx = 4;
  else if (a < 3.5f) idx = 5;
  else if (a <= 5.0f) idx = 6;
  else idx = 7;
  if (idx == 0) return 0;
  return static_cast<uint8_t>((x < 0.0f ? 0x8u : 0x0u) | idx);
}

// Exact fast-reciprocal primitive used by vLLM's pinned NVFP4 quantizer
// (`csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh:207-211`).  Keep the
// two-stage expression below intact: `rcp(sf * rcp(global))` is observably not
// identical to the algebraically simplified `global * rcp(sf)` at FP4 bucket
// boundaries.
__device__ __forceinline__ float ReciprocalApproximateFtz(float value) {
  float reciprocal;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(value));
  return reciprocal;
}

// Opt-in switch for the NATIVE block-scaled fp4xfp4 MMA + its matching
// fast-reciprocal activation quant (mirror of vllm's flashinfer-cutlass sm120a
// path). DEFAULT OFF: the default forward keeps the bf16-dequant WMMA GEMM +
// exact-division quant, which is byte-identical to the CPU RefScaledFp4Quant /
// RunNvfp4Emulation reference (and reproduces vllm's EMULATION token stream).
// Set VT_NVFP4_FP4_NATIVE=1 to select the native sm120a-mirror path (validated
// bit-exact vs the CPU reference; on the 27B near-tie it still lands on vllm's
// emulation token, not cutlass's — see notes §7.2 / the qwen27 gate).
inline bool NativeFp4MmaEnabled() {
  const char* e = std::getenv("VT_NVFP4_FP4_NATIVE");
  return e && e[0] == '1';
}

__device__ __forceinline__ int64_t CutlassScaleOffset(int64_t row, int64_t col,
                                                      int64_t padded_cols) {
  const int64_t m_tile = row / 128;
  const int64_t outer_m = row % 32;
  const int64_t inner_m = (row % 128) / 32;
  const int64_t k_tile = col / 4;
  const int64_t inner_k = col % 4;
  return ((((m_tile * (padded_cols / 4) + k_tile) * 32 + outer_m) * 4 +
            inner_m) *
               4 +
           inner_k);
}

template <bool kSwizzled>
__device__ __forceinline__ void StoreFp4Scale(uint8_t* scale, int64_t row,
                                              int64_t col, int64_t groups,
                                              int64_t padded_cols,
                                              uint8_t value) {
  if constexpr (kSwizzled) {
    scale[CutlassScaleOffset(row, col, padded_cols)] = value;
  } else {
    scale[row * groups + col] = value;
  }
}

// --- NUMERICS-NEUTRAL vectorized-load fast path (VT_FP4_QUANT_FAST /
//     VT_SILU_FP4_FAST) --------------------------------------------------------
// Load the 16 consecutive Tin elements at [p, p+16) into v[16] with a SINGLE
// 16-byte (uint4) vectorized global load when the address is naturally 16-byte
// aligned, else the identical scalar loads. The loaded VALUES and their order are
// bit-identical either way, so every downstream computation (fmaxf group amax,
// F32ToFp8Dev scale, CastToFp4NibbleDev pack) is BYTE-EXACT vs the scalar kernel.
// This mirrors upstream vLLM's own quant load — one thread loads
// CVT_FP4_ELTS_PER_THREAD=16 elements via ld256_cg / ld128_cg into a PackedVec
// (csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu:56-80,126-149 @
// e24d1b24) — here as a plain aligned uint4 load (not the .cg streaming variant),
// so the 32 threads of a warp coalesce their group spans into contiguous 128-byte
// transactions instead of 16 strided 2-byte loads.
__device__ __forceinline__ void LoadFp4Group16(float (&v)[16],
                                               const __nv_bfloat16* p) {
  if ((reinterpret_cast<uintptr_t>(p) & 0xF) == 0) {
    // 16 bf16 = 32 bytes = two 128-bit (uint4) loads.
    const uint4 w0 = reinterpret_cast<const uint4*>(p)[0];
    const uint4 w1 = reinterpret_cast<const uint4*>(p)[1];
    const uint32_t words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      // Each 32-bit word packs two consecutive bf16 (little-endian: .x = elem
      // 2k, .y = elem 2k+1). bf16->f32 is a lossless widen, byte-identical to the
      // scalar __bfloat162float(p[j]).
      const float2 f = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(words[k]));
      v[2 * k] = f.x;
      v[2 * k + 1] = f.y;
    }
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) v[j] = __bfloat162float(p[j]);
  }
}

__device__ __forceinline__ void LoadFp4Group16(float (&v)[16], const float* p) {
  if ((reinterpret_cast<uintptr_t>(p) & 0xF) == 0) {
    // 16 f32 = 64 bytes = four 128-bit (uint4) loads.
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = reinterpret_cast<const uint4*>(p)[c];
      v[4 * c + 0] = __uint_as_float(w.x);
      v[4 * c + 1] = __uint_as_float(w.y);
      v[4 * c + 2] = __uint_as_float(w.z);
      v[4 * c + 3] = __uint_as_float(w.w);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) v[j] = p[j];
  }
}

// Shared quant epilogue over an already-materialized 16-element group v[16].
// This is the EXACT per-element math of ScaledFp4QuantKernel / SiluAndMul...Kernel
// (fmaxf group amax -> input_global_scale scaling -> [-448,448] clamp ->
// F32ToFp8Dev scale byte -> exact-or-approx reciprocal out_scale -> per-element
// [-6,6] clamp -> CastToFp4NibbleDev nibble pack), factored so the fast kernels
// reuse it verbatim and the ONLY difference vs the scalar kernels is the load
// pattern. `out_base` is the element offset of this group's packed output row
// (ScaledFp4Quant: row*k_dim + g*16; SiluAndMul: row*i_dim + g*16).
template <bool kSwizzled>
__device__ __forceinline__ void ScaledFp4QuantEpilogue(
    uint8_t* packed, uint8_t* scale, const float (&v)[16],
    float input_global_scale, int64_t row, int64_t g, int64_t groups,
    int64_t scale_cols, int64_t out_base, bool approx_recip) {
  float vmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) vmax = fmaxf(vmax, fabsf(v[j]));
  const float inverse_six =
      approx_recip ? ReciprocalApproximateFtz(6.0f) : (1.0f / 6.0f);
  float sc = input_global_scale * (vmax * inverse_six);
  sc = fminf(fmaxf(sc, -448.0f), 448.0f);
  const uint8_t sc8 = F32ToFp8Dev(sc);
  StoreFp4Scale<kSwizzled>(scale, row, g, groups, scale_cols, sc8);
  const float sfv = F8E4M3ToF32Dev(sc8);
  float out_scale = 0.0f;
  if (sfv != 0.0f) {
    if (approx_recip) {
      out_scale = ReciprocalApproximateFtz(
          sfv * ReciprocalApproximateFtz(input_global_scale));
    } else {
      out_scale = input_global_scale / sfv;
    }
  }
  // Pack the 8 output bytes (one E2M1 nibble pair per byte), then store the whole
  // 16-element group with ONE 64-bit write when its packed span is 8-byte aligned
  // (numerics-neutral: byte-identical to the 8 scalar stores, one coalesced STG.64
  // instead of eight 1-byte STG.U8 — mirrors upstream vLLM's uint64 packed store,
  // nvfp4_quant_kernels.cu:98). out_base is even and i_dim%16==0, so out_base/2 =
  // row*(i_dim/2)+g*8 is 8-aligned whenever the base pointer is; otherwise byte-wise.
  uint8_t bytes[8];
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    const float lo = fminf(fmaxf(v[j] * out_scale, -6.0f), 6.0f);
    const float hi = fminf(fmaxf(v[j + 1] * out_scale, -6.0f), 6.0f);
    bytes[j / 2] =
        static_cast<uint8_t>(CastToFp4NibbleDev(lo) | (CastToFp4NibbleDev(hi) << 4));
  }
  uint8_t* out = packed + out_base / 2;
  if ((reinterpret_cast<uintptr_t>(out) & 0x7) == 0) {
    uint64_t word = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) word |= static_cast<uint64_t>(bytes[j]) << (8 * j);
    *reinterpret_cast<uint64_t*>(out) = word;  // little-endian: out[j] == bytes[j]
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = bytes[j];
  }
}

// ScaledFp4Quant: one thread per linear or padded (row, 16-group) scale slot.
// The swizzled specialization writes the CUTLASS scale-factor atom address
// directly and zeroes padded rows/columns in this same launch. Reciprocal is
// selected by approx_recip (notes §7.2). Tin f32/bf16.
template <typename Tin, bool kSwizzled>
__global__ void ScaledFp4QuantKernel(uint8_t* packed, uint8_t* scale, const Tin* x,
                                     float input_global_scale, int64_t m_rows, int64_t k_dim,
                                     int64_t scale_rows, int64_t scale_cols,
                                     bool approx_recip) {
  const int64_t groups = k_dim / 16;
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t columns = kSwizzled ? scale_cols : groups;
  const int64_t rows = kSwizzled ? scale_rows : m_rows;
  if (gid >= rows * columns) return;
  const int64_t row = gid / columns;
  const int64_t g = gid % columns;
  if constexpr (kSwizzled) {
    if (row >= m_rows || g >= groups) {
      StoreFp4Scale<true>(scale, row, g, groups, scale_cols, uint8_t{0});
      return;
    }
  }
  const int64_t base = row * k_dim + g * 16;
  float vmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) vmax = fmaxf(vmax, fabsf(Load(x, base + j)));
  const float inverse_six =
      approx_recip ? ReciprocalApproximateFtz(6.0f) : (1.0f / 6.0f);
  float sc = input_global_scale * (vmax * inverse_six);
  sc = fminf(fmaxf(sc, -448.0f), 448.0f);
  const uint8_t sc8 = F32ToFp8Dev(sc);
  StoreFp4Scale<kSwizzled>(scale, row, g, groups, scale_cols, sc8);
  // outputScale = SFScaleVal / SFValue. In native mode (approx_recip) this uses
  // the fast approximate reciprocal, mirroring the NATIVE vllm cvt_warp_fp16_to_fp4
  // (nvfp4_utils.cuh:283-286 reciprocal_approximate_ftz) — the sub-bucket delta
  // the sm120a oracle uses. Default (emulation-grade) uses exact division, which
  // is byte-identical to the CPU RefScaledFp4Quant reference (notes §7.2).
  const float sfv = F8E4M3ToF32Dev(sc8);
  float out_scale = 0.0f;
  if (sfv != 0.0f) {
    if (approx_recip) {
      out_scale = ReciprocalApproximateFtz(
          sfv * ReciprocalApproximateFtz(input_global_scale));
    } else {
      out_scale = input_global_scale / sfv;
    }
  }
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    const float lo = fminf(fmaxf(Load(x, base + j) * out_scale, -6.0f), 6.0f);
    const float hi = fminf(fmaxf(Load(x, base + j + 1) * out_scale, -6.0f), 6.0f);
    packed[(base + j) / 2] =
        static_cast<uint8_t>(CastToFp4NibbleDev(lo) | (CastToFp4NibbleDev(hi) << 4));
  }
}

// NUMERICS-NEUTRAL fast variant of ScaledFp4QuantKernel: identical thread/group
// mapping and the identical ScaledFp4QuantEpilogue, differing ONLY in that each
// thread loads its 16-element group with LoadFp4Group16 (one 16-byte vectorized
// global load) instead of 16 scalar Load()s. Selected by VT_FP4_QUANT_FAST; see
// fp4_quant_fast.h and the byte-exact old-vs-new parity cases in
// tests/vt/test_ops_nvfp4_fp4.cpp.
template <typename Tin, bool kSwizzled>
__global__ void ScaledFp4QuantFastKernel(uint8_t* packed, uint8_t* scale,
                                         const Tin* x, float input_global_scale,
                                         int64_t m_rows, int64_t k_dim,
                                         int64_t scale_rows, int64_t scale_cols,
                                         bool approx_recip) {
  const int64_t groups = k_dim / 16;
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t columns = kSwizzled ? scale_cols : groups;
  const int64_t rows = kSwizzled ? scale_rows : m_rows;
  if (gid >= rows * columns) return;
  const int64_t row = gid / columns;
  const int64_t g = gid % columns;
  if constexpr (kSwizzled) {
    if (row >= m_rows || g >= groups) {
      StoreFp4Scale<true>(scale, row, g, groups, scale_cols, uint8_t{0});
      return;
    }
  }
  const int64_t base = row * k_dim + g * 16;
  float v[16];
  LoadFp4Group16(v, x + base);
  ScaledFp4QuantEpilogue<kSwizzled>(packed, scale, v, input_global_scale, row, g,
                                    groups, scale_cols, base, approx_recip);
}

void ScaledFp4QuantKernelCuda(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& x,
                              float input_global_scale_inv,
                              Fp4ScaleLayout scale_layout) {
  const int64_t m = x.shape[0], k = x.shape[1];
  if (m == 0 || k == 0) return;
  cudaStream_t s = AsStream(q);
  const bool swizzled = scale_layout == Fp4ScaleLayout::kCutlassSwizzled;
  const int64_t scale_rows = out_scale.shape[0];
  const int64_t scale_cols = out_scale.shape[1];
  const int64_t total = swizzled ? out_scale.Numel() : m * (k / 16);
  constexpr int kBlock = 256;
  const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
  auto* pk = out_packed.Ptr<uint8_t>();
  auto* sc = out_scale.Ptr<uint8_t>();
  const bool approx = NativeFp4MmaEnabled();  // native path -> fast-reciprocal quant
  // NUMERICS-NEUTRAL vectorized-load fast path (opt-in, default OFF). getenv is
  // read per call so in-process CUDA tests can flip the selection; the parse is
  // the CPU-tested predicate in fp4_quant_fast.h.
  const bool fast = Fp4QuantFastFlagIsOn(std::getenv("VT_FP4_QUANT_FAST"));
  switch (x.dtype) {
    case DType::kF32:
      if (swizzled) {
        if (fast)
          ScaledFp4QuantFastKernel<float, true><<<grid, kBlock, 0, s>>>(
              pk, sc, x.Ptr<float>(), input_global_scale_inv, m, k, scale_rows,
              scale_cols, approx);
        else
          ScaledFp4QuantKernel<float, true><<<grid, kBlock, 0, s>>>(
              pk, sc, x.Ptr<float>(), input_global_scale_inv, m, k, scale_rows,
              scale_cols, approx);
      } else {
        if (fast)
          ScaledFp4QuantFastKernel<float, false><<<grid, kBlock, 0, s>>>(
              pk, sc, x.Ptr<float>(), input_global_scale_inv, m, k, scale_rows,
              scale_cols, approx);
        else
          ScaledFp4QuantKernel<float, false><<<grid, kBlock, 0, s>>>(
              pk, sc, x.Ptr<float>(), input_global_scale_inv, m, k, scale_rows,
              scale_cols, approx);
      }
      break;
    case DType::kBF16:
      if (swizzled) {
        if (fast)
          ScaledFp4QuantFastKernel<__nv_bfloat16, true><<<grid, kBlock, 0, s>>>(
              pk, sc, x.Ptr<__nv_bfloat16>(), input_global_scale_inv, m, k,
              scale_rows, scale_cols, approx);
        else
          ScaledFp4QuantKernel<__nv_bfloat16, true><<<grid, kBlock, 0, s>>>(
              pk, sc, x.Ptr<__nv_bfloat16>(), input_global_scale_inv, m, k,
              scale_rows, scale_cols, approx);
      } else {
        if (fast)
          ScaledFp4QuantFastKernel<__nv_bfloat16, false><<<grid, kBlock, 0, s>>>(
              pk, sc, x.Ptr<__nv_bfloat16>(), input_global_scale_inv, m, k,
              scale_rows, scale_cols, approx);
        else
          ScaledFp4QuantKernel<__nv_bfloat16, false><<<grid, kBlock, 0, s>>>(
              pk, sc, x.Ptr<__nv_bfloat16>(), input_global_scale_inv, m, k,
              scale_rows, scale_cols, approx);
      }
      break;
    default: VT_CHECK(false, "cuda scaled_fp4_quant: unsupported x dtype (f32/bf16 only)");
  }
  Check(cudaGetLastError(), "scaled_fp4_quant kernel launch");
}

// FUSED silu-mul -> NVFP4 quant (mirror vllm ActivationQuantFusionPass /
// silu_and_mul_nvfp4_quant, act_quant_fusion.py:40 + activation_nvfp4_quant_fusion
// _kernels.cu). Removes the bf16 intermediate the unfused MoeSiluMul(->bf16) +
// ScaledFp4Quant writes+reads on the memory-bound prefill. One thread per
// (row, 16-group), same grid as ScaledFp4QuantKernel. gate/up are [M,I] (our
// two-input MoeSiluMul form, qwen3_5.cpp DenseMlpBlock). BIT-IDENTICAL to
// MoeSiluMul(->bf16) then ScaledFp4Quant: silu(gate)*up is ROUNDED THROUGH bf16
// (MoeSiluMul stores bf16) before quant, reusing the exact ScaledFp4Quant epilogue.
template <typename Tin, bool kSwizzled>
__global__ void SiluMulFp4QuantKernel(uint8_t* packed, uint8_t* scale, const Tin* gate,
                                      const Tin* up, float input_global_scale, int64_t m_rows,
                                      int64_t i_dim, int64_t scale_rows,
                                      int64_t scale_cols, bool approx_recip) {
  const int64_t groups = i_dim / 16;
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t columns = kSwizzled ? scale_cols : groups;
  const int64_t rows = kSwizzled ? scale_rows : m_rows;
  if (gid >= rows * columns) return;
  const int64_t row = gid / columns;
  const int64_t g = gid % columns;
  if constexpr (kSwizzled) {
    if (row >= m_rows || g >= groups) {
      StoreFp4Scale<true>(scale, row, g, groups, scale_cols, uint8_t{0});
      return;
    }
  }
  const int64_t base = row * i_dim + g * 16;
  // silu(gate)*up, rounded through bf16 (== MoeSiluMul store to bf16, cuda_ops.cu:149).
  float v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float gg = Load(gate, base + j);
    const float uu = Load(up, base + j);
    const float sm = (gg / (1.0f + expf(-gg))) * uu;
    v[j] = __bfloat162float(__float2bfloat16(sm));
  }
  // --- exact ScaledFp4QuantKernel epilogue (cuda_matmul_nvfp4.cu:960-989) over v ---
  float vmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) vmax = fmaxf(vmax, fabsf(v[j]));
  const float inverse_six =
      approx_recip ? ReciprocalApproximateFtz(6.0f) : (1.0f / 6.0f);
  float sc = input_global_scale * (vmax * inverse_six);
  sc = fminf(fmaxf(sc, -448.0f), 448.0f);
  const uint8_t sc8 = F32ToFp8Dev(sc);
  StoreFp4Scale<kSwizzled>(scale, row, g, groups, scale_cols, sc8);
  const float sfv = F8E4M3ToF32Dev(sc8);
  float out_scale = 0.0f;
  if (sfv != 0.0f) {
    if (approx_recip) {
      out_scale = ReciprocalApproximateFtz(
          sfv * ReciprocalApproximateFtz(input_global_scale));
    } else {
      out_scale = input_global_scale / sfv;
    }
  }
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    const float lo = fminf(fmaxf(v[j] * out_scale, -6.0f), 6.0f);
    const float hi = fminf(fmaxf(v[j + 1] * out_scale, -6.0f), 6.0f);
    packed[(base + j) / 2] =
        static_cast<uint8_t>(CastToFp4NibbleDev(lo) | (CastToFp4NibbleDev(hi) << 4));
  }
}

void SiluMulFp4QuantKernelCuda(Queue& q, Tensor& out_packed, Tensor& out_scale,
                               const Tensor& gate, const Tensor& up,
                               float input_global_scale_inv,
                               Fp4ScaleLayout scale_layout) {
  const int64_t m = gate.shape[0], i = gate.shape[1];
  if (m == 0 || i == 0) return;
  cudaStream_t s = AsStream(q);
  const bool swizzled = scale_layout == Fp4ScaleLayout::kCutlassSwizzled;
  const int64_t scale_rows = out_scale.shape[0];
  const int64_t scale_cols = out_scale.shape[1];
  const int64_t total = swizzled ? out_scale.Numel() : m * (i / 16);
  constexpr int kBlock = 256;
  const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
  auto* pk = out_packed.Ptr<uint8_t>();
  auto* sc = out_scale.Ptr<uint8_t>();
  const bool approx = NativeFp4MmaEnabled();  // pair fused/unfused reciprocal choice
  switch (gate.dtype) {
    case DType::kF32:
      if (swizzled) {
        SiluMulFp4QuantKernel<float, true><<<grid, kBlock, 0, s>>>(
            pk, sc, gate.Ptr<float>(), up.Ptr<float>(), input_global_scale_inv,
            m, i, scale_rows, scale_cols, approx);
      } else {
        SiluMulFp4QuantKernel<float, false><<<grid, kBlock, 0, s>>>(
            pk, sc, gate.Ptr<float>(), up.Ptr<float>(), input_global_scale_inv,
            m, i, scale_rows, scale_cols, approx);
      }
      break;
    case DType::kBF16:
      if (swizzled) {
        SiluMulFp4QuantKernel<__nv_bfloat16, true><<<grid, kBlock, 0, s>>>(
            pk, sc, gate.Ptr<__nv_bfloat16>(), up.Ptr<__nv_bfloat16>(),
            input_global_scale_inv, m, i, scale_rows, scale_cols, approx);
      } else {
        SiluMulFp4QuantKernel<__nv_bfloat16, false><<<grid, kBlock, 0, s>>>(
            pk, sc, gate.Ptr<__nv_bfloat16>(), up.Ptr<__nv_bfloat16>(),
            input_global_scale_inv, m, i, scale_rows, scale_cols, approx);
      }
      break;
    default: VT_CHECK(false, "cuda silu_mul_fp4_quant: unsupported dtype (f32/bf16 only)");
  }
  Check(cudaGetLastError(), "silu_mul_fp4_quant kernel launch");
}

// FUSED full-attention sigmoid output gate + NVFP4 activation quant. Mirror of
// SiluMulFp4QuantKernel, substituting attn*sigmoid(gate) for silu(gate)*up — the
// vLLM Inductor triton_poi_fused_mul_scaled_fp4_quant_sigmoid fold. attn is f32 OR
// bf16 (FA-2 hands bf16; exact upcast); gate is always f32 (sigmoid input must not
// round). Bit-identical to SigmoidGateBf16(attn,gate -> bf16) + ScaledFp4Quant.
template <typename Tattn, bool kSwizzled>
__global__ void SigmoidGateFp4QuantKernel(uint8_t* packed, uint8_t* scale, const Tattn* attn,
                                          const float* gate, float input_global_scale,
                                          int64_t m_rows, int64_t i_dim, int64_t scale_rows,
                                          int64_t scale_cols, bool approx_recip) {
  const int64_t groups = i_dim / 16;
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t columns = kSwizzled ? scale_cols : groups;
  const int64_t rows = kSwizzled ? scale_rows : m_rows;
  if (gid >= rows * columns) return;
  const int64_t row = gid / columns;
  const int64_t g = gid % columns;
  if constexpr (kSwizzled) {
    if (row >= m_rows || g >= groups) {
      StoreFp4Scale<true>(scale, row, g, groups, scale_cols, uint8_t{0});
      return;
    }
  }
  const int64_t base = row * i_dim + g * 16;
  // attn*sigmoid(gate), rounded through bf16 (== SigmoidGateBf16 store to bf16,
  // cuda_glue.cu:164). sigmoid(x)=1/(1+exp(-x)).
  float v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float aa = Load(attn, base + j);
    const float gg = Load(gate, base + j);
    const float sm = aa * (1.0f / (1.0f + expf(-gg)));
    v[j] = __bfloat162float(__float2bfloat16(sm));
  }
  // --- exact ScaledFp4QuantKernel epilogue (cuda_matmul_nvfp4.cu:960-989) over v ---
  float vmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) vmax = fmaxf(vmax, fabsf(v[j]));
  const float inverse_six =
      approx_recip ? ReciprocalApproximateFtz(6.0f) : (1.0f / 6.0f);
  float sc = input_global_scale * (vmax * inverse_six);
  sc = fminf(fmaxf(sc, -448.0f), 448.0f);
  const uint8_t sc8 = F32ToFp8Dev(sc);
  StoreFp4Scale<kSwizzled>(scale, row, g, groups, scale_cols, sc8);
  const float sfv = F8E4M3ToF32Dev(sc8);
  float out_scale = 0.0f;
  if (sfv != 0.0f) {
    if (approx_recip) {
      out_scale = ReciprocalApproximateFtz(
          sfv * ReciprocalApproximateFtz(input_global_scale));
    } else {
      out_scale = input_global_scale / sfv;
    }
  }
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    const float lo = fminf(fmaxf(v[j] * out_scale, -6.0f), 6.0f);
    const float hi = fminf(fmaxf(v[j + 1] * out_scale, -6.0f), 6.0f);
    packed[(base + j) / 2] =
        static_cast<uint8_t>(CastToFp4NibbleDev(lo) | (CastToFp4NibbleDev(hi) << 4));
  }
}

void SigmoidGateFp4QuantKernelCuda(Queue& q, Tensor& out_packed, Tensor& out_scale,
                                   const Tensor& attn, const Tensor& gate,
                                   float input_global_scale_inv, Fp4ScaleLayout scale_layout) {
  const int64_t m = attn.shape[0], i = attn.shape[1];
  if (m == 0 || i == 0) return;
  cudaStream_t s = AsStream(q);
  const bool swizzled = scale_layout == Fp4ScaleLayout::kCutlassSwizzled;
  const int64_t scale_rows = out_scale.shape[0];
  const int64_t scale_cols = out_scale.shape[1];
  const int64_t total = swizzled ? out_scale.Numel() : m * (i / 16);
  constexpr int kBlock = 256;
  const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
  auto* pk = out_packed.Ptr<uint8_t>();
  auto* sc = out_scale.Ptr<uint8_t>();
  const bool approx = NativeFp4MmaEnabled();  // pair fused/unfused reciprocal choice
  const float* gp = gate.Ptr<float>();
  switch (attn.dtype) {
    case DType::kF32:
      if (swizzled) {
        SigmoidGateFp4QuantKernel<float, true><<<grid, kBlock, 0, s>>>(
            pk, sc, attn.Ptr<float>(), gp, input_global_scale_inv, m, i, scale_rows,
            scale_cols, approx);
      } else {
        SigmoidGateFp4QuantKernel<float, false><<<grid, kBlock, 0, s>>>(
            pk, sc, attn.Ptr<float>(), gp, input_global_scale_inv, m, i, scale_rows,
            scale_cols, approx);
      }
      break;
    case DType::kBF16:
      if (swizzled) {
        SigmoidGateFp4QuantKernel<__nv_bfloat16, true><<<grid, kBlock, 0, s>>>(
            pk, sc, attn.Ptr<__nv_bfloat16>(), gp, input_global_scale_inv, m, i,
            scale_rows, scale_cols, approx);
      } else {
        SigmoidGateFp4QuantKernel<__nv_bfloat16, false><<<grid, kBlock, 0, s>>>(
            pk, sc, attn.Ptr<__nv_bfloat16>(), gp, input_global_scale_inv, m, i,
            scale_rows, scale_cols, approx);
      }
      break;
    default: VT_CHECK(false, "cuda sigmoid_gate_fp4_quant: unsupported attn dtype (f32/bf16)");
  }
  Check(cudaGetLastError(), "sigmoid_gate_fp4_quant kernel launch");
}

// The sm12x-only packed BF16 producer is compiled in
// cuda_nvfp4_sm12x.cu. This common TU retains the portable producer below for
// every release SM and calls the native producer through an exact-arch seam.
// Exact one-input custom-op form used by vLLM's ActivationQuantFusionPass.
// Upstream: vllm@e24d1b24
//   csrc/libtorch_stable/quantization/fp4/
//     activation_nvfp4_quant_fusion_kernels.cu:27-116,120-163
// Input is the MergedColumnParallelLinear result [M,2I], with gate then up in
// each row. The BF16 round below is intentional: it equals SiluAndMul's BF16
// output boundary before ScaledFp4Quant while avoiding the intermediate write
// and read. The quant epilogue is kept identical to the existing two-input
// fused producer above.
template <typename Tin, bool kSwizzled>
__global__ void SiluAndMulFp4QuantKernel(
    uint8_t* packed, uint8_t* scale, const Tin* gate_up,
    float input_global_scale, int64_t m_rows, int64_t i_dim,
    int64_t scale_rows, int64_t scale_cols, bool approx_recip) {
  const int64_t groups = i_dim / 16;
  const int64_t gid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t columns = kSwizzled ? scale_cols : groups;
  const int64_t rows = kSwizzled ? scale_rows : m_rows;
  if (gid >= rows * columns) return;
  const int64_t row = gid / columns;
  const int64_t g = gid % columns;
  if constexpr (kSwizzled) {
    if (row >= m_rows || g >= groups) {
      StoreFp4Scale<true>(scale, row, g, groups, scale_cols, uint8_t{0});
      return;
    }
  }
  const int64_t output_base = row * i_dim + g * 16;
  const int64_t input_row_base = row * (2 * i_dim);
  const int64_t gate_base = input_row_base + g * 16;
  const int64_t up_base = input_row_base + i_dim + g * 16;

  float v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float gate = Load(gate_up, gate_base + j);
    const float up = Load(gate_up, up_base + j);
    const float silu_mul = (gate / (1.0f + expf(-gate))) * up;
    v[j] = __bfloat162float(__float2bfloat16(silu_mul));
  }

  float vmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) vmax = fmaxf(vmax, fabsf(v[j]));
  const float inverse_six =
      approx_recip ? ReciprocalApproximateFtz(6.0f) : (1.0f / 6.0f);
  float sc = input_global_scale * (vmax * inverse_six);
  sc = fminf(fmaxf(sc, -448.0f), 448.0f);
  const uint8_t sc8 = F32ToFp8Dev(sc);
  StoreFp4Scale<kSwizzled>(scale, row, g, groups, scale_cols, sc8);
  const float sfv = F8E4M3ToF32Dev(sc8);
  float out_scale = 0.0f;
  if (sfv != 0.0f) {
    if (approx_recip) {
      out_scale = ReciprocalApproximateFtz(
          sfv * ReciprocalApproximateFtz(input_global_scale));
    } else {
      out_scale = input_global_scale / sfv;
    }
  }
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    const float lo =
        fminf(fmaxf(v[j] * out_scale, -6.0f), 6.0f);
    const float hi =
        fminf(fmaxf(v[j + 1] * out_scale, -6.0f), 6.0f);
    packed[(output_base + j) / 2] = static_cast<uint8_t>(
        CastToFp4NibbleDev(lo) | (CastToFp4NibbleDev(hi) << 4));
  }
}

// NUMERICS-NEUTRAL fast variant of SiluAndMulFp4QuantKernel: identical
// thread/group mapping, identical silu(gate)*up bf16 round, and the identical
// ScaledFp4QuantEpilogue, differing ONLY in that each thread loads its 16-element
// gate and up spans with LoadFp4Group16 (one 16-byte vectorized global load each)
// instead of 32 scalar Load()s. Selected by VT_SILU_FP4_FAST; see fp4_quant_fast.h
// and the byte-exact old-vs-new parity cases in tests/vt/test_ops_nvfp4_fp4.cpp.
template <typename Tin, bool kSwizzled>
__global__ void SiluAndMulFp4QuantFastKernel(
    uint8_t* packed, uint8_t* scale, const Tin* gate_up,
    float input_global_scale, int64_t m_rows, int64_t i_dim,
    int64_t scale_rows, int64_t scale_cols, bool approx_recip) {
  const int64_t groups = i_dim / 16;
  const int64_t gid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t columns = kSwizzled ? scale_cols : groups;
  const int64_t rows = kSwizzled ? scale_rows : m_rows;
  if (gid >= rows * columns) return;
  const int64_t row = gid / columns;
  const int64_t g = gid % columns;
  if constexpr (kSwizzled) {
    if (row >= m_rows || g >= groups) {
      StoreFp4Scale<true>(scale, row, g, groups, scale_cols, uint8_t{0});
      return;
    }
  }
  const int64_t output_base = row * i_dim + g * 16;
  const int64_t input_row_base = row * (2 * i_dim);
  const int64_t gate_base = input_row_base + g * 16;
  const int64_t up_base = input_row_base + i_dim + g * 16;

  float gate[16];
  float up[16];
  LoadFp4Group16(gate, gate_up + gate_base);
  LoadFp4Group16(up, gate_up + up_base);
  // silu(gate)*up, rounded through bf16 (== MoeSiluMul store to bf16); identical
  // per-element math to the scalar kernel above.
  float v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float silu_mul = (gate[j] / (1.0f + expf(-gate[j]))) * up[j];
    v[j] = __bfloat162float(__float2bfloat16(silu_mul));
  }
  ScaledFp4QuantEpilogue<kSwizzled>(packed, scale, v, input_global_scale, row, g,
                                    groups, scale_cols, output_base, approx_recip);
}

void SiluAndMulFp4QuantKernelCuda(Queue& q, Tensor& out_packed,
                                  Tensor& out_scale, const Tensor& gate_up,
                                  float input_global_scale_inv,
                                  Fp4ScaleLayout scale_layout) {
  const int64_t m = gate_up.shape[0], i = gate_up.shape[1] / 2;
  if (m == 0 || i == 0) return;
  cudaStream_t s = AsStream(q);
  const bool swizzled = scale_layout == Fp4ScaleLayout::kCutlassSwizzled;
  const int64_t scale_rows = out_scale.shape[0];
  const int64_t scale_cols = out_scale.shape[1];
  const int64_t total = swizzled ? out_scale.Numel() : m * (i / 16);
  constexpr int kBlock = 256;
  const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
  auto* pk = out_packed.Ptr<uint8_t>();
  auto* sc = out_scale.Ptr<uint8_t>();
  const bool approx = NativeFp4MmaEnabled();
#if defined(VT_FP4_MMA_SM120A)
  if (swizzled && gate_up.dtype == DType::kBF16 &&
      TryLaunchSiluAndMulFp4QuantPackedSm12x(
          s, pk, sc, gate_up.data, input_global_scale_inv, m, i, scale_cols,
          out_scale.Numel(), approx)) {
    return;
  }
#endif
  // NUMERICS-NEUTRAL vectorized-load fast path (opt-in, default OFF). Distinct
  // from the native packed sm120a path above (VT_FP4_FUSED_VEC, hardware cvt):
  // this fast kernel keeps the bit-exact scalar-cvt epilogue and only vectorizes
  // the gate/up loads. getenv is read per call so in-process CUDA tests can flip
  // the selection; the parse is the CPU-tested predicate in fp4_quant_fast.h.
  const bool fast = SiluFp4FastFlagIsOn(std::getenv("VT_SILU_FP4_FAST"));
  switch (gate_up.dtype) {
    case DType::kF32:
      if (swizzled) {
        if (fast)
          SiluAndMulFp4QuantFastKernel<float, true><<<grid, kBlock, 0, s>>>(
              pk, sc, gate_up.Ptr<float>(), input_global_scale_inv, m, i,
              scale_rows, scale_cols, approx);
        else
          SiluAndMulFp4QuantKernel<float, true><<<grid, kBlock, 0, s>>>(
              pk, sc, gate_up.Ptr<float>(), input_global_scale_inv, m, i,
              scale_rows, scale_cols, approx);
      } else {
        if (fast)
          SiluAndMulFp4QuantFastKernel<float, false><<<grid, kBlock, 0, s>>>(
              pk, sc, gate_up.Ptr<float>(), input_global_scale_inv, m, i,
              scale_rows, scale_cols, approx);
        else
          SiluAndMulFp4QuantKernel<float, false><<<grid, kBlock, 0, s>>>(
              pk, sc, gate_up.Ptr<float>(), input_global_scale_inv, m, i,
              scale_rows, scale_cols, approx);
      }
      break;
    case DType::kBF16:
      if (swizzled) {
        if (fast)
          SiluAndMulFp4QuantFastKernel<__nv_bfloat16, true><<<grid, kBlock, 0, s>>>(
              pk, sc, gate_up.Ptr<__nv_bfloat16>(), input_global_scale_inv, m, i,
              scale_rows, scale_cols, approx);
        else
          SiluAndMulFp4QuantKernel<__nv_bfloat16, true><<<grid, kBlock, 0, s>>>(
              pk, sc, gate_up.Ptr<__nv_bfloat16>(), input_global_scale_inv, m, i,
              scale_rows, scale_cols, approx);
      } else {
        if (fast)
          SiluAndMulFp4QuantFastKernel<__nv_bfloat16, false><<<grid, kBlock, 0, s>>>(
              pk, sc, gate_up.Ptr<__nv_bfloat16>(), input_global_scale_inv, m, i,
              scale_rows, scale_cols, approx);
        else
          SiluAndMulFp4QuantKernel<__nv_bfloat16, false><<<grid, kBlock, 0, s>>>(
              pk, sc, gate_up.Ptr<__nv_bfloat16>(), input_global_scale_inv, m, i,
              scale_rows, scale_cols, approx);
      }
      break;
    default:
      VT_CHECK(false,
               "cuda silu_and_mul_fp4_quant: unsupported dtype (f32/bf16 only)");
  }
  Check(cudaGetLastError(), "silu_and_mul_fp4_quant kernel launch");
}

// Naive fp4xfp4 GEMM (small-M / decode). out[m,n] = alpha * sum_k
// (a_fp4[m,k]·f8(a_scale[m,k/16]))·(b_fp4[n,k]·f8(b_scale[n,k/16])). One thread
// per output; decode both operands on the fly. Bit-consistent with the WMMA path
// (same per-element decode), only K-order differs.
template <typename Tout>
__global__ void MatmulNvfp4Fp4Naive(Tout* out, const uint8_t* a_packed, const uint8_t* a_scale,
                                    const uint8_t* b_packed, const uint8_t* b_scale, float alpha,
                                    int64_t m_rows, int64_t n_cols, int64_t k_dim) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t m = blockIdx.y;
  if (n >= n_cols || m >= m_rows) return;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;
  const uint8_t* aprow = a_packed + m * packed_cols;
  const uint8_t* asrow = a_scale + m * groups;
  const uint8_t* bprow = b_packed + n * packed_cols;
  const uint8_t* bsrow = b_scale + n * groups;
  float acc = 0.0f;
  for (int64_t g = 0; g < groups; ++g) {
    const float asf = F8E4M3ToF32Dev(asrow[g]);
    const float bsf = F8E4M3ToF32Dev(bsrow[g]);
    for (int j = 0; j < 8; ++j) {
      const uint8_t ab = aprow[g * 8 + j];
      const uint8_t bb = bprow[g * 8 + j];
      const float a_lo = kE2M1[ab & 0x7u] * ((ab & 0x8u) ? -1.0f : 1.0f) * asf;
      const float a_hi = kE2M1[(ab >> 4) & 0x7u] * ((ab & 0x80u) ? -1.0f : 1.0f) * asf;
      const float b_lo = kE2M1[bb & 0x7u] * ((bb & 0x8u) ? -1.0f : 1.0f) * bsf;
      const float b_hi = kE2M1[(bb >> 4) & 0x7u] * ((bb & 0x80u) ? -1.0f : 1.0f) * bsf;
      acc += a_lo * b_lo + a_hi * b_hi;
    }
  }
  Store(out, m * n_cols + n, alpha * acc);
}

// Coalesced M=1 (decode) fp4xfp4 GEMV. The Naive kernel puts ONE THREAD per output
// column n, so adjacent threads read weight rows strided by k/2 bytes -> UNCOALESCED,
// and the native MMA tile wastes 15 of 16 rows on M=1 (nsys: ~72us/expert GEMM, ~12%
// of the 240 GB/s peak). Here ONE WARP owns column n: the 32 lanes cooperatively
// stride the packed weight row `bprow` (lane L reads bytes L, L+32, ... -> COALESCED
// within the warp), each dequants + accumulates the fp4·fp4 dot, then a warp reduce.
// The shared activation `aprow` is re-read per warp but hits L2 (small). Near-tie vs
// Naive/MMA (same per-element decode; only the K-reduction order differs). Mirrors the
// ds4 Q8_0 sub-warp GEMV that reached ~76% BW.
template <typename Tout, int CPW>
__global__ void MatmulNvfp4Fp4Gemv(Tout* out, const uint8_t* a_packed, const uint8_t* a_scale,
                                   const uint8_t* b_packed, const uint8_t* b_scale, float alpha,
                                   int64_t n_cols, int64_t k_dim) {
  const int64_t warp = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int64_t n0 = warp * CPW;                 // this warp owns columns [n0, n0+CPW)
  if (n0 >= n_cols) return;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;
  const uint8_t* aprow = a_packed;               // m == 0 (shared across all columns)
  const uint8_t* asrow = a_scale;
  float acc[CPW];
#pragma unroll
  for (int c = 0; c < CPW; ++c) acc[c] = 0.0f;
  for (int64_t bidx = lane; bidx < packed_cols; bidx += 32) {
    const int64_t g = bidx >> 3;                 // group = byte / 8
    const float asf = F8E4M3ToF32Dev(asrow[g]);
    const uint8_t ab = aprow[bidx];              // activation loaded + decoded ONCE
    const float a_lo = kE2M1[ab & 0x7u] * ((ab & 0x8u) ? -1.0f : 1.0f) * asf;
    const float a_hi = kE2M1[(ab >> 4) & 0x7u] * ((ab & 0x80u) ? -1.0f : 1.0f) * asf;
#pragma unroll
    for (int c = 0; c < CPW; ++c) {
      const int64_t n = n0 + c;
      if (n >= n_cols) continue;
      const uint8_t* bprow = b_packed + n * packed_cols;
      const uint8_t* bsrow = b_scale + n * groups;
      const float bsf = F8E4M3ToF32Dev(bsrow[g]);
      const uint8_t bb = bprow[bidx];
      const float b_lo = kE2M1[bb & 0x7u] * ((bb & 0x8u) ? -1.0f : 1.0f) * bsf;
      const float b_hi = kE2M1[(bb >> 4) & 0x7u] * ((bb & 0x80u) ? -1.0f : 1.0f) * bsf;
      acc[c] += a_lo * b_lo + a_hi * b_hi;
    }
  }
#pragma unroll
  for (int c = 0; c < CPW; ++c) {
    float a = acc[c];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    const int64_t n = n0 + c;
    if (lane == 0 && n < n_cols) Store(out, n, alpha * a);
  }
}

inline bool Fp4GemvEnabled() {
  const char* e = std::getenv("VT_NVFP4_FP4_GEMV");
  return e == nullptr || e[0] != '0';  // default ON; the coalesced M=1 decode path
}

// Decode a fp4 tile [ROWS][BK] (packed [ROWS][K/2], fp8 group scales [ROWS][K/16])
// into shared bf16 `dst` [ROWS][BK]. `outer0` = the tile's first row in the full
// [ROWS_total][K] operand; used by both the A (M) and B (N) operands (symmetric).
template <int BK>
__device__ __forceinline__ void DecodeFp4TileToShared(__nv_bfloat16* dst, const uint8_t* packed,
                                                      const uint8_t* scale, int64_t outer0,
                                                      int64_t outer_max, int64_t kt, int64_t k_dim,
                                                      int tid, int kThreads, int tile_rows) {
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;
  for (int idx = tid; idx < tile_rows * (BK / 2); idx += kThreads) {
    const int nl = idx / (BK / 2), bc = idx % (BK / 2), kl = 2 * bc;
    const int64_t gn = outer0 + nl;
    if (gn < outer_max && kt + kl < k_dim) {
      const uint8_t b = packed[gn * packed_cols + kt / 2 + bc];
      const float gs = F8E4M3ToF32Dev(scale[gn * groups + (kt + kl) / 16]);
      float w_lo, w_hi;
      DecodeFp4Byte(b, gs, w_lo, w_hi);
      dst[nl * BK + kl] = __float2bfloat16(w_lo);
      dst[nl * BK + kl + 1] = __float2bfloat16(w_hi);
    } else {
      dst[nl * BK + kl] = __float2bfloat16(0.0f);
      dst[nl * BK + kl + 1] = __float2bfloat16(0.0f);
    }
  }
}

// Tensor-core (bf16 WMMA) fp4xfp4 GEMM: decode BOTH operands to bf16 in shared,
// run wmma m16n16k16, scale the accumulator by alpha at store. Same tiling as
// MatmulNvfp4Wmma; the only additions are the A-operand fp4 decode and the alpha.
template <typename Tout, int BM, int BN, int BK, int WARPS_M, int WARPS_N>
__global__ void MatmulNvfp4Fp4Wmma(Tout* out, const uint8_t* a_packed, const uint8_t* a_scale,
                                   const uint8_t* b_packed, const uint8_t* b_scale, float alpha,
                                   int64_t m_rows, int64_t n_cols, int64_t k_dim) {
// bf16 WMMA tensor-core body is Ampere+ (bf16 fragments are complete only for
// __CUDA_ARCH__ >= 800). Guard so Turing/Volta/Pascal device passes compile
// this TU; sm_80+ keeps the body verbatim (preprocessor identity ->
// byte-identical). This TU is compiled for EVERY arch (CMakeLists.txt:903)
// because it also carries the generic bf16 MoE GEMMs, so it cannot simply be
// gated behind its fp4-mma cell. See cuda-arch-breadth-fp16.md §V0-b.
#if __CUDA_ARCH__ >= 800
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  constexpr int WMPER = BM / WARPS_M, WNPER = BN / WARPS_N;
  constexpr int MF = WMPER / 16, NF = WNPER / 16;
  __shared__ __nv_bfloat16 As[BM * BK];
  __shared__ __nv_bfloat16 Ws[BN * BK];
  __shared__ float Cs[BM * BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NF];
#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni) wmma::fill_fragment(acc[mi][ni], 0.0f);

  for (int64_t kt = 0; kt < k_dim; kt += BK) {
    DecodeFp4TileToShared<BK>(As, a_packed, a_scale, row0, m_rows, kt, k_dim, tid, kThreads, BM);
    DecodeFp4TileToShared<BK>(Ws, b_packed, b_scale, col0, n_cols, kt, k_dim, tid, kThreads, BN);
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[MF];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[NF];
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
        wmma::load_matrix_sync(af[mi], &As[(wm * WMPER + mi * 16) * BK + kk * 16], BK);
#pragma unroll
      for (int ni = 0; ni < NF; ++ni)
        wmma::load_matrix_sync(bf[ni], &Ws[(wn * WNPER + ni * 16) * BK + kk * 16], BK);
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
#pragma unroll
        for (int ni = 0; ni < NF; ++ni) wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni)
      wmma::store_matrix_sync(&Cs[(wm * WMPER + mi * 16) * BN + (wn * WNPER + ni * 16)],
                              acc[mi][ni], BN, wmma::mem_row_major);
  __syncthreads();

  for (int idx = tid; idx < BM * BN; idx += kThreads) {
    const int r = idx / BN, c = idx % BN;
    const int64_t gr = row0 + r, gc = col0 + c;
    if (gr < m_rows && gc < n_cols) Store(out, gr * n_cols + gc, alpha * Cs[idx]);
  }
#else
  __trap();
#endif
}


template <typename Tout>
void LaunchFp4Fp4(cudaStream_t s, const DeviceCaps& caps, Tensor& out, const Tensor& a_packed,
                  const Tensor& a_scale, const Tensor& b_packed, const Tensor& b_scale, float alpha,
                  int64_t m, int64_t n, int64_t k) {
  auto* ap = a_packed.Ptr<uint8_t>();
  auto* as = a_scale.Ptr<uint8_t>();
  auto* bp = b_packed.Ptr<uint8_t>();
  auto* bs = b_scale.Ptr<uint8_t>();
  // M=1 DECODE: the coalesced warp-per-column GEMV (the native MMA/Naive are ~12% of
  // peak BW for M=1). Bit-consistent per-element decode with the tile paths (near-tie
  // K-order). VT_NVFP4_FP4_GEMV=0 falls through to the tile dispatch for an A/B.
  if (m == 1 && Fp4GemvEnabled()) {
    constexpr int kThreads = 128;   // 4 warps/block
    // kCpw = output columns per warp. MEASURED (2026-08-01, real ckpt A/B): kCpw=1
    // (one warp per column) is FASTEST — decode 0.15 s/tok; kCpw=4 REGRESSED to 0.21
    // (the 4x-fewer-warps occupancy loss beats the activation-reuse gain; Laguna's fp4
    // shape differs from the ds4 Q8_0 GEMV where multi-row ILP helped). Kept templated
    // so the knob is re-measurable per arch/shape.
    constexpr int kCpw = 1;
    const int64_t warps_needed = (n + kCpw - 1) / kCpw;
    const dim3 grid(static_cast<unsigned>((warps_needed * 32 + kThreads - 1) / kThreads));
    MatmulNvfp4Fp4Gemv<Tout, kCpw><<<grid, kThreads, 0, s>>>(out.Ptr<Tout>(), ap, as, bp, bs,
                                                             alpha, n, k);
    Check(cudaGetLastError(), "matmul_nvfp4_fp4 kernel launch (gemv)");
    return;
  }
  // Runtime SM dispatch: ask the registry for a tactic that supports THIS device.
  // With one registered tactic and GB10's sm_121 this reduces exactly to the old
  // `#if VT_FP4_MMA_SM120A && NativeFp4MmaEnabled()` test — same kernel, same
  // grid, same stream, bit-identical output.
  if (const ArchTactic* tactic = SelectArchTactic(TacticFamily::kNvfp4Fp4Mma, caps)) {
    Nvfp4Fp4MmaArgs args;
    args.stream = s;
    args.out = out.Ptr<Tout>();
    args.out_dtype = out.dtype;
    args.a_packed = ap;
    args.a_scale = as;
    args.b_packed = bp;
    args.b_scale = bs;
    args.alpha = alpha;
    args.m = m;
    args.n = n;
    args.k = k;
    if (tactic->launch(caps, &args)) return;
  }
  if (m < kTileMinRows || !WmmaEnabled()) {
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(m));
    MatmulNvfp4Fp4Naive<Tout><<<grid, kBlock, 0, s>>>(out.Ptr<Tout>(), ap, as, bp, bs, alpha, m, n,
                                                      k);
    Check(cudaGetLastError(), "matmul_nvfp4_fp4 kernel launch (naive)");
    return;
  }
  constexpr int BM = 64, BN = 64, BK = 32, WARPS_M = 2, WARPS_N = 2;
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>((m + BM - 1) / BM));
  MatmulNvfp4Fp4Wmma<Tout, BM, BN, BK, WARPS_M, WARPS_N><<<grid, kThreads, 0, s>>>(
      out.Ptr<Tout>(), ap, as, bp, bs, alpha, m, n, k);
  Check(cudaGetLastError(), "matmul_nvfp4_fp4 kernel launch (wmma)");
}

void MatmulNvfp4Fp4KernelCuda(Queue& q, Tensor& out, const Tensor& a_packed, const Tensor& a_scale,
                              const Tensor& b_packed, const Tensor& b_scale, float alpha) {
  const int64_t m = a_packed.shape[0], k = a_packed.shape[1] * 2, n = b_packed.shape[0];
  if (m == 0 || n == 0) return;
  cudaStream_t s = AsStream(q);
  // The capability is resolved ONCE per process (cached probe) and threaded into
  // the launcher — seam-gap #4: before this, the kernel layer had no access to
  // the (major, minor) that only CudaPlatform knew.
  const DeviceCaps& caps = GetDeviceCaps();
  switch (out.dtype) {
    case DType::kF32:
      LaunchFp4Fp4<float>(s, caps, out, a_packed, a_scale, b_packed, b_scale, alpha, m, n, k);
      break;
    case DType::kBF16:
      LaunchFp4Fp4<__nv_bfloat16>(s, caps, out, a_packed, a_scale, b_packed, b_scale, alpha, m, n,
                                  k);
      break;
    default: VT_CHECK(false, "cuda matmul_nvfp4_fp4: unsupported out dtype (f32/bf16 only)");
  }
}

// Registers the CUDA NVFP4 dequant-GEMM during static init (table fill only, no
// CUDA calls — see cuda_ops.cu for the pre-main discipline rationale).
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulNvfp4, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulNvfp4Fn>(&MatmulNvfp4KernelCuda)));
    RegisterOp(OpId::kScaledFp4Quant, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ScaledFp4QuantFn>(&ScaledFp4QuantKernelCuda)));
    RegisterOp(OpId::kSiluMulFp4Quant, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<SiluMulFp4QuantFn>(&SiluMulFp4QuantKernelCuda)));
    RegisterOp(
        OpId::kSiluAndMulFp4Quant, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<SiluAndMulFp4QuantFn>(
            &SiluAndMulFp4QuantKernelCuda)));
    RegisterOp(
        OpId::kSigmoidGateFp4Quant, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<SigmoidGateFp4QuantFn>(
            &SigmoidGateFp4QuantKernelCuda)));
    RegisterOp(OpId::kMatmulNvfp4Fp4, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulNvfp4Fp4Fn>(&MatmulNvfp4Fp4KernelCuda)));
    RegisterOp(OpId::kMoeGroupedGemmNvfp4, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MoeGroupedGemmNvfp4Fn>(&MoeGroupedGemmNvfp4KernelCuda)));
    RegisterOp(OpId::kMoeGroupedGemmBf16, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MoeGroupedGemmBf16Fn>(&MoeGroupedGemmBf16KernelCuda)));
    RegisterOp(OpId::kMoeGroupedGemmBf16GateUpSilu, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MoeGroupedGemmBf16GateUpSiluFn>(
                   &MoeGroupedGemmBf16GateUpSiluCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda

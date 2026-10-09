// EXL3 (exllamav3 trellis) fused linear — ROCm arm. Row BACKEND-ROCM-EXL3,
// issue #2433.
//
// THE DONOR IS THE PORTABLE CPU REFERENCE, NOT THE CUDA KERNEL. This is the one
// decision worth reading before the code:
//
//   src/vt/cpu/cpu_exl3_kernels.cpp   HadBlock128, ShuffleHadWarp, HadRowBlock,
//                                     Exl3GemmKernelCpu
//   src/vt/cpu/cpu_exl3_dequant.cpp   Exl3TileCodeword, Exl3DecodeCodeword,
//                                     Exl3TileRowMajorIndex, Exl3DecodeTile
//
// which are themselves 1:1 ports of exllamav3 @
// 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT) — hadamard_inner.cuh:17-44 and
// :93-279, exl3_gemm_kernel.cuh:8-80, exl3_dq.cuh:15-31, codebook.cuh:56-90,
// quantize.py:22-42. vLLM implements no EXL3 at the parity pin, so exllamav3 is
// this format's registered secondary oracle (.agents/oracles/exllamav3.md).
//
// WHY NOT src/vt/cuda/cuda_exl3.cu. Four structural blockers, of which the first
// alone settles it. The point is not that the PTX would need translating; it is
// that the kernel's geometry does not exist on this hardware.
//
//   cuda_exl3.cu:103  `constexpr int kSmemMax = 90 * 1024;`, static_asserted at
//                     :694 and installed at :1893 through
//                     cudaFuncAttributeMaxDynamicSharedMemorySize. AMD LDS is
//                     64 KiB per workgroup. The tile arithmetic at :694 is sized
//                     against 90 KiB, so no launch bound or flag recovers it.
//   :168, :1248       `mma.sync.aligned.m16n8k16`. m16n8k16 has no AMD shape —
//                     RDNA3.5 WMMA is 16x16x16 — and FragA/FragB/FragC at
//                     :151-155 are NVIDIA's fragment layout, not a width.
//   :257              `ldmatrix.sync.aligned.m8n8.x4.shared.b16`. No equivalent.
//   :241,:247,:251    `cp.async` + commit_group + wait_group, which is what the
//                     SH_STAGES pipeline IS. gfx1151 has no async global->LDS.
//
// src/vt/rocm/ is hand-written HIP ported from vLLM's ROCm csrc/, never a
// hipify of our CUDA, and this tree ships no hipify script. That is preserved.
//
// THE EXEMPLAR MIRRORED is src/vt/rocm/rocm_grouped_gemm.hip, which serves
// kMatmulBTQuant: the same "decode the weight INSIDE the GEMM" shape, the same
// bit-exact scalar codec instead of a vendor intrinsic, and the same
// grow-only per-stream scratch discipline.
//
// WHAT THIS ARM CLAIMS, AND WHY IT IS A BYTE CLAIM. All three steps run the
// same IEEE f32 operations on the same values in the SAME ORDER as
// Exl3GemmKernelCpu: the Hadamard because it is ShuffleHadWarp's five xor-partner
// levels with __syncthreads() where the host has a memcpy, and the GEMM because
// one thread owns one output element and walks `ti` then `rr` ascending, which
// is the host loop nest read down a single accumulator. So the gate on this file
// is byte equality with the CPU arm, NOT the 1.0e-3 bound the CUDA arm carries —
// that bound exists because mma.m16n8k16's internal accumulation order is
// unspecified and a split-K reduction sits on top of it, and this arm has
// neither. `-ffp-contract=off` is already set for HIP in CMakeLists.txt, which
// is what stops a compiler-generated FMA from contracting `acc += xv * w` into
// a different value.
//
// NO matrix cores, NO async copies, NO cooperative launch, and 3 KiB of LDS
// against a 64 KiB budget. The blockers above are not worked around; they are
// not encountered. Speed is a later row and this file offers none.
#include <hip/hip_runtime.h>
#include "rocm_env.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"
#include "vt/rocm/rocm_f16_codec.h"
#include "vt/rocm/rocm_exl3_decode.h"
#include "vt/rocm/rocm_exl3_fold.h"

namespace vt::rocm {
// The m<=8 GEMV arm lives in rocm_exl3_gemv.hip (spec
// .agents/specs/rocm-exl3-gemv.md): upstream's own `exl3_gemv_kernel.cuh`
// geometry, ROCm-ported via exllamav3-rocm. It writes the same f32 `raw`
// scratch, so step 3 below is unchanged whichever arm runs step 2. A false
// return is a DECLINE — the same contract the CUDA arm's `Exl3GemvTryLaunch`
// carries — and the call falls through to `Exl3GemmK`. `fold` offers step 3
// to the m=1 direct arms (rocm_exl3_fold.h).
bool Exl3GemvTryLaunchRocm(hipStream_t st, const Tensor& a_had, const Tensor& trellis,
                           float* raw, int size_m, int size_k, int size_n, int device,
                           const Exl3GemmArgs& args, Exl3GemvOutFold* fold);

// Spec rocm-exl3-gemv-residual: the dot arm (rocm_exl3_dot.hip) — a fused
// decode+dot kernel at every width 1..8, for shapes the upstream GEMV
// envelope declines (bits > 4 included) and for mid-m through 8-row chunks.
bool Exl3DotTryLaunchRocm(hipStream_t st, const Tensor& a_had, const Tensor& trellis,
                          float* raw, int size_m, int size_k, int size_n, int device,
                          const Exl3GemmArgs& args);
// Spec gfx1100-exl3-decode-60tps: the fused in-had m=1 arm. When it accepts,
// it computes a_had inside the kernel (HadBlock128ToH2, bit-identical to
// HadK<true,true>) and the caller MUST NOT launch the separate input
// hadamard. a_raw is the UNTRANSFORMED input (f16 storage), suh_ptr the
// pre-scale vector. Same decline contract as Exl3GemvTryLaunchRocm.
bool Exl3GemvM1FusedTryLaunchRocm(hipStream_t st, const void* a_raw,
                                  const uint16_t* suh_ptr,
                                  const Tensor& trellis, float* raw,
                                  int size_k, int size_n, int device,
                                  const Exl3GemmArgs& args, bool a_bf16);
bool Exl3DotChunkedTryLaunchRocm(hipStream_t st, const Tensor& a_had,
                                 const Tensor& trellis, float* raw, int size_m,
                                 int size_k, int size_n, int device,
                                 const Exl3GemmArgs& args);

// The reconstruct+hipBLAS arm (rocm_exl3_recon.hip), registered as
// kExl3ReconstructGemm and reused here for shapes the fused path cannot
// accelerate: bits > 4 (the 6 bpw lm_head) and mid-m prefill. An EMPTY
// w_scratch selects its persistent per-stream scratch, matching the contract
// include/vt/ops.h documents for the registered entry.
void Exl3ReconstructGemmKernelRocm(Queue& q, Tensor& c, const Tensor& a,
                                   const Tensor& trellis, const Tensor& suh,
                                   const Tensor& svh, Tensor& a_had,
                                   Tensor& w_scratch, const Exl3GemmArgs& args);

// The had_r_128 stages, callable by the reconstruct+GEMM arm in
// rocm_exl3_recon.hip, which runs the same transform before and after its
// hipBLAS GEMM. `half_in`/`half_out` select among the four instantiations the
// fused path uses; anything else is refused rather than silently widened.
void Exl3LaunchHadRocm(hipStream_t st, void* out, const void* in, const uint16_t* pre,
                       const uint16_t* post, float r_scale, int64_t rows, int64_t cols,
                       int in_dt, int out_dt);

namespace {

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess)
    throw std::runtime_error(std::string("vt rocm exl3: ") + what + ": " +
                             hipGetErrorString(err));
}
inline hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }

// 1/sqrt(128) spelled as upstream spells it (hadamard.cu:107) and as
// cpu_exl3_kernels.cpp spells it. Kept as the literal and never recomputed: a
// recomputed 1/sqrt(128) can differ in the last f32 bit, and that bit is the
// difference between this file's byte gate passing and failing.
constexpr float kInvSqrt128 = 0.088388347648f;

// ─── numeric helpers, transcribed from the CPU arm ───────────────────────────

// cpu_exl3_kernels.cpp NegBySignBit. XORing the sign bit, not `0.0f - v`: the
// two differ on a zero and this arm reproduces the host operation.
__device__ inline float NegBySignBit(float v) {
  return __int_as_float(static_cast<int>(__float_as_uint(v) ^ 0x80000000u));
}

// cpu_exl3_kernels.cpp MulF16 — the fp16 multiply __hmul2 performs. The exact
// product of two fp16 significands is 22 bits and fits f32 exactly, so rounding
// once from f32 is the value an fp16 multiply produces.
__device__ inline uint16_t MulF16(uint16_t a, uint16_t b) {
  return DF32ToF16(DF16ToF32(a) * DF16ToF32(b));
}
// ─── the trellis decode lives in rocm_exl3_decode.h ─────────────────────────
// Exl3TileCodeword / Exl3DecodeCodeword / Exl3TileRowMajorIndex, shared with
// rocm_exl3_recon.hip (the reconstruct+GEMM arm) so the two ops an EXL3
// checkpoint decodes through read identical codebooks rather than two
// drifting copies.


// ─── step 1 and step 3: had_r_128 ────────────────────────────────────────────
//
// One wave32 per 128-block, four waves per workgroup. Lane t holds the four
// values of columns 4t..4t+3, so lane bit `i` carries Hadamard level 4*i and
// levels 4..64 are exactly the five xor-partner steps — ShuffleHadWarp. The
// partners are exchanged in registers: DPP row_xmask for xor 1/2/4/8 and
// v_permlanex16 for xor 16 (ISA §7.7.1), so a block needs no LDS and no
// barrier; each lane's per-element operations (own-first adds, sign-bit
// flips) are the LDS formulation's, in the same order.
//
// IN-PLACE IS SAFE: a wave owns its whole 128-block, and every lane's four
// loads precede the first exchange, which no store can overtake. Upstream
// says the same of its own kernel ("Works inplace if y == x",
// hadamard.cu:86) and Exl3Gemm's contract permits `a_had` to alias `a`.
// InD/OutD are element dtype codes: 0 = f32, 1 = f16, 2 = bf16. bf16 exists
// because the EXL3 linear callers sit on a bf16 residual stream: folding the
// CastF16 stage into the load (DF32ToF16(bf16->f32) is bit-identical to
// CastF16K<bf16>) and the CastBf16 into the store (__float2bfloat16, the same
// op CastBf16K runs) erases two elementwise launches per call.
// Lane t ^ I's value of v, in registers.
template <int I>
__device__ __forceinline__ float HadXorLane(float v) {
  const int x = __float_as_int(v);
  if constexpr (I == 16)
    return __int_as_float(
        __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false));
  else
    return __int_as_float(__builtin_amdgcn_mov_dpp(x, 0x160 + I, 0xf, 0xf, false));
}

template <int I>
__device__ __forceinline__ void HadXorLevel(float (&x)[4], int t) {
  const bool flip = (t & I) != 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float par = HadXorLane<I>(x[j]);
    const float own = flip ? NegBySignBit(x[j]) : x[j];
    x[j] = own + par;  // own FIRST, as the host adds
  }
}

__device__ __forceinline__ void Unpack4x16(uint2 w, uint16_t (&h)[4]) {
  h[0] = static_cast<uint16_t>(w.x);
  h[1] = static_cast<uint16_t>(w.x >> 16);
  h[2] = static_cast<uint16_t>(w.y);
  h[3] = static_cast<uint16_t>(w.y >> 16);
}

template <int InD, int OutD>
__global__ void HadK(void* __restrict__ out, const void* __restrict__ in,
                     const uint16_t* __restrict__ pre, const uint16_t* __restrict__ post,
                     float r_scale, int64_t nblocks, int64_t blocks_per_row, int64_t cols,
                     int64_t total_groups) {
  const int ty = static_cast<int>(threadIdx.y);
  const int t = static_cast<int>(threadIdx.x);

  for (int64_t g = blockIdx.x; g < total_groups; g += gridDim.x) {
    const int64_t gb = g * 4 + ty;
    if (gb >= nblocks) continue;  // wave-uniform: a wave is one 128-block
    const int64_t row = gb / blocks_per_row;
    const int64_t blk = gb % blocks_per_row;
    const int64_t base = row * cols + blk * 128 + 4 * t;
    const int64_t sbase = blk * 128 + 4 * t;  // the scale vectors are indexed by COLUMN

    // load + pre-scale (hadamard_inner.cuh:109-114 for the half arm, :167-174
    // for the float arm: an fp16 multiply BEFORE the widen, versus a widen and
    // an f32 multiply).
    uint16_t ps[4] = {0, 0, 0, 0};
    if (pre != nullptr) Unpack4x16(*reinterpret_cast<const uint2*>(pre + sbase), ps);
    float v[4];
    if constexpr (InD == 0) {
      const float4 f = *reinterpret_cast<const float4*>(static_cast<const float*>(in) + base);
      const float fv[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] = pre != nullptr ? fv[j] * DF16ToF32(ps[j]) : fv[j];
    } else {
      uint16_t h[4];
      Unpack4x16(*reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(in) + base), h);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        // bf16 source: fold the caller's CastF16 into the load — widen then
        // DF32ToF16 round, the exact chain CastF16K<bf16> runs, so the f16
        // value the transform sees is bit-identical.
        if constexpr (InD == 2) h[j] = DF32ToF16(__uint_as_float(static_cast<uint32_t>(h[j]) << 16));
        const uint16_t w = pre != nullptr ? MulF16(h[j], ps[j]) : h[j];
        v[j] = DF16ToF32(w);
      }
    }

    // levels 1 and 2, on the four values this lane holds
    // (hadamard_inner.cuh:118-129).
    const float s0 = v[0] + v[1];
    const float d0 = v[0] - v[1];
    const float s1 = v[2] + v[3];
    const float d1 = v[2] - v[3];
    float x[4] = {s0 + s1, d0 + d1, s0 - s1, d0 - d1};

    // levels 4..64: ShuffleHadWarp, five xor-partner steps.
    HadXorLevel<1>(x, t);
    HadXorLevel<2>(x, t);
    HadXorLevel<4>(x, t);
    HadXorLevel<8>(x, t);
    HadXorLevel<16>(x, t);

    // r_scale, then the store and the post-scale. Both half-output inners round
    // FIRST and apply the post scale as an fp16 multiply afterwards
    // (hadamard_inner.cuh:137-146); the float arms multiply in f32.
    uint16_t qs[4] = {0, 0, 0, 0};
    if (post != nullptr) Unpack4x16(*reinterpret_cast<const uint2*>(post + sbase), qs);
    if constexpr (OutD == 0) {
      float o[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float res = x[j] * r_scale;
        o[j] = post != nullptr ? res * DF16ToF32(qs[j]) : res;
      }
      *reinterpret_cast<float4*>(static_cast<float*>(out) + base) =
          make_float4(o[0], o[1], o[2], o[3]);
    } else {
      uint16_t o[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float res = x[j] * r_scale;
        if constexpr (OutD == 1) {
          const uint16_t r = DF32ToF16(res);
          o[j] = post != nullptr ? MulF16(r, qs[j]) : r;
        } else {
          // bf16 destination: one RN round of res*post, bit-identical to the
          // separate CastBf16K the caller would run on the f32 form.
          const float r = post != nullptr ? res * DF16ToF32(qs[j]) : res;
          o[j] = __bfloat16_as_ushort(__float2bfloat16(r));
        }
      }
      *reinterpret_cast<uint2*>(static_cast<uint16_t*>(out) + base) =
          make_uint2(o[0] | (static_cast<uint32_t>(o[1]) << 16),
                     o[2] | (static_cast<uint32_t>(o[3]) << 16));
    }
  }
}

void LaunchHadIO(void* out, const void* in, const uint16_t* pre, const uint16_t* post,
                 float r_scale, int64_t rows, int64_t cols, hipStream_t st,
                 int in_dt, int out_dt) {
  const int64_t blocks_per_row = cols / 128;
  const int64_t nblocks = rows * blocks_per_row;
  if (nblocks <= 0) return;
  const int64_t total_groups = (nblocks + 3) / 4;
  const unsigned grid =
      static_cast<unsigned>(total_groups > 65535 ? 65535 : total_groups);
#define VT_HAD_LAUNCH(I_, O_)                                               \
  if (in_dt == I_ && out_dt == O_) {                                        \
    HadK<I_, O_><<<grid, dim3(32, 4), 0, st>>>(out, in, pre, post, r_scale, \
                                             nblocks, blocks_per_row, cols, \
                                             total_groups);                 \
    Check(hipGetLastError(), "had_r_128");                                  \
    return;                                                                 \
  }
  // Every instantiated pair — the callers' real dtype set: f32 raw scratch
  // in/out, f16 a_had, bf16 residual. Anything else (f16->f32 included)
  // throws below.
  VT_HAD_LAUNCH(1, 1) VT_HAD_LAUNCH(0, 0) VT_HAD_LAUNCH(0, 1)
  VT_HAD_LAUNCH(2, 1) VT_HAD_LAUNCH(0, 2)
#undef VT_HAD_LAUNCH
  throw std::runtime_error("vt rocm exl3: had_r_128 has no arm for dtype pair (" +
                           std::to_string(in_dt) + ", " + std::to_string(out_dt) +
                           ") — 0=f32 1=f16 2=bf16");
}

// ─── step 2: the tiled GEMM against the decoded trellis ──────────────────────
//
// Exl3GemmKernelCpu step 2, with its loop nest read down a SINGLE accumulator.
// One workgroup owns one 16-column output tile and up to 16 rows; one thread
// owns one output element. Per k-tile the 256 threads decode the 256 codewords
// of that tile into LDS — which is Exl3DecodeTile with its `t` loop unrolled
// across threads, and the ONLY place a thread reads a codeword it does not own.
//
// THE ACCUMULATION ORDER IS THE HOST'S: `ti` ascending in the outer loop, `rr`
// ascending inside, into one f32 register. The host's `if (xv == 0.0f) continue`
// is kept and is not an optimisation here — with acc == -0.0f, adding a +0.0f
// product flips the sign of the zero, so dropping the guard would make this arm
// differ from its own reference on exactly the values a byte gate exists to
// catch.
__global__ void Exl3GemmK(float* __restrict__ raw, const uint16_t* __restrict__ ah,
                          const uint16_t* __restrict__ trellis, int64_t m, int64_t k,
                          int64_t n, int bits, int codebook, int half_flag) {
  __shared__ float tile[256];  // 1 KiB
  const int64_t tiles_n = n / 16;
  // A frac tile (K+0.5, mul1) is 16*KA+8 uint16 words; the reader switches
  // with it. `half_flag` is validated on the host (codebook == 2, KA 1..7).
  const int64_t tile_words =
      16 * static_cast<int64_t>(bits) + (half_flag ? 8 : 0);
  const int64_t tj = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y) * 16 + tid / 16;
  const int cc = tid % 16;
  const bool live = row < m;

  float acc = 0.0f;
  for (int64_t ti = 0; ti < k / 16; ++ti) {
    const uint16_t* tp = trellis + (ti * tiles_n + tj) * tile_words;
    __syncthreads();  // the previous iteration's readers are done with `tile`
    const uint16_t cw = half_flag ? Exl3FracTileCodeword(tp, bits, tid)
                                  : Exl3TileCodeword(tp, bits, tid);
    tile[Exl3TileRowMajorIndex(tid)] = Exl3DecodeCodeword(cw, codebook);
    __syncthreads();
    if (live) {
      const uint16_t* arow = ah + row * k + ti * 16;
      for (int rr = 0; rr < 16; ++rr) {
        const float xv = DF16ToF32(arow[rr]);
        if (xv == 0.0f) continue;
        acc += xv * tile[rr * 16 + cc];
      }
    }
  }
  if (live) raw[row * n + tj * 16 + cc] = acc;
}

// ─── the f32 staging buffer ──────────────────────────────────────────────────
//
// The CPU arm holds `raw` in a std::vector; a device needs the same [m, n] f32
// between step 2 and step 3. Grow-only, per stream, allocated stream-ordered and
// never freed — the discipline rocm_grouped_gemm.hip's EnsureQuantScratch
// documents, and for its reason: hipMalloc/hipFree/hipStreamSynchronize per call
// are illegal under hipGraph stream capture, and a captured graph may have baked
// the pointer. Bounded: it grows O(log(max/min)) times over a process.
//
// NAMED AS A COST rather than hidden. A fused kernel would not need it. A fused
// kernel is the later speed row, and this one buys the end of the reference tier.
struct StreamScratch {
  void* buf = nullptr;
  size_t bytes = 0;
  uint32_t* cnt = nullptr;  // the output fold's per-128-column counters
  size_t ncnt = 0;
};

StreamScratch& ScratchFor(hipStream_t s) {
  static std::mutex mu;
  static std::unordered_map<hipStream_t, StreamScratch> pools;
  std::lock_guard<std::mutex> lk(mu);
  return pools[s];
}

float* EnsureRawScratch(size_t need, hipStream_t s) {
  StreamScratch& sc = ScratchFor(s);
  if (need > sc.bytes) {
    // Stream-ordered free of the superseded buffer: queued behind the work
    // that still references it, so a grow mid-launch is safe.
    if (sc.buf != nullptr) Check(hipFreeAsync(sc.buf, s), "raw scratch free");
    Check(hipMallocAsync(&sc.buf, need, s), "raw scratch grow");
    sc.bytes = need;
  }
  return static_cast<float*>(sc.buf);
}

// Same grow-only, per-stream, never-freed discipline as the raw scratch. The
// kernels leave every counter zero, so the zeroing memset runs only on growth;
// a growth inside a capture bakes a memset node that rewrites zeros.
uint32_t* EnsureFoldCounters(size_t need, hipStream_t s) {
  StreamScratch& sc = ScratchFor(s);
  if (need > sc.ncnt) {
    if (sc.cnt != nullptr)
      Check(hipFreeAsync(sc.cnt, s), "fold counters free");
    Check(hipMallocAsync(reinterpret_cast<void**>(&sc.cnt), need * sizeof(uint32_t), s),
          "fold counters grow");
    Check(hipMemsetAsync(sc.cnt, 0, need * sizeof(uint32_t), s), "fold counters zero");
    sc.ncnt = need;
  }
  return sc.cnt;
}

}  // namespace

void Exl3LaunchHadRocm(hipStream_t st, void* out, const void* in, const uint16_t* pre,
                       const uint16_t* post, float r_scale, int64_t rows, int64_t cols,
                       int in_dt, int out_dt) {
  // in_dt/out_dt: 0=f32 1=f16 2=bf16 — the dtype codes HadK templated on.
  // Every uninstantiated pair (f16->f32 included) throws out of LaunchHadIO.
  LaunchHadIO(out, in, pre, post, r_scale, rows, cols, st, in_dt, out_dt);
}

// The fused chain, step for step as Exl3GemmKernelCpu runs it:
//   1. A_had = had_r_128(A, pre_scale = suh)
//   2. C_raw = A_had @ reconstruct(trellis)        [f32 accumulation]
//   3. C     = had_r_128(C_raw, post_scale = svh)
// Argument validation (shapes, dtypes, contiguity, the 16/128 multiples) is done
// once in `vt::Exl3Gemm` for every backend and is not repeated here. What IS
// checked here is what selects device code: `bits` and `codebook` reach the
// kernel as values, and an out-of-range one would decode silently rather than
// throw, because a device has no VT_CHECK.
void Exl3GemmKernelRocm(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                        const Tensor& suh, const Tensor& svh, Tensor& a_had,
                        const Exl3GemmArgs& args) {
  EnsureQueueDevice(q);
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = c.shape[1];
  if (m == 0 || k == 0 || n == 0) return;
  VT_CHECK(args.bits >= 1 && args.bits <= 8,
           "vt rocm exl3: bits must be in [1, 8]; got " + std::to_string(args.bits));
  VT_CHECK(args.codebook >= 0 && args.codebook <= 2,
           "vt rocm exl3: codebook " + std::to_string(args.codebook) +
               " is not implemented (0 == 3INST, 1 == MCG, 2 == mul1). Upstream defines "
               "no other value: `decode_3inst<cb>` (codebook.cuh:56-90) has arms for 0, "
               "1 and 2 and falls off the end for anything else.");
  // HALF-INTEGER rate (BACKEND-ROCM frac rates): K+0.5 is mul1-only, and the
  // GEMV/m1-fused arms below carry no frac instantiation — a frac tensor must
  // not fall into an integer-(KA, 2) GEMV arm, which would pass every existing
  // gate and decode the wrong bitstream. It routes to the dot arm, and past
  // the dot arm's shape envelope to Exl3GemmK (whose decode reads the frac
  // windows through Exl3FracTileCodeword).
  VT_CHECK(!args.half || args.codebook == 2,
           "vt rocm exl3: a half-integer rate (K+0.5) requires the mul1 codebook (2); "
           "got codebook " + std::to_string(args.codebook));

  hipStream_t st = AsStream(q);

  // Spec rocm-exl3-gemv-residual, R2/R3: shapes the fused path cannot run
  // fast delegate BEFORE the input transform, because the reconstruct arm
  // runs Hadamard stages of its own. All of them honour VT_EXL3_GEMV=0 (and
  // force_gemv=0 via the ops layer): mode 0 means the byte-exact
  // transcription for EVERY shape on this backend.
  const bool fast_paths = args.force_gemv != 0 &&
                          (Exl3GemvMode() != 0 || args.force_gemv > 0);
  // R3: shapes upstream's GEMV envelope cannot express — this checkpoint's
  // 6 bpw lm_head, uninstantiated (bits, cb) — take the dot arm (fused
  // decode+dot, capture-safe, all widths) instead of the transcription.
  // The reconstruct arm stays for mid-m above the chunk cap; routing
  // m<=8+chunks to recon was tried first and measured ~40 ms/token on the
  // lm_head (decode to fp16 scratch + hipBLAS is 2.5x the trellis traffic).
  const bool dot_shape = (k % 128 == 0) && (n % 128 == 0);
  // The reconstruct arm's Hadamards and hipBLAS GEMM are f16-in, f16/f32-out
  // only; a bf16 a or c (admitted by ops.cpp for the folded casts) takes the
  // Exl3GemmK route below, whose Hadamards carry the dtype codes.
  const bool recon_dtypes = a.dtype == DType::kF16 &&
                            (c.dtype == DType::kF16 || c.dtype == DType::kF32);
  // No frac reconstruct arm (spec ## Owed): a half-integer rate must not reach
  // ReconstructDecodeSliceK, which would decode the frac bitstream as integer
  // KA. Exl3GemmK decodes it correctly.
  const bool want_recon = fast_paths && args.force_gemv <= 0 && m > 32 &&
                          dot_shape && recon_dtypes && !args.half;
  if (want_recon) {
    // Mid-m above the chunked cap: B would be re-read per 8-row chunk, so
    // past 32 rows the one-pass decode + hipBLAS GEMM wins.
    Tensor w_scratch;  // empty: the arm's persistent per-stream scratch
    Exl3ReconstructGemmKernelRocm(q, c, a, trellis, suh, svh, a_had, w_scratch, args);
    return;
  }

  // 1. the input transform, into the caller's scratch (which may alias A).
  // For m == 1 the fused arm is tried BEFORE this launches: it computes a_had
  // in LDS inside the GEMV kernel (HadBlock128ToH2, bit-identical to
  // HadK<true,true>) so the launch below is skipped only when it accepted.
  float* raw = EnsureRawScratch(static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float),
                                st);
  bool gemv = false;
  // VT_EXL3_DOT_FIRST=1 is an A/B lever, not a contract: it asks for the dot
  // kernel BEFORE the m<=8 GEMV so its own numbers (it reads each trellis
  // word once and carries no prefetch-ring bookkeeping) can be compared
  // against the emulated-fragment arm on real serving. Default keeps
  // upstream's order.
  static const bool kDotFirstEnv =
      EnvIntStrict("VT_EXL3_DOT_FIRST", 0) != 0;
  const bool dot_first = fast_paths && kDotFirstEnv;
  // The fused arm is NOT tried when VT_EXL3_DOT_FIRST is set: that lever asks
  // for the dot arm ahead of every GEMV variant, and the dot arm consumes the
  // already-transformed a_had, so it cannot run ahead of the input transform
  // the way the fused arm does. It also honours the same fast_paths gate as
  // every fast arm — VT_EXL3_GEMV=0 must still select the transcription.
  if (m == 1 && fast_paths && !args.half && !(dot_first && dot_shape)) {
    gemv = Exl3GemvM1FusedTryLaunchRocm(
        st, a.data, suh.data ? suh.Ptr<uint16_t>() : nullptr, trellis, raw,
        static_cast<int>(k), static_cast<int>(n), q.device.index, args,
        a.dtype == DType::kBF16);
  }
  if (!gemv) {
    // bf16 a arrives only on this backend (ops.cpp gates the dtype on ROCm);
    // the bf16-in arm folds the caller's CastF16 bit-identically.
    const int a_dt = (a.dtype == DType::kBF16) ? 2 : 1;
    LaunchHadIO(a_had.data, a.data, suh.Ptr<uint16_t>(), nullptr,
                kInvSqrt128, m, k, st, a_dt, 1);
  }

  Exl3GemvOutFold fold;
  if (m <= kExl3GemvMaxM) {
    if (dot_first && dot_shape)
      gemv = Exl3DotTryLaunchRocm(st, a_had, trellis, raw, static_cast<int>(m),
                                  static_cast<int>(k), static_cast<int>(n),
                                  q.device.index, args);
    if (!gemv) {
      if (m == 1 && c.dtype == DType::kBF16 && svh.data != nullptr) {
        fold.out = c.data;
        fold.post = svh.Ptr<uint16_t>();
        fold.cnt = EnsureFoldCounters(static_cast<size_t>(n / 128), st);
        fold.r_scale = kInvSqrt128;
      }
      gemv = Exl3GemvTryLaunchRocm(st, a_had, trellis, raw, static_cast<int>(m),
                                   static_cast<int>(k), static_cast<int>(n),
                                   q.device.index, args, &fold);
    }
    if (!gemv && fast_paths && dot_shape)
      gemv = Exl3DotTryLaunchRocm(st, a_had, trellis, raw, static_cast<int>(m),
                                  static_cast<int>(k), static_cast<int>(n),
                                  q.device.index, args);
  } else if (fast_paths && dot_shape && m <= 32) {
    gemv = Exl3DotChunkedTryLaunchRocm(st, a_had, trellis, raw, static_cast<int>(m),
                                       static_cast<int>(k), static_cast<int>(n),
                                       q.device.index, args);
  }
  if (!gemv && args.force_gemv > 0) {
    throw std::runtime_error(
        "vt rocm exl3: exl3_gemm was asked to force the m<=8 GEMV arm (force_gemv=1) but "
        "the arm declined: bits " + std::to_string(args.bits) + " codebook " +
        std::to_string(args.codebook) + " m " + std::to_string(m) + " k " +
        std::to_string(k) + " n " + std::to_string(n) +
        ". Upstream's own direct entry point refuses the same way (exl3_gemv.cu:238).");
  }
  if (!gemv) {
    const unsigned gx = static_cast<unsigned>(n / 16);
    const unsigned gy = static_cast<unsigned>((m + 15) / 16);
    const int half_flag = args.half ? 1 : 0;
    Exl3GemmK<<<dim3(gx, gy), 256, 0, st>>>(raw, a_had.Ptr<uint16_t>(),
                                           trellis.Ptr<uint16_t>(), m, k, n, args.bits,
                                           args.codebook, half_flag);
    Check(hipGetLastError(), "exl3_gemm");
  }

  // 3. the output transform — had_ff for an f32 C, had_fh for an fp16 one, the
  // same two arms the CPU reference picks between off `c.dtype`. Already done
  // when the m=1 GEMV ran it in its epilogue.
  if (fold.applied) return;
  const int c_dt = (c.dtype == DType::kBF16) ? 2 : (c.dtype == DType::kF16 ? 1 : 0);
  LaunchHadIO(c.data, raw, nullptr, svh.Ptr<uint16_t>(), kInvSqrt128, m, n, st,
              0, c_dt);
}

}  // namespace vt::rocm

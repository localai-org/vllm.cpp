// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA baseline kernels for rmsnorm / silu_and_mul / embedding / rope_neox.
// Correctness-grade (M0.6): plain grid-stride / one-block-per-row kernels, f32
// accumulation, double-precision RoPE angles matching the CPU reference.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cub/cub.cuh>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "vt/cuda/cuda_embedding_quant.h"  // the KGATHER seam: block-table gather + its error record
#include "vt/cuda/rmsnorm_decode_fast.h"
#include "vt/ops.h"
#include "vt/op_provider.h"
#include "vt/dflash_attn_grid.h"
#include "vt/dflash_attn_mask.h"

namespace vt::cuda {

#ifdef VLLM_CPP_FLASH_ATTN
// Defined in cuda_flash_attn_fa2.cu (same TU set, gated on the same feature).
// Declared here at vt::cuda scope — NOT inside the anonymous namespace below —
// exactly as cuda_paged_attn.cu:53-64 declares the paged FA-2 launchers.
void LaunchDenseFA2Bf16(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& key,
                        const Tensor& value, float scale, bool causal);
#endif

namespace {

constexpr int kBlock = 256;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

unsigned GridFor(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
}

// f32 load/store overloads: bf16 converts on the way in/out, math is f32.
__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__half* p, int64_t i, float v) { p[i] = __float2half_rn(v); }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, same as host F32ToBF16
}

// VECTOR load of N CONSECUTIVE elements into f32, for the attention inner loops.
// N is a compile-time width, so N==4/2 collapse to one LDG.128/LDG.64 (f32) or
// LDG.64/LDG.32 (bf16) instead of N separate scalar loads. Callers guarantee the
// element offset is a multiple of N, which is what makes the wide load legal.
// GAMMA-ONLY element type (#2503). `vt::RmsNorm` and `vt::RmsNormQuantFp8` both
// admit `IsFloat(weight.dtype)` (ops.cpp:24, :1030, :748), which includes kF16,
// and the CPU siblings widen a kF16 gamma like any other (`WidenRowToF32`,
// cpu_ops.cpp:554-557; `LoadF32`, cpu_ops.cpp:1042) -- so the CUDA arm refusing
// it was a device arm refusing a dtype its CPU sibling accepts, which
// `cuda_qwen4_exp.cu:60-62` says must not stand unrecorded. vLLM's own
// `GemmaRMSNorm` upcasts whatever the gamma is (`self.weight.float()`,
// vllm/models/qwen4_exp/nvidia/ple_layer.py:80 at vLLM origin/main cdefd9d499 --
// AHEAD of this project's pin 5559679229, a forward reference, #2502), and
// `CudaPlatform::supported_dtypes` already promises kF16 (platforms/cuda.cpp:112).
//
// The earlier gamma-only extension left activation/output support owed (#2542).
// The shared FP16 scalar arm now also admits half activations and outputs;
// the gamma loader above remains valid for all three scalar formats.
__device__ inline float Load(const __half* p, int64_t i) { return __half2float(p[i]); }

template <int N>
__device__ inline void LoadVec(const float* p, int64_t i, float* o) {
  if (N == 4) {
    const float4 v = *reinterpret_cast<const float4*>(p + i);
    o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
  } else if (N == 2) {
    const float2 v = *reinterpret_cast<const float2*>(p + i);
    o[0] = v.x; o[1] = v.y;
  } else {
#pragma unroll
    for (int c = 0; c < N; ++c) o[c] = p[i + c];
  }
}
template <int N>
__device__ inline void LoadVec(const __nv_bfloat16* p, int64_t i, float* o) {
  if (N == 4) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p + i);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
    o[0] = a.x; o[1] = a.y; o[2] = b.x; o[3] = b.y;
  } else if (N == 2) {
    const unsigned raw = *reinterpret_cast<const unsigned*>(p + i);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw));
    o[0] = a.x; o[1] = a.y;
  } else {
#pragma unroll
    for (int c = 0; c < N; ++c) o[c] = __bfloat162float(p[i + c]);
  }
}

// Round-trip a f32 value through the residual store dtype so the variance below
// squares the SAME rounded value that gets written back to the residual stream —
// mirrors vLLM fused_add_rms_norm, whose bf16 residual add (`z += residual`) rounds
// to the model dtype before the f32 variance (layernorm_kernels.cu). Identity for a
// f32 residual, so the previous f32-residual path stays byte-for-byte unchanged.
template <typename Tres> __device__ inline float ResRound(float v);
template <> __device__ inline float ResRound<float>(float v) { return v; }
template <> __device__ inline float ResRound<__half>(float v) { return __half2float(__float2half_rn(v)); }
template <> __device__ inline float ResRound<__nv_bfloat16>(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

// ---------------------------------------------------------------------------
// rmsnorm: one block per row, shared-memory f32 tree reduction.
// Upstream csrc counterpart: csrc/layernorm_kernels.cu (rms_norm_kernel / fused_add_rms_norm_kernel) — align signatures post-MVP.

// `Tw` is the GAMMA's own dtype and is INDEPENDENT of `Tin` (#2477). It used to be
// `Tin`, which welded the two and forced the dispatcher to refuse the pairing the
// qwen4_exp QSA block issues: an **f32 ACTIVATION against a bf16 GAMMA**, at
// qwen4_exp_qsa_block.cpp:705, where `vt::AttnGateSplit` fills a `DType::kF32`
// `q_f32` (:694) and the gamma is bf16.
//
// NOT the reverse, and an earlier revision of this comment said the reverse. The
// released GGUF stores norm tensors as F32 ON DISK, but the loader converts:
// `LoadNormBf16` (qwen4_exp_weights.cpp) runs `DequantAll` into an f32 vector and
// returns `Bf16From(...)` = `MakeTensor(kBF16, ...)` (glm_moe_dsa_loader.cpp:145-151).
// On-disk dtype is not runtime dtype.
//
// THIS IS A SYMPTOM FIX AND THE SPEC SAYS SO. The cause is #2488: upstream chunks
// `q` off the bf16 QKV GEMM and hands it to `q_norm` unwidened
// (vllm/model_executor/models/qwen3_next.py:384-440), so the f32 buffer is the
// divergence. Narrowing it is blocked on `AttnGateSplit`'s f32 output being welded
// into its own CUDA signature (cuda_glue.cu:181) and shared with Qwen3.5.
//
// Decoupling the two dtypes is nevertheless right on its own terms: vLLM never
// couples them (`GemmaRMSNorm` does `self.weight.float()`, vllm/models/qwen4_exp/
// nvidia/ple_layer.py:80 -- read at vLLM origin/main 25efcfa788, which is AHEAD of
// this project's pin 5559679229; see #2502), this file's CPU sibling widens `w` and
// `x` separately (cpu_ops.cpp:554-557, :576), and the CUDA grouped-norm sibling
// already carries an independent weight tag (cuda_rms_norm_group.cu:194).
// `Load()` is overloaded for both element types above, so the body is unchanged and
// the f32 arithmetic below is bit-identical for every pairing that already worked.
template <typename Tin, typename Tw, typename Tout, typename Tres>
__global__ void RmsNormRowKernel(Tout* out, const Tin* x, const Tw* w, Tres* residual,
                                 int64_t h, float eps, bool gemma, int64_t x_stride, int64_t res_stride) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * x_stride;
  Tout* orow = out + row * h;
  Tres* rrow = residual == nullptr ? nullptr : residual + row * res_stride;

  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float v = Load(xrow, j);
    if (rrow != nullptr) {
      v = ResRound<Tres>(v + Load(rrow, j));  // new residual stream: f32 add, round to Tres
      Store(rrow, j, v);                       // updated in place (f32 or bf16)
    }
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    const float v = rrow != nullptr ? Load(rrow, j) : Load(xrow, j);
    float wj = Load(w, j);
    if (gemma) wj += 1.0f;
    Store(orow, j, v * inv * wj);
  }
}

// ---------------------------------------------------------------------------
// Decode-fast rmsnorm variant (VT_RMSNORM_DECODE_FAST). Re-expressed 2026-07-17
// (KERNEL-EW-NORM-ACT numerics rework, CLAIM-EW-NORM-ACT-2) to be BIT-IDENTICAL
// to the shipped RmsNormRowKernel above (the through-stack 235/235 bit-reference
// that matches vLLM's production greedy stream), not merely ≤1-ulp close.
//
// WHY BIT-IDENTICAL (not ≤1-ulp): the 27B greedy token 6 is a RAZOR near-tie
// (198 "\n" prod vs 271 "\n\n" emu; the logit gap is ~zero). The shipped
// RmsNormRowKernel + GDN cubin = 235/235 (198). The prior fast kernel differed
// from shipped by ≤1 ulp (residual-add rounding + a different variance reduction
// ORDER + rsqrtf); accumulated over 64 layers alongside the GDN cubin's own
// ≤1-ulp perturbation, that tipped the tie to 271 (233/235), forcing the default
// back OFF (a875397). Making the fast output the SAME BITS as shipped removes the
// perturbation by construction: fast+cubin ≡ shipped+cubin ≡ 198 always.
//
// The three divergences vs shipped RmsNormRowKernel (cuda_ops.cu:62-93) and the
// fix that makes each bit-exact:
//   (1) residual add. shipped:62-79 does v = ResRound<bf16>(f32(x)+f32(res)) — a
//       f32 add of the two bf16 operands then a SINGLE round to bf16 (double
//       rounding through f32). The prior fast used __hadd2 (single-round bf16
//       add), which differs from the shipped double-round on rare carries. Fixed:
//       __float2bfloat16(f32(x)+f32(res)) reproduces ResRound exactly.
//   (2) variance sum ORDER. shipped:70-85 uses kBlock=256 threads, per-thread
//       partial acc = Σ_m sq[tid + 256*m] (increasing m), then the shared-memory
//       binary tree `for s=128; s>0; s>>=1`. f32 add is non-associative, so the
//       sum's bits depend on this exact 256-partial + tree structure. The prior
//       fast used cub::BlockReduce<float,1024> over 1024 threads (a different
//       thread count AND a different reduction tree) => a different f32 variance.
//       Fixed: this kernel launches with kBlock(=256) threads and reproduces
//       shipped's scalar-strided Pass 1 and shared-tree byte-for-byte.
//   (3) inv. shipped:86 uses `1.0f / sqrtf(...)` (correctly-rounded sqrt+div);
//       the prior fast used rsqrtf (a ≤2-ulp reciprocal-sqrt approximation).
//       Fixed: `1.0f / sqrtf(partial[0]/h + eps)` verbatim.
//
// The ONLY thing that legitimately differs from shipped — and the whole source of
// any speedup — is Pass 2 (normalize): out = bf16((f32(res)*inv)*(f32(w)+gemma))
// is ELEMENT-INDEPENDENT, so vectorizing it with 16-byte (4×bf162) loads/stores
// changes memory traffic, NOT the arithmetic, and stays bit-identical. Pass 1
// (the variance) is order-sensitive and is kept byte-for-byte shipped.
//
// Scope: bf16-in / bf16-out / bf16-residual, H%8==0, H>=1024 (the 129
// input/post-attn/final residual RMSNorm launches, H=5120 on the 27B / H=2048 on
// the 35B). Other dtype/residual/small-H keep RmsNormRowKernel. The q/k head norms
// take no residual => not this path. 16-byte load = 4 packed bf162.
struct alignas(16) RmsNormBf16x8 {
  __nv_bfloat162 d[4];
};

// Launch geometry. The variance REDUCTION is byte-for-byte the shipped
// RmsNormRowKernel's (kBlock=256 partials + tree), but the memory passes run with
// kFastBlock=1024 threads: at decode the RMSNorm launches only `rows` (= num
// decode tokens, ~16 at c16) blocks, so the GPU is block-starved and thread-level
// parallelism per block — not occupancy — hides the memory latency. That
// thread-count is the ORIGINAL fast kernel's win; this rework keeps it while
// making the arithmetic bit-identical to shipped. kFastMaxH bounds the f32
// square-buffer below (guard rejects larger H); 8192 covers the 27B H=5120 / 35B
// H=2048 decode norms with headroom, at 32 KB static shared (< the 48 KB no-opt-in
// cap, and free here because only ~16 blocks are resident).
constexpr int kFastBlock = 1024;
constexpr int kFastMaxH = 8192;

// One block per row. Pass 1 (vectorized, kFastBlock threads): residual =
// bf16(f32(x)+f32(res)) [== shipped ResRound], stored bf16, with each element's
// f32 square written to the shared buffer ssq. Reduction (kBlock=256 threads):
// p_i = Σ_m ssq[i + 256*m] then the shared-memory binary tree — the EXACT shipped
// summation ORDER (cuda_ops.cu:70-86), reading the same bf16-rounded squares, so
// the f32 variance is bit-identical despite the vectorized loads. inv =
// 1.0f/sqrtf (shipped:86, not rsqrtf). Pass 2 (vectorized): normalize — element-
// independent, so identical bits at higher bandwidth. Every op matches shipped
// bit-for-bit; only the memory access WIDTH and thread COUNT differ.
__global__ void RmsNormRowFastKernel(__nv_bfloat16* __restrict__ out,
                                     const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ w,
                                     __nv_bfloat16* __restrict__ residual, int h, float eps,
                                     bool gemma) {
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t base = static_cast<int64_t>(blockIdx.x) * h;
  const int vh = h / 8;
  const RmsNormBf16x8* xv = reinterpret_cast<const RmsNormBf16x8*>(x + base);
  RmsNormBf16x8* rv = reinterpret_cast<RmsNormBf16x8*>(residual + base);
  const RmsNormBf16x8* wv = reinterpret_cast<const RmsNormBf16x8*>(w);
  RmsNormBf16x8* ov = reinterpret_cast<RmsNormBf16x8*>(out + base);

  __shared__ float ssq[kFastMaxH];   // per-element v^2 (v = the bf16-rounded residual)
  __shared__ float partial[kBlock];  // 256 partials for shipped's exact tree order

  // Pass 1 — vectorized residual add + store + per-element square into ssq. The
  // residual add is bf16(f32(x)+f32(res)) (== ResRound<bf16>, shipped:75) and the
  // square is of that bf16-rounded value (shipped:78), so ssq[j] is byte-for-byte
  // shipped's v*v term for every element.
  for (int vi = tid; vi < vh; vi += kFastBlock) {
    RmsNormBf16x8 t = xv[vi];
    RmsNormBf16x8 r = rv[vi];
    RmsNormBf16x8 nr;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 xf = __bfloat1622float2(t.d[k]);
      float2 rf = __bfloat1622float2(r.d[k]);
      nr.d[k] = __floats2bfloat162_rn(xf.x + rf.x, xf.y + rf.y);  // per-lane == __float2bfloat16
      float2 nf = __bfloat1622float2(nr.d[k]);                    // bf16 value back to f32
      ssq[vi * 8 + 2 * k] = nf.x * nf.x;
      ssq[vi * 8 + 2 * k + 1] = nf.y * nf.y;
    }
    rv[vi] = nr;  // residual store (bf16), bit-identical to shipped
  }
  __syncthreads();  // publish ssq (and residual) before the strided reduction reads them

  // Reduction — BYTE-FOR-BYTE shipped (cuda_ops.cu:80-86). Threads >=256 idle.
  if (tid < kBlock) {
    float acc = 0.0f;
    for (int j = tid; j < h; j += kBlock) acc += ssq[j];  // p_i = Σ_m ssq[i+256m], increasing m
    partial[tid] = acc;
  }
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);

  // Pass 2 — vectorized normalize (bit-identical to shipped:87-92; each output
  // element is (f32(res)*inv)*(f32(w)+gemma) rounded to bf16, independent of the
  // others). Each thread reloads the residual vector it wrote in Pass 1.
  for (int vi = tid; vi < vh; vi += kFastBlock) {
    RmsNormBf16x8 r = rv[vi];
    RmsNormBf16x8 wr = wv[vi];
    RmsNormBf16x8 o;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 rf = __bfloat1622float2(r.d[k]);
      float2 wf = __bfloat1622float2(wr.d[k]);
      float w0 = wf.x, w1 = wf.y;
      if (gemma) {
        w0 += 1.0f;
        w1 += 1.0f;
      }
      o.d[k] = __floats2bfloat162_rn(rf.x * inv * w0, rf.y * inv * w1);
    }
    ov[vi] = o;
  }
}

// f32-residual sibling of RmsNormRowFastKernel — BYTE-IDENTICAL to the shipped
// RmsNormRowKernel<float,float,float> (residual!=null, f32 in/out/residual). The Laguna
// NVFP4 decode's post-attn residual RMSNorm runs this shipped kernel as a <<<rows=1,256>>>
// single block (ncu: launch__waves_per_multiprocessor≈0.00, sm__throughput≈0.06% — one SM
// of ~100+, latency-bound). Same fix as the bf16 fast kernel: float4-vectorized memory
// passes over kFastBlock(=1024) threads hide the per-block memory latency (decode launches
// only `rows` blocks), while the variance REDUCTION reproduces the shipped kBlock(=256)
// strided partials + tree over the SAME per-element squares in the SAME Σ_m ssq[i+256m]
// order → bit-identical f32 variance. residual add is f32 x+res (ResRound<float> is
// identity, shipped:75), inv = 1.0f/sqrtf (shipped:86), normalize out=res*inv*(w[+gemma])
// element-independent — every bit matches shipped. Scope guard (h%4==0, 1024<=h<=kFastMaxH,
// 16B-aligned) in TryLaunchRmsNormDecodeFastF32.
__global__ void RmsNormRowFastF32Kernel(float* __restrict__ out, const float* __restrict__ x,
                                        const float* __restrict__ w, float* __restrict__ residual,
                                        int h, float eps, bool gemma) {
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t base = static_cast<int64_t>(blockIdx.x) * h;
  const int vh = h >> 2;
  const float4* xv = reinterpret_cast<const float4*>(x + base);
  float4* rv = reinterpret_cast<float4*>(residual + base);
  const float4* wv = reinterpret_cast<const float4*>(w);
  float4* ov = reinterpret_cast<float4*>(out + base);
  __shared__ float sv[kFastMaxH];     // residual VALUE v (not v²): the reduction squares with
  __shared__ float partial[kBlock];   // shipped's `acc += v*v` expression so nvcc emits the
                                      // SAME fma (f32 v² is NOT exact — pre-squaring the value
                                      // rounds it and diverges by ≤1 ulp; the bf16 sibling can
                                      // pre-square because a bf16² is exactly representable).

  // Pass 1 — residual = x + res (f32, == shipped ResRound<float> identity), store residual + v.
  for (int vi = tid; vi < vh; vi += kFastBlock) {
    float4 t = xv[vi], r = rv[vi], nr;
    nr.x = t.x + r.x;
    nr.y = t.y + r.y;
    nr.z = t.z + r.z;
    nr.w = t.w + r.w;
    rv[vi] = nr;
    const int e = vi << 2;
    sv[e] = nr.x;
    sv[e + 1] = nr.y;
    sv[e + 2] = nr.z;
    sv[e + 3] = nr.w;
  }
  __syncthreads();
  // Reduction — BYTE-FOR-BYTE shipped (cuda_ops.cu:80-86): thread t squares+accumulates
  // v[t],v[t+256],… with the IDENTICAL `acc += v*v` expression (same nvcc fma). Threads >=256 idle.
  if (tid < kBlock) {
    float acc = 0.0f;
    for (int j = tid; j < h; j += kBlock) acc += sv[j] * sv[j];
    partial[tid] = acc;
  }
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);
  // Pass 2 — normalize (element-independent, bit-identical to shipped:87-92).
  for (int vi = tid; vi < vh; vi += kFastBlock) {
    float4 r = rv[vi], wr = wv[vi], o;
    float w0 = wr.x, w1 = wr.y, w2 = wr.z, w3 = wr.w;
    if (gemma) {
      w0 += 1.0f;
      w1 += 1.0f;
      w2 += 1.0f;
      w3 += 1.0f;
    }
    o.x = r.x * inv * w0;
    o.y = r.y * inv * w1;
    o.z = r.z * inv * w2;
    o.w = r.w * inv * w3;
    ov[vi] = o;
  }
}

// Runtime predicate + launch for the decode-fast path. Returns true iff it ran.
// Guard: bf16 in/out/weight/residual, 16-byte-aligned pointers (vectorized loads),
// H%8==0, 1024<=H<=kFastMaxH (H>=1024 scopes to the big residual RMSNorm launches;
// H<=kFastMaxH bounds the f32 square-buffer shared array). The launch uses
// kFastBlock threads for the memory passes, while the reduction reproduces the
// shipped RmsNormRowKernel's kBlock=256 partial + tree order, so the output is
// bit-identical. Out-of-scope shapes keep RmsNormRowKernel.
inline bool TryLaunchRmsNormDecodeFast(cudaStream_t s, Tensor& out, const Tensor& x,
                                       const Tensor& w, const RmsNormArgs& args,
                                       Tensor* residual, unsigned rows, int64_t h) {
  if (!RmsNormDecodeFastFlagIsOn(std::getenv("VT_RMSNORM_DECODE_FAST"))) return false;
  if (!x.IsContiguous() || (residual != nullptr && !residual->IsContiguous())) return false;
  if (out.dtype != DType::kBF16 || x.dtype != DType::kBF16 || w.dtype != DType::kBF16)
    return false;
  if (residual == nullptr || residual->dtype != DType::kBF16) return false;
  if (h % 8 != 0 || h < 1024 || h > kFastMaxH) return false;
  auto aligned16 = [](const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & 0xF) == 0;
  };
  if (!aligned16(out.data) || !aligned16(x.data) || !aligned16(w.data) ||
      !aligned16(residual->data))
    return false;
  RmsNormRowFastKernel<<<rows, kFastBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), x.Ptr<__nv_bfloat16>(),
                                             w.Ptr<__nv_bfloat16>(),
                                             residual->Ptr<__nv_bfloat16>(),
                                             static_cast<int>(h), args.eps, args.gemma);
  return true;
}

// f32-residual sibling of TryLaunchRmsNormDecodeFast. Same VT_RMSNORM_DECODE_FAST contract
// (default ON, '0' = rollback), same bit-identity guarantee (RmsNormRowFastF32Kernel). Scope:
// f32 in/out/weight AND a f32 residual (the add+RMSNorm decode launch — the Laguna NVFP4
// post-attn residual norm), h%4==0, 1024<=h<=kFastMaxH, 16-byte-aligned. Every other case
// keeps RmsNormRowKernel. float4 vectorization needs h%4==0 (vs the bf16 path's h%8==0).
inline bool TryLaunchRmsNormDecodeFastF32(cudaStream_t s, Tensor& out, const Tensor& x,
                                          const Tensor& w, const RmsNormArgs& args,
                                          Tensor* residual, unsigned rows, int64_t h) {
  if (!RmsNormDecodeFastFlagIsOn(std::getenv("VT_RMSNORM_DECODE_FAST"))) return false;
  if (!x.IsContiguous() || (residual != nullptr && !residual->IsContiguous())) return false;
  if (out.dtype != DType::kF32 || x.dtype != DType::kF32 || w.dtype != DType::kF32) return false;
  if (residual == nullptr || residual->dtype != DType::kF32) return false;
  if (h % 4 != 0 || h < 1024 || h > kFastMaxH) return false;
  auto aligned16 = [](const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & 0xF) == 0;
  };
  if (!aligned16(out.data) || !aligned16(x.data) || !aligned16(w.data) ||
      !aligned16(residual->data))
    return false;
  RmsNormRowFastF32Kernel<<<rows, kFastBlock, 0, s>>>(out.Ptr<float>(), x.Ptr<float>(),
                                                      w.Ptr<float>(), residual->Ptr<float>(),
                                                      static_cast<int>(h), args.eps, args.gemma);
  return true;
}

// Dispatch the residual store dtype (f32 or bf16). A bf16 residual mirrors vLLM's
// bf16 model dtype (model_config.dtype=bfloat16): the residual stream is bf16, only
// the variance/normalize accumulation below stays f32. A f32 residual (or none)
// takes the byte-identical previous path.
template <typename Tin, typename Tw, typename Tout>
void LaunchRmsNormRes(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                      const RmsNormArgs& args, Tensor* residual, unsigned rows, int64_t h) {
  if (residual != nullptr && residual->dtype == DType::kBF16) {
    RmsNormRowKernel<Tin, Tw, Tout, __nv_bfloat16><<<rows, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tw>(), residual->Ptr<__nv_bfloat16>(), h,
        args.eps, args.gemma, x.stride[0], residual == nullptr ? h : residual->stride[0]);
  } else if (residual != nullptr && residual->dtype == DType::kF16) {
    RmsNormRowKernel<Tin, Tw, Tout, __half><<<rows, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tw>(), residual->Ptr<__half>(), h,
        args.eps, args.gemma, x.stride[0], residual->stride[0]);
  } else {
    float* res = residual == nullptr ? nullptr : residual->Ptr<float>();
    RmsNormRowKernel<Tin, Tw, Tout, float><<<rows, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tw>(), res, h, args.eps, args.gemma, x.stride[0], residual == nullptr ? h : residual->stride[0]);
  }
}

template <typename Tin, typename Tw>
void LaunchRmsNorm(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return;
  const unsigned rows = static_cast<unsigned>(t);
  // Decode-fast path (VT_RMSNORM_DECODE_FAST, default ON; '0' = rollback): only
  // engages for the bf16 add+RMSNorm decode launches; every other case keeps
  // RmsNormRowKernel. Output is bit-identical to RmsNormRowKernel by construction.
  if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
    if (TryLaunchRmsNormDecodeFast(s, out, x, w, args, residual, rows, h)) {
      Check(cudaGetLastError(), "rmsnorm fast launch");
      return;
    }
  }
  // f32-residual decode-fast path (Laguna NVFP4 post-attn residual norm). Bit-identical
  // to RmsNormRowKernel<float,float,float>; same VT_RMSNORM_DECODE_FAST contract.
  if constexpr (std::is_same_v<Tin, float>) {
    if (TryLaunchRmsNormDecodeFastF32(s, out, x, w, args, residual, rows, h)) {
      Check(cudaGetLastError(), "rmsnorm fast-f32 launch");
      return;
    }
  }
  switch (out.dtype) {
    case DType::kF32:
      LaunchRmsNormRes<Tin, Tw, float>(s, out, x, w, args, residual, rows, h);
      break;
    case DType::kF16:
      LaunchRmsNormRes<Tin, Tw, __half>(s, out, x, w, args, residual, rows, h);
      break;
    case DType::kBF16:
      LaunchRmsNormRes<Tin, Tw, __nv_bfloat16>(s, out, x, w, args, residual, rows, h);
      break;
    default: VT_CHECK(false, "cuda rmsnorm: unsupported out dtype");
  }
  Check(cudaGetLastError(), "rmsnorm launch");
}

// The gamma's dtype is dispatched SEPARATELY from the activation's (#2477). This
// read `VT_CHECK(w.dtype == x.dtype)`, which was an honest mirror of the kernel's
// old `const Tin* w` -- accepting the mismatch without retyping the kernel would
// have read the bf16 `[256]` gamma through a `const float*` and run off its end,
// because at qwen4_exp_qsa_block.cpp:705 `Tin` IS `float` and `head_dim` is 256.
// (`[128]` here previously; that is the indexer's head_dim, from the abandoned
// reading in which :632 was the refusing site.) The
// refusal is therefore NARROWED, not deleted: an unsupported gamma dtype is still
// refused, and now says which one it got.
template <typename Tin>
void DispatchRmsNormWeight(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                           const RmsNormArgs& args, Tensor* residual) {
  switch (w.dtype) {
    case DType::kF32: LaunchRmsNorm<Tin, float>(s, out, x, w, args, residual); break;
    // kF16 (#2503): the seam admits it and the CPU arm serves it, so this arm has
    // to as well. Added rather than recorded as a divergence because vLLM upcasts
    // any float gamma and this platform advertises kF16 (see the `Load(const
    // __half*)` comment above). The `default:` below is what stops a future dtype
    // from being read through the wrong pointer type.
    case DType::kF16: LaunchRmsNorm<Tin, __half>(s, out, x, w, args, residual); break;
    case DType::kBF16: LaunchRmsNorm<Tin, __nv_bfloat16>(s, out, x, w, args, residual); break;
    default:
      VT_CHECK(false, std::string("cuda rmsnorm: unsupported weight dtype "
                                  "(f32/f16/bf16 only), got ") +
                          Name(w.dtype));
  }
}

void RmsNormKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                       const RmsNormArgs& args, Tensor* residual) {
  switch (x.dtype) {
    case DType::kF32:
      DispatchRmsNormWeight<float>(AsStream(q), out, x, w, args, residual);
      break;
    case DType::kF16:
      DispatchRmsNormWeight<__half>(AsStream(q), out, x, w, args, residual);
      break;
    case DType::kBF16:
      DispatchRmsNormWeight<__nv_bfloat16>(AsStream(q), out, x, w, args, residual);
      break;
    default: VT_CHECK(false, "cuda rmsnorm: unsupported input dtype (f32/f16/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// rmsnorm + static fp8 quant, fused: emit the fp8 activation (and optionally the
// bf16 normed activation) directly from the RMSNorm's normalize loop, so the
// standalone QuantFp8Static pass + its bf16 round-trip disappear. Mirror of
// vLLM's Inductor fused_add_rms_norm_static_fp8_quant
// (vllm/compilation/passes/fusion/rms_quant_fusion.py:124). Same reduction as
// RmsNormRowKernel; BIT-IDENTICAL to RmsNorm(bf16)+QuantFp8Static because the
// fp8 is taken from the SAME bf16-rounded value the split path quantizes
// (F32ToFp8Dev(__bfloat162float(bf16(n)) * inv_scale)).
__device__ __forceinline__ uint8_t RmsNormF32ToFp8Dev(float f) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3));
}

// `Tw` is the GAMMA's own dtype and is INDEPENDENT of `Tin` -- the same decoupling
// `RmsNormRowKernel` above got in #2477, applied to its fused twin (#2492). It used
// to be `Tin`, which welded the two and forced `RmsNormQuantFp8KernelCuda` to
// refuse `w.dtype != x.dtype` while its CPU sibling read the gamma through
// `LoadF32(w, j)` (cpu_ops.cpp:1042) and accepted every float dtype. `Load()` is
// overloaded for all three gamma element types above, so the reduction, the
// bf16 intermediate and the fp8 conversion below are unchanged and bit-identical
// for every pairing that already worked.
template <typename Tin, typename Tw, typename Tres>
__global__ void RmsNormQuantFp8RowKernel(uint8_t* out_fp8, __nv_bfloat16* out_bf16, const Tin* x,
                                         const Tw* w, Tres* residual, int64_t h, float eps,
                                         bool gemma, float inv_scale) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * h;
  uint8_t* orow = out_fp8 + row * h;
  __nv_bfloat16* brow = out_bf16 == nullptr ? nullptr : out_bf16 + row * h;
  Tres* rrow = residual == nullptr ? nullptr : residual + row * h;

  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float v = Load(xrow, j);
    if (rrow != nullptr) {
      v = ResRound<Tres>(v + Load(rrow, j));  // new residual stream: f32 add, round to Tres
      Store(rrow, j, v);                       // updated in place (f32 or bf16)
    }
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    const float v = rrow != nullptr ? Load(rrow, j) : Load(xrow, j);
    float wj = Load(w, j);
    if (gemma) wj += 1.0f;
    // bf16-intermediate (matches RmsNorm's bf16 store then QuantFp8Static's bf16 load).
    const __nv_bfloat16 nb = __float2bfloat16(v * inv * wj);
    if (brow != nullptr) brow[j] = nb;
    orow[j] = RmsNormF32ToFp8Dev(__bfloat162float(nb) * inv_scale);
  }
}

template <typename Tin, typename Tw>
void LaunchRmsNormQuantFp8(cudaStream_t s, Tensor& out_fp8, Tensor* out_bf16, const Tensor& x,
                           const Tensor& w, const RmsNormArgs& args, Tensor* residual,
                           float input_scale) {
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return;
  const unsigned rows = static_cast<unsigned>(t);
  const float inv_scale = 1.0f / input_scale;
  __nv_bfloat16* bf16 = out_bf16 == nullptr ? nullptr : out_bf16->Ptr<__nv_bfloat16>();
  if (residual != nullptr && residual->dtype == DType::kBF16) {
    RmsNormQuantFp8RowKernel<Tin, Tw, __nv_bfloat16><<<rows, kBlock, 0, s>>>(
        out_fp8.Ptr<uint8_t>(), bf16, x.Ptr<Tin>(), w.Ptr<Tw>(),
        residual->Ptr<__nv_bfloat16>(), h, args.eps, args.gemma, inv_scale);
  } else {
    float* res = residual == nullptr ? nullptr : residual->Ptr<float>();
    RmsNormQuantFp8RowKernel<Tin, Tw, float><<<rows, kBlock, 0, s>>>(
        out_fp8.Ptr<uint8_t>(), bf16, x.Ptr<Tin>(), w.Ptr<Tw>(), res, h, args.eps, args.gemma,
        inv_scale);
  }
  Check(cudaGetLastError(), "rmsnorm_quant_fp8 launch");
}

// The gamma's dtype is dispatched SEPARATELY from the activation's (#2492),
// exactly as `DispatchRmsNormWeight` does for the unfused op. The refusal is
// NARROWED, not deleted: accepting a mismatch without retyping the kernel would
// have read a bf16 gamma through a `const float*` and run off the end of it.
template <typename Tin>
void DispatchRmsNormQuantFp8Weight(cudaStream_t s, Tensor& out_fp8, Tensor* out_bf16,
                                   const Tensor& x, const Tensor& w, const RmsNormArgs& args,
                                   Tensor* residual, float input_scale) {
  switch (w.dtype) {
    case DType::kF32:
      LaunchRmsNormQuantFp8<Tin, float>(s, out_fp8, out_bf16, x, w, args, residual, input_scale);
      break;
    case DType::kF16:
      LaunchRmsNormQuantFp8<Tin, __half>(s, out_fp8, out_bf16, x, w, args, residual, input_scale);
      break;
    case DType::kBF16:
      LaunchRmsNormQuantFp8<Tin, __nv_bfloat16>(s, out_fp8, out_bf16, x, w, args, residual,
                                                input_scale);
      break;
    default:
      VT_CHECK(false, std::string("cuda rmsnorm_quant_fp8: unsupported weight dtype "
                                  "(f32/f16/bf16 only), got ") +
                          Name(w.dtype));
  }
}

void RmsNormQuantFp8KernelCuda(Queue& q, Tensor& out_fp8, Tensor* out_bf16, const Tensor& x,
                               const Tensor& w, const RmsNormArgs& args, Tensor* residual,
                               float input_scale) {
  switch (x.dtype) {
    case DType::kF32:
      DispatchRmsNormQuantFp8Weight<float>(AsStream(q), out_fp8, out_bf16, x, w, args, residual,
                                           input_scale);
      break;
    case DType::kBF16:
      DispatchRmsNormQuantFp8Weight<__nv_bfloat16>(AsStream(q), out_fp8, out_bf16, x, w, args,
                                                   residual, input_scale);
      break;
    default: VT_CHECK(false, "cuda rmsnorm_quant_fp8: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// silu_and_mul: grid-stride over the T*D output elements.
// Upstream csrc counterpart: csrc/activation_kernels.cu (act_and_mul_kernel<silu>) — align post-MVP.

template <typename Tin, typename Tout>
__global__ void SiluAndMulKernel(Tout* out, const Tin* x, int64_t n, int64_t d) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t i = idx / d;
    const int64_t j = idx - i * d;
    const float gate = Load(x, i * 2 * d + j);
    const float up = Load(x, i * 2 * d + d + j);
    // vLLM a7c23ac96d csrc/libtorch_stable/activation_kernels.cu:165-167
    // returns input scalar_t before compute():44 multiplies by up.
    const float silu = static_cast<float>(static_cast<Tin>(gate / (1.0f + expf(-gate))));
    Store(out, idx, silu * up);
  }
}

template <typename Tin>
void LaunchSiluAndMul(cudaStream_t s, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  const int64_t n = t * d;
  if (n == 0) return;
  switch (out.dtype) {
    case DType::kF32:
      SiluAndMulKernel<Tin, float>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<float>(), x.Ptr<Tin>(), n, d);
      break;
    case DType::kF16:
      SiluAndMulKernel<Tin, __half><<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__half>(), x.Ptr<Tin>(), n, d);
      break;
    case DType::kBF16:
      SiluAndMulKernel<Tin, __nv_bfloat16>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), x.Ptr<Tin>(), n, d);
      break;
    default: VT_CHECK(false, "cuda silu_and_mul: unsupported out dtype");
  }
  Check(cudaGetLastError(), "silu_and_mul launch");
}

void SiluAndMulKernelCuda(Queue& q, Tensor& out, const Tensor& x) {
  switch (x.dtype) {
    case DType::kF16: LaunchSiluAndMul<__half>(AsStream(q), out, x); break;
    case DType::kF32: LaunchSiluAndMul<float>(AsStream(q), out, x); break;
    case DType::kBF16: LaunchSiluAndMul<__nv_bfloat16>(AsStream(q), out, x); break;
    default: VT_CHECK(false, "cuda silu_and_mul: unsupported input dtype (f32/f16/bf16 only)");
  }
}

// ClampedSwiGLU (DeepSeek-V4 activation, activation.py:197-201): gate clamped
// MAX-ONLY, up clamped BOTH sides, then silu(gate)*up at alpha=1, beta=0.
// Same kernel structure as SiluAndMulKernel with per-element clamping.
template <typename Tin, typename Tout>
__global__ void ClampedSwiGLUKernel(Tout* out, const Tin* x, int64_t n, int64_t d,
                                     float limit) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t i = idx / d;
    const int64_t j = idx - i * d;
    const float gate = fminf(Load(x, i * 2 * d + j), limit);
    const float up = fminf(fmaxf(Load(x, i * 2 * d + d + j), -limit), limit);
    const float silu = gate / (1.0f + expf(-gate));
    Store(out, idx, silu * up);
  }
}

template <typename Tin>
void LaunchClampedSwiGLU(cudaStream_t s, Tensor& out, const Tensor& gate_up, float limit) {
  const int64_t t = gate_up.shape[0], d = gate_up.shape[1] / 2;
  const int64_t n = t * d;
  if (n == 0) return;
  switch (out.dtype) {
    case DType::kF32:
      ClampedSwiGLUKernel<Tin, float>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<float>(), gate_up.Ptr<Tin>(), n, d, limit);
      break;
    case DType::kBF16:
      ClampedSwiGLUKernel<Tin, __nv_bfloat16>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), gate_up.Ptr<Tin>(), n, d,
                                         limit);
      break;
    default: VT_CHECK(false, "cuda clamped_swiglu: unsupported out dtype");
  }
  Check(cudaGetLastError(), "clamped_swiglu launch");
}

void ClampedSwiGLUKernelCuda(Queue& q, Tensor& out, const Tensor& gate_up, float limit) {
  switch (gate_up.dtype) {
    case DType::kF32: LaunchClampedSwiGLU<float>(AsStream(q), out, gate_up, limit); break;
    case DType::kBF16:
      LaunchClampedSwiGLU<__nv_bfloat16>(AsStream(q), out, gate_up, limit);
      break;
    default:
      VT_CHECK(false, "cuda clamped_swiglu: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// gelu_and_mul (Gemma GeGLU): out = gelu_tanh(gate) * up. gelu_tanh is the
// exact `gelu_pytorch_tanh` / F.gelu(approximate="tanh") — computed in f32 then
// stored, mirroring vLLM GeluAndMul(approximate="tanh").

template <typename Tin, typename Tout>
__global__ void GeluAndMulKernel(Tout* out, const Tin* x, int64_t n, int64_t d) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t i = idx / d;
    const int64_t j = idx - i * d;
    const float g = Load(x, i * 2 * d + j);
    const float up = Load(x, i * 2 * d + d + j);
    // 0.5*g*(1 + tanh( sqrt(2/pi) * (g + 0.044715*g^3) )); sqrt(2/pi)=0.7978845608...
    const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    const float gelu = 0.5f * g * (1.0f + tanhf(inner));
    Store(out, idx, gelu * up);
  }
}

template <typename Tin>
void LaunchGeluAndMul(cudaStream_t s, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  const int64_t n = t * d;
  if (n == 0) return;
  switch (out.dtype) {
    case DType::kF32:
      GeluAndMulKernel<Tin, float>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<float>(), x.Ptr<Tin>(), n, d);
      break;
    case DType::kBF16:
      GeluAndMulKernel<Tin, __nv_bfloat16>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), x.Ptr<Tin>(), n, d);
      break;
    default: VT_CHECK(false, "cuda gelu_and_mul: unsupported out dtype");
  }
  Check(cudaGetLastError(), "gelu_and_mul launch");
}

void GeluAndMulKernelCuda(Queue& q, Tensor& out, const Tensor& x) {
  switch (x.dtype) {
    case DType::kF32: LaunchGeluAndMul<float>(AsStream(q), out, x); break;
    case DType::kBF16: LaunchGeluAndMul<__nv_bfloat16>(AsStream(q), out, x); break;
    default: VT_CHECK(false, "cuda gelu_and_mul: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// mul_scalar: out[i] = x[i] * scalar (f32 compute, out-dtype store). The Gemma
// embedding normalizer `embed * sqrt(hidden_size)`.

template <typename Tin, typename Tout>
__global__ void MulScalarKernel(Tout* out, const Tin* x, int64_t n, float scalar) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step)
    Store(out, idx, Load(x, idx) * scalar);
}

template <typename Tin>
void LaunchMulScalar(cudaStream_t s, Tensor& out, const Tensor& x, float scalar) {
  const int64_t n = x.Numel();
  if (n == 0) return;
  switch (out.dtype) {
    case DType::kF32:
      MulScalarKernel<Tin, float>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<float>(), x.Ptr<Tin>(), n, scalar);
      break;
    case DType::kBF16:
      MulScalarKernel<Tin, __nv_bfloat16>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), x.Ptr<Tin>(), n, scalar);
      break;
    default: VT_CHECK(false, "cuda mul_scalar: unsupported out dtype");
  }
  Check(cudaGetLastError(), "mul_scalar launch");
}

void MulScalarKernelCuda(Queue& q, Tensor& out, const Tensor& x, double scalar) {
  const float s = static_cast<float>(scalar);
  switch (x.dtype) {
    case DType::kF32: LaunchMulScalar<float>(AsStream(q), out, x, s); break;
    case DType::kBF16: LaunchMulScalar<__nv_bfloat16>(AsStream(q), out, x, s); break;
    default: VT_CHECK(false, "cuda mul_scalar: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// soft_cap: out[i] = cap * tanh(x[i] / cap) (f32 compute, out-dtype store). The
// Gemma-2 final logit soft-cap (gemma2.py:344-345). Mirrors torch
// logits.div_(cap).tanh_().mul_(cap).

template <typename Tin, typename Tout>
__global__ void SoftCapKernel(Tout* out, const Tin* x, int64_t n, float cap) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step)
    Store(out, idx, cap * tanhf(static_cast<float>(Load(x, idx)) / cap));
}

template <typename Tin>
void LaunchSoftCap(cudaStream_t s, Tensor& out, const Tensor& x, float cap) {
  const int64_t n = x.Numel();
  if (n == 0) return;
  switch (out.dtype) {
    case DType::kF32:
      SoftCapKernel<Tin, float>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<float>(), x.Ptr<Tin>(), n, cap);
      break;
    case DType::kBF16:
      SoftCapKernel<Tin, __nv_bfloat16>
          <<<GridFor(n), kBlock, 0, s>>>(out.Ptr<__nv_bfloat16>(), x.Ptr<Tin>(), n, cap);
      break;
    default: VT_CHECK(false, "cuda soft_cap: unsupported out dtype");
  }
  Check(cudaGetLastError(), "soft_cap launch");
}

void SoftCapKernelCuda(Queue& q, Tensor& out, const Tensor& x, double cap) {
  const float c = static_cast<float>(cap);
  switch (x.dtype) {
    case DType::kF32: LaunchSoftCap<float>(AsStream(q), out, x, c); break;
    case DType::kBF16: LaunchSoftCap<__nv_bfloat16>(AsStream(q), out, x, c); break;
    default: VT_CHECK(false, "cuda soft_cap: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// embedding: grid-stride gather. Ids live on the device, so bounds are checked
// in-kernel: bad ids are clamped for the gather (no OOB read) and the first bad
// id is recorded in a device-side flag via atomicCAS. The host wrapper
// synchronizes the stream, reads the flag back, and throws — CUDA Embedding is
// synchronizing for now (M0.6 decision, see ops.h; revisit for full async in
// M0.9/M2).
// No direct csrc counterpart (upstream uses torch embedding); keep vt-native.

// The error record moved to vt/cuda/cuda_embedding_quant.h when the BLOCK arm
// landed: the block kernels live in cuda_quant_dot.cu (the only TU that may
// include the device codebooks) and write this same latch, and two hand-mirrored
// layouts would report a wrong id without ever failing to compile.
using EmbeddingErr = EmbeddingQuantErr;

template <typename Tin, typename Tout, typename Tid>
__global__ void EmbeddingKernel(Tout* out, const Tin* table, const Tid* ids, int64_t n,
                                int64_t h, int64_t v, EmbeddingErr* err) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t i = idx / h;
    const int64_t j = idx - i * h;
    int64_t id = static_cast<int64_t>(ids[i]);
    if (id < 0 || id >= v) {
      if (atomicCAS(&err->status, 0, 1) == 0) err->id = static_cast<long long>(id);
      id = id < 0 ? 0 : v - 1;  // clamp: keep the gather in-bounds
    }
    Store(out, idx, Load(table, id * h + j));
  }
}

template <typename Tin, typename Tout>
cudaError_t LaunchEmbedding(cudaStream_t s, Tensor& out, const Tensor& table,
                            const Tensor& ids, EmbeddingErr* err) {
  const int64_t t = ids.shape[0], h = table.shape[1], v = table.shape[0];
  const int64_t n = t * h;
  if (ids.dtype == DType::kI32) {
    EmbeddingKernel<Tin, Tout, int32_t><<<GridFor(n), kBlock, 0, s>>>(
        out.Ptr<Tout>(), table.Ptr<Tin>(), ids.Ptr<int32_t>(), n, h, v, err);
  } else {
    EmbeddingKernel<Tin, Tout, int64_t><<<GridFor(n), kBlock, 0, s>>>(
        out.Ptr<Tout>(), table.Ptr<Tin>(), ids.Ptr<int64_t>(), n, h, v, err);
  }
  return cudaGetLastError();
}

template <typename Tin>
cudaError_t LaunchEmbeddingIn(cudaStream_t s, Tensor& out, const Tensor& table,
                              const Tensor& ids, EmbeddingErr* err) {
  if (out.dtype == DType::kF32) return LaunchEmbedding<Tin, float>(s, out, table, ids, err);
  if (out.dtype == DType::kF16) return LaunchEmbedding<Tin, __half>(s, out, table, ids, err);
  return LaunchEmbedding<Tin, __nv_bfloat16>(s, out, table, ids, err);
}

// Out-of-range reporting WITHOUT a per-call barrier.
//
// The original shape was cudaMalloc + kernel + D2H + cudaStreamSynchronize +
// cudaFree on EVERY call. All three of those driver calls SYNCHRONIZE the device,
// and embedding runs once per engine step at the very front of the forward, so
// the sequence is a hard barrier between consecutive steps. It costs almost
// nothing while the engine is serialized anyway (measured 23.7 us/call), and it
// costs the ENTIRE overlap the moment the engine stops being serialized — which
// is exactly what ENG-ASYNC-SCHED is for. See
// .agents/specs/async-discrete-device-combine.md W4e.
//
// The replacement is a small RING of persistent slots. Each slot owns a device
// flag, a PINNED host mirror and an event; a call takes the next slot, resets and
// launches into it, and records the event — no sync, no allocation. The check is
// then DEFERRED: before reusing a slot, its event is consumed. If the event has
// already completed the check is free (cudaEventQuery); only a genuinely
// in-flight slot blocks, which bounds the ring's memory without ever making the
// common path wait.
//
// Semantics change in exactly one way, deliberately: a bad id is reported up to
// kSlots calls later than it was, instead of on the offending call. It is still
// LOUD (the same exception, the same message, on the same queue) and it is still
// memory-safe on the offending call itself, because the kernel clamps the gather
// — an out-of-range id has never produced an out-of-bounds read, only a wrong
// row. The one report the ring cannot deliver is an error in the final embedding
// of a process that then exits without another embedding; that residue is
// covered by the token-exactness gates, which compare the produced ids.
struct EmbeddingErrSlot {
  EmbeddingErr* dev = nullptr;   // device-side flag the kernel atomically sets
  EmbeddingErr* host = nullptr;  // pinned mirror the async D2H lands in
  cudaEvent_t done = nullptr;    // completion of that D2H
  bool pending = false;          // event recorded and not yet consumed
  int64_t vocab = 0;             // table rows of the call that armed the slot
};

// One embedding call per engine step per queue, so a handful of slots covers any
// realistic overlap depth; the ring only has to outlive the in-flight window.
constexpr int kEmbeddingErrSlots = 4;

struct EmbeddingErrRing {
  EmbeddingErrSlot slots[kEmbeddingErrSlots];
  int next = 0;
};

// One ring per process: the forward runs on a single device and a single host
// thread (the same assumption DevicePool and the resident-weight caches make).
EmbeddingErrRing& ErrRing() {
  static EmbeddingErrRing ring;
  return ring;
}

// Consume a slot's outstanding result, blocking only if `force`. Returns the
// error to report, if any, WITHOUT throwing: the caller decides when to throw so
// a partially-armed slot is never left behind.
bool ConsumeEmbeddingErr(EmbeddingErrSlot& slot, bool force, EmbeddingErr* err_out,
                         int64_t* vocab_out) {
  if (!slot.pending) return false;
  if (!force) {
    const cudaError_t q = cudaEventQuery(slot.done);
    if (q == cudaErrorNotReady) return false;  // still in flight: check it later
    if (q != cudaSuccess) Check(q, "embedding flag event query");
  } else {
    Check(cudaEventSynchronize(slot.done), "embedding flag event sync");
  }
  slot.pending = false;
  if (slot.host->status == 0) return false;
  *err_out = *slot.host;
  *vocab_out = slot.vocab;
  return true;
}

void EmbeddingKernelCuda(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids) {
  // Validate dtypes before touching the ring so a throw cannot leave a slot armed.
  //
  // BLOCK-QUANTIZED tables are admitted here (KGATHER). This kernel asserted
  // f32/bf16 until then, which is why `DeviceQuantGatherSupported` refused every
  // non-CPU device and a kept table expanded at load instead -- 26.822 GiB of
  // IQ4_NL becoming 95.368 GiB of bf16 for the shipped n-gram table, which fits
  // nothing in this fleet. The decode is one row per gathered id, mirroring the
  // CPU arm (cpu_ops.cpp EmbeddingKernel) and `ggml_compute_forward_get_rows_q`.
  const bool block_table = IsBlockQuant(table.dtype);
  VT_CHECK(block_table || table.dtype == DType::kF32 || table.dtype == DType::kBF16 || table.dtype == DType::kF16,
           "cuda embedding: unsupported table dtype (f32/bf16 or a block quant)");
  // A block dtype with NO device decoder must say so by name. It cannot be
  // reached from the loader, whose `DeviceQuantGatherSupported` gate is pinned
  // to this same list, but a direct `vt::Embedding` caller can reach it and the
  // alternative to a named refusal is a silently untouched output buffer.
  VT_CHECK(!block_table || EmbeddingQuantSupported(table.dtype),
           std::string("cuda embedding: no device row decoder for block dtype ") +
               Name(table.dtype));
  VT_CHECK(out.dtype == DType::kF32 || out.dtype == DType::kBF16 || out.dtype == DType::kF16,
           "cuda embedding: unsupported out dtype");
  const int64_t n = ids.shape[0] * table.shape[1];
  if (n == 0) return;
  // v == 0 with nonempty ids can never gather anything valid, and the in-kernel
  // clamp (v - 1) would go out of bounds — throw loudly before launching.
  VT_CHECK(table.shape[0] > 0, "cuda embedding: empty table (vocab 0) with nonempty ids");
  cudaStream_t s = AsStream(q);

  EmbeddingErrRing& ring = ErrRing();
  EmbeddingErrSlot& slot = ring.slots[ring.next];
  ring.next = (ring.next + 1) % kEmbeddingErrSlots;

  if (slot.dev == nullptr) {
    // First use of this slot: the ONLY allocation this path ever makes. Pinned
    // host memory so the D2H is a real async copy rather than a staged one.
    Check(cudaMalloc(&slot.dev, sizeof(EmbeddingErr)), "cudaMalloc embedding flag");
    Check(cudaHostAlloc(reinterpret_cast<void**>(&slot.host), sizeof(EmbeddingErr),
                        cudaHostAllocDefault),
          "cudaHostAlloc embedding flag mirror");
    Check(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming),
          "cudaEventCreate embedding flag");
    slot.host->status = 0;
  }

  // Reusing this slot means its previous result must be consumed first. Force the
  // wait: the slot is about to be overwritten, so skipping the check here would
  // DROP a report rather than defer it. With kSlots slots in the ring this only
  // blocks when more than kSlots embeddings are genuinely in flight.
  EmbeddingErr prev{};
  int64_t prev_vocab = 0;
  const bool had_prev = ConsumeEmbeddingErr(slot, /*force=*/true, &prev, &prev_vocab);

  cudaError_t st = cudaMemsetAsync(slot.dev, 0, sizeof(EmbeddingErr), s);
  if (st == cudaSuccess) {
    st = block_table ? LaunchEmbeddingQuant(s, out, table, ids, slot.dev)
         : table.dtype == DType::kF32
             ? LaunchEmbeddingIn<float>(s, out, table, ids, slot.dev)
             : table.dtype == DType::kF16
             ? LaunchEmbeddingIn<__half>(s, out, table, ids, slot.dev)
             : LaunchEmbeddingIn<__nv_bfloat16>(s, out, table, ids, slot.dev);
  }
  if (st == cudaSuccess) {
    st = cudaMemcpyAsync(slot.host, slot.dev, sizeof(EmbeddingErr), cudaMemcpyDeviceToHost, s);
  }
  if (st == cudaSuccess) st = cudaEventRecord(slot.done, s);
  if (st == cudaSuccess) {
    slot.pending = true;
    slot.vocab = table.shape[0];
  }

  // A launch/copy failure is reported before a deferred out-of-range id: it is
  // the more immediate fault, and it may be why the older flag never arrived.
  Check(st, "embedding");
  if (had_prev) {
    throw std::runtime_error("vt cuda: embedding: id " + std::to_string(prev.id) +
                             " out of range [0, " + std::to_string(prev_vocab) + ")");
  }

  // Opportunistically drain every OTHER slot whose copy has already landed, so a
  // bad id surfaces at the next call rather than only when its slot comes round
  // again. Free: cudaEventQuery on a completed event does not block.
  for (int i = 0; i < kEmbeddingErrSlots; ++i) {
    EmbeddingErrSlot& other = ring.slots[i];
    if (&other == &slot) continue;
    EmbeddingErr err{};
    int64_t vocab = 0;
    if (ConsumeEmbeddingErr(other, /*force=*/false, &err, &vocab)) {
      throw std::runtime_error("vt cuda: embedding: id " + std::to_string(err.id) +
                               " out of range [0, " + std::to_string(vocab) + ")");
    }
  }
}

// ---------------------------------------------------------------------------
// rope_neox: grid-stride over (token, head, rotation pair) across q and k.
// Angle math in double (pow/cos/sin) to match the CPU reference numerics.
// Upstream csrc counterpart: csrc/pos_encoding_kernels.cu (rotary_embedding_kernel) — align post-MVP.

// Llama-3 rope frequency rescale (vLLM Llama3RotaryEmbedding._compute_inv_freq,
// rotary_embedding/llama3_rope.py:33-54). `freq` is the base inv_freq
// (base^(-2i/rot)); when scaling_factor <= 0 this is a no-op (plain RoPE). The
// piecewise low/high wavelength interpolation matches vLLM's torch.where ladder.
__device__ __host__ inline double Llama3ScaleFreq(double freq, double scaling_factor,
                                                  double low_ff, double high_ff,
                                                  double orig_max) {
  if (!(scaling_factor > 0.0)) return freq;
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const double low_freq_wavelen = orig_max / low_ff;
  const double high_freq_wavelen = orig_max / high_ff;
  const double wave_len = kTwoPi / freq;
  double smooth = 0.0;
  if (low_ff != high_ff)
    smooth = (orig_max / wave_len - low_ff) / (high_ff - low_ff);
  if (wave_len < high_freq_wavelen) return freq;                  // high-freq: keep
  if (wave_len > low_freq_wavelen) return freq / scaling_factor;  // low-freq: scale
  return (1.0 - smooth) * freq / scaling_factor + smooth * freq;  // mid: interpolate
}

template <typename T, typename Tid>
__global__ void RopeNeoxKernel(T* qs, T* ks, const Tid* pos, int64_t hq, int64_t hk,
                               int64_t d, int64_t half, int rot, double base,
                               double l3_sf, double l3_lo, double l3_hi, double l3_omax,
                               int64_t n) {
  const int64_t heads = hq + hk;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t pair = idx % half;
    const int64_t head = (idx / half) % heads;
    const int64_t tok = idx / (half * heads);
    T* ptr;
    int64_t off;
    if (head < hq) {
      ptr = qs;
      off = (tok * hq + head) * d;
    } else {
      ptr = ks;
      off = (tok * hk + (head - hq)) * d;
    }
    const int64_t p = static_cast<int64_t>(pos[tok]);
    double freq = pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
    freq = Llama3ScaleFreq(freq, l3_sf, l3_lo, l3_hi, l3_omax);
    const double angle = static_cast<double>(p) * freq;
    float c = static_cast<float>(cos(angle));
    float sn = static_cast<float>(sin(angle));
    if constexpr (std::is_same_v<T, __half>) { c = ResRound<__half>(c); sn = ResRound<__half>(sn); }
    const float x = Load(ptr, off + pair);
    const float y = Load(ptr, off + pair + half);
    Store(ptr, off + pair, x * c - y * sn);
    Store(ptr, off + pair + half, x * sn + y * c);
  }
}

template <typename T>
void LaunchRope(cudaStream_t s, Tensor& qs, Tensor& ks, const Tensor& pos,
                const RopeArgs& args) {
  const int64_t t = qs.shape[0], hq = qs.shape[1], hk = ks.shape[1], d = qs.shape[2];
  const int64_t half = args.rotary_dim / 2;
  const int64_t n = t * (hq + hk) * half;
  if (n == 0) return;
  const double base = static_cast<double>(args.base);
  const double l3_sf = static_cast<double>(args.llama3_scaling_factor);
  const double l3_lo = static_cast<double>(args.llama3_low_freq_factor);
  const double l3_hi = static_cast<double>(args.llama3_high_freq_factor);
  const double l3_omax = static_cast<double>(args.llama3_orig_max_position);
  if (pos.dtype == DType::kI32) {
    RopeNeoxKernel<T, int32_t><<<GridFor(n), kBlock, 0, s>>>(
        qs.Ptr<T>(), ks.Ptr<T>(), pos.Ptr<int32_t>(), hq, hk, d, half, args.rotary_dim, base,
        l3_sf, l3_lo, l3_hi, l3_omax, n);
  } else {
    RopeNeoxKernel<T, int64_t><<<GridFor(n), kBlock, 0, s>>>(
        qs.Ptr<T>(), ks.Ptr<T>(), pos.Ptr<int64_t>(), hq, hk, d, half, args.rotary_dim, base,
        l3_sf, l3_lo, l3_hi, l3_omax, n);
  }
  Check(cudaGetLastError(), "rope_neox launch");
}

void RopeNeoxKernelCuda(Queue& q, Tensor& qs, Tensor& ks, const Tensor& pos,
                        const RopeArgs& args) {
  switch (qs.dtype) {
    case DType::kF16: LaunchRope<__half>(AsStream(q), qs, ks, pos, args); break;
    case DType::kF32: LaunchRope<float>(AsStream(q), qs, ks, pos, args); break;
    case DType::kBF16: LaunchRope<__nv_bfloat16>(AsStream(q), qs, ks, pos, args); break;
    default: VT_CHECK(false, "cuda rope: unsupported dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// Supplied-cache RoPE. Ported from pinned vLLM base.py:160-252,
// csrc/libtorch_stable/pos_encoding_kernels.cu:8-200, and the 3-axis selection
// in mrope.py:14-187,263-375 @ e24d1b24fe96. This hot kernel performs only
// cache lookup + rotation; YaRN formula construction happens once on the host.

__device__ inline int MropeAxisForPair(int64_t pair, int section_t,
                                       int section_h, int section_w,
                                       bool interleaved) {
  if (interleaved) {
    if (pair % 3 == 1 && pair <= 3LL * section_h) return 1;
    if (pair % 3 == 2 && pair <= 3LL * section_w) return 2;
    return 0;
  }
  if (pair < section_t) return 0;
  if (pair < static_cast<int64_t>(section_t) + section_h) return 1;
  return 2;
}

template <typename T, typename Tid>
__global__ void RopeFromCacheKernel(
    T* qs, T* ks, const Tid* positions, const T* cache, int64_t cache_rows,
    int64_t tokens, int64_t hq, int64_t hk, int64_t q_tok_stride,
    int64_t q_head_stride, int64_t k_tok_stride, int64_t k_head_stride,
    int rotary_dim, int64_t half, bool is_neox_style, bool is_mrope,
    int section_t, int section_h, int section_w, bool mrope_interleaved,
    int64_t n) {
  const int64_t heads = hq + hk;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
       idx < n; idx += step) {
    const int64_t pair = idx % half;
    const int64_t head = (idx / half) % heads;
    const int64_t token = idx / (half * heads);
    const int axis =
        is_mrope ? MropeAxisForPair(pair, section_t, section_h, section_w,
                                   mrope_interleaved)
                 : 0;
    const int64_t position_offset =
        is_mrope ? static_cast<int64_t>(axis) * tokens + token : token;
    const int64_t position = static_cast<int64_t>(positions[position_offset]);
    // The public contract, like upstream's custom op, requires positions to be
    // valid cache rows. Avoid an out-of-bounds read if a broken caller violates
    // it; CPU validation reports the exact error in reference tests.
    if (position < 0 || position >= cache_rows) continue;
    const int64_t cache_offset = position * rotary_dim;
    const float c = Load(cache, cache_offset + pair);
    const float sn = Load(cache, cache_offset + half + pair);

    // MLA campaign W6: STRIDE-driven addressing. DeepSeek's decoupled RoPE
    // rotates only the trailing rope slice of each query head and its k_pe is a
    // column block of the fused kv_a projection — both are strided views. For a
    // contiguous tensor these offsets are integer-identical to the previous
    // (token * heads + head) * head_dim formula, so existing callers are
    // bit-identical by construction.
    T* states = head < hq ? qs : ks;
    const int64_t local_head = head < hq ? head : head - hq;
    const int64_t row = head < hq
                            ? token * q_tok_stride + local_head * q_head_stride
                            : token * k_tok_stride + local_head * k_head_stride;
    const int64_t first = is_neox_style ? pair : pair * 2;
    const int64_t second = is_neox_style ? pair + half : pair * 2 + 1;
    const float x = Load(states, row + first);
    const float y = Load(states, row + second);
    Store(states, row + first, x * c - y * sn);
    Store(states, row + second, x * sn + y * c);
  }
}

template <typename T, typename Tid>
void LaunchRopeFromCacheTyped(cudaStream_t stream, Tensor& qs, Tensor* ks,
                              const Tensor& positions, const Tensor& cache,
                              const RopeArgs& args) {
  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t hk = ks == nullptr ? 0 : ks->shape[1];
  const int64_t half = args.rotary_dim / 2;
  const int64_t n = tokens * (hq + hk) * half;
  if (n == 0) return;
  RopeFromCacheKernel<T, Tid><<<GridFor(n), kBlock, 0, stream>>>(
      qs.Ptr<T>(), ks == nullptr ? nullptr : ks->Ptr<T>(),
      positions.Ptr<Tid>(), cache.Ptr<T>(), cache.shape[0], tokens, hq, hk,
      qs.stride[0], qs.stride[1], ks == nullptr ? 0 : ks->stride[0],
      ks == nullptr ? 0 : ks->stride[1], args.rotary_dim, half,
      args.is_neox_style, positions.rank == 2, args.mrope_section[0],
      args.mrope_section[1], args.mrope_section[2], args.mrope_interleaved, n);
}

template <typename T>
void LaunchRopeFromCache(cudaStream_t stream, Tensor& qs, Tensor* ks,
                         const Tensor& positions, const Tensor& cache,
                         const RopeArgs& args) {
  if (positions.dtype == DType::kI32) {
    LaunchRopeFromCacheTyped<T, int32_t>(stream, qs, ks, positions, cache,
                                         args);
  } else {
    LaunchRopeFromCacheTyped<T, int64_t>(stream, qs, ks, positions, cache,
                                         args);
  }
  Check(cudaGetLastError(), "rope_from_cache launch");
}

void RopeFromCacheKernelCuda(Queue& q, Tensor& qs, Tensor* ks,
                             const Tensor& positions, const Tensor& cache,
                             const RopeArgs& args) {
  switch (qs.dtype) {
    case DType::kF16:
      LaunchRopeFromCache<__half>(AsStream(q), qs, ks, positions, cache, args);
      break;
    case DType::kF32:
      LaunchRopeFromCache<float>(AsStream(q), qs, ks, positions, cache, args);
      break;
    case DType::kBF16:
      LaunchRopeFromCache<__nv_bfloat16>(AsStream(q), qs, ks, positions,
                                         cache, args);
      break;
    default:
      VT_CHECK(false,
               "cuda rope_from_cache: unsupported dtype (f32/bf16 only)");
  }
}

// Fused MLA norm-rope (kFusedNormRope): one block per token. The RMS reduction
// over the latent slice [0,off) reuses RmsNormRowKernel's EXACT tree reduce +
// scale (cuda_ops.cu:62-93), and the decoupled-pe rotation over [off,off+rot)
// reuses RopeFromCacheKernel's EXACT cache read + rotation (cuda_ops.cu:896-943).
// The two halves address DISJOINT dims, so the fused output is BIT-IDENTICAL to
// {RmsNorm(x[:,:off]) ; RopeFromCache(x[:,off:])} — the CPU FusedNormRopeKernel
// runs exactly that composite as the golden.
template <typename T, typename Tid>
__global__ void FusedNormRopeKernel(T* latent_out, T* pe_out, const T* x, const T* w,
                                    const Tid* positions, const T* cache, int64_t cache_rows,
                                    int64_t off, int rot, int64_t half, int64_t x_row_stride,
                                    int64_t lat_row_stride, int64_t pe_row_stride,
                                    bool is_neox_style, bool gemma, float eps) {
  const int64_t row = blockIdx.x;
  const T* xrow = x + row * x_row_stride;
  T* lrow = latent_out + row * lat_row_stride;
  T* prow = pe_out + row * pe_row_stride;

  // --- latent RMSNorm over [0, off) — identical to RmsNormRowKernel. ----------
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < off; j += kBlock) {
    const float v = Load(xrow, j);
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(off) + eps);
  for (int64_t j = threadIdx.x; j < off; j += kBlock) {
    float wj = Load(w, j);
    if (gemma) wj += 1.0f;
    Store(lrow, j, Load(xrow, j) * inv * wj);
  }

  // --- decoupled-pe RopeFromCache over [off, off+rot) — identical to
  //     RopeFromCacheKernel (single vector, base rope, positions rank-1). ------
  const int64_t position = static_cast<int64_t>(positions[row]);
  if (position < 0 || position >= cache_rows) return;
  const int64_t cache_offset = position * rot;
  for (int64_t pair = threadIdx.x; pair < half; pair += kBlock) {
    const float c = Load(cache, cache_offset + pair);
    const float sn = Load(cache, cache_offset + half + pair);
    const int64_t first = is_neox_style ? pair : pair * 2;
    const int64_t second = is_neox_style ? pair + half : pair * 2 + 1;
    const float xr = Load(xrow, off + first);
    const float yr = Load(xrow, off + second);
    Store(prow, first, xr * c - yr * sn);
    Store(prow, second, xr * sn + yr * c);
  }
}

template <typename T>
void LaunchFusedNormRope(cudaStream_t stream, Tensor& latent_out, Tensor& pe_out,
                         const Tensor& x, const Tensor& w, const Tensor& positions,
                         const Tensor& cache, const RmsNormArgs& norm_args,
                         const RopeArgs& rope_args) {
  const int64_t t = x.shape[0];
  const int64_t off = w.shape[0];
  const int rot = rope_args.rotary_dim;
  const int64_t half = rot / 2;
  if (t == 0) return;
  const unsigned rows = static_cast<unsigned>(t);
  if (positions.dtype == DType::kI32) {
    FusedNormRopeKernel<T, int32_t><<<rows, kBlock, 0, stream>>>(
        latent_out.Ptr<T>(), pe_out.Ptr<T>(), x.Ptr<T>(), w.Ptr<T>(),
        positions.Ptr<int32_t>(), cache.Ptr<T>(), cache.shape[0], off, rot, half,
        x.stride[0], latent_out.stride[0], pe_out.stride[0], rope_args.is_neox_style,
        norm_args.gemma, norm_args.eps);
  } else {
    FusedNormRopeKernel<T, int64_t><<<rows, kBlock, 0, stream>>>(
        latent_out.Ptr<T>(), pe_out.Ptr<T>(), x.Ptr<T>(), w.Ptr<T>(),
        positions.Ptr<int64_t>(), cache.Ptr<T>(), cache.shape[0], off, rot, half,
        x.stride[0], latent_out.stride[0], pe_out.stride[0], rope_args.is_neox_style,
        norm_args.gemma, norm_args.eps);
  }
  Check(cudaGetLastError(), "fused_norm_rope launch");
}

void FusedNormRopeKernelCuda(Queue& q, Tensor& latent_out, Tensor& pe_out, const Tensor& x,
                             const Tensor& w, const Tensor& positions, const Tensor& cache,
                             const RmsNormArgs& norm_args, const RopeArgs& rope_args) {
  switch (x.dtype) {
    case DType::kF32:
      LaunchFusedNormRope<float>(AsStream(q), latent_out, pe_out, x, w, positions, cache,
                                 norm_args, rope_args);
      break;
    case DType::kBF16:
      LaunchFusedNormRope<__nv_bfloat16>(AsStream(q), latent_out, pe_out, x, w, positions, cache,
                                         norm_args, rope_args);
      break;
    default:
      VT_CHECK(false, "cuda fused_norm_rope: unsupported dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// rope_cos_sin_cache: precompute the batch's cos|sin ONCE per step (grid-stride
// over (token, pair)) so the fused preamble below does zero in-kernel
// transcendentals. Angle math in DOUBLE + f32 cast — bit-for-bit RopeNeoxKernel's
// c/sn. cos_sin[t, i]=cos, cos_sin[t, half+i]=sin. Mirrors vLLM's cos_sin_cache
// (RotaryEmbedding._compute_cos_sin_cache; read by fla fused_qk_norm_rope.py:95).

template <typename Tid>
__global__ void RopeCosSinCacheKernel(float* cos_sin, const Tid* pos, int64_t t, int rot,
                                      int64_t half, double base, double l3_sf, double l3_lo,
                                      double l3_hi, double l3_omax, float linear) {
  const int64_t n = t * half;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t pair = idx % half;
    const int64_t tok = idx / half;
    const int64_t p = static_cast<int64_t>(pos[tok]);
    if (linear > 0.f) {
      const float exponent = static_cast<float>(2 * pair) / static_cast<float>(rot);
      const float inv = 1.f / powf(static_cast<float>(base), exponent);
      const float angle = (static_cast<float>(p) / linear) * inv;
      cos_sin[tok * rot + pair] = cosf(angle);
      cos_sin[tok * rot + half + pair] = sinf(angle);
      continue;
    }
    double freq = pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
    freq = Llama3ScaleFreq(freq, l3_sf, l3_lo, l3_hi, l3_omax);
    const double angle = static_cast<double>(p) * freq;
    cos_sin[tok * rot + pair] = static_cast<float>(cos(angle));
    cos_sin[tok * rot + half + pair] = static_cast<float>(sin(angle));
  }
}

void RopeCosSinCacheKernelCuda(Queue& q, Tensor& cos_sin, const Tensor& pos, const RopeArgs& args) {
  const int64_t t = cos_sin.shape[0];
  const int64_t half = args.rotary_dim / 2;
  const int64_t n = t * half;
  if (n == 0) return;
  cudaStream_t s = AsStream(q);
  const double base = static_cast<double>(args.base);
  const double l3_sf = static_cast<double>(args.llama3_scaling_factor);
  const double l3_lo = static_cast<double>(args.llama3_low_freq_factor);
  const double l3_hi = static_cast<double>(args.llama3_high_freq_factor);
  const double l3_omax = static_cast<double>(args.llama3_orig_max_position);
  if (pos.dtype == DType::kI32) {
    RopeCosSinCacheKernel<int32_t><<<GridFor(n), kBlock, 0, s>>>(
        cos_sin.Ptr<float>(), pos.Ptr<int32_t>(), t, args.rotary_dim, half, base,
        l3_sf, l3_lo, l3_hi, l3_omax, args.linear_scaling_factor);
  } else {
    RopeCosSinCacheKernel<int64_t><<<GridFor(n), kBlock, 0, s>>>(
        cos_sin.Ptr<float>(), pos.Ptr<int64_t>(), t, args.rotary_dim, half, base,
        l3_sf, l3_lo, l3_hi, l3_omax, args.linear_scaling_factor);
  }
  Check(cudaGetLastError(), "rope_cos_sin_cache launch");
}

// ---------------------------------------------------------------------------
// attn_qk_norm_rope_gate: the fused full-attention preamble — one launch replacing
// AttnGateSplit + RmsNorm(q) + RmsNorm(k) + RopeNeox. Grid (T, Hq+Hkv); one block
// per (token, head) does the shared-mem gemma-RMSNorm tree reduction over Dh (reuses
// RmsNormRowKernel's math), then partial NeoX RoPE reading the precomputed cos_sin
// cache (no per-element pow/cos/sin), then passes the gate half through (q heads).
// Structure mirrors GdnPostConvKernel (cuda_gdn.cu). Bit-for-bit equal to the four
// composed ops for f32 out; templated on Tsrc (qgate/kf) and Tout (q/k/gate).
// Mirrors vLLM's fused_qk_rmsnorm_rope (fla fused_qk_norm_rope.py:95-102).

// gemma-RMSNorm one element: (v*inv)*(gemma ? w+1 : w) — matches RmsNormRowKernel.
__device__ inline float GemmaNormElem(float v, float inv, float w, bool gemma) {
  float wj = w;
  if (gemma) wj += 1.0f;
  return v * inv * wj;
}

// Tqk = q_out/k_out store dtype, Tgate = gate_out store dtype. All math stays
// f32; a bf16 Tqk store is the RN round of the exact f32 value, so (Tqk=bf16,
// Tgate=f32) is bit-identical to the f32 path followed by CastBf16 on q/k — the
// FA-2 prefill combo (bf16 q feeds FA-2, bf16 k feeds the bf16 KV-cache write,
// gate stays f32 because sigmoid(gate) must see the un-rounded f32 value).
//
// NOTE (2026-07-18, CLAIM-35B-FA2-FLIP-1): a "round normed q/k to bf16 BEFORE
// RoPE" tighten (mirror of vLLM fused_qk_norm_rope.py:67) was implemented and
// op-level VALIDATED bit-identical to the unfused bf16 path, but it flipped the
// 27B's known tok6 whitespace near-tie AWAY from the vLLM oracle (want_prod →
// want_emu) in COMBINATION with our other sub-ULP op diffs (a compensating-error
// interaction — the RMSNorm-saga lesson). The 35B passes 315/315 with OR without
// the round, so the round is NOT shipped: the untightened preamble keeps BOTH
// the 27B (235/235) and the 35B (315/315) token-exact to their graphed oracles.
template <typename Tsrc, typename Tqk, typename Tgate>
__global__ void AttnQkNormRopeGateKernel(Tqk* q_out, Tqk* k_out, Tgate* gate_out,
                                         const Tsrc* qgate, const Tsrc* kf, const float* q_norm,
                                         const float* k_norm, const float* cos_sin, int64_t hq,
                                         int64_t hkv, int64_t dh,
                                         int64_t qgate_row_stride,
                                         int64_t kf_row_stride, int rot,
                                         int64_t half, float eps, bool gemma) {
  const int64_t tok = blockIdx.x;
  const int64_t head = blockIdx.y;  // [0, hq+hkv)
  const bool is_q = head < hq;

  const Tsrc* src;
  const float* w;
  Tqk* out;
  int64_t gate_base = 0;  // only meaningful for q heads
  if (is_q) {
    const int64_t qrow = tok * qgate_row_stride + head * 2 * dh;
    src = qgate + qrow;       // q half [0,dh)
    gate_base = qrow + dh;    // gate half [dh,2dh)
    w = q_norm;
    out = q_out + (tok * hq + head) * dh;
  } else {
    const int64_t hk = head - hq;
    src = kf + tok * kf_row_stride + hk * dh;
    w = k_norm;
    out = k_out + (tok * hkv + hk) * dh;
  }

  // ---- gemma-RMSNorm reduction over Dh (f32 variance) ----
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < dh; j += kBlock) {
    const float v = Load(src, j);
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(dh) + eps);

  // ---- partial NeoX RoPE + write (recompute the paired normed elems; no shared
  // round-trip). Matches RopeNeoxKernel: out[i]=x*c - y*sn, out[i+half]=x*sn + y*c
  // with x=normed[i], y=normed[i+half]; dims [rot,Dh) normed but unrotated. ----
  const float* cs = cos_sin + tok * rot;
  for (int64_t j = threadIdx.x; j < dh; j += kBlock) {
    if (j < half) {
      const float ni = GemmaNormElem(Load(src, j), inv, w[j], gemma);
      const float nih = GemmaNormElem(Load(src, j + half), inv, w[j + half], gemma);
      Store(out, j, ni * cs[j] - nih * cs[half + j]);
    } else if (j < rot) {
      const int64_t i = j - half;
      const float ni = GemmaNormElem(Load(src, i), inv, w[i], gemma);
      const float nih = GemmaNormElem(Load(src, i + half), inv, w[i + half], gemma);
      Store(out, j, ni * cs[half + i] + nih * cs[i]);
    } else {
      Store(out, j, GemmaNormElem(Load(src, j), inv, w[j], gemma));
    }
  }

  // ---- gate passthrough (q heads only): the raw gate half, no norm/rope ----
  if (is_q) {
    Tgate* go = gate_out + (tok * hq + head) * dh;
    for (int64_t j = threadIdx.x; j < dh; j += kBlock) Store(go, j, Load(qgate, gate_base + j));
  }
}

template <typename Tsrc, typename Tqk, typename Tgate>
void LaunchAttnPreamble(cudaStream_t s, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                        const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                        const Tensor& k_norm, const Tensor& cos_sin, const RmsNormArgs& na,
                        const RopeArgs& ra) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  const int64_t half = ra.rotary_dim / 2;
  if (t == 0) return;
  dim3 grid(static_cast<unsigned>(t), static_cast<unsigned>(hq + hkv));
  AttnQkNormRopeGateKernel<Tsrc, Tqk, Tgate><<<grid, kBlock, 0, s>>>(
      q_out.Ptr<Tqk>(), k_out.Ptr<Tqk>(), gate_out.Ptr<Tgate>(), qgate.Ptr<Tsrc>(),
      kf.Ptr<Tsrc>(), q_norm.Ptr<float>(), k_norm.Ptr<float>(), cos_sin.Ptr<float>(), hq, hkv, dh,
      qgate.stride[0], kf.stride[0], ra.rotary_dim, half, na.eps, na.gemma);
  Check(cudaGetLastError(), "attn_qk_norm_rope_gate launch");
}

template <typename Tsrc>
void LaunchAttnPreambleOut(cudaStream_t s, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                           const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                           const Tensor& k_norm, const Tensor& cos_sin, const RmsNormArgs& na,
                           const RopeArgs& ra) {
  // (q/k out, gate out) combos: (f32,f32) — the default token-exact path;
  // (bf16,bf16) — all-bf16; (bf16,f32) — the FA-2 prefill combo (bf16 q/k for
  // FA-2 + the bf16 KV-cache write, f32 gate for the sigmoid). Validation in
  // ops.cpp admits exactly these.
  switch (q_out.dtype) {
    case DType::kF32:
      LaunchAttnPreamble<Tsrc, float, float>(s, q_out, k_out, gate_out, qgate, kf, q_norm, k_norm,
                                             cos_sin, na, ra);
      break;
    case DType::kBF16:
      if (gate_out.dtype == DType::kF32) {
        LaunchAttnPreamble<Tsrc, __nv_bfloat16, float>(s, q_out, k_out, gate_out, qgate, kf,
                                                       q_norm, k_norm, cos_sin, na, ra);
      } else {
        LaunchAttnPreamble<Tsrc, __nv_bfloat16, __nv_bfloat16>(s, q_out, k_out, gate_out, qgate,
                                                               kf, q_norm, k_norm, cos_sin, na, ra);
      }
      break;
    default: VT_CHECK(false, "cuda attn_qk_norm_rope_gate: unsupported out dtype");
  }
}

void AttnQkNormRopeGateKernelCuda(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                                  const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                                  const Tensor& k_norm, const Tensor& cos_sin,
                                  const RmsNormArgs& na, const RopeArgs& ra) {
  cudaStream_t s = AsStream(q);
  switch (qgate.dtype) {
    case DType::kF32:
      LaunchAttnPreambleOut<float>(s, q_out, k_out, gate_out, qgate, kf, q_norm, k_norm, cos_sin,
                                   na, ra);
      break;
    case DType::kBF16:
      LaunchAttnPreambleOut<__nv_bfloat16>(s, q_out, k_out, gate_out, qgate, kf, q_norm, k_norm,
                                           cos_sin, na, ra);
      break;
    default: VT_CHECK(false, "cuda attn_qk_norm_rope_gate: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// attention: one block per (query i, q-head h); block threads cooperate over
// the head_dim and stream the keys with an online (flash-style) softmax. The
// online update is algebraically identical to the CPU two-pass reference
// (qwen36-forward-notes.md §5); f32 accumulation. Correctness-grade (M0.9).

template <typename Tin, typename Tout>
__global__ void AttentionKernel(Tout* out, const Tin* query, const Tin* key, const Tin* value,
                                int64_t hq, int64_t hk, int64_t d, int64_t t, float scale,
                                bool causal) {
  const int64_t i = blockIdx.x;  // query position
  const int64_t h = blockIdx.y;  // q-head
  const int64_t g = h / (hq / hk);
  const int64_t jmax = causal ? i : t - 1;
  const int64_t qoff = (i * hq + h) * d;

  extern __shared__ float smem[];
  float* acc = smem;                    // [d] running output accumulator
  float* red = smem + d;                // [blockDim.x] reduction scratch
  __shared__ float s_score, s_m, s_l;   // block-wide score / running max / denom
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) acc[e] = 0.0f;
  if (threadIdx.x == 0) {
    s_m = -CUDART_INF_F;
    s_l = 0.0f;
  }
  __syncthreads();

  for (int64_t j = 0; j <= jmax; ++j) {
    const int64_t koff = (j * hk + g) * d;
    float part = 0.0f;
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      part += Load(query, qoff + e) * Load(key, koff + e);
    red[threadIdx.x] = part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) s_score = red[0] * scale;
    __syncthreads();

    const float s = s_score;
    const float m_new = fmaxf(s_m, s);
    const float corr = expf(s_m - m_new);  // 0 on the first key (s_m == -inf)
    const float p = expf(s - m_new);
    const int64_t voff = (j * hk + g) * d;
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      acc[e] = acc[e] * corr + p * Load(value, voff + e);
    __syncthreads();
    if (threadIdx.x == 0) {
      s_l = s_l * corr + p;
      s_m = m_new;
    }
    __syncthreads();
  }

  const float inv = 1.0f / s_l;
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) Store(out, qoff + e, acc[e] * inv);
}

template <typename Tin>
void LaunchAttention(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& key,
                     const Tensor& value, const AttentionArgs& args) {
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  if (t == 0 || hq == 0 || d == 0) return;
  const dim3 grid(static_cast<unsigned>(t), static_cast<unsigned>(hq));
  const size_t shmem = (static_cast<size_t>(d) + kBlock) * sizeof(float);
  switch (out.dtype) {
    case DType::kF32:
      AttentionKernel<Tin, float><<<grid, kBlock, shmem, s>>>(
          out.Ptr<float>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), hq, hk, d, t,
          args.scale, args.causal);
      break;
    case DType::kBF16:
      AttentionKernel<Tin, __nv_bfloat16><<<grid, kBlock, shmem, s>>>(
          out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), hq, hk,
          d, t, args.scale, args.causal);
      break;
    default: VT_CHECK(false, "cuda attention: unsupported out dtype");
  }
  Check(cudaGetLastError(), "attention launch");
}

void AttentionKernelCuda(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                         const Tensor& value, const AttentionArgs& args) {
  switch (query.dtype) {
    case DType::kF32: LaunchAttention<float>(AsStream(q), out, query, key, value, args); break;
    case DType::kBF16:
      LaunchAttention<__nv_bfloat16>(AsStream(q), out, query, key, value, args);
      break;
    default: VT_CHECK(false, "cuda attention: unsupported input dtype (f32/bf16 only)");
  }
}

// SPEC-DFLASH2 W12 D1 (#2087) — resolve one GLOBAL query row to the KEY span it
// attends over and to its COMBINED intra-block offset. `qcu` is the query cu and
// `cu` the key cu; every kernel below reads BOTH through this one function so a
// query-cu that is honoured in four kernels and forgotten in the fifth is not a
// shape a reviewer has to find by reading five mask derivations (spec R2).
//
// The launcher passes the SAME pointer for both when the caller set no query cu.
// Then `ke - ks == qcu[r+1] - qcu[r]` and `ic == i - qcu[r]`, i.e. exactly the
// `ii` these kernels computed before, so the null case is not a branch — it is
// the same arithmetic with a zero offset.
struct DFlashRowSpan {
  int64_t ks;  // first GLOBAL key row of this query's request
  int64_t ke;  // one past the last
  int64_t ic;  // this query's COMBINED intra-block offset, 0-based within [ks,ke)
};
__device__ __forceinline__ DFlashRowSpan DFlashResolveRow(const int32_t* qcu,
                                                          const int32_t* cu, int num_reqs,
                                                          int64_t i) {
  DFlashRowSpan sp;
  sp.ks = 0;
  sp.ke = 0;
  sp.ic = 0;
  for (int r = 0; r < num_reqs; ++r) {
    if (i >= qcu[r] && i < qcu[r + 1]) {
      sp.ks = cu[r];
      sp.ke = cu[r + 1];
      // The query block is the BOTTOM-RIGHT suffix of the key block.
      sp.ic = (sp.ke - sp.ks) - (qcu[r + 1] - qcu[r]) + (i - qcu[r]);
      break;
    }
  }
  return sp;
}

// ---------------------------------------------------------------------------
// DFlash in-block attention (SPEC-DFLASH D2, DF-DRAFT-MODEL) — the project's FIRST
// non-causal / bidirectional attention CUDA kernel. Mirrors AttentionKernel's
// one-block-per-(query,head) block-reduction online-softmax recurrence EXACTLY
// (same f32 accumulation), generalized to (a) per-request BLOCK bounds so a query
// attends only within its own (1+k) block and (b) a BIDIRECTIONAL (non-causal) or
// causal-within-window mask per DFlashQwen3Attention (_resolve_layer_attention,
// qwen3_dflash.py:86-146). Kept a SEPARATE op so the causal kAttention used by the
// text/audio paths is byte-identical. `cu` is a device copy of cu_seqlens
// (num_reqs+1); each block linearly finds its request (num_reqs is small).
template <typename Tin, typename Tout>
__global__ void DFlashBlockAttentionKernel(Tout* out, const Tin* query, const Tin* key,
                                           const Tin* value, const int32_t* cu,
                                           const int32_t* qcu, int num_reqs, int64_t hq,
                                           int64_t hk, int64_t d, float scale, bool causal,
                                           int64_t window) {
  const int64_t i = blockIdx.x;  // GLOBAL query row
  const int64_t h = blockIdx.y;  // q-head
  const int64_t g = h / (hq / hk);
  // Find the request block owning query row i (small num_reqs; the uniform DFlash
  // case is num_reqs blocks of 1+k). qs/qe are GLOBAL KEY bounds of the block.
  const DFlashRowSpan sp = DFlashResolveRow(qcu, cu, num_reqs, i);
  const int64_t qs = sp.ks, qe = sp.ke;
  const int64_t ii = sp.ic;                        // combined intra-block offset
  // #2784: the ONE shared bound (vt/dflash_attn_mask.h). A NON-CAUSAL layer
  // carrying a window attends SYMMETRICALLY within it; it used to attend over
  // the whole block.
  const vt::DFlashMaskSpan span = vt::DFlashMaskSpanOf(ii, qe - qs, causal, window);
  const int64_t jhi = span.hi;  // last visible intra-block key
  const int64_t jlo = span.lo;
  const int64_t qoff = (i * hq + h) * d;

  extern __shared__ float smem[];
  float* acc = smem;                   // [d] running output accumulator
  float* red = smem + d;               // [blockDim.x] reduction scratch
  __shared__ float s_score, s_m, s_l;  // block-wide score / running max / denom
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) acc[e] = 0.0f;
  if (threadIdx.x == 0) {
    s_m = -CUDART_INF_F;
    s_l = 0.0f;
  }
  __syncthreads();

  for (int64_t jj = jlo; jj <= jhi; ++jj) {
    const int64_t koff = ((qs + jj) * hk + g) * d;
    float part = 0.0f;
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      part += Load(query, qoff + e) * Load(key, koff + e);
    red[threadIdx.x] = part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) s_score = red[0] * scale;
    __syncthreads();

    const float s = s_score;
    const float m_new = fmaxf(s_m, s);
    const float corr = expf(s_m - m_new);  // 0 on the first key (s_m == -inf)
    const float p = expf(s - m_new);
    const int64_t voff = ((qs + jj) * hk + g) * d;
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      acc[e] = acc[e] * corr + p * Load(value, voff + e);
    __syncthreads();
    if (threadIdx.x == 0) {
      s_l = s_l * corr + p;
      s_m = m_new;
    }
    __syncthreads();
  }

  const float inv = 1.0f / s_l;
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) Store(out, qoff + e, acc[e] * inv);
}

// WARP-PER-QUERY fast path for the common head_dim (a multiple of the warp width).
//
// The general kernel above spends one block per query row and walks keys ONE AT A
// TIME, doing a shared-memory tree reduction plus ~11 __syncthreads() PER KEY. At
// seq 3264 that is ~36k barriers per query row, and Q is re-read from global for
// every key — measured ~36 GFLOP/s, which made attention dominate everything at
// realistic sequence lengths.
//
// Here ONE WARP owns a query row: Q and the output accumulator live in REGISTERS
// (d/32 each per lane), the dot product is a warp shuffle reduction, and the online
// softmax runs inside the warp. No __syncthreads at all, and Q is read once.
//
// The arithmetic is deliberately IDENTICAL to the general kernel — same sequential
// key order, same online-softmax recurrence (m/corr/p/l), same f32 accumulation — so
// this is a scheduling change, not a numerics change.
// Q-BLOCKED warp attention: one warp carries kQ queries at once.
//
// *** MEASURED NEGATIVE ON sm_110 AND DELIBERATELY NOT DISPATCHED. Do not wire
// *** this up again on this hardware without re-reading the numbers below.
//
// The idea: the one-query-per-warp kernel re-reads every K and V row for EVERY
// query, so global traffic is O(seq^2 * d) per head -- ~341 TB per step at H3's
// default canvas (seq 15424, 56 heads, 50 layers, d 64) against Thor's 273 GB/s.
// Holding kQ queries per warp loads each K/V row ONCE and reuses it kQ times.
//
// It was dispatched, gated (all five DFlash/attention suites green at kQ 4 and 8,
// with a RED proof that the kernel really produced the output) and MEASURED on the
// real H3 denoise loop, 2 paired reps per arm, order reversed between reps:
//
//   864x480/124f, seq 15424:  per-query 557.80 s   kQ=4 553.14 s (-0.84%)   kQ=8 765.57 s (+37%)
//   512x512/33f,  seq  3264:  per-query  28.04 s   kQ=4  28.98 s (+3.3%)    kQ=8  36.33 s (+30%)
//
// So the premise is WRONG here, and the reason is the same one that killed the
// shared-memory tiling attempt: the K/V re-reads were never coming from HBM. One
// head's K+V at seq 15424 is 15424*64*2*2 B = 3.9 MB against Thor's 32 MB L2, and
// every warp on a given blockIdx.y streams the SAME rows, so the traffic this
// kernel removes was already L2-resident and nearly free. What it does NOT remove
// is the per-key warp-shuffle reduction -- it still performs kQ of them per K row
// loaded, i.e. exactly as many in total as before. It only adds cost: registers go
// 64 -> 108 (kQ 4 -> 8) at kPerLane 2, which cuts occupancy, and the kQ online-
// softmax updates per key form a serial expf dependency chain, which is why kQ=8
// is far worse than kQ=4 rather than twice as good.
//
// Kept as source (it compiles to nothing while uninstantiated) because the
// conclusion is HARDWARE-SPECIFIC: on a part with a small L2 relative to one head's
// K/V the arithmetic could flip. Full disposition in docs/BENCHMARKS.md.
//
// Deliberately uses NO SHARED MEMORY. A previous attempt staged K/V tiles in
// shared and measured 23% SLOWER: 32 KB per block against Thor's 48 KB/block limit
// allowed ~one block per SM and occupancy collapsed. Registers cost nothing here --
// kQ * kPerLane accumulators plus the same for Q, which fits comfortably.
//
// Per-query key ranges are computed up front, so causal, sliding-window and
// multi-request layouts all work: the mask is warp-UNIFORM for each query (j, qs,
// qe and ii are all warp-wide), so the skip costs no divergence.
//
// Arithmetic is unchanged -- same sequential key order, same online-softmax
// recurrence, same f32 accumulation -- so this is a scheduling change.
template <typename Tin, typename Tout, int kPerLane, int kQ>
__global__ void DFlashAttnQBlockKernel(Tout* out, const Tin* query, const Tin* key,
                                       const Tin* value, const int32_t* cu, const int32_t* qcu,
                                       int num_reqs, int64_t rows, int64_t hq, int64_t hk,
                                       int64_t d, float scale, bool causal, int64_t window) {
  const int lane = threadIdx.x & 31;
  const int64_t wid =
      (static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  const int64_t i0 = wid * kQ;
  if (i0 >= rows) return;
  const int64_t h = blockIdx.y;
  const int64_t g = h / (hq / hk);

  float qreg[kQ][kPerLane], acc[kQ][kPerLane], m[kQ], l[kQ];
  int64_t jlo[kQ], jhi[kQ];
  bool act[kQ];
  int64_t lo = rows, hi = 0;

#pragma unroll
  for (int u = 0; u < kQ; ++u) {
    const int64_t i = i0 + u;
    act[u] = (i < rows);
    m[u] = -CUDART_INF_F;
    l[u] = 0.0f;
#pragma unroll
    for (int c = 0; c < kPerLane; ++c) acc[u][c] = 0.0f;
    jlo[u] = 0;
    jhi[u] = -1;
    if (!act[u]) continue;
    const DFlashRowSpan sp = DFlashResolveRow(qcu, cu, num_reqs, i);
    const int64_t qs = sp.ks, qe = sp.ke;
    const int64_t ii = sp.ic;
    // #2784: the ONE shared bound (vt/dflash_attn_mask.h).
    const vt::DFlashMaskSpan span = vt::DFlashMaskSpanOf(ii, qe - qs, causal, window);
    jlo[u] = qs + span.lo;
    jhi[u] = qs + span.hi;
    lo = min(lo, jlo[u]);
    hi = max(hi, jhi[u]);
    const int64_t qoff = (i * hq + h) * d;
#pragma unroll
    for (int c = 0; c < kPerLane; ++c) {
      qreg[u][c] = Load(query, qoff + static_cast<int64_t>(c) * 32 + lane);
    }
  }
  if (hi < lo) return;

  for (int64_t j = lo; j <= hi; ++j) {
    const int64_t koff = (j * hk + g) * d;
    float kreg[kPerLane], vreg[kPerLane];
#pragma unroll
    for (int c = 0; c < kPerLane; ++c) {
      kreg[c] = Load(key, koff + static_cast<int64_t>(c) * 32 + lane);
      vreg[c] = Load(value, koff + static_cast<int64_t>(c) * 32 + lane);
    }
#pragma unroll
    for (int u = 0; u < kQ; ++u) {
      if (!act[u] || j < jlo[u] || j > jhi[u]) continue;  // warp-uniform
      float part = 0.0f;
#pragma unroll
      for (int c = 0; c < kPerLane; ++c) part += qreg[u][c] * kreg[c];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xFFFFFFFFu, part, off);
      const float s = __shfl_sync(0xFFFFFFFFu, part, 0) * scale;
      const float m_new = fmaxf(m[u], s);
      const float corr = expf(m[u] - m_new);
      const float pw = expf(s - m_new);
#pragma unroll
      for (int c = 0; c < kPerLane; ++c) acc[u][c] = acc[u][c] * corr + pw * vreg[c];
      l[u] = l[u] * corr + pw;
      m[u] = m_new;
    }
  }

#pragma unroll
  for (int u = 0; u < kQ; ++u) {
    if (!act[u] || l[u] == 0.0f) continue;
    const int64_t qoff = ((i0 + u) * hq + h) * d;
    const float inv = 1.0f / l[u];
#pragma unroll
    for (int c = 0; c < kPerLane; ++c) {
      Store(out, qoff + static_cast<int64_t>(c) * 32 + lane, acc[u][c] * inv);
    }
  }
}

template <typename Tin, typename Tout, int kPerLane>
__global__ void DFlashBlockAttentionWarpKernel(Tout* out, const Tin* query, const Tin* key,
                                              const Tin* value, const int32_t* cu,
                                              const int32_t* qcu, int num_reqs, int64_t hq,
                                              int64_t hk, int64_t d, float scale, bool causal,
                                              int64_t window) {
  const int lane = threadIdx.x & 31;
  const int64_t warp_id =
      (static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  const int64_t total = gridDim.y;  // unused; kept for symmetry with the grid below
  (void)total;
  const int64_t i = warp_id;        // GLOBAL query row
  const int64_t h = blockIdx.y;     // q-head
  const int64_t rows = qcu[num_reqs];  // QUERY rows: what this grid covers
  if (i >= rows) return;
  const int64_t g = h / (hq / hk);

  const DFlashRowSpan sp = DFlashResolveRow(qcu, cu, num_reqs, i);
  const int64_t qs = sp.ks, qe = sp.ke;
  const int64_t ii = sp.ic;
  // #2784: the ONE shared bound (vt/dflash_attn_mask.h).
  const vt::DFlashMaskSpan span = vt::DFlashMaskSpanOf(ii, qe - qs, causal, window);
  const int64_t jhi = span.hi;
  const int64_t jlo = span.lo;
  const int64_t qoff = (i * hq + h) * d;

  // Q and the accumulator in registers: lane L holds elements L, L+32, L+64, ...
  float qreg[kPerLane], acc[kPerLane];
#pragma unroll
  for (int c = 0; c < kPerLane; ++c) {
    qreg[c] = Load(query, qoff + static_cast<int64_t>(c) * 32 + lane);
    acc[c] = 0.0f;
  }
  float m = -CUDART_INF_F, l = 0.0f;

  for (int64_t jj = jlo; jj <= jhi; ++jj) {
    const int64_t koff = ((qs + jj) * hk + g) * d;
    float part = 0.0f;
#pragma unroll
    for (int c = 0; c < kPerLane; ++c) {
      part += qreg[c] * Load(key, koff + static_cast<int64_t>(c) * 32 + lane);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xFFFFFFFFu, part, off);
    float s = __shfl_sync(0xFFFFFFFFu, part, 0) * scale;

    const float m_new = fmaxf(m, s);
    const float corr = expf(m - m_new);
    const float pw = expf(s - m_new);
    const int64_t voff = ((qs + jj) * hk + g) * d;
#pragma unroll
    for (int c = 0; c < kPerLane; ++c) {
      acc[c] = acc[c] * corr + pw * Load(value, voff + static_cast<int64_t>(c) * 32 + lane);
    }
    l = l * corr + pw;
    m = m_new;
  }

  const float inv = 1.0f / l;
#pragma unroll
  for (int c = 0; c < kPerLane; ++c) {
    Store(out, qoff + static_cast<int64_t>(c) * 32 + lane, acc[c] * inv);
  }
}

// VT_DFLASH_ATTN_WARP=1 falls back to the older per-key warp-reduction kernel, so
// the two forms can be A/B'd on ONE binary (same-binary A/B is the benchmark
// protocol here). Cached once; getenv is not hot-path-safe.
inline bool UseDflashAttnWarpKernel() {
  static const bool on = [] {
    const char* e = std::getenv("VT_DFLASH_ATTN_WARP");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// VT_DFLASH_ATTN_KEYLANE=1 selects the MEASURED-NEGATIVE one-key-per-lane form
// (kept for the same-binary 3-way A/B that recorded the verdict).
inline bool UseDflashAttnKeyLaneKernel() {
  static const bool on = [] {
    const char* e = std::getenv("VT_DFLASH_ATTN_KEYLANE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// ---------------------------------------------------------------------------
// TWO-PHASE, ONE-KEY-PER-LANE attention.
//
// *** MEASURED NEGATIVE ON sm_110: 28.90 s/step against 18.73 s/step for the
// *** per-key warp kernel on the SAME binary (H3 --denoise-only, 512x512/33f,
// *** seq 3224, 3 steps) -- 54% SLOWER. NOT the default; reachable only via
// *** VT_DFLASH_ATTN_KEYLANE=1. It is kept because it is the experiment that
// *** located the real constraint, and the reason is recorded below.
// ***
// *** WHY IT LOST. Giving each lane a WHOLE K row removes the cross-lane
// *** reduction, exactly as intended -- but it also makes every K load
// *** instruction 32-WAY SCATTERED (lane L addresses row base+L, one 32-byte
// *** sector per lane, ~32 sectors per instruction against 2-4 for a coalesced
// *** one). The rows are L1-resident so nothing goes to HBM, yet the load/store
// *** unit still replays every one of those addresses, and the work simply moved
// *** from the shuffle pipe to the memory pipe. The lesson is the one the
// *** SUCCESSOR kernel below is built on: kill the per-key reduction WITHOUT
// *** giving up lane==element coalescing.
//
// WHY IT WAS TRIED. The warp-per-query kernel above is 47x slower than this
// file's own GEMMs
// on the same chip: MEASURED 0.63 TFLOP/s of attention against 30.0 TFLOP/s of
// cuBLASLt GEMM and a ~7.7 TFLOP/s CUDA-core f32 ceiling. Two causes, neither of
// them memory:
//
//   (1) ~32% of its instructions are USEFUL. Per (query,key) pair a lane does
//       d/32 (= 2 at H3's head_dim 64) FMAs and then a FIVE-STAGE
//       __shfl_down_sync tree plus a broadcast to collapse them to one scalar.
//   (2) A SERIAL DEPENDENCY CHAIN. Every key's (m, l, acc) update reads the
//       previous key's, so the warp is LATENCY-bound: ~8% of the CUDA-core
//       ceiling rather than the ~32% its instruction mix would allow.
//
// This kernel removes both. One warp still owns one query row, but keys are
// processed 32 AT A TIME in two phases with a different lane->data mapping in
// each, and NEITHER phase contains a cross-lane reduction of a dot product:
//
//   Phase A -- LANE == KEY. Lane L computes the WHOLE dot product for key
//     base+L by looping all d elements. Q comes from SHARED, where every lane
//     reads the SAME address (a broadcast, free). Each lane owns a complete
//     score, so there is nothing to reduce. K is strided ACROSS lanes but each
//     lane streams its own row contiguously in 4-element vector loads; the 32
//     live rows are ~4-8 KB, i.e. L1-resident, so the strided form costs
//     nothing here.
//   Then ONE warp max + ONE warp sum + ONE online-softmax update per 32 keys.
//     That is the 32x shortening of the dependency chain.
//   Phase B -- LANE == ELEMENT. Lane L owns the CONTIGUOUS element block
//     [L*kPerLane, (L+1)*kPerLane) and loops the 32 probabilities (broadcast
//     from shared), so V loads are COALESCED and vectorized, and again no lane
//     ever reduces against another.
//
// Useful-FMA accounting per 32 keys per warp: d (phase A) + d (phase B) = 2d
// FMA instructions, against ~800 total instructions for the same work before.
//
// MEMORY IS NOT THE BOUND, AND TWO EARLIER ATTEMPTS PROVED IT. Shared-memory K/V
// tiling measured 23% SLOWER (32 KB/block against Thor's 48 KB limit collapsed
// occupancy) and register Q-blocking measured -0.84% at seq 15424 / +3.3% at seq
// 3264 (DFlashAttnQBlockKernel above). One head's K+V at seq 15424 is 3.9 MB
// against a 32 MB L2, so the traffic those attempts removed was already free.
// This kernel therefore keeps its shared footprint DELIBERATELY TINY -- q (d
// floats) + p (32 floats) per warp, ~5 KB for a 8-warp block at d=128 -- so
// occupancy stays where the warp kernel had it.
//
// NUMERICS. Chunked max/sum is mathematically the same softmax but NOT bitwise
// identical to the per-key recurrence (fewer rescalings, pairwise accumulation
// inside a chunk -- if anything better conditioned). Gated on tolerance against
// the CPU reference, not on bit-equality.
//
// MASKS. jlo/jhi are per-query GLOBAL key bounds computed exactly as the warp
// kernel computes them, so non-causal, causal, sliding-window and multi-request
// (cu_seqlens) layouts all work, including a warp whose neighbours live in a
// different document -- each warp is independent, and a partial final chunk is
// handled by the `valid` predicate rather than by a shape guard. (A previous
// long-sequence kernel here was guarded to num_reqs == 1 and so silently never
// ran on H3's packed {0, used, seq_len} TWO-document layout while the suite
// stayed green; there is no such guard in this one.)
template <typename Tin, typename Tout, int kPerLane>
__global__ void DFlashAttnKeyLaneKernel(Tout* out, const Tin* query, const Tin* key,
                                        const Tin* value, const int32_t* cu, const int32_t* qcu,
                                        int num_reqs, int64_t hq, int64_t hk, float scale,
                                        bool causal, int64_t window) {
  constexpr int kD = kPerLane * 32;  // head_dim, compile-time (dispatch guarantees it)
  extern __shared__ float dfa_smem[];

  const int lane = threadIdx.x & 31;
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int warps = static_cast<int>(blockDim.x >> 5);
  float* qsh = dfa_smem + static_cast<size_t>(warp) * (kD + 32);
  float* psh = qsh + kD;

  const int64_t i = static_cast<int64_t>(blockIdx.x) * warps + warp;  // GLOBAL query row
  const int64_t rows = qcu[num_reqs];  // QUERY rows: what this grid covers
  if (i >= rows) return;  // whole-warp exit; this kernel never uses __syncthreads
  const int64_t h = blockIdx.y;
  const int64_t g = h / (hq / hk);

  const DFlashRowSpan sp = DFlashResolveRow(qcu, cu, num_reqs, i);
  const int64_t qs = sp.ks, qe = sp.ke;
  const int64_t ii = sp.ic;
  // #2784: the ONE shared bound (vt/dflash_attn_mask.h).
  const vt::DFlashMaskSpan span = vt::DFlashMaskSpanOf(ii, qe - qs, causal, window);
  const int64_t jhi = qs + span.hi;  // last visible GLOBAL key
  const int64_t jlo = qs + span.lo;

  const int64_t qoff = (i * hq + h) * kD;
  for (int e = lane; e < kD; e += 32) qsh[e] = Load(query, qoff + e);
  __syncwarp();

  float acc[kPerLane];
#pragma unroll
  for (int c = 0; c < kPerLane; ++c) acc[c] = 0.0f;
  float m = -CUDART_INF_F, l = 0.0f;

  const int64_t ebase = static_cast<int64_t>(lane) * kPerLane;  // this lane's element block

  for (int64_t base = jlo; base <= jhi; base += 32) {
    // ---- Phase A: lane == key. One FULL dot product per lane, no reduction. --
    const int64_t j = base + lane;
    const bool valid = (j <= jhi);
    float s = -CUDART_INF_F;
    if (valid) {
      const int64_t koff = (j * hk + g) * kD;
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;  // 4 independent FMA chains
#pragma unroll
      for (int c = 0; c < kD; c += 4) {
        float qv[4], kv[4];
        LoadVec<4>(qsh, static_cast<int64_t>(c), qv);
        LoadVec<4>(key, koff + c, kv);
        a0 += qv[0] * kv[0];
        a1 += qv[1] * kv[1];
        a2 += qv[2] * kv[2];
        a3 += qv[3] * kv[3];
      }
      s = ((a0 + a1) + (a2 + a3)) * scale;
    }

    // ---- ONE online-softmax update per 32 keys (the dependency-chain fix). ---
    float mx = s;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
    const float m_new = fmaxf(m, mx);  // mx is finite: lane 0 of every chunk is valid
    const float corr = expf(m - m_new);
    const float p = valid ? expf(s - m_new) : 0.0f;
    float sum = p;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
    l = l * corr + sum;
    m = m_new;

    psh[lane] = p;
    __syncwarp();

    // ---- Phase B: lane == element. Coalesced V, probabilities broadcast. ----
    const int cnt = static_cast<int>(jhi - base + 1) < 32 ? static_cast<int>(jhi - base + 1) : 32;
    float t0[kPerLane], t1[kPerLane];
#pragma unroll
    for (int c = 0; c < kPerLane; ++c) { t0[c] = 0.0f; t1[c] = 0.0f; }
    int jj = 0;
    for (; jj + 1 < cnt; jj += 2) {
      float v0[kPerLane], v1[kPerLane];
      LoadVec<kPerLane>(value, ((base + jj) * hk + g) * kD + ebase, v0);
      LoadVec<kPerLane>(value, ((base + jj + 1) * hk + g) * kD + ebase, v1);
      const float p0 = psh[jj], p1 = psh[jj + 1];
#pragma unroll
      for (int c = 0; c < kPerLane; ++c) {
        t0[c] += p0 * v0[c];
        t1[c] += p1 * v1[c];
      }
    }
    if (jj < cnt) {
      float v0[kPerLane];
      LoadVec<kPerLane>(value, ((base + jj) * hk + g) * kD + ebase, v0);
      const float p0 = psh[jj];
#pragma unroll
      for (int c = 0; c < kPerLane; ++c) t0[c] += p0 * v0[c];
    }
#pragma unroll
    for (int c = 0; c < kPerLane; ++c) acc[c] = acc[c] * corr + (t0[c] + t1[c]);
    __syncwarp();  // psh is rewritten by the next chunk
  }

  const float inv = 1.0f / l;
#pragma unroll
  for (int c = 0; c < kPerLane; ++c) Store(out, qoff + ebase + c, acc[c] * inv);
}

// The per-lane ELEMENT PARTITION used by the chunked attention kernel. Any
// partition of the head_dim across the 32 lanes computes the same dot product,
// so the only thing that matters is which one loads FASTEST:
//
//   kPerLane 2 or 4 -> CONTIGUOUS blocks ([L*n, (L+1)*n)). One vector load per
//     lane covers the whole block, the warp reads 32*n consecutive elements, and
//     the access is perfectly coalesced with n-times FEWER instructions.
//   otherwise (1, 3) -> STRIDED (elements L, L+32, L+64...). A contiguous block
//     of 3 would make each warp instruction read at a 3-element stride, spanning
//     ~3x the sectors it needs; strided scalar loads stay perfectly coalesced.
//     head_dim 96 -- MiniMax-H3's production shape, 5376/56 -- lands here.
//
// Q, K, V and the output all use the SAME partition, so it never escapes the
// kernel.
template <int N, typename T>
__device__ inline void LoadLane(const T* p, int64_t row, int lane, float* o) {
  if (N == 2 || N == 4) {
    LoadVec<N>(p, row + static_cast<int64_t>(lane) * N, o);
  } else {
#pragma unroll
    for (int c = 0; c < N; ++c) o[c] = Load(p, row + static_cast<int64_t>(c) * 32 + lane);
  }
}
template <int N, typename T>
__device__ inline void StoreLane(T* p, int64_t row, int lane, const float* v) {
  if (N == 2 || N == 4) {
#pragma unroll
    for (int c = 0; c < N; ++c) Store(p, row + static_cast<int64_t>(lane) * N + c, v[c]);
  } else {
#pragma unroll
    for (int c = 0; c < N; ++c) Store(p, row + static_cast<int64_t>(c) * 32 + lane, v[c]);
  }
}

// ---------------------------------------------------------------------------
// CHUNKED attention with a warp REDUCE-SCATTER (the D15 default).
//
// This is the kernel that actually beats the per-key warp form. It keeps that
// kernel's COALESCED lane==element loads -- which is what killed the key-lane
// variant above -- and attacks the two things the diagnosis actually named: the
// per-key cross-lane REDUCTION and the per-key SERIAL DEPENDENCY CHAIN.
//
// Per warp = one query row; keys are processed 32 AT A TIME:
//
//   1. For each of the 32 keys in the chunk, every lane computes its PARTIAL of
//      the dot product from the kPerLane elements it owns -- a plain coalesced
//      vector load, exactly as before, but the partial is KEPT IN A REGISTER
//      instead of being reduced immediately. After the chunk each lane holds a
//      32-entry partial vector p[], indexed only by compile-time constants so it
//      stays in registers.
//   2. ONE butterfly REDUCE-SCATTER (5 rounds, 16+8+4+2+1 = 31 shuffles) turns
//      that 32x32 partial matrix into "lane L holds the complete score for key
//      base+L". The per-key form needs 5 shuffles + a broadcast for EVERY key,
//      i.e. 192 shuffles per 32 keys; this needs 31. Each round exchanges only
//      the half of the live range the partner is responsible for, so the array
//      indices stay compile-time (a lane-dependent SELECT, never a
//      lane-dependent INDEX -- a dynamic index into p[] would spill it to local
//      memory and lose everything).
//   3. ONE online-softmax update per 32 keys instead of 32: one warp max, one
//      warp sum, TWO expf calls per chunk against 64. That is the 32x shortening
//      of the (m, l, acc) dependency chain, which is why the old kernel sat at
//      ~8% of the CUDA-core ceiling rather than the ~32% its instruction mix
//      allowed -- it was latency-bound, not throughput-bound.
//   4. The 32 probabilities go through shared (32 floats per warp, a broadcast
//      read) and V is accumulated with the same coalesced lane==element mapping,
//      into TWO alternating accumulator sets so the FMA chain has ILP.
//
// The element mapping is whichever of contiguous/strided LOADS FASTEST for the
// head_dim (see LoadLane above): contiguous vector loads at head_dim 64/128,
// strided scalar loads at 96 -- H3's production shape -- where a contiguous
// 3-element block would span ~3x the sectors it needs.
//
// MEMORY IS NOT THE BOUND -- THREE MEASUREMENTS SAY SO. Shared-memory K/V
// tiling: 23% SLOWER (32 KB/block against Thor's 48 KB limit collapsed
// occupancy). Register Q-blocking: -0.84% at seq 15424, +3.3% at seq 3264
// (DFlashAttnQBlockKernel above). One-key-per-lane with Q in shared
// (DFlashAttnKeyLaneKernel above): 54% SLOWER, because giving each lane a whole
// K row makes every load instruction 32-way scattered -- it removes the
// reduction but pays for it in the load/store pipe. One head's K+V at seq 15424
// is 3.9 MB against a 32 MB L2, so K/V were never coming from HBM in the first
// place. This kernel therefore changes NOTHING about where K and V come from,
// and its shared footprint is 32 floats per warp.
//
// NUMERICS. Chunked max/sum is the same softmax but not bitwise identical to the
// per-key recurrence (far fewer rescalings, pairwise accumulation within a
// chunk -- if anything better conditioned). Gated on tolerance against the CPU
// reference.
//
// MASKS. jlo/jhi are the per-query GLOBAL key bounds, computed exactly as the
// per-key kernel computes them, so non-causal, causal, sliding-window and
// multi-request (cu_seqlens) layouts all work, including warps whose neighbours
// live in a different document. Full chunks and the partial tail chunk run the
// SAME body under a compile-time kFull flag: the tail clamps its row index to
// stay in bounds and zeroes the out-of-range probabilities, so there is no
// shape guard anywhere. (A previous long-sequence kernel here was guarded to
// num_reqs == 1 and therefore silently never ran on H3's packed
// {0, used, seq_len} TWO-document layout while the suite stayed green.)
template <typename Tin, int kPerLane, bool kFull>
__device__ inline void DFlashAttnChunk(const Tin* key, const Tin* value, int64_t base, int cnt,
                                       int64_t hk, int64_t g, int lane, const float* qreg,
                                       float scale, float* psh, float& m, float& l, float* acc) {
  constexpr int kD = kPerLane * 32;

  // (1) Per-key partials, coalesced, kept in registers.
  float p[32];
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    const int ti = kFull ? t : (t < cnt ? t : cnt - 1);  // clamp keeps loads in bounds
    float kv[kPerLane];
    LoadLane<kPerLane>(key, ((base + ti) * hk + g) * kD, lane, kv);
    float d0 = 0.0f;
#pragma unroll
    for (int c = 0; c < kPerLane; ++c) d0 += qreg[c] * kv[c];
    p[t] = (kFull || t < cnt) ? d0 : 0.0f;
  }

  // (2) Butterfly reduce-scatter: 31 shuffles total; lane L ends with key L.
#pragma unroll
  for (int half = 16; half >= 1; half >>= 1) {
    const bool hi = (lane & half) != 0;
#pragma unroll
    for (int t = 0; t < half; ++t) {
      const float send = hi ? p[t] : p[t + half];
      const float keep = hi ? p[t + half] : p[t];
      p[t] = keep + __shfl_xor_sync(0xFFFFFFFFu, send, half);
    }
  }

  // (3) ONE online-softmax update for the whole chunk.
  float s = p[0] * scale;
  const bool live = kFull || (lane < cnt);
  if (!live) s = -CUDART_INF_F;
  float mx = s;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
  const float m_new = fmaxf(m, mx);  // mx is finite: lane 0 of every chunk is live
  const float corr = expf(m - m_new);
  const float pv = live ? expf(s - m_new) : 0.0f;
  float sum = pv;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
  l = l * corr + sum;
  m = m_new;

  psh[lane] = pv;
  __syncwarp();

  // (4) V accumulation, same coalesced mapping, two interleaved FMA chains.
  float t0[kPerLane], t1[kPerLane];
#pragma unroll
  for (int c = 0; c < kPerLane; ++c) { t0[c] = 0.0f; t1[c] = 0.0f; }
#pragma unroll
  for (int t = 0; t < 32; t += 2) {
    const int i0 = kFull ? t : (t < cnt ? t : cnt - 1);
    const int i1 = kFull ? (t + 1) : (t + 1 < cnt ? t + 1 : cnt - 1);
    float v0[kPerLane], v1[kPerLane];
    LoadLane<kPerLane>(value, ((base + i0) * hk + g) * kD, lane, v0);
    LoadLane<kPerLane>(value, ((base + i1) * hk + g) * kD, lane, v1);
    const float p0 = psh[t], p1 = psh[t + 1];  // 0 beyond cnt, so the clamp is harmless
#pragma unroll
    for (int c = 0; c < kPerLane; ++c) {
      t0[c] += p0 * v0[c];
      t1[c] += p1 * v1[c];
    }
  }
#pragma unroll
  for (int c = 0; c < kPerLane; ++c) acc[c] = acc[c] * corr + (t0[c] + t1[c]);
  __syncwarp();  // psh is rewritten by the next chunk
}

template <typename Tin, typename Tout, int kPerLane>
__global__ void DFlashAttnChunkKernel(Tout* out, const Tin* query, const Tin* key,
                                      const Tin* value, const int32_t* cu, const int32_t* qcu,
                                      int num_reqs, int64_t hq, int64_t hk, float scale,
                                      bool causal, int64_t window) {
  constexpr int kD = kPerLane * 32;  // head_dim, compile-time (dispatch guarantees it)
  extern __shared__ float dfa_smem[];

  const int lane = threadIdx.x & 31;
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int warps = static_cast<int>(blockDim.x >> 5);
  float* psh = dfa_smem + static_cast<size_t>(warp) * 32;

  const int64_t i = static_cast<int64_t>(blockIdx.x) * warps + warp;  // GLOBAL query row
  const int64_t rows = qcu[num_reqs];  // QUERY rows: what this grid covers
  if (i >= rows) return;  // whole-warp exit; this kernel never uses __syncthreads
  const int64_t h = blockIdx.y;
  const int64_t g = h / (hq / hk);

  const DFlashRowSpan sp = DFlashResolveRow(qcu, cu, num_reqs, i);
  const int64_t qs = sp.ks, qe = sp.ke;
  const int64_t ii = sp.ic;
  // #2784: the ONE shared bound (vt/dflash_attn_mask.h).
  const vt::DFlashMaskSpan span = vt::DFlashMaskSpanOf(ii, qe - qs, causal, window);
  const int64_t jhi = qs + span.hi;  // last visible GLOBAL key
  const int64_t jlo = qs + span.lo;

  const int64_t qoff = (i * hq + h) * kD;
  float qreg[kPerLane], acc[kPerLane];
  LoadLane<kPerLane>(query, qoff, lane, qreg);
#pragma unroll
  for (int c = 0; c < kPerLane; ++c) acc[c] = 0.0f;
  float m = -CUDART_INF_F, l = 0.0f;

  const int64_t nkeys = jhi - jlo + 1;
  const int64_t tail = jlo + (nkeys & ~static_cast<int64_t>(31));  // first partial-chunk base
  for (int64_t base = jlo; base < tail; base += 32) {
    DFlashAttnChunk<Tin, kPerLane, true>(key, value, base, 32, hk, g, lane, qreg, scale, psh, m,
                                         l, acc);
  }
  if (tail <= jhi) {
    DFlashAttnChunk<Tin, kPerLane, false>(key, value, tail, static_cast<int>(jhi - tail + 1), hk,
                                          g, lane, qreg, scale, psh, m, l, acc);
  }

  const float inv = 1.0f / l;
  float res[kPerLane];
#pragma unroll
  for (int c = 0; c < kPerLane; ++c) res[c] = acc[c] * inv;
  StoreLane<kPerLane>(out, qoff, lane, res);
}

// ---------------------------------------------------------------------------
// TENSOR-CORE (bf16 mma.sync) FlashAttention — the D16 default for bf16 streams.
//
// Every kernel above computes attention on the CUDA CORES: at head_dim 128 the
// chunked reduce-scatter form measured 1.13 TFLOP/s against a ~7.7 TFLOP/s f32
// CUDA-core ceiling and 30.0 TFLOP/s of cuBLASLt GEMM on the SAME chip. That gap
// is not a scheduling problem any more — it is the wrong MATH UNIT. This kernel
// moves both GEMMs of attention onto the bf16 tensor cores with
// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, which is Ampere-and-later
// (it assembles, launches and accumulates on sm_110; the FP4 `kind::mxf4nvf4`
// restriction recorded for this box applies to the block-scaled FP4 MMA ONLY).
//
// SHAPE. One WARP owns 16 query rows and the FULL head_dim of output; a block is
// kMmaWarps warps, so 64 query rows share one staged K/V tile of kMmaKeys keys.
//
//   S[16q x 32k] = Q[16q x d] · Kᵀ[d x 32k]   — d/16 k-steps x 4 n-tiles
//   O[16q x d]  += P[16q x 32k] · V[32k x d]  — 2 k-steps x d/8 n-tiles
//
// SHARED MEMORY IS A FIRST-CLASS CONSTRAINT here, because the previous
// shared-memory attempt in this file died of occupancy (32 KB/block, 23% SLOWER).
// So ONLY K and V are staged: 2 * 32 * (d+8) * 2 B = 17.4 KB/block at d=128,
// 4.6 KB at d=64. S, P and the whole O accumulator stay in REGISTERS. That is
// possible because the m16n8k16 C fragment maps EXACTLY onto the next MMA's A
// fragment: the accumulator lane layout (rows groupID / groupID+8, cols
// threadInGroup*2 + {0,1}) is the same partition the A operand wants, so P needs
// zero shuffles and zero shared round-trip — the reason FlashAttention-2 uses raw
// mma.sync rather than the nvcuda::wmma API (whose fragments are opaque and must
// go through memory; see PagedFlashWmmaKernel in cuda_paged_attn.cu, which pays
// exactly that cost).
//
// FRAGMENT LAYOUT (PTX ISA "Matrix Fragments for mma.m16n8k16"), gid = lane>>2,
// tig = lane&3:
//   A (16x16 row): reg0={row gid, col tig*2, +1}   reg1={row gid+8, same cols}
//                  reg2={row gid, col tig*2+8, +9} reg3={row gid+8, same cols}
//   B (16x8 col):  reg0={col gid, row tig*2, +1}   reg1={col gid, row tig*2+8, +9}
//   C (16x8 f32):  c0,c1={row gid,   col tig*2, +1}
//                  c2,c3={row gid+8, col tig*2, +1}
// So every lane owns exactly TWO query rows for its whole lifetime, and the online
// softmax state (m, l) is two registers reduced across the 4 lanes of a QUAD
// (__shfl_xor 1 and 2) — no block barrier, no shared scratch.
//
// SHARED BANKS. K/V rows are padded to d+8 elements. d is a multiple of 16, so the
// row stride in 4-byte banks is (d+8)/2 = an ODD multiple of 4, and the 8 distinct
// rows a warp touches land on banks 4*gid (mod 32) — all distinct — while tig adds
// 0..3. The K fragment load is therefore CONFLICT-FREE. The V fragment reads a
// COLUMN (b0/b1 are consecutive KEYS, not consecutive elements), so it costs four
// 16-bit shared loads with a 2-way conflict; that is the one structural tax of
// keeping V row-major, and it is paid against 2x the MMA work it feeds.
//
// MASKS. Identical bookkeeping to the chunked kernel, evaluated per (row, key)
// on the f32 S fragment: non-causal, causal, sliding-window and multi-request
// (cu_seqlens) all work, including a warp whose 16 rows straddle a document
// boundary. The block walks the UNION of its 64 rows' key ranges so the staged
// tile is block-uniform (every barrier is reached by every warp), and rows mask
// out what they cannot see. Query rows past qcu[num_reqs] clamp their row index
// and skip the store. Since W12 D1 (#2087) the query grid and the key span are
// counted SEPARATELY here (`qrows` and `krows`), because a caller may hand this
// op fewer query rows than key rows.
//
// NUMERICS. Q, K and V are already bf16 in the production stream, so QKᵀ loses
// NOTHING (bf16 in, f32 accumulate — the CUDA-core path converts the same bf16 to
// f32 and accumulates in f32). The PV GEMM must round P to bf16, exactly as
// FlashAttention does; that is a real ~2^-9 relative error on the probabilities
// and it is why this path is gated at a bf16 tolerance rather than the f32 path's
// 2e-5. f32 inputs therefore stay on DFlashAttnChunkKernel, whose f32 tolerance
// this could not hold.
[[maybe_unused]] __device__ __forceinline__ void MmaBf16M16N8K16(float* c, const unsigned* a, const unsigned* b) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  (void)c;
  (void)a;
  (void)b;
#endif
}

constexpr int kMmaWarps = 4;   // warps per block (4 * 16 = 64 query rows per block)
constexpr int kMmaQ = 16;      // query rows per warp = the MMA's M
constexpr int kMmaKeys = 32;   // keys per staged K/V tile = 4 n-tiles of the QKᵀ MMA
constexpr int kMmaPad = 8;     // shared row padding, in elements (see SHARED BANKS)

// kDT = head_dim / 16 (the QKᵀ k-step count). head_dim = 16 * kDT is a compile-time
// constant so every fragment index below is a compile-time index into a register
// array — a dynamic index would spill the accumulator to local memory.
template <typename Tout, int kDT>
__global__ __launch_bounds__(kMmaWarps * 32) void DFlashAttnMmaKernel(
    Tout* out, const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const int32_t* cu, const int32_t* qcu, int num_reqs,
    int64_t hq, int64_t hk, float scale, bool causal, int64_t window,
    int64_t tiles_per_req) {
#if __CUDA_ARCH__ >= 800
  constexpr int kD = kDT * 16;        // head_dim
  constexpr int kNT = kD / 8;         // P·V n-tiles (8 output columns each)
  constexpr int kStride = kD + kMmaPad;

  extern __shared__ uint4 dfa_mma_smem[];  // uint4 => guaranteed 16-byte aligned
  __nv_bfloat16* ksh = reinterpret_cast<__nv_bfloat16*>(dfa_mma_smem);
  __nv_bfloat16* vsh = ksh + kMmaKeys * kStride;
  const unsigned short* vsh_u = reinterpret_cast<const unsigned short*>(vsh);

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int gid = lane >> 2;  // 0..7 : which of the 8 row-pairs this lane owns
  const int tig = lane & 3;   // 0..3 : which column pair

  // D1 (#2087): TWO row counts now. `qrows` bounds the query grid and every
  // query-side clamp; `krows` bounds the KEY staging clamp. They are equal
  // whenever the caller set no query cu, which is why one name sufficed before.
  const int64_t qrows = qcu[num_reqs];
  const int64_t krows = cu[num_reqs];
  const int64_t h = blockIdx.y;
  const int64_t g = h / (hq / hk);
  // #2202: ONE REQUEST PER BLOCK. The grid used to tile the query axis globally,
  // so at c=8 (Tq = 8 x 9 = 72) block 0 spanned every request and staged the
  // UNION of their key runs -- the whole combined sequence, ~303 tiles of 32
  // keys at ctx 2048, while each of its rows is live for only its own request's
  // ~38. The mapping is `vt::DFlashResolveQueryBlock`, which lives in a header
  // and is coverage-tested on the CPU (tests/vt/test_dflash_attn_grid.cpp),
  // because a wrong mapping drops output rows silently rather than crashing.
  const vt::DFlashQueryBlock qb = vt::DFlashResolveQueryBlock(
      qcu, num_reqs, tiles_per_req, kMmaWarps * kMmaQ, static_cast<int64_t>(blockIdx.x));
  if (!qb.live) return;  // block-uniform: a padded tile of a narrower request
  const int64_t qblk = qb.qblk;
  const int64_t qend = qb.qend;
  const int64_t rq = qb.req;
  (void)qrows;

  // --- key range: THIS REQUEST's run. No union, because the block cannot span
  // a request boundary any more. Same arithmetic the union loop applied to the
  // one intersecting request: `lo` was max(qrs, qblk) == qblk and `hi` was
  // min(qre, qend) - 1 == qend - 1, both by construction of the mapping.
  const int64_t qrs = qcu[rq], qre = qcu[rq + 1];
  const int64_t rs = cu[rq], re = cu[rq + 1];
  const int64_t off = (re - rs) - (qre - qrs);  // bottom-right anchor
  // #2784: the block's STAGING range is the union of its rows' masks. Both
  // bounds of `vt::DFlashMaskSpanOf` are non-decreasing in the query row
  // (asserted in tests/vt/test_dflash_attn_mask.cpp), so the union is the FIRST
  // row's `lo` and the LAST row's `hi` — no third formula to keep in step.
  const vt::DFlashMaskSpan sfirst =
      vt::DFlashMaskSpanOf(off + (qblk - qrs), re - rs, causal, window);
  const vt::DFlashMaskSpan slast =
      vt::DFlashMaskSpanOf(off + (qend - 1 - qrs), re - rs, causal, window);
  const int64_t khi = rs + slast.hi;
  const int64_t klo = rs + sfirst.lo;
  if (khi < klo) return;  // block-uniform

  // --- this lane's TWO query rows and their per-row mask bounds -------------
  const int64_t qbase = qblk + static_cast<int64_t>(warp) * kMmaQ;
  int64_t qrow[2], rlo[2], rhi[2];
  bool live[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int64_t i = qbase + gid + 8 * u;
    // #2202: bounded by THIS REQUEST's end, not the batch's. A block no longer
    // spans requests, so a row past `qend` belongs to the next one and must not
    // be attended or written.
    live[u] = i < qend;
    const int64_t ic = live[u] ? i : (qend - 1);  // clamp: reads stay in bounds
    qrow[u] = ic;
    const DFlashRowSpan sp = DFlashResolveRow(qcu, cu, num_reqs, ic);
    const int64_t qs = sp.ks, qe = sp.ke;
    const int64_t ii = sp.ic;
    // #2784: the ONE shared bound (vt/dflash_attn_mask.h).
    const vt::DFlashMaskSpan rspan = vt::DFlashMaskSpanOf(ii, qe - qs, causal, window);
    rhi[u] = qs + rspan.hi;
    rlo[u] = qs + rspan.lo;
  }

  // --- Q A-fragments, read ONCE straight from global into registers ---------
  unsigned qfrag[kDT][4];
#pragma unroll
  for (int kd = 0; kd < kDT; ++kd) {
    const int64_t c0 = static_cast<int64_t>(kd) * 16 + tig * 2;
    const int64_t o0 = (qrow[0] * hq + h) * kD;
    const int64_t o1 = (qrow[1] * hq + h) * kD;
    qfrag[kd][0] = *reinterpret_cast<const unsigned*>(query + o0 + c0);
    qfrag[kd][1] = *reinterpret_cast<const unsigned*>(query + o1 + c0);
    qfrag[kd][2] = *reinterpret_cast<const unsigned*>(query + o0 + c0 + 8);
    qfrag[kd][3] = *reinterpret_cast<const unsigned*>(query + o1 + c0 + 8);
  }

  float acc[kNT][4];
#pragma unroll
  for (int nt = 0; nt < kNT; ++nt) {
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[nt][c] = 0.0f;
  }
  float mrow[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float lrow[2] = {0.0f, 0.0f};

  for (int64_t base = klo; base <= khi; base += kMmaKeys) {
    // (1) stage the K and V tiles: 8 bf16 (16 B) per thread per step, coalesced.
    __syncthreads();
    for (int idx = tid * 8; idx < kMmaKeys * kD; idx += static_cast<int>(blockDim.x) * 8) {
      const int kk = idx / kD, col = idx % kD;
      const int64_t j = base + kk;
      const int64_t jc = j < krows ? j : (krows - 1);  // clamp; masked keys score -inf
      const int64_t off = (jc * hk + g) * kD + col;
      const int sh = kk * kStride + col;
      *reinterpret_cast<uint4*>(ksh + sh) = *reinterpret_cast<const uint4*>(key + off);
      *reinterpret_cast<uint4*>(vsh + sh) = *reinterpret_cast<const uint4*>(value + off);
    }
    __syncthreads();

    // (2) S = Q·Kᵀ on the tensor cores. 4 independent n-tiles => 4-way ILP over
    //     the kDT-long serial accumulate chain.
    float s[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
      for (int c = 0; c < 4; ++c) s[nt][c] = 0.0f;
#pragma unroll
      for (int kd = 0; kd < kDT; ++kd) {
        const int krow = nt * 8 + gid;
        unsigned b[2];
        b[0] = *reinterpret_cast<const unsigned*>(ksh + krow * kStride + kd * 16 + tig * 2);
        b[1] = *reinterpret_cast<const unsigned*>(ksh + krow * kStride + kd * 16 + tig * 2 + 8);
        MmaBf16M16N8K16(s[nt], qfrag[kd], b);
      }
    }

    // (3) scale + MASK, then the per-row max over this lane's 8 columns.
    float rmax[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int u = c >> 1;
        const int64_t j = base + nt * 8 + tig * 2 + (c & 1);
        float v = s[nt][c] * scale;
        if (j < rlo[u] || j > rhi[u]) v = -CUDART_INF_F;
        s[nt][c] = v;
        rmax[u] = fmaxf(rmax[u], v);
      }
    }
    // The 8 columns of a row live in the 4 lanes of a QUAD: reduce with xor 1,2.
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      rmax[u] = fmaxf(rmax[u], __shfl_xor_sync(0xFFFFFFFFu, rmax[u], 1));
      rmax[u] = fmaxf(rmax[u], __shfl_xor_sync(0xFFFFFFFFu, rmax[u], 2));
    }

    // (4) online softmax. A row can see NOTHING in this tile (the block walks the
    //     union of 64 rows' ranges), so guard the -inf minus -inf that would NaN.
    float corr[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const bool any = rmax[u] > -CUDART_INF_F;
      const float mn = any ? fmaxf(mrow[u], rmax[u]) : mrow[u];
      corr[u] = any ? __expf(mrow[u] - mn) : 1.0f;  // 0 on the first live tile
      mrow[u] = mn;
    }
    float rsum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int u = c >> 1;
        const float p = s[nt][c] > -CUDART_INF_F ? __expf(s[nt][c] - mrow[u]) : 0.0f;
        s[nt][c] = p;
        rsum[u] += p;
      }
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      rsum[u] += __shfl_xor_sync(0xFFFFFFFFu, rsum[u], 1);
      rsum[u] += __shfl_xor_sync(0xFFFFFFFFu, rsum[u], 2);
      lrow[u] = lrow[u] * corr[u] + rsum[u];
    }
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
      acc[nt][0] *= corr[0];
      acc[nt][1] *= corr[0];
      acc[nt][2] *= corr[1];
      acc[nt][3] *= corr[1];
    }

    // (5) P -> bf16 A-fragments. The C layout of step (2) IS the A layout, so this
    //     is a pure pack: n-tiles {2kk, 2kk+1} become the 16-key k-step kk.
    unsigned pf[2][4];
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const __nv_bfloat162 r0 = __floats2bfloat162_rn(s[2 * kk][0], s[2 * kk][1]);
      const __nv_bfloat162 r1 = __floats2bfloat162_rn(s[2 * kk][2], s[2 * kk][3]);
      const __nv_bfloat162 r2 = __floats2bfloat162_rn(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      const __nv_bfloat162 r3 = __floats2bfloat162_rn(s[2 * kk + 1][2], s[2 * kk + 1][3]);
      pf[kk][0] = *reinterpret_cast<const unsigned*>(&r0);
      pf[kk][1] = *reinterpret_cast<const unsigned*>(&r1);
      pf[kk][2] = *reinterpret_cast<const unsigned*>(&r2);
      pf[kk][3] = *reinterpret_cast<const unsigned*>(&r3);
    }

    // (6) O += P·V on the tensor cores. B wants consecutive KEYS in one register,
    //     which V's row-major tile does not give: four 16-bit shared reads.
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
      const int dcol = nt * 8 + gid;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int r0 = kk * 16 + tig * 2;
        unsigned b[2];
        b[0] = static_cast<unsigned>(vsh_u[r0 * kStride + dcol]) |
               (static_cast<unsigned>(vsh_u[(r0 + 1) * kStride + dcol]) << 16);
        b[1] = static_cast<unsigned>(vsh_u[(r0 + 8) * kStride + dcol]) |
               (static_cast<unsigned>(vsh_u[(r0 + 9) * kStride + dcol]) << 16);
        MmaBf16M16N8K16(acc[nt], pf[kk], b);
      }
    }
  }

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    if (!live[u]) continue;
    const float inv = lrow[u] > 0.0f ? 1.0f / lrow[u] : 0.0f;
    const int64_t ooff = (qrow[u] * hq + h) * kD;
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
      const int64_t col = nt * 8 + tig * 2;
      Store(out, ooff + col, acc[nt][2 * u] * inv);
      Store(out, ooff + col + 1, acc[nt][2 * u + 1] * inv);
    }
  }
#else
  (void)out; (void)query; (void)key; (void)value; (void)cu; (void)qcu; (void)num_reqs;
  (void)hq; (void)hk; (void)scale; (void)causal; (void)window;
  __trap();  // never launched below sm_80 (host gate: DFlashMmaSupported)
#endif
}

// The bf16 tensor-core path needs BOTH a compile-time and a runtime guarantee:
// mma.sync's bf16 form is Ampere-and-later, so it must be in the compiled arch set
// AND on the device we are actually running. VT_DFLASH_ATTN_MMA=0 disables it for
// the same-binary A/B this project's benchmark protocol requires.
#if defined(__CUDA_ARCH_LIST__)
constexpr int kBuiltArchList[] = {__CUDA_ARCH_LIST__};
constexpr bool BuiltArchesAllHaveBf16Mma() {
  for (int a : kBuiltArchList) {
    if (a < 800) return false;
  }
  return true;
}
#else
constexpr bool BuiltArchesAllHaveBf16Mma() { return false; }
#endif

inline bool DFlashMmaSupported() {
  static const bool on = [] {
    if (!BuiltArchesAllHaveBf16Mma()) return false;
    const char* e = std::getenv("VT_DFLASH_ATTN_MMA");
    if (e != nullptr && e[0] == '0') return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
      return false;
    return major >= 8;
  }();
  return on;
}

template <typename Tin>
void LaunchDFlashBlockAttention(cudaStream_t s, Tensor& out, const Tensor& query,
                                const Tensor& key, const Tensor& value,
                                const DFlashBlockAttentionArgs& args) {
  // D1 (#2087): the grid runs over the QUERY rows, which are no longer the key
  // rows. `t` is the query count -- every grid expression below already used
  // query.shape[0], which is why the grid shrinks with no change to its shape.
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  if (t == 0 || hq == 0 || d == 0) return;
  // Upload cu_seqlens (host, num_reqs+1) to a stream-ordered device scratch, and
  // the query cu right behind it in the SAME allocation. When the caller set no
  // query cu the kernels get the same pointer twice, so the null case costs no
  // extra byte and takes no branch inside a kernel.
  const size_t cun = static_cast<size_t>(args.num_reqs) + 1;
  const bool split_q = args.cu_seqlens_q != nullptr;
  const size_t cub = cun * sizeof(int32_t);
  int32_t* d_cu = nullptr;
  Check(cudaMallocAsync(&d_cu, split_q ? 2 * cub : cub, s), "dflash-block-attn cu malloc");
  Check(cudaMemcpyAsync(d_cu, args.cu_seqlens, cub, cudaMemcpyHostToDevice, s),
        "dflash-block-attn cu upload");
  int32_t* d_qcu = d_cu;
  if (split_q) {
    d_qcu = d_cu + cun;
    Check(cudaMemcpyAsync(d_qcu, args.cu_seqlens_q, cub, cudaMemcpyHostToDevice, s),
          "dflash-block-attn query cu upload");
  }
  const dim3 grid(static_cast<unsigned>(t), static_cast<unsigned>(hq));
  // TENSOR-CORE fast path: bf16 in, head_dim a multiple of the MMA's k (16), up to
  // 128 (the largest head this port uses). f32 streams stay on the CUDA-core kernel
  // below -- rounding them to bf16 to reach the tensor cores would trade the f32
  // path's 2e-5 tolerance for a bf16 one, which is not a trade this op may make.
  if (std::is_same<Tin, __nv_bfloat16>::value && d % 16 == 0 && d >= 16 && d <= 128 &&
      DFlashMmaSupported()) {
    // #2202: the query axis is tiled PER REQUEST, so a block never spans a
    // request boundary and stages only that request's key run. `tiles_per_req`
    // is the widest request's tile count; a narrower request's extra tiles
    // return immediately. At c=8 (8 requests of 9 rows) this is 8 blocks of one
    // tile instead of 2 blocks of 64 rows, where the first of those 2 used to
    // stage the entire combined sequence.
    const int32_t* host_qcu = split_q ? args.cu_seqlens_q : args.cu_seqlens;
    const int64_t tiles_per_req =
        vt::DFlashQueryTilesPerReq(host_qcu, args.num_reqs, kMmaWarps * kMmaQ);
    if (tiles_per_req == 0) return;
    const dim3 mgrid(static_cast<unsigned>(static_cast<int64_t>(args.num_reqs) * tiles_per_req),
                     static_cast<unsigned>(hq));
    const unsigned mblock = kMmaWarps * 32;
    const size_t mshmem =
        2u * kMmaKeys * static_cast<size_t>(d + kMmaPad) * sizeof(__nv_bfloat16);
#define VT_DFLASH_MMA(DT)                                                                     \
  do {                                                                                        \
    if (out.dtype == DType::kF32) {                                                           \
      DFlashAttnMmaKernel<float, DT><<<mgrid, mblock, mshmem, s>>>(                           \
          out.Ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(query.data),               \
          reinterpret_cast<const __nv_bfloat16*>(key.data),                                   \
          reinterpret_cast<const __nv_bfloat16*>(value.data), d_cu, d_qcu, args.num_reqs,     \
          hq, hk, args.scale, args.causal, args.sliding_window, tiles_per_req);          \
    } else {                                                                                  \
      DFlashAttnMmaKernel<__nv_bfloat16, DT><<<mgrid, mblock, mshmem, s>>>(                   \
          out.Ptr<__nv_bfloat16>(), reinterpret_cast<const __nv_bfloat16*>(query.data),       \
          reinterpret_cast<const __nv_bfloat16*>(key.data),                                   \
          reinterpret_cast<const __nv_bfloat16*>(value.data), d_cu, d_qcu, args.num_reqs,     \
          hq, hk, args.scale, args.causal, args.sliding_window, tiles_per_req);          \
    }                                                                                         \
  } while (0)
    switch (d / 16) {
      case 1: VT_DFLASH_MMA(1); break;
      case 2: VT_DFLASH_MMA(2); break;
      case 3: VT_DFLASH_MMA(3); break;
      case 4: VT_DFLASH_MMA(4); break;
      case 5: VT_DFLASH_MMA(5); break;
      case 6: VT_DFLASH_MMA(6); break;
      case 7: VT_DFLASH_MMA(7); break;
      default: VT_DFLASH_MMA(8); break;
    }
#undef VT_DFLASH_MMA
    Check(cudaGetLastError(), "dflash-block-attn mma launch");
    Check(cudaFreeAsync(d_cu, s), "dflash-block-attn cu free");
    return;
  }
  // Warp-per-query fast path when head_dim is a whole number of warp widths (64 and
  // 128 cover every head this port uses). Three forms live here so the verdict can
  // be reproduced on ONE binary (same-binary A/B is this project's benchmark rule):
  // the CHUNKED reduce-scatter kernel is the default, VT_DFLASH_ATTN_KEYLANE=1
  // selects the measured-negative one-key-per-lane form, and VT_DFLASH_ATTN_WARP=1
  // selects the original per-key warp-reduction form.
  if (d % 32 == 0 && d / 32 <= 4 && !UseDflashAttnWarpKernel()) {
    constexpr int kWarpsPerBlock = 8;
    const dim3 kgrid(static_cast<unsigned>((t + kWarpsPerBlock - 1) / kWarpsPerBlock),
                     static_cast<unsigned>(hq));
    const unsigned kblock = kWarpsPerBlock * 32;
    const bool keylane = UseDflashAttnKeyLaneKernel();
    const size_t kshmem =
        static_cast<size_t>(kWarpsPerBlock) * (keylane ? (d + 32) : 32) * sizeof(float);
#define VT_DFLASH_CHUNK(PER_LANE)                                                               \
  do {                                                                                          \
    if (out.dtype == DType::kF32) {                                                             \
      DFlashAttnChunkKernel<Tin, float, PER_LANE><<<kgrid, kblock, kshmem, s>>>(                \
          out.Ptr<float>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), d_cu, d_qcu,    \
          args.num_reqs, hq, hk, args.scale, args.causal, args.sliding_window);                 \
    } else {                                                                                    \
      DFlashAttnChunkKernel<Tin, __nv_bfloat16, PER_LANE><<<kgrid, kblock, kshmem, s>>>(        \
          out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), d_cu,   \
          d_qcu, args.num_reqs, hq, hk, args.scale, args.causal, args.sliding_window);          \
    }                                                                                           \
  } while (0)
    if (!keylane) {
      switch (d / 32) {
        case 1: VT_DFLASH_CHUNK(1); break;
        case 2: VT_DFLASH_CHUNK(2); break;
        case 3: VT_DFLASH_CHUNK(3); break;
        default: VT_DFLASH_CHUNK(4); break;
      }
#undef VT_DFLASH_CHUNK
      Check(cudaGetLastError(), "dflash-block-attn chunk launch");
      Check(cudaFreeAsync(d_cu, s), "dflash-block-attn cu free");
      return;
    }
#define VT_DFLASH_KEYLANE(PER_LANE)                                                             \
  do {                                                                                          \
    if (out.dtype == DType::kF32) {                                                             \
      DFlashAttnKeyLaneKernel<Tin, float, PER_LANE><<<kgrid, kblock, kshmem, s>>>(              \
          out.Ptr<float>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), d_cu, d_qcu,    \
          args.num_reqs, hq, hk, args.scale, args.causal, args.sliding_window);                 \
    } else {                                                                                    \
      DFlashAttnKeyLaneKernel<Tin, __nv_bfloat16, PER_LANE><<<kgrid, kblock, kshmem, s>>>(      \
          out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), d_cu,   \
          d_qcu, args.num_reqs, hq, hk, args.scale, args.causal, args.sliding_window);          \
    }                                                                                           \
  } while (0)
    switch (d / 32) {
      case 1: VT_DFLASH_KEYLANE(1); break;
      case 2: VT_DFLASH_KEYLANE(2); break;
      case 3: VT_DFLASH_KEYLANE(3); break;
      default: VT_DFLASH_KEYLANE(4); break;
    }
#undef VT_DFLASH_KEYLANE
    Check(cudaGetLastError(), "dflash-block-attn key-lane launch");
    Check(cudaFreeAsync(d_cu, s), "dflash-block-attn cu free");
    return;
  }
  if (d % 32 == 0 && d / 32 <= 4) {
    constexpr int kWarpsPerBlock = 4;
    const dim3 wgrid(static_cast<unsigned>((t + kWarpsPerBlock - 1) / kWarpsPerBlock),
                     static_cast<unsigned>(hq));
    const unsigned wblock = kWarpsPerBlock * 32;
#define VT_DFLASH_WARP(PER_LANE)                                                              \
  do {                                                                                        \
    if (out.dtype == DType::kF32) {                                                           \
      DFlashBlockAttentionWarpKernel<Tin, float, PER_LANE><<<wgrid, wblock, 0, s>>>(          \
          out.Ptr<float>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), d_cu, d_qcu,  \
          args.num_reqs, hq, hk, d, args.scale, args.causal, args.sliding_window);            \
    } else {                                                                                  \
      DFlashBlockAttentionWarpKernel<Tin, __nv_bfloat16, PER_LANE><<<wgrid, wblock, 0, s>>>(  \
          out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), d_cu, \
          d_qcu, args.num_reqs, hq, hk, d, args.scale, args.causal, args.sliding_window);     \
    }                                                                                         \
  } while (0)
    switch (d / 32) {
      case 1: VT_DFLASH_WARP(1); break;
      case 2: VT_DFLASH_WARP(2); break;
      case 3: VT_DFLASH_WARP(3); break;
      default: VT_DFLASH_WARP(4); break;
    }
#undef VT_DFLASH_WARP
    Check(cudaGetLastError(), "dflash-block-attn warp launch");
    Check(cudaFreeAsync(d_cu, s), "dflash-block-attn cu free");
    return;
  }

  const size_t shmem = (static_cast<size_t>(d) + kBlock) * sizeof(float);
  switch (out.dtype) {
    case DType::kF32:
      DFlashBlockAttentionKernel<Tin, float><<<grid, kBlock, shmem, s>>>(
          out.Ptr<float>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), d_cu, d_qcu,
          args.num_reqs, hq, hk, d, args.scale, args.causal, args.sliding_window);
      break;
    case DType::kBF16:
      DFlashBlockAttentionKernel<Tin, __nv_bfloat16><<<grid, kBlock, shmem, s>>>(
          out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), d_cu,
          d_qcu, args.num_reqs, hq, hk, d, args.scale, args.causal, args.sliding_window);
      break;
    default: VT_CHECK(false, "cuda dflash-block-attn: unsupported out dtype");
  }
  Check(cudaGetLastError(), "dflash-block-attn launch");
  Check(cudaFreeAsync(d_cu, s), "dflash-block-attn cu free");
}

void DFlashBlockAttentionKernelCuda(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                                    const Tensor& value, const DFlashBlockAttentionArgs& args) {
  switch (query.dtype) {
    case DType::kF32:
      LaunchDFlashBlockAttention<float>(AsStream(q), out, query, key, value, args);
      break;
    case DType::kBF16:
      LaunchDFlashBlockAttention<__nv_bfloat16>(AsStream(q), out, query, key, value, args);
      break;
    default: VT_CHECK(false, "cuda dflash-block-attn: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// DFlash PAGED in-block attention (SPEC-DFLASH D12 Part B) — the CAPTURE-SAFE form
// of DFlashBlockAttentionKernel. Same block-cooperative f32 online-softmax
// recurrence, but each (1+k) block query attends over [PAGED context ; its own
// (1+k) block] with the context supplied as DATA: a paged K/V cache
// ([pages, block_size, Hkv, D]) + per-request seq_lens (context length C_r) +
// block_table (logical page -> physical page), mirroring PagedAttentionKernel's
// paged read. Grid is STATIC over the fixed Nq=(1+k)*num_reqs query rows; EVERY
// metadata input is a persistent device pointer read in place — NO cudaMallocAsync
// / cudaMemcpyAsync of a function-local host buffer (the eager DFlash launcher's
// cu_seqlens upload was the cudagraph-capture-bakes-stack-addresses UAF class).
// The D2 mask is applied over the COMBINED index: context rows occupy combined
// positions [0,C_r) (position-ordered) and block rows [C_r, C_r+blen_r), so a
// query's combined offset is C_r+ii and the mask/order match the materialized
// combined buffer bit-for-bit.
template <typename Tin, typename Tout>
__global__ void DFlashPagedBlockAttentionKernel(
    Tout* out, const Tin* query, const Tin* block_key, const Tin* block_value,
    const Tin* ctx_key, const Tin* ctx_value, const int32_t* cu, const int32_t* slen,
    const int32_t* btbl, int num_reqs, int64_t hq, int64_t hk, int64_t d, int64_t block_size,
    int64_t max_pages, int64_t ck_blk, int64_t ck_pg, int64_t ck_hd, float scale, bool causal,
    int64_t window) {
  const int64_t i = blockIdx.x;  // GLOBAL block-query row
  const int64_t h = blockIdx.y;  // q-head
  const int64_t g = h / (hq / hk);
  int req = -1;
  int64_t qs = 0, qe = 0;
  for (int r = 0; r < num_reqs; ++r) {
    if (i >= cu[r] && i < cu[r + 1]) {
      qs = cu[r];
      qe = cu[r + 1];
      req = r;
      break;
    }
  }
  if (req < 0) return;  // padding row beyond the last request
  const int64_t blen = qe - qs;
  const int64_t C = slen[req];
  const int64_t N = C + blen;  // combined key length
  const int64_t ii_comb = C + (i - qs);
  // #2784: the ONE shared bound (vt/dflash_attn_mask.h).
  const vt::DFlashMaskSpan span = vt::DFlashMaskSpanOf(ii_comb, N, causal, window);
  const int64_t jhi = span.hi;
  const int64_t jlo = span.lo;
  const int64_t qoff = (i * hq + h) * d;

  extern __shared__ float smem[];
  float* acc = smem;                   // [d] running output accumulator
  float* red = smem + d;               // [blockDim.x] reduction scratch
  __shared__ float s_score, s_m, s_l;  // block-wide score / running max / denom
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) acc[e] = 0.0f;
  if (threadIdx.x == 0) {
    s_m = -CUDART_INF_F;
    s_l = 0.0f;
  }
  __syncthreads();

  for (int64_t cj = jlo; cj <= jhi; ++cj) {
    int64_t koff;
    const Tin* ksrc;
    const Tin* vsrc;
    if (cj < C) {  // paged context key/value (k and v share the [pages,bs,Hkv,D] layout)
      const int64_t page = btbl[req * max_pages + cj / block_size];
      const int64_t off = cj % block_size;
      koff = page * ck_blk + off * ck_pg + g * ck_hd;
      ksrc = ctx_key;
      vsrc = ctx_value;
    } else {  // contiguous block key/value
      const int64_t brow = qs + (cj - C);
      koff = (brow * hk + g) * d;
      ksrc = block_key;
      vsrc = block_value;
    }
    float part = 0.0f;
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      part += Load(query, qoff + e) * Load(ksrc, koff + e);
    red[threadIdx.x] = part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) s_score = red[0] * scale;
    __syncthreads();

    const float s = s_score;
    const float m_new = fmaxf(s_m, s);
    const float corr = expf(s_m - m_new);
    const float p = expf(s - m_new);
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      acc[e] = acc[e] * corr + p * Load(vsrc, koff + e);
    __syncthreads();
    if (threadIdx.x == 0) {
      s_l = s_l * corr + p;
      s_m = m_new;
    }
    __syncthreads();
  }

  const float inv = 1.0f / s_l;
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) Store(out, qoff + e, acc[e] * inv);
}

// D14 WARP-scoped variant of DFlashPagedBlockAttentionKernel — the SAME paged/
// block combined-index read, GQA broadcast, and causal/SWA mask, but ONE WARP
// (not a whole kBlock=256 block) owns each (block-query row, q-head): the
// head_dim dot product is a butterfly `__shfl_xor` reduction (no __syncthreads),
// the output accumulator lives in registers, and the online-softmax stats are
// per-lane. This removes the block kernel's per-key __syncthreads storm (2 barriers
// + a 256-wide shared-mem tree reduction PER context key, over C~500-640 keys) that
// nsys attributed ~1.8% of the graphed DFlash step to (median ~460 us/call vs
// vLLM's fused flash draft-attn ~0.15%) — exactly the AttentionDenseFast fix applied
// to the ViT tower. It is NOT bit-identical to the block kernel (the head_dim
// partial-sum grouping over 32 lanes differs), but is the SAME f32-online-softmax
// math within the bf16 envelope (CUDA==CPU gates f32 eps 1e-4 / bf16 eps 3e-2), and
// spec-decode output is exact by construction (the target verify is unchanged; only
// which draft proposals are accepted can shift, within the ratified ±4 gate). Reads
// only persistent device pointers (capture-safe, no shared mem, no host uploads).
// Grounded in the shipped AttentionWarpKernel + the online-softmax recurrence
// (src/vt/cuda/flash_attn/).
template <typename Tin, typename Tout>
__global__ void DFlashPagedBlockAttentionWarpKernel(
    Tout* out, const Tin* query, const Tin* block_key, const Tin* block_value,
    const Tin* ctx_key, const Tin* ctx_value, const int32_t* cu, const int32_t* slen,
    const int32_t* btbl, int num_reqs, int64_t hq, int64_t hk, int64_t d, int64_t block_size,
    int64_t max_pages, int64_t ck_blk, int64_t ck_pg, int64_t ck_hd, float scale, bool causal,
    int64_t window) {
  constexpr int kMaxPerLane = 8;  // head_dim up to 256
  const int64_t i = blockIdx.x;   // GLOBAL block-query row
  const int64_t h = blockIdx.y;   // q-head
  const int64_t g = h / (hq / hk);
  const int lane = static_cast<int>(threadIdx.x);  // 0..31
  int req = -1;
  int64_t qs = 0, qe = 0;
  for (int r = 0; r < num_reqs; ++r) {
    if (i >= cu[r] && i < cu[r + 1]) {
      qs = cu[r];
      qe = cu[r + 1];
      req = r;
      break;
    }
  }
  if (req < 0) return;  // padding row beyond the last request
  const int64_t blen = qe - qs;
  const int64_t C = slen[req];
  const int64_t N = C + blen;  // combined key length
  const int64_t ii_comb = C + (i - qs);
  // #2784: the ONE shared bound (vt/dflash_attn_mask.h).
  const vt::DFlashMaskSpan span = vt::DFlashMaskSpanOf(ii_comb, N, causal, window);
  const int64_t jhi = span.hi;
  const int64_t jlo = span.lo;
  const int64_t qoff = (i * hq + h) * d;
  const int npl = static_cast<int>((d + 31) / 32);
  float qreg[kMaxPerLane];
  float acc[kMaxPerLane];
#pragma unroll
  for (int k = 0; k < kMaxPerLane; ++k) {
    qreg[k] = 0.0f;
    acc[k] = 0.0f;
  }
  for (int k = 0; k < npl; ++k) {
    const int e = lane + 32 * k;
    if (e < d) qreg[k] = Load(query, qoff + e);
  }
  float m = -CUDART_INF_F, l = 0.0f;
  for (int64_t cj = jlo; cj <= jhi; ++cj) {
    int64_t koff;
    const Tin* ksrc;
    const Tin* vsrc;
    if (cj < C) {  // paged context key/value (k and v share the [pages,bs,Hkv,D] layout)
      const int64_t page = btbl[req * max_pages + cj / block_size];
      const int64_t off = cj % block_size;
      koff = page * ck_blk + off * ck_pg + g * ck_hd;
      ksrc = ctx_key;
      vsrc = ctx_value;
    } else {  // contiguous block key/value
      const int64_t brow = qs + (cj - C);
      koff = (brow * hk + g) * d;
      ksrc = block_key;
      vsrc = block_value;
    }
    float part = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerLane; ++k) {
      const int e = lane + 32 * k;
      if (k < npl && e < d) part += qreg[k] * Load(ksrc, koff + e);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
    const float s = part * scale;
    const float m_new = fmaxf(m, s);
    const float corr = expf(m - m_new);  // 0 on the first key (m == -inf)
    const float p = expf(s - m_new);
#pragma unroll
    for (int k = 0; k < kMaxPerLane; ++k) {
      const int e = lane + 32 * k;
      if (k < npl && e < d) acc[k] = acc[k] * corr + p * Load(vsrc, koff + e);
    }
    l = l * corr + p;
    m = m_new;
  }
  const float inv = 1.0f / l;
  for (int k = 0; k < npl; ++k) {
    const int e = lane + 32 * k;
    if (e < d) Store(out, qoff + e, acc[k] * inv);
  }
}

// Select the block kernel (bit-identical, the D12/D13 reference) only when
// VT_DFLASH_ATTN_BLOCK=1; the WARP kernel is the D14 default (same math, bf16
// envelope, ~no __syncthreads storm). Cached once (getenv is not hot-path-safe).
inline bool UseDflashAttnBlockKernel() {
  static const bool on = [] {
    const char* e = std::getenv("VT_DFLASH_ATTN_BLOCK");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

template <typename Tin>
void LaunchDFlashPagedBlockAttention(cudaStream_t s, Tensor& out, const Tensor& query,
                                     const Tensor& block_key, const Tensor& block_value,
                                     const Tensor& ctx_key, const Tensor& ctx_value,
                                     const Tensor& cu_seqlens, const Tensor& seq_lens,
                                     const Tensor& block_table,
                                     const DFlashPagedBlockAttentionArgs& args) {
  const int64_t nq = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = block_key.shape[1];
  if (nq == 0 || hq == 0 || d == 0) return;
  const int64_t block_size = args.block_size;
  const int64_t max_pages = block_table.shape[1];
  // Contiguous paged cache strides: [num_pages, block_size, Hkv, D].
  const int64_t ck_blk = block_size * hk * d, ck_pg = hk * d, ck_hd = d;
  const dim3 grid(static_cast<unsigned>(nq), static_cast<unsigned>(hq));
  if (!UseDflashAttnBlockKernel()) {
    VT_CHECK(d <= 256, "cuda dflash-paged-block-attn(warp): head_dim <= 256 only");
    switch (out.dtype) {
      case DType::kF32:
        DFlashPagedBlockAttentionWarpKernel<Tin, float><<<grid, 32, 0, s>>>(
            out.Ptr<float>(), query.Ptr<Tin>(), block_key.Ptr<Tin>(), block_value.Ptr<Tin>(),
            ctx_key.Ptr<Tin>(), ctx_value.Ptr<Tin>(), cu_seqlens.Ptr<int32_t>(),
            seq_lens.Ptr<int32_t>(), block_table.Ptr<int32_t>(), args.num_reqs, hq, hk, d,
            block_size, max_pages, ck_blk, ck_pg, ck_hd, args.scale, args.causal,
            args.sliding_window);
        break;
      case DType::kBF16:
        DFlashPagedBlockAttentionWarpKernel<Tin, __nv_bfloat16><<<grid, 32, 0, s>>>(
            out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), block_key.Ptr<Tin>(),
            block_value.Ptr<Tin>(), ctx_key.Ptr<Tin>(), ctx_value.Ptr<Tin>(),
            cu_seqlens.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(), block_table.Ptr<int32_t>(),
            args.num_reqs, hq, hk, d, block_size, max_pages, ck_blk, ck_pg, ck_hd, args.scale,
            args.causal, args.sliding_window);
        break;
      default: VT_CHECK(false, "cuda dflash-paged-block-attn: unsupported out dtype");
    }
    Check(cudaGetLastError(), "dflash-paged-block-attn(warp) launch");
    return;
  }
  const size_t shmem = (static_cast<size_t>(d) + kBlock) * sizeof(float);
  switch (out.dtype) {
    case DType::kF32:
      DFlashPagedBlockAttentionKernel<Tin, float><<<grid, kBlock, shmem, s>>>(
          out.Ptr<float>(), query.Ptr<Tin>(), block_key.Ptr<Tin>(), block_value.Ptr<Tin>(),
          ctx_key.Ptr<Tin>(), ctx_value.Ptr<Tin>(), cu_seqlens.Ptr<int32_t>(),
          seq_lens.Ptr<int32_t>(), block_table.Ptr<int32_t>(), args.num_reqs, hq, hk, d, block_size,
          max_pages, ck_blk, ck_pg, ck_hd, args.scale, args.causal, args.sliding_window);
      break;
    case DType::kBF16:
      DFlashPagedBlockAttentionKernel<Tin, __nv_bfloat16><<<grid, kBlock, shmem, s>>>(
          out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), block_key.Ptr<Tin>(),
          block_value.Ptr<Tin>(), ctx_key.Ptr<Tin>(), ctx_value.Ptr<Tin>(),
          cu_seqlens.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(), block_table.Ptr<int32_t>(),
          args.num_reqs, hq, hk, d, block_size, max_pages, ck_blk, ck_pg, ck_hd, args.scale,
          args.causal, args.sliding_window);
      break;
    default: VT_CHECK(false, "cuda dflash-paged-block-attn: unsupported out dtype");
  }
  Check(cudaGetLastError(), "dflash-paged-block-attn launch");
}

void DFlashPagedBlockAttentionKernelCuda(Queue& q, Tensor& out, const Tensor& query,
                                         const Tensor& block_key, const Tensor& block_value,
                                         const Tensor& ctx_key, const Tensor& ctx_value,
                                         const Tensor& cu_seqlens, const Tensor& seq_lens,
                                         const Tensor& block_table,
                                         const DFlashPagedBlockAttentionArgs& args) {
  switch (query.dtype) {
    case DType::kF32:
      LaunchDFlashPagedBlockAttention<float>(AsStream(q), out, query, block_key, block_value,
                                             ctx_key, ctx_value, cu_seqlens, seq_lens, block_table,
                                             args);
      break;
    case DType::kBF16:
      LaunchDFlashPagedBlockAttention<__nv_bfloat16>(AsStream(q), out, query, block_key,
                                                     block_value, ctx_key, ctx_value, cu_seqlens,
                                                     seq_lens, block_table, args);
      break;
    default:
      VT_CHECK(false, "cuda dflash-paged-block-attn: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// AttentionDenseFast — a WARP-scoped variant of the AttentionKernel above with
// the IDENTICAL online-softmax recurrence (flash-style, f32 accumulation), for
// dense full/causal attention over small head_dim (the Qwen3-VL vision tower:
// head_dim 72, non-causal). One WARP (not a whole block) owns each (query,head):
// the head_dim reduction is a butterfly `__shfl_xor` (NO __syncthreads), the
// output accumulator lives in registers, and the softmax stats are tracked per
// lane. This removes the naive kernel's per-key block-reduction __syncthreads
// storm — nsys attributed ~99% of the ViT tower forward to that kernel (56 ms
// per block at 784 tokens). It is NOT bit-identical to AttentionKernel (the
// head_dim partial-sum grouping over 32 lanes differs from the block version),
// but is the same f32-online-softmax math within the tower's bf16 envelope.
// Registered as a SEPARATE op (kAttentionDenseFast) so kAttention — used by the
// text/audio paths — is byte-identical. Grounded in the online-softmax /
// FlashAttention recurrence we already ship (src/vt/cuda/flash_attn/).
template <typename Tin, typename Tout>
__global__ void AttentionWarpKernel(Tout* out, const Tin* query, const Tin* key, const Tin* value,
                                    int64_t hq, int64_t hk, int64_t d, int64_t t, float scale,
                                    bool causal) {
  constexpr int kMaxPerLane = 8;  // head_dim up to 256
  const int64_t i = blockIdx.x;   // query position
  const int64_t h = blockIdx.y;   // q-head
  const int64_t g = h / (hq / hk);
  const int lane = static_cast<int>(threadIdx.x);  // 0..31
  const int64_t jmax = causal ? i : t - 1;
  const int64_t qoff = (i * hq + h) * d;
  const int npl = static_cast<int>((d + 31) / 32);  // elements this lane owns
  float qreg[kMaxPerLane];
  float acc[kMaxPerLane];
#pragma unroll
  for (int k = 0; k < kMaxPerLane; ++k) {
    qreg[k] = 0.0f;
    acc[k] = 0.0f;
  }
  for (int k = 0; k < npl; ++k) {
    const int e = lane + 32 * k;
    if (e < d) qreg[k] = Load(query, qoff + e);
  }
  float m = -CUDART_INF_F, l = 0.0f;
  for (int64_t j = 0; j <= jmax; ++j) {
    const int64_t koff = (j * hk + g) * d;
    float part = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerLane; ++k) {
      const int e = lane + 32 * k;
      if (k < npl && e < d) part += qreg[k] * Load(key, koff + e);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
    const float s = part * scale;
    const float m_new = fmaxf(m, s);
    const float corr = expf(m - m_new);  // 0 on the first key (m == -inf)
    const float p = expf(s - m_new);
    const int64_t voff = (j * hk + g) * d;
#pragma unroll
    for (int k = 0; k < kMaxPerLane; ++k) {
      const int e = lane + 32 * k;
      if (k < npl && e < d) acc[k] = acc[k] * corr + p * Load(value, voff + e);
    }
    l = l * corr + p;
    m = m_new;
  }
  const float inv = 1.0f / l;
  for (int k = 0; k < npl; ++k) {
    const int e = lane + 32 * k;
    if (e < d) Store(out, qoff + e, acc[k] * inv);
  }
}

template <typename Tin>
void LaunchAttentionWarp(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& key,
                         const Tensor& value, const AttentionArgs& args) {
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  if (t == 0 || hq == 0 || d == 0) return;
  VT_CHECK(d <= 256, "cuda attention-dense-fast: head_dim <= 256 only");
  const dim3 grid(static_cast<unsigned>(t), static_cast<unsigned>(hq));
  switch (out.dtype) {
    case DType::kF32:
      AttentionWarpKernel<Tin, float><<<grid, 32, 0, s>>>(
          out.Ptr<float>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), hq, hk, d, t,
          args.scale, args.causal);
      break;
    case DType::kBF16:
      AttentionWarpKernel<Tin, __nv_bfloat16><<<grid, 32, 0, s>>>(
          out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), hq, hk,
          d, t, args.scale, args.causal);
      break;
    default: VT_CHECK(false, "cuda attention-dense-fast: unsupported out dtype");
  }
  Check(cudaGetLastError(), "attention-dense-fast launch");
}

void AttentionDenseFastKernelCuda(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                                  const Tensor& value, const AttentionArgs& args) {
  switch (query.dtype) {
    case DType::kF32:
      LaunchAttentionWarp<float>(AsStream(q), out, query, key, value, args);
      break;
    case DType::kBF16:
      LaunchAttentionWarp<__nv_bfloat16>(AsStream(q), out, query, key, value, args);
      break;
    default:
      VT_CHECK(false, "cuda attention-dense-fast: unsupported input dtype (f32/bf16 only)");
  }
}

// AttentionDenseFlash — a SHARED-MEMORY-TILED flash variant of AttentionWarpKernel
// with the BIT-IDENTICAL per-warp online-softmax recurrence, structured so that a
// block of `kFlashBr` query-warps SHARES each streamed K/V tile out of shared
// memory instead of re-reading all K/V from global once PER (query,head). This is
// the classic FlashAttention K/V-tiling: K and V are streamed into shared memory in
// tiles of `kFlashBc` columns and reused across the whole query block, killing the
// O(t^2) redundant global K/V reads that make AttentionWarpKernel memory-bound
// (§13.5: ~21 ms/layer of K reads alone at 1500 frames, no cross-query tile reuse).
// Ported 1:1 in STRUCTURE from the vendored FlashAttention-2 forward kernel
// (src/vt/cuda/flash_attn/src/flash_fwd_kernel.h compute_attn_1rowblock :52 — sK/sV
// shared tiles :163-165, the `for (int n_block ...)` K/V-tile stream + online
// rescale), and cross-checked to vLLM's non-causal encoder attention dispatch
// (vllm/model_executor/models/whisper.py WhisperEncoderAttention:255). Unlike the
// cute/MMA FA2 (head_dim {128,192,256}, paged-KV), this is a scalar warp-per-query
// kernel for the DENSE single-request head_dim-64 encoder layout — so the ARITHMETIC
// is byte-for-byte the AttentionWarpKernel recurrence (same per-lane head_dim
// grouping lane+32k, same butterfly __shfl_xor, same sequential j-order 0..t-1, same
// f32 accumulation) with K/V bytes sourced from shared memory rather than global.
// Because every float op and its order are unchanged, the output is BIT-IDENTICAL to
// AttentionDenseFast ⇒ token-identical by construction; kAttention (text decode) is
// untouched. One q-head per CTA (all warps share the same GQA kv-head g).
constexpr int kFlashBr = 16;  // query-warps per CTA (= K/V global-read reuse factor)
constexpr int kFlashBc = 64;  // key/value columns streamed per shared-memory tile
// The register blocking: head_dim elements each of the 32 lanes holds. It lives at
// file scope rather than inside the kernel body precisely so the static_assert below
// can READ it -- a per-kernel local is invisible here, and asserting the literal 8
// instead would compare the header's constant against a number nothing else uses.
constexpr int kFlashMaxPerLane = 8;  // head_dim up to 8 * 32
// The head_dim bound this kernel advertises is computed in include/vt/ops.h so a box
// with no GPU can execute it. These tie the two together: change the tile width or the
// register blocking here and the arithmetic there stops describing this kernel, which
// is how the op came to advertise a head_dim it could not launch (#1544).
static_assert(kFlashBc == kAttentionDenseFlashTileCols,
              "AttentionDenseFlashSmemBytes must use this kernel's tile width");
static_assert(kFlashMaxPerLane * 32 == kAttentionDenseMaxHeadDim,
              "kFlashMaxPerLane * warp size must equal the advertised register bound");

template <typename Tin, typename Tout>
__global__ void AttentionDenseFlashKernel(Tout* out, const Tin* query, const Tin* key,
                                          const Tin* value, int64_t hq, int64_t hk, int64_t d,
                                          int64_t t, float scale, bool causal) {
  extern __shared__ __align__(16) char flash_smem[];
  Tin* sK = reinterpret_cast<Tin*>(flash_smem);
  Tin* sV = sK + static_cast<int64_t>(kFlashBc) * d;

  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int nthreads = kFlashBr * 32;
  const int64_t h = blockIdx.y;             // q-head (one per CTA)
  const int64_t g = h / (hq / hk);          // shared kv-head for every warp in CTA
  const int64_t qi = static_cast<int64_t>(blockIdx.x) * kFlashBr + warp;  // this warp's query
  const bool active = qi < t;
  const int npl = static_cast<int>((d + 31) / 32);  // head_dim elements this lane owns

  // This warp's query row, in registers (identical layout to AttentionWarpKernel).
  float qreg[kFlashMaxPerLane];
  float acc[kFlashMaxPerLane];
#pragma unroll
  for (int k = 0; k < kFlashMaxPerLane; ++k) {
    qreg[k] = 0.0f;
    acc[k] = 0.0f;
  }
  if (active) {
    const int64_t qoff = (qi * hq + h) * d;
    for (int k = 0; k < npl; ++k) {
      const int e = lane + 32 * k;
      if (e < d) qreg[k] = Load(query, qoff + e);
    }
  }
  float m = -CUDART_INF_F, l = 0.0f;

  // Non-causal: every warp scans keys [0, t). Causal: up to the max query in this
  // block; each warp then stops at its own qi (same 0..jmax order as the warp kernel).
  const int64_t block_qmax =
      min(static_cast<int64_t>(blockIdx.x) * kFlashBr + (kFlashBr - 1), t - 1);
  const int64_t key_end = causal ? (block_qmax + 1) : t;

  for (int64_t c0 = 0; c0 < key_end; c0 += kFlashBc) {
    const int tile = static_cast<int>(min(static_cast<int64_t>(kFlashBc), key_end - c0));
    // Cooperative load of this K/V tile into shared memory (all warps participate).
    __syncthreads();
    for (int idx = static_cast<int>(threadIdx.x); idx < tile * static_cast<int>(d);
         idx += nthreads) {
      const int jj = idx / static_cast<int>(d);
      const int e = idx % static_cast<int>(d);
      const int64_t off = ((c0 + jj) * hk + g) * d + e;
      sK[idx] = key[off];
      sV[idx] = value[off];
    }
    __syncthreads();
    if (!active) continue;
    // This warp's online-softmax update over the tile — bit-identical math to
    // AttentionWarpKernel, K/V now read from shared memory.
    const int64_t jstop = causal ? min(static_cast<int64_t>(tile), qi - c0 + 1)
                                 : static_cast<int64_t>(tile);
    for (int64_t j = 0; j < jstop; ++j) {
      const int64_t base = j * d;
      float part = 0.0f;
#pragma unroll
      for (int k = 0; k < kFlashMaxPerLane; ++k) {
        const int e = lane + 32 * k;
        if (k < npl && e < d) part += qreg[k] * Load(sK, base + e);
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
      const float s = part * scale;
      const float m_new = fmaxf(m, s);
      const float corr = expf(m - m_new);
      const float p = expf(s - m_new);
#pragma unroll
      for (int k = 0; k < kFlashMaxPerLane; ++k) {
        const int e = lane + 32 * k;
        if (k < npl && e < d) acc[k] = acc[k] * corr + p * Load(sV, base + e);
      }
      l = l * corr + p;
      m = m_new;
    }
  }
  if (!active) return;
  const int64_t qoff = (qi * hq + h) * d;
  const float inv = 1.0f / l;
  for (int k = 0; k < npl; ++k) {
    const int e = lane + 32 * k;
    if (e < d) Store(out, qoff + e, acc[k] * inv);
  }
}

template <typename Tin>
void LaunchAttentionDenseFlash(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& key,
                               const Tensor& value, const AttentionArgs& args) {
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  if (t == 0 || hq == 0 || d == 0) return;
  // The honest bound, not the register bound. `kFlashMaxPerLane` allows head_dim 256, but
  // the K/V tile below asks for `2*kFlashBc*d*sizeof(Tin)` bytes of DYNAMIC shared
  // memory, and no `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemory
  // Size, ...)` exists anywhere in src/vt/cuda/ — so the driver caps the request at
  // its default 48 KiB and the real ceiling is 192 (bf16) / 96 (f32). This op used to
  // advertise `d <= 256` and hand a wider caller a bare launch failure from the
  // `cudaGetLastError` at the bottom of this function, naming nothing it could do
  // about it (#1544). REFUSING here rather than falling back to AttentionDenseFast is
  // deliberate: that rung re-reads K and V from global once per (query, head), which
  // is the exact redundancy this kernel exists to remove, so taking it silently would
  // be an unannounced slowdown. Name it and let the caller choose.
  const int64_t dmax = AttentionDenseFlashMaxHeadDim(static_cast<int64_t>(sizeof(Tin)));
  VT_CHECK(d <= dmax,
           std::string("cuda attention-dense-flash: head_dim ") + std::to_string(d) +
               " needs " +
               std::to_string(AttentionDenseFlashSmemBytes(
                   d, static_cast<int64_t>(sizeof(Tin)))) +
               " bytes of dynamic shared memory, over CUDA's default cap of " +
               std::to_string(kCudaDefaultDynamicSmemBytes) +
               "; this kernel serves head_dim <= " + std::to_string(dmax) +
               " for this input dtype. Use vt::AttentionDenseFast, which uses no "
               "shared memory and serves head_dim <= " +
               std::to_string(kAttentionDenseMaxHeadDim));
  const unsigned nblk = static_cast<unsigned>((t + kFlashBr - 1) / kFlashBr);
  const dim3 grid(nblk, static_cast<unsigned>(hq));
  // The request the guard above admitted. They are two functions, and
  // AttentionDenseFlashMaxHeadDim RE-DERIVES the division rather than inverting
  // AttentionDenseFlashSmemBytes, so agreement is a property to be tested, not one
  // the code makes structural. tests/vt/test_ops_attention.cpp pins it in both
  // directions at the inclusive edge; mutating the `2 *` in SmemBytes to `3 *` turns
  // those cases RED, which is what keeps the two halves honest. Open-coding the byte
  // count here instead is how this contract drifted the first time.
  const size_t shmem = static_cast<size_t>(
      AttentionDenseFlashSmemBytes(d, static_cast<int64_t>(sizeof(Tin))));  // sK + sV
  switch (out.dtype) {
    case DType::kF32:
      AttentionDenseFlashKernel<Tin, float><<<grid, kFlashBr * 32, shmem, s>>>(
          out.Ptr<float>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), hq, hk, d, t,
          args.scale, args.causal);
      break;
    case DType::kBF16:
      AttentionDenseFlashKernel<Tin, __nv_bfloat16><<<grid, kFlashBr * 32, shmem, s>>>(
          out.Ptr<__nv_bfloat16>(), query.Ptr<Tin>(), key.Ptr<Tin>(), value.Ptr<Tin>(), hq, hk, d,
          t, args.scale, args.causal);
      break;
    default: VT_CHECK(false, "cuda attention-dense-flash: unsupported out dtype");
  }
  Check(cudaGetLastError(), "attention-dense-flash launch");
}

void AttentionDenseFlashKernelCuda(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                                   const Tensor& value, const AttentionArgs& args) {
  switch (query.dtype) {
    case DType::kF32:
      LaunchAttentionDenseFlash<float>(AsStream(q), out, query, key, value, args);
      break;
    case DType::kBF16:
      LaunchAttentionDenseFlash<__nv_bfloat16>(AsStream(q), out, query, key, value, args);
      break;
    default:
      VT_CHECK(false, "cuda attention-dense-flash: unsupported input dtype (f32/bf16 only)");
  }
}

// AttentionDenseFa2 — the same dense non-causal contract run on the VENDORED
// FlashAttention-2 forward's tensor cores (multimodal-speed §17). The scalar
// AttentionDenseFlash above tiles K/V through shared memory but still walks the keys
// with one warp per query through a dependent online-softmax chain, which is
// serial-latency-bound; FA-2 replaces that with an `mma.sync` block reduction, and it
// is the kernel vLLM itself dispatches for this shape.
//
// The fast path is bf16, head_dim 64 or 128, non-causal, MHA, and the vendored
// kernels compiled in — the set of compiled NON-SPLIT instantiations, which is
// `run_mha_fwd_<bfloat16_t, {64,128}, false>` and nothing else. head_dim 64 came
// with the Whisper encoder (multimodal-speed §17); head_dim 128 came with the
// LTX-2.5 DiT (#1551), whose video stream is 32 heads x 128 over 2352 tokens and
// which could not reach a tensor core until it existed. Anything outside that set
// falls through to AttentionDenseFlash, and every such caller is byte-identical to
// the flash-tiled path by construction.
//
// WIDENING THIS GATE IS NOT FREE AND MUST NOT BE GUESSED. The head dim is a
// template parameter: admitting one the vendored TU does not instantiate is a link
// error at best, and admitting a shape the launcher's other guards do not cover is
// a wrong answer. Add the instantiation and its CMake entry first, then this test.
//
// The fall-through is not a promise that every shape runs. AttentionDenseFlash has
// its own head_dim domain — the K/V tile's shared-memory request against CUDA's
// default 48 KiB cap, so 192 in bf16 and 96 in f32 (#1544) — and refuses above it
// naming vt::AttentionDenseFast. A shape wider than that reaches a NAMED refusal
// through here, not a kernel.
//
// The two domains are NOT nested, and #1551 is what makes that matter. The vendored
// launcher raises its own dynamic shared-memory cap
// (flash_attn/src/flash_fwd_launch_template.h:85-88), so the FA-2 arm has no 48 KiB
// bound at all, while the fall-through arm does. bf16 head_dim 128 is inside both.
// f32 head_dim 128 is inside NEITHER — f32 is not an FA-2 shape and its 65,536 B
// tile does not fit — so it reaches AttentionDenseFlash's named refusal. That is
// the LTX-2.5 f32 parity arm at production geometry, already disclosed and owned by
// #1612; production renders bf16.
#ifdef VLLM_CPP_FLASH_ATTN
// Same-binary A/B + RED knob, mirroring VT_FA2_PREFILL / VT_FA2_DECODE
// (cuda_paged_attn.cu): VT_FA2_DENSE=0 restores the scalar flash-tiled kernel so both
// arms run from one build. DEFAULT ON (parity-enabler-as-default). Read fresh each
// call — this is a host path taken once per encoder layer.
bool Fa2DenseEnabled() {
  const char* e = std::getenv("VT_FA2_DENSE");
  return e == nullptr || e[0] != '0';
}
#endif

void AttentionDenseFa2KernelCuda(Queue& q, Tensor& out, const Tensor& query, const Tensor& key,
                                 const Tensor& value, const AttentionArgs& args) {
#ifdef VLLM_CPP_FLASH_ATTN
  const int64_t fa2_d = query.shape[2];
  const bool fa2_shape = query.dtype == DType::kBF16 && key.dtype == DType::kBF16 &&
                         value.dtype == DType::kBF16 && out.dtype == DType::kBF16 &&
                         (fa2_d == 64 || fa2_d == 128) && key.shape[1] == query.shape[1] &&
                         value.shape[1] == query.shape[1] && !args.causal;
  if (fa2_shape && Fa2DenseEnabled()) {
    // args.causal is false here (fa2_shape requires !args.causal); it is threaded
    // through so the launcher can REFUSE a causal request instead of answering a
    // different question. See LaunchDenseFA2Bf16's causal guard.
    LaunchDenseFA2Bf16(AsStream(q), out, query, key, value, args.scale, args.causal);
    return;
  }
#endif
  AttentionDenseFlashKernelCuda(q, out, query, key, value, args);
}

// ---------------------------------------------------------------------------
// fused_chain (TDR): the Tier-1 single-pass INTERPRETER over the canonical
// (out, x, weight, residual) 4-operand shape. The Tier-0 composite is device-
// agnostic (ops.cpp), dispatching each opcode to the standalone vt:: op. This
// kernel is reached ONLY for Tier-1-able recipes (steps in {kAdd,kMul,kSilu,
// kSigmoid,kRmsNorm}) when VT_FUSED_TIER=1; the general FusedChain wrapper
// resolves the canonical operand order [0=x,1=weight,2=residual,3=out] and
// forwards here. One block per row, shared-mem tree reduction (the
// RmsNormRowKernel skeleton). Bit-for-bit equal to the CUDA RmsNorm(residual)
// golden: same f32 tree reduction, gemma (1+w), Tres-rounded residual add.
// Operand indices resolve to typed pointers: 0->Tin x, 1->Tin w, 2->Tres res,
// 3->Tout out. The recipe POD is passed BY VALUE into the kernel (small, no heap).

template <typename Tin, typename Tout, typename Tres>
struct FusedCtx {
  const Tin* x;
  const Tin* w;
  Tres* res;
  Tout* out;
  int64_t h;
};

template <typename Tin, typename Tout, typename Tres>
__device__ inline float FusedLoadDev(uint8_t idx, const FusedCtx<Tin, Tout, Tres>& c, int64_t row,
                                     int64_t j) {
  switch (idx) {
    case 0: return Load(c.x, row * c.h + j);
    case 1: return Load(c.w, j);
    case 2: return Load(c.res, row * c.h + j);
    case 3: return Load(c.out, row * c.h + j);
    default: return 0.0f;  // read-only / out-of-range (validated host-side)
  }
}

template <typename Tin, typename Tout, typename Tres>
__device__ inline void FusedStoreDev(uint8_t idx, const FusedCtx<Tin, Tout, Tres>& c, int64_t row,
                                     int64_t j, float v) {
  switch (idx) {
    case 2: Store(c.res, row * c.h + j, v); break;
    case 3: Store(c.out, row * c.h + j, v); break;
    default: break;  // read-only operands (validated host-side)
  }
}

__device__ inline float FSigmoidDev(float x) { return 1.0f / (1.0f + expf(-x)); }

// Tier 1 — interpreter: one block per row walks the whole recipe.
template <typename Tin, typename Tout, typename Tres>
__global__ void FusedChainInterpKernel(FusedCtx<Tin, Tout, Tres> c, float eps, FusedRecipe r) {
  const int64_t row = blockIdx.x;
  const int64_t h = c.h;
  __shared__ float partial[kBlock];
  for (int s = 0; s < r.n; ++s) {
    const FStep st = r.steps[s];
    if (st.op == FOp::kRmsNorm) {
      float acc = 0.0f;
      for (int64_t j = threadIdx.x; j < h; j += kBlock) {
        const float v = FusedLoadDev(st.in[0], c, row, j);
        acc += v * v;  // kMeanSquare, f32
      }
      partial[threadIdx.x] = acc;
      __syncthreads();
      for (int stride = kBlock / 2; stride > 0; stride /= 2) {
        if (static_cast<int>(threadIdx.x) < stride)
          partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
      }
      const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);
      for (int64_t j = threadIdx.x; j < h; j += kBlock) {
        const float v = FusedLoadDev(st.in[0], c, row, j);
        float wj = FusedLoadDev(st.in[1], c, row, j);
        if (st.gemma) wj += 1.0f;
        FusedStoreDev(st.out, c, row, j, v * inv * wj);
      }
      __syncthreads();
    } else if (st.op == FOp::kSilu || st.op == FOp::kSigmoid) {  // unary elementwise
      for (int64_t j = threadIdx.x; j < h; j += kBlock) {
        const float a = FusedLoadDev(st.in[0], c, row, j);
        FusedStoreDev(st.out, c, row, j, st.op == FOp::kSilu ? a * FSigmoidDev(a) : FSigmoidDev(a));
      }
      __syncthreads();
    } else {  // kAdd / kMul — binary elementwise over the row
      for (int64_t j = threadIdx.x; j < h; j += kBlock) {
        const float a = FusedLoadDev(st.in[0], c, row, j);
        const float b = FusedLoadDev(st.in[1], c, row, j);
        FusedStoreDev(st.out, c, row, j, st.op == FOp::kAdd ? a + b : a * b);
      }
      __syncthreads();  // writes (e.g. residual) visible before the next step reads
    }
  }
}

template <typename Tin, typename Tout, typename Tres>
void LaunchFusedInterp(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& weight,
                       Tensor* residual, const FusedRecipe& r, float eps) {
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return;
  FusedCtx<Tin, Tout, Tres> c{x.Ptr<Tin>(), weight.Ptr<Tin>(),
                              residual == nullptr ? nullptr : residual->Ptr<Tres>(),
                              out.Ptr<Tout>(), h};
  FusedChainInterpKernel<Tin, Tout, Tres><<<static_cast<unsigned>(t), kBlock, 0, s>>>(c, eps, r);
  Check(cudaGetLastError(), "fused_chain interp launch");
}

// Resolve the (Tin, Tout, Tres) triple and launch the interpreter. Tres follows
// the residual dtype (f32 or bf16); with no residual it is unused (pass float).
template <typename Tin, typename Tout>
void LaunchFusedInterpRes(cudaStream_t s, Tensor& out, const Tensor& x, const Tensor& weight,
                          Tensor* residual, const FusedRecipe& r, float eps) {
  if (residual != nullptr && residual->dtype == DType::kBF16) {
    LaunchFusedInterp<Tin, Tout, __nv_bfloat16>(s, out, x, weight, residual, r, eps);
  } else if (residual != nullptr && residual->dtype == DType::kF16) {
    LaunchFusedInterp<Tin, Tout, __half>(s, out, x, weight, residual, r, eps);
  } else {
    LaunchFusedInterp<Tin, Tout, float>(s, out, x, weight, residual, r, eps);
  }
}

// Registered kernel: the Tier-1 interpreter (the general wrapper only dispatches
// here for Tier-1-able recipes; Tier-0 composite is device-agnostic in ops.cpp).
void FusedChainKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                          Tensor* residual, const FusedRecipe& r, float eps) {
  VT_CHECK(weight.dtype == x.dtype, "cuda fused_chain: weight dtype must match x");
  cudaStream_t s = AsStream(q);
  switch (x.dtype) {
    case DType::kF32:
      switch (out.dtype) {
        case DType::kF32:
          LaunchFusedInterpRes<float, float>(s, out, x, weight, residual, r, eps);
          break;
        case DType::kF16:
          LaunchFusedInterpRes<float, __half>(s, out, x, weight, residual, r, eps);
          break;
        case DType::kBF16:
          LaunchFusedInterpRes<float, __nv_bfloat16>(s, out, x, weight, residual, r, eps);
          break;
        default: VT_CHECK(false, "cuda fused_chain: unsupported out dtype");
      }
      break;
    case DType::kF16:
      switch (out.dtype) {
        case DType::kF16: LaunchFusedInterpRes<__half, __half>(s, out, x, weight, residual, r, eps); break;
        case DType::kBF16: LaunchFusedInterpRes<__half, __nv_bfloat16>(s, out, x, weight, residual, r, eps); break;
        case DType::kF32: LaunchFusedInterpRes<__half, float>(s, out, x, weight, residual, r, eps); break;
        default: VT_CHECK(false, "cuda fused_chain: unsupported out dtype");
      }
      break;
    case DType::kBF16:
      switch (out.dtype) {
        case DType::kF32:
          LaunchFusedInterpRes<__nv_bfloat16, float>(s, out, x, weight, residual, r, eps);
          break;
        case DType::kF16:
          LaunchFusedInterpRes<__nv_bfloat16, __half>(s, out, x, weight, residual, r, eps);
          break;
        case DType::kBF16:
          LaunchFusedInterpRes<__nv_bfloat16, __nv_bfloat16>(s, out, x, weight, residual, r, eps);
          break;
        default: VT_CHECK(false, "cuda fused_chain: unsupported out dtype");
      }
      break;
    default: VT_CHECK(false, "cuda fused_chain: unsupported input dtype (f32/bf16 only)");
  }
}

// ---------------------------------------------------------------------------
// DFlash2 grouped dynamic depthwise convolution (SPEC-DFLASH2 W2, #1314) — the
// CUDA MIRROR of the CPU reference `DFlashGroupedConvKernel` (cpu_ops.cpp),
// which carries the full port note and the upstream anchor.
//
// One thread per (row, channel). Every step is elementwise, so this kernel has
// NO reduction-order freedom and is asserted BIT-IDENTICAL to the CPU reference
// rather than within an envelope
// (tests/vt/test_ops_dflash2_grouped_conv.cpp).
//
// The two rounding helpers are `__fadd_rn`/`__fmul_rn` rather than `+`/`*` on
// purpose: on the f32 arm `ResRound` is the identity, so `acc + k * x` is an
// FMA contraction pattern and nvcc would fuse it by default, which the CPU
// reference (built with `-ffp-contract=off`) does not. The intrinsics forbid
// the contraction, so the two arms answer the same bits.
template <typename T>
__global__ void DFlashGroupedConvKernel(T* out, const T* x, const T* coefficients, const T* base,
                                        int64_t rows, int64_t h, int64_t taps, int64_t groups,
                                        int64_t gsize, int64_t sides, int64_t side,
                                        int64_t block, bool pot) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= rows * h) return;
  const int64_t i = idx / h;
  const int64_t c = idx - i * h;
  const int64_t g = c / gsize;
  const int64_t pos = pot ? (i & (block - 1)) : (i % block);
  float acc = 0.0f;
  for (int64_t t = 0; t < taps && t <= pos; ++t) {
    const float b = Load(base, (side * taps + t) * h + c);
    const float d = Load(coefficients, ((i * sides + side) * taps + t) * groups + g);
    const float k = ResRound<T>(__fadd_rn(b, d));
    const float term = ResRound<T>(__fmul_rn(k, Load(x, (i - t) * h + c)));
    acc = (t == 0) ? term : ResRound<T>(__fadd_rn(acc, term));
  }
  Store(out, idx, acc);
}

void DFlashGroupedConvKernelCuda(Queue& q, Tensor& out, const Tensor& x,
                                 const Tensor& coefficients, const Tensor& base,
                                 const DFlashGroupedConvArgs& args) {
  const int64_t rows = x.shape[0];
  const int64_t h = args.num_groups * args.group_size;
  const int64_t sides = coefficients.shape[1];
  const int64_t n = rows * h;
  if (n == 0) return;
  constexpr int kBlock = 256;
  const int64_t grid = (n + kBlock - 1) / kBlock;
  const bool pot = (args.block_size & (args.block_size - 1)) == 0;
  cudaStream_t s = AsStream(q);
  switch (x.dtype) {
    case DType::kF32:
      DFlashGroupedConvKernel<float><<<static_cast<unsigned>(grid), kBlock, 0, s>>>(
          out.Ptr<float>(), x.Ptr<float>(), coefficients.Ptr<float>(), base.Ptr<float>(), rows, h,
          args.taps, args.num_groups, args.group_size, sides, args.side, args.block_size, pot);
      break;
    case DType::kBF16:
      DFlashGroupedConvKernel<__nv_bfloat16><<<static_cast<unsigned>(grid), kBlock, 0, s>>>(
          out.Ptr<__nv_bfloat16>(), x.Ptr<__nv_bfloat16>(), coefficients.Ptr<__nv_bfloat16>(),
          base.Ptr<__nv_bfloat16>(), rows, h, args.taps, args.num_groups, args.group_size, sides,
          args.side, args.block_size, pot);
      break;
    default:
      VT_CHECK(false, "cuda dflash2-grouped-conv: unsupported dtype (f32/bf16 only)");
  }
  VT_CHECK(cudaGetLastError() == cudaSuccess, "cuda dflash2-grouped-conv: launch failed");
}

// ---------------------------------------------------------------------------
// DFlash2 candidate-selector edge lattice (SPEC-DFLASH2 W3, #1314) — the CUDA
// MIRROR of the CPU reference `Dflash2SelectorEdgesKernel` (cpu_ops.cpp), which
// carries the full port note and the upstream anchor.
//
// One block per (request, step, predecessor slot). The block first materializes
// upstream's `predecessors * hidden[:, :, None]` for its own slot into dynamic
// shared memory — rounded to the codebook dtype, because upstream materializes
// that tensor — and then block-reduces one rank contraction per child candidate.
//
// UNLIKE vt::DFlashGroupedConv this kernel is NOT bit-identical to the CPU
// reference and is not specified to be: the rank contraction is a REDUCTION, and
// this tree reduction sums in a different order than the CPU reference's serial
// loop. It is gated within an f32 envelope. The two ROUNDING PLACEMENTS are
// still exact — the elementwise product and the single round of the completed
// sum — because those are what upstream's materializations pin.
template <typename T>
__global__ void Dflash2SelectorEdgesKernel(float* scores, const T* pred_codebook,
                                           const T* succ_codebook, const int64_t* cand,
                                           const float* unary, const T* hidden,
                                           const int64_t* anchors, int64_t L, int64_t K,
                                           int64_t R) {
  extern __shared__ float gated[];
  const int64_t slot = blockIdx.x;
  const int64_t p = slot % K;
  const int64_t idx = slot / K;  // flattened (b, l)
  const int64_t b = idx / L, l = idx - b * L;
  const int64_t pid = (l == 0) ? anchors[b] : cand[(b * L + (l - 1)) * K + p];
  for (int64_t r = threadIdx.x; r < R; r += blockDim.x)
    gated[r] = ResRound<T>(__fmul_rn(Load(pred_codebook, pid * R + r),
                                     Load(hidden, idx * R + r)));
  __syncwarp();
  for (int64_t c = 0; c < K; ++c) {
    const int64_t cid = cand[idx * K + c];
    float acc = 0.0f;
    for (int64_t r = threadIdx.x; r < R; r += blockDim.x)
      acc += gated[r] * Load(succ_codebook, cid * R + r);
    // ONE WARP per block, so the contraction reduces with __shfl_xor_sync and
    // needs no shared scratch and no __syncthreads inside this loop.
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
    if (threadIdx.x == 0)
      scores[idx * K * K + p * K + c] = unary[idx * K + c] + ResRound<T>(acc);
  }
}

void Dflash2SelectorEdgesKernelCuda(Queue& q, Tensor& scores, const Tensor& pred_codebook,
                                    const Tensor& succ_codebook, const Tensor& candidate_ids,
                                    const Tensor& unary, const Tensor& hidden,
                                    const Tensor& anchors,
                                    const Dflash2SelectorEdgesArgs& args) {
  const int64_t B = candidate_ids.shape[0], L = candidate_ids.shape[1];
  const int64_t K = args.top_k, R = pred_codebook.shape[1];
  const int64_t blocks = B * L * K;
  if (blocks == 0) return;
  constexpr int kThreads = 32;  // ONE warp: the contraction reduces by shuffle
  const size_t shared = static_cast<size_t>(R) * sizeof(float);
  cudaStream_t s = AsStream(q);
  switch (pred_codebook.dtype) {
    case DType::kF32:
      Dflash2SelectorEdgesKernel<float>
          <<<static_cast<unsigned>(blocks), kThreads, shared, s>>>(
              scores.Ptr<float>(), pred_codebook.Ptr<float>(), succ_codebook.Ptr<float>(),
              candidate_ids.Ptr<int64_t>(), unary.Ptr<float>(), hidden.Ptr<float>(),
              anchors.Ptr<int64_t>(), L, K, R);
      break;
    case DType::kBF16:
      Dflash2SelectorEdgesKernel<__nv_bfloat16>
          <<<static_cast<unsigned>(blocks), kThreads, shared, s>>>(
              scores.Ptr<float>(), pred_codebook.Ptr<__nv_bfloat16>(),
              succ_codebook.Ptr<__nv_bfloat16>(), candidate_ids.Ptr<int64_t>(),
              unary.Ptr<float>(), hidden.Ptr<__nv_bfloat16>(), anchors.Ptr<int64_t>(), L, K, R);
      break;
    default:
      VT_CHECK(false, "cuda dflash2-selector-edges: unsupported dtype (f32/bf16 only)");
  }
  VT_CHECK(cudaGetLastError() == cudaSuccess, "cuda dflash2-selector-edges: launch failed");
}

// MIRROR of the CPU reference `Dflash2PathWalkKernel` (cpu_ops.cpp), which
// carries the full port note. `Dflash2PathWalkArgs` (include/vt/ops.h) carries
// the contract.
//
// ONE BLOCK PER REQUEST, which is upstream's own grid: `_selector_walk_kernel`
// launches `(num_reqs,)` programs with `num_warps=1` and keeps the step loop
// INSIDE the program, so a k-token block costs one launch instead of k. Spec
// `## Risks/decisions` D3 requires that shape from the first landing -- the same
// sequential walk shipped host-side in DSpark and measured 28% of the 27B draft
// step (#436) before it was moved.
//
// The block is one warp. `previous` is a per-thread register that every lane
// computes identically from the same shuffle reduction, so the carry needs no
// shared memory and no __syncthreads: after the __shfl_xor_sync butterfly every
// lane holds the same winner.
//
// BIT-EXACT with the CPU arm by construction, unlike Dflash2SelectorEdgesKernel:
// there is no arithmetic here to reorder, only comparisons and a gather. The
// reduction is seeded the same way on both arms (-inf with slot index `top_k`,
// strict `>`), which is what makes the all -inf row, the tie rows and a
// NaN-bearing row agree rather than merely usually agreeing.
//
// WHERE THE TIE RULE LIVES IS NOT ARBITRARY. The per-lane scan is STRICT ONLY,
// because `j` ascends within a lane and a strict `>` therefore already keeps
// the LOWEST-indexed maximum -- the CPU arm's answer, reached the CPU arm's
// way. The lower-slot preference belongs to the cross-lane BUTTERFLY, which
// combines lane winners in no particular slot order and would otherwise resolve
// a tie by lane geometry. Until W4's fresh review the lane scan ALSO carried
// `|| (v == best && j < slot)`. That clause is unreachable once a lane has
// claimed anything, so its only effect was at the SEED: a lane holding -inf
// compared equal to the -inf seed and claimed a slot, which the CPU arm's
// strict scan refuses. On a NaN-bearing row the arms then answered differently
// (`[NaN,-inf]` read cpu 0 / cuda 1) while every NaN-free row still agreed, so
// no fixture without a NaN could see it. The clause is deleted rather than the
// bit-exactness claim narrowed. THE DELETION IS VERIFIED, and #1518 corrects an
// earlier sentence here calling it unverified: this translation unit is
// compiled for ten architectures by CI's `build-cuda-fat` job on every pull
// request, and the operator RAN `test_ops_dflash2_path_walk` on `dgx:gpu0`
// (GB10, sm_121a) at the W4 merge commit -- 83 assertions on device against 49
// on CPU, `Status: SUCCESS!`, zero `no CUDA backend; skipping` lines, the NaN
// row among them. The AUTHORING HOST has no `nvcc` and still skips the case
// locally, and the remainder of `## Owed` O11 in
// .agents/specs/dflash2-spec-decode.md stands.
__global__ void Dflash2PathWalkKernel(int64_t* tokens, const float* scores,
                                      const int64_t* cand, int64_t L, int64_t K) {
  const int64_t b = blockIdx.x;
  const int lane = static_cast<int>(threadIdx.x);
  const int width = static_cast<int>(blockDim.x);
  int64_t previous = 0;
  for (int64_t l = 0; l < L; ++l) {
    const int64_t flat = b * L + l;
    const float* row = scores + (flat * K + previous) * K;
    float best = -CUDART_INF_F;
    int slot = static_cast<int>(K);
    for (int64_t j = lane; j < K; j += width) {
      const float v = row[j];
      // STRICT `>` and nothing else, which is the CPU arm's own scan: `j`
      // ascends, so the first maximum a lane meets is the lowest-indexed one it
      // holds. A `v == best` arm here would let a lane claim a slot on the -inf
      // seed, which is where the two arms used to part on a NaN row.
      if (v > best) {
        best = v;
        slot = static_cast<int>(j);
      }
    }
    // The lower slot wins an exact tie HERE, because lanes combine out of slot
    // order. `best == -inf` implies `slot == top_k` on every lane (the strict
    // scan above never claims on -inf), so the tie arm compares real slots or
    // two seeds and never a real slot against a seed.
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(0xFFFFFFFFu, best, off);
      const int os = __shfl_xor_sync(0xFFFFFFFFu, slot, off);
      if (ov > best || (ov == best && os < slot)) {
        best = ov;
        slot = os;
      }
    }
    if (slot == static_cast<int>(K)) slot = 0;  // an all -inf (fully masked) row
    previous = slot;
    if (lane == 0) tokens[flat] = cand[flat * K + slot];
  }
}

void Dflash2PathWalkKernelCuda(Queue& q, Tensor& tokens, const Tensor& scores,
                               const Tensor& candidate_ids,
                               const Dflash2PathWalkArgs& args) {
  const int64_t B = candidate_ids.shape[0], L = candidate_ids.shape[1];
  const int64_t K = args.top_k;
  if (B == 0 || L == 0) return;
  constexpr int kThreads = 32;  // ONE warp: the argmax reduces by shuffle
  Dflash2PathWalkKernel<<<static_cast<unsigned>(B), kThreads, 0, AsStream(q)>>>(
      tokens.Ptr<int64_t>(), scores.Ptr<float>(), candidate_ids.Ptr<int64_t>(), L, K);
  VT_CHECK(cudaGetLastError() == cudaSuccess, "cuda dflash2-path-walk: launch failed");
}

// Registers the CUDA kernels during static init (pre-main, like the CPU ops).
// Filling the op table is harmless on machines without a GPU: the kCUDA
// backend never registers there, so no CUDA queue can exist to dispatch with.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kRmsNorm, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernelCuda)));
    RegisterOpScalarType(OpId::kRmsNorm, DeviceType::kCUDA, kNativeProviderName, DType::kF16);
    RegisterOpScalarType(OpId::kRmsNorm, DeviceType::kCUDA, kNativeProviderName, DType::kF32);
    RegisterOpScalarType(OpId::kRmsNorm, DeviceType::kCUDA, kNativeProviderName, DType::kBF16);
    RegisterOp(OpId::kRmsNormQuantFp8, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<RmsNormQuantFp8Fn>(&RmsNormQuantFp8KernelCuda)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernelCuda)));
    RegisterOpScalarType(OpId::kSiluAndMul, DeviceType::kCUDA, kNativeProviderName, DType::kF16);
    RegisterOp(OpId::kClampedSwiGLU, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<ClampedSwiGLUFn>(&ClampedSwiGLUKernelCuda)));
    RegisterOp(OpId::kGeluAndMul, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GeluAndMulFn>(&GeluAndMulKernelCuda)));
    RegisterOp(OpId::kMulScalar, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MulScalarFn>(&MulScalarKernelCuda)));
    RegisterOp(OpId::kSoftCap, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<SoftCapFn>(&SoftCapKernelCuda)));
    RegisterOp(OpId::kEmbedding, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernelCuda)));
    RegisterOpScalarType(OpId::kEmbedding, DeviceType::kCUDA, kNativeProviderName, DType::kF16);
    // KGATHER. `EmbeddingKernelCuda` branches on `IsBlockQuant(table.dtype)` and
    // routes a block table to `LaunchEmbeddingQuant` (decoders in
    // vt/cuda/cuda_quant_dequant.cuh). Registering it under the quant id is what
    // makes this device's block-gather capability visible to `OpRegistered`,
    // which is what `DeviceQuantGatherSupported` asks -- so this call IS the
    // GGUF residency flip, and there is no separate boolean anywhere.
    //
    // It was withheld until a GPU had executed the decoders, because registering
    // it early would route EVERY GGUF model's block-typed gather table on CUDA
    // into never-executed code, with no throw and no log if a decode were wrong.
    // MEASURED on thor:gpu0 (Jetson Thor, sm_110, nvcc 13.0.88, aarch64),
    // 2026-08-31, job e53a20f5: the RED leg without this arm failed BY NAME on
    // 4 of 5 cases at 32 assertions ("cuda embedding: unsupported table dtype"),
    // and with it `tests/vt/test_cuda_embedding_quant.cpp` passed 6 of 6 cases
    // at 231 of 231 assertions -- every one of the 18 block encodings gathered
    // BIT-EXACTLY against the CPU arm, in f32 and bf16 out, i32 and i64 ids.
    // Deleting this call reds that gate, which is how the mutation leg proved it
    // is the reachable call site and not a duplicate.
    RegisterCudaBlockGather();
    RegisterOp(OpId::kRopeNeox, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernelCuda)));
    RegisterOpScalarType(OpId::kRopeNeox, DeviceType::kCUDA, kNativeProviderName, DType::kF16);
    RegisterOp(
        OpId::kRopeFromCache, DeviceType::kCUDA,
        reinterpret_cast<void*>(
            static_cast<RopeFromCacheFn>(&RopeFromCacheKernelCuda)));
    RegisterOpScalarType(OpId::kRopeFromCache, DeviceType::kCUDA, kNativeProviderName, DType::kF16);
    RegisterOp(
        OpId::kFusedNormRope, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<FusedNormRopeFn>(&FusedNormRopeKernelCuda)));
    RegisterOp(
        OpId::kRopeCosSinCache, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernelCuda)));
    RegisterOp(OpId::kAttnQkNormRopeGate, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<AttnQkNormRopeGateFn>(&AttnQkNormRopeGateKernelCuda)));
    RegisterOp(OpId::kAttention, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<AttentionFn>(&AttentionKernelCuda)));
    RegisterOp(OpId::kAttentionDenseFast, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<AttentionFn>(&AttentionDenseFastKernelCuda)));
    RegisterOp(OpId::kAttentionDenseFlash, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<AttentionFn>(&AttentionDenseFlashKernelCuda)));
    RegisterOp(OpId::kAttentionDenseFa2, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<AttentionFn>(&AttentionDenseFa2KernelCuda)));
    RegisterOp(OpId::kDFlashBlockAttention, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<DFlashBlockAttentionFn>(&DFlashBlockAttentionKernelCuda)));
    RegisterOp(OpId::kDFlashPagedBlockAttention, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<DFlashPagedBlockAttentionFn>(&DFlashPagedBlockAttentionKernelCuda)));
    RegisterOp(OpId::kFusedChain, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernelCuda)));
    RegisterOpScalarType(OpId::kFusedChain, DeviceType::kCUDA, kNativeProviderName, DType::kF16);
    RegisterOp(OpId::kDFlashGroupedConv, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<DFlashGroupedConvFn>(&DFlashGroupedConvKernelCuda)));
    RegisterOp(OpId::kDflash2PathWalk, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<Dflash2PathWalkFn>(&Dflash2PathWalkKernelCuda)));
    RegisterOp(OpId::kDflash2SelectorEdges, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<Dflash2SelectorEdgesFn>(&Dflash2SelectorEdgesKernelCuda)));
  }
} registrar;

}  // namespace
// See vt/cuda/cuda_embedding_quant.h. Defined out here, outside the anonymous
// namespace, because `EmbeddingKernelCuda` has internal linkage and the gate
// cannot name it -- this function is the only way in, which is also what keeps
// the flip auditable as a single call site.
void RegisterCudaBlockGather() {
  RegisterOp(OpId::kEmbeddingQuant, DeviceType::kCUDA,
             reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernelCuda)));
    RegisterOpScalarType(OpId::kEmbeddingQuant, DeviceType::kCUDA, kNativeProviderName, DType::kF16);
}

}  // namespace vt::cuda

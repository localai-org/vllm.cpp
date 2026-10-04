// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CUDA Qwen3.6 elementwise "glue" ops (M0.9 forward): the small reshape/split/
// activation fusions that sit between the big decode ops, moved on-device so the
// whole decode step is CUDA-graph capturable. Correctness-grade — plain
// grid-stride kernels matching the CPU reference math in src/vt/cpu/cpu_ops.cpp
// element for element. All math is f32; dims are inferred from tensor shapes.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::cuda {
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

// f32 load/store overloads: bf16 converts on the way in/out, math is f32
// (mirrors cuda_moe.cu; __float2bfloat16 is round-to-nearest-even, same as the
// host F32ToBF16).
__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16(v); }
__device__ inline void Store(__half* p, int64_t i, float v) { p[i] = __float2half(v); }

__device__ inline float SigmoidF(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// cast_bf16: out[i] = bf16(in[i]). Thread per element, grid-stride. Input may be
// a torch.split-style packed view (merged QKV): each logical row is dense while
// row_stride spans the parent Q+K+V tensor (symmetric with CastF32Kernel).
__global__ void CastBf16Kernel(__nv_bfloat16* out, const float* in, int64_t n,
                               int64_t row_size, int64_t row_stride) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step) {
    const int64_t row = i / row_size;
    const int64_t col = i - row * row_size;
    Store(out, i, Load(in, row * row_stride + col));
  }
}

void CastBf16KernelCuda(Queue& q, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  if (n == 0) return;
  const int64_t rows = in.shape[0];
  const int64_t row_size = n / rows;
  CastBf16Kernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(out.Ptr<__nv_bfloat16>(), in.Ptr<float>(),
                                                         n, row_size, in.stride[0]);
  Check(cudaGetLastError(), "cast_bf16 launch");
}

// cast_f16: out[i] = f16(in[i]), from an f32 OR bf16 source. QUANT-EXL3 W1a
// (#2181) — the narrowing cast an EXL3 linear needs on the way in, because
// `Exl3Gemm` reads its activation as fp16 and nothing else. Templated on the
// source so the bf16 arm widens exactly through f32 before the single rounding
// store, rather than reinterpreting. Same packed-view row handling as the two
// casts either side of it.
template <typename Src>
__global__ void CastF16Kernel(__half* out, const Src* in, int64_t n, int64_t row_size,
                              int64_t row_stride) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step) {
    const int64_t row = i / row_size;
    const int64_t col = i - row * row_size;
    Store(out, i, Load(in, row * row_stride + col));
  }
}

void CastF16KernelCuda(Queue& q, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  if (n == 0) return;
  const int64_t rows = in.shape[0];
  const int64_t row_size = n / rows;
  if (in.dtype == DType::kF32) {
    CastF16Kernel<float><<<GridFor(n), kBlock, 0, AsStream(q)>>>(
        out.Ptr<__half>(), in.Ptr<float>(), n, row_size, in.stride[0]);
  } else {
    CastF16Kernel<__nv_bfloat16><<<GridFor(n), kBlock, 0, AsStream(q)>>>(
        out.Ptr<__half>(), in.Ptr<__nv_bfloat16>(), n, row_size, in.stride[0]);
  }
  Check(cudaGetLastError(), "cast_f16 launch");
}

// cast_f32: bf16 -> f32 upcast. Input may be a torch.split-style packed view:
// each logical row is dense, while row_stride spans the parent Q+K+V tensor.
__global__ void CastF32Kernel(float* out, const __nv_bfloat16* in, int64_t n,
                              int64_t row_size, int64_t row_stride) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += step) {
    const int64_t row = i / row_size;
    const int64_t col = i - row * row_size;
    out[i] = __bfloat162float(in[row * row_stride + col]);
  }
}

void CastF32KernelCuda(Queue& q, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  if (n == 0) return;
  const int64_t rows = in.shape[0];
  const int64_t row_size = n / rows;
  CastF32Kernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(out.Ptr<float>(),
                                                        in.Ptr<__nv_bfloat16>(), n,
                                                        row_size, in.stride[0]);
  Check(cudaGetLastError(), "cast_f32 launch");
}

// T25: permute V heads from grouped (k*rpk+r) to tiled (r*num_k+k) order.
// CPU sibling: cpu_ops.cpp PermuteVHeadsKernel.
__global__ void PermuteVHeadsKernel(__nv_bfloat16* out,
                                    const __nv_bfloat16* in, int64_t n,
                                    int64_t value_dim, int64_t num_k,
                                    int64_t rpk, int64_t dv) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += step) {
    const int64_t row = i / value_dim;
    const int64_t rem = i - row * value_dim;
    const int64_t t = rem / dv;
    const int64_t h = rem - t * dv;
    const int64_t r = t / num_k;
    const int64_t k = t - r * num_k;
    const int64_t g = k * rpk + r;
    out[i] = in[row * value_dim + g * dv + h];
  }
}

void PermuteVHeadsKernelCuda(Queue& q, Tensor& out, const Tensor& in,
                             int64_t T, int64_t num_k, int64_t rpk,
                             int64_t dv) {
  const int64_t value_dim = num_k * rpk * dv;
  const int64_t n = T * value_dim;
  if (n == 0) return;
  PermuteVHeadsKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
      out.Ptr<__nv_bfloat16>(), in.Ptr<__nv_bfloat16>(), n, value_dim, num_k,
      rpk, dv);
  Check(cudaGetLastError(), "permute_v_heads launch");
}

// mul_col_vec_f32: x[m,n] *= col[n]. x f32 OR bf16 [M,N] with row stride
// row_stride (inner-contiguous rows), col always f32 [N]. Thread per logical
// element (flat over M*N); recover (row,col-index) and apply the broadcast
// column scalar.
//
// PERF-FP8-ALPHA-FOLD / #417: `Tx` is the STORE width, added as a dtype axis on
// THIS kernel rather than as a second kernel — the same widening
// SigmoidGateBf16Kernel took for its `Tattn` operand just below. The pass is a
// full read-modify-write of the merged FP8 GDN in_proj output, measured
// bandwidth-bound at 209.5 GB/s = 77% of the GB10's 273.1 GB/s peak, so its cost
// IS its width and a bf16 x moves half the bytes.
//
// The ARITHMETIC is unchanged on both arms: Load() upcasts to f32, the multiply
// is the same single IEEE f32 multiply, and only Store() differs. The f32 arm is
// byte-identical to the `*=` it replaces (for float, Store(x,i,Load(x,i)*c) IS
// `x[i] *= c`); the bf16 arm rounds the product to bf16, which is the whole
// point of the narrowing and is why the caller opts in.
//
// `col` stays f32 on BOTH arms deliberately: it is the resident folded alpha
// (input_scale * that shard's weight_scale), it costs [N] bytes and not [M,N],
// so rounding it would perturb every column for no bandwidth saving at all.
template <typename Tx>
__global__ void MulColVecF32Kernel(Tx* x, const float* col, int64_t n_elem,
                                   int64_t row_size, int64_t row_stride) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_elem; i += step) {
    const int64_t row = i / row_size;
    const int64_t c = i - row * row_size;
    const int64_t off = row * row_stride + c;
    Store(x, off, Load(x, off) * col[c]);
  }
}

void MulColVecF32KernelCuda(Queue& q, Tensor& x, const Tensor& col) {
  const int64_t rows = x.shape[0], row_size = x.shape[1];
  const int64_t n_elem = rows * row_size;
  if (n_elem == 0) return;
  switch (x.dtype) {
    case DType::kF32:
      MulColVecF32Kernel<float><<<GridFor(n_elem), kBlock, 0, AsStream(q)>>>(
          x.Ptr<float>(), col.Ptr<float>(), n_elem, row_size, x.stride[0]);
      break;
    case DType::kBF16:
      MulColVecF32Kernel<__nv_bfloat16><<<GridFor(n_elem), kBlock, 0, AsStream(q)>>>(
          x.Ptr<__nv_bfloat16>(), col.Ptr<float>(), n_elem, row_size, x.stride[0]);
      break;
    default: VT_CHECK(false, "cuda mul_col_vec_f32: x must be f32 or bf16");
  }
  Check(cudaGetLastError(), "mul_col_vec_f32 launch");
}

// ---------------------------------------------------------------------------
// attn_gate_split: qgate [T, Hq*2*Dh] -> q_out/gate_out [T,Hq,Dh]. Thread per
// output element (flat index over T*Hq*Dh); (i,h,d) recovered from it.
//
// `Tout` is the QUERY half's store width, and it exists for the same reason
// `Tin` does. `Tin` was templated because VT_BF16_GEMM_OUT makes the `q_proj`
// GEMM emit bf16; `Tout` is templated because vLLM's `torch.chunk` does not
// widen either side of that split (qwen3_next.py:430 at vLLM `cdefd9d499`, an
// unpinned forward reference), so a caller whose qgate is already the bf16 model
// dtype has nothing to gain from an f32 destination and pays twice the bytes for
// it. Both arms read through Load()/Store(), so `Tout = float` is byte-identical
// to the hardcoded `float*` this replaces.
//
// `gate_out` stays `float*` on both arms: `vt::SigmoidGateBf16` takes an f32
// gate on all four of its backends. #2488 carries that half.
template <typename Tout, typename Tin>
__global__ void AttnGateSplitKernel(Tout* q_out, float* gate_out, const Tin* qgate, int64_t t,
                                    int64_t hq, int64_t dh) {
  const int64_t n = t * hq * dh;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t d = idx % dh;
    const int64_t h = (idx / dh) % hq;
    const int64_t i = idx / (dh * hq);
    const int64_t base = i * (hq * 2 * dh) + h * 2 * dh;  // start of (i,h) pair
    Store(q_out, idx, Load(qgate, base + d));
    Store(gate_out, idx, Load(qgate, base + dh + d));
  }
}

template <typename Tin>
void LaunchAttnGateSplit(Queue& q, Tensor& q_out, Tensor& gate_out, const Tensor& qgate, int64_t t,
                         int64_t hq, int64_t dh, int64_t n) {
  switch (q_out.dtype) {
    case DType::kF32:
      AttnGateSplitKernel<float, Tin><<<GridFor(n), kBlock, 0, AsStream(q)>>>(
          q_out.Ptr<float>(), gate_out.Ptr<float>(), qgate.Ptr<Tin>(), t, hq, dh);
      break;
    case DType::kBF16:
      AttnGateSplitKernel<__nv_bfloat16, Tin><<<GridFor(n), kBlock, 0, AsStream(q)>>>(
          q_out.Ptr<__nv_bfloat16>(), gate_out.Ptr<float>(), qgate.Ptr<Tin>(), t, hq, dh);
      break;
    default: VT_CHECK(false, "attn_gate_split: q_out must be f32 or bf16");
  }
}

void AttnGateSplitKernelCuda(Queue& q, Tensor& q_out, Tensor& gate_out, const Tensor& qgate) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t n = t * hq * dh;
  if (n == 0) return;
  // qgate may be f32 (default) or bf16 (VT_BF16_GEMM_OUT — the q_proj GEMM emits bf16),
  // and q_out may be f32 (Qwen3.5, whose qk-norm/RoPE chain reads f32) or bf16 (qwen4_exp,
  // whose model dtype it is). gate_out is f32 on every arm. Load()/Store() convert.
  switch (qgate.dtype) {
    case DType::kF32: LaunchAttnGateSplit<float>(q, q_out, gate_out, qgate, t, hq, dh, n); break;
    case DType::kBF16:
      LaunchAttnGateSplit<__nv_bfloat16>(q, q_out, gate_out, qgate, t, hq, dh, n);
      break;
    default: VT_CHECK(false, "attn_gate_split: qgate must be f32 or bf16");
  }
  Check(cudaGetLastError(), "attn_gate_split launch");
}

// ---------------------------------------------------------------------------
// sigmoid_gate_bf16: out[i] = bf16(attn[i] * sigmoid(gate[i])). Tattn f32 or
// bf16 (the FA-2 prefill path hands a bf16 attention out); the bf16 Load upcast
// is exact, so bf16-attn is bit-identical to f32-attn holding the same
// bf16-representable values. The gate is always f32 (sigmoid input must not be
// rounded).
template <typename Tattn>
__global__ void SigmoidGateBf16Kernel(__nv_bfloat16* out, const Tattn* attn, const float* gate,
                                      int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step)
    Store(out, i, Load(attn, i) * SigmoidF(Load(gate, i)));
}

void SigmoidGateBf16KernelCuda(Queue& q, Tensor& out, const Tensor& attn, const Tensor& gate) {
  const int64_t n = out.Numel();
  if (n == 0) return;
  if (attn.dtype == DType::kBF16) {
    SigmoidGateBf16Kernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
        out.Ptr<__nv_bfloat16>(), attn.Ptr<__nv_bfloat16>(), gate.Ptr<float>(), n);
  } else {
    SigmoidGateBf16Kernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
        out.Ptr<__nv_bfloat16>(), attn.Ptr<float>(), gate.Ptr<float>(), n);
  }
  Check(cudaGetLastError(), "sigmoid_gate_bf16 launch");
}

// ---------------------------------------------------------------------------
// gdn_g_beta (gdn-semantics.md §6): g/beta from raw a/b/A_log/dt_bias. Thread
// per output element (flat idx over T*Hv); hv = idx % Hv indexes a_log/dt_bias.
template <typename Tgate>
__global__ void GdnGBetaKernel(float* g_out, float* beta_out,
                               const Tgate* araw, const Tgate* braw,
                               const float* a_log, const float* dt_bias,
                               int64_t t, int64_t hv, int64_t a_row_stride,
                               int64_t b_row_stride) {
  const int64_t n = t * hv;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t h = idx % hv;
    const int64_t row = idx / hv;
    const float av = Load(araw, row * a_row_stride + h);
    const float bv = Load(braw, row * b_row_stride + h);
    const float x = av + dt_bias[h];
    const float sp = x > 20.0f ? x : log1pf(expf(x));  // softplus, threshold 20
    g_out[idx] = -expf(a_log[h]) * sp;
    beta_out[idx] = SigmoidF(bv);
  }
}

void GdnGBetaKernelCuda(Queue& q, Tensor& g_out, Tensor& beta_out, const Tensor& araw,
                        const Tensor& braw, const Tensor& a_log, const Tensor& dt_bias) {
  const int64_t t = g_out.shape[0], hv = g_out.shape[1];
  const int64_t n = t * hv;
  if (n == 0) return;
  if (araw.dtype == DType::kBF16) {
    GdnGBetaKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
        g_out.Ptr<float>(), beta_out.Ptr<float>(),
        araw.Ptr<__nv_bfloat16>(), braw.Ptr<__nv_bfloat16>(),
        a_log.Ptr<float>(), dt_bias.Ptr<float>(), t, hv, araw.stride[0],
        braw.stride[0]);
  } else {
    GdnGBetaKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
        g_out.Ptr<float>(), beta_out.Ptr<float>(), araw.Ptr<float>(),
        braw.Ptr<float>(), a_log.Ptr<float>(), dt_bias.Ptr<float>(), t, hv,
        araw.stride[0], braw.stride[0]);
  }
  Check(cudaGetLastError(), "gdn_g_beta launch");
}

// ---------------------------------------------------------------------------
// gdn_conv_split: conv [T, 2*key_dim+value_dim] -> q/k [T,key_dim], v
// [T,value_dim]. Thread per q/k output element (flat idx over T*key_dim); the v
// copy is folded in for idx < T*value_dim so both halves ride one launch.
// Templated on the q/k/v output dtype (Tqkv): f32 by default, or bf16 for the
// coupled GDN bf16 path (VT_GDN_BF16) so the split activations feed the WMMA
// chunk-scan as native bf16 (halved traffic + bf16 tensor-core fragments). Tconv
// = the conv-output dtype: f32 (default) or bf16 under the input-side bf16 GDN
// path (VT_GDN_IN_BF16); the conv read upcasts to f32 via Load(), math rounds to
// Tqkv on store via the Store overloads.
template <typename Tqkv, typename Tconv>
__global__ void GdnConvSplitKernel(Tqkv* q_out, Tqkv* k_out, Tqkv* v_out, const Tconv* conv,
                                   int64_t t, int64_t key_dim, int64_t value_dim) {
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t nq = t * key_dim, nv = t * value_dim;
  const int64_t n = nq > nv ? nq : nv;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    if (idx < nq) {
      const int64_t i = idx / key_dim, j = idx % key_dim;
      const int64_t row = i * conv_dim;
      Store(q_out, idx, Load(conv, row + j));
      Store(k_out, idx, Load(conv, row + key_dim + j));
    }
    if (idx < nv) {
      const int64_t i = idx / value_dim, j = idx % value_dim;
      Store(v_out, idx, Load(conv, i * conv_dim + 2 * key_dim + j));
    }
  }
}

void GdnConvSplitKernelCuda(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out,
                            const Tensor& conv) {
  const int64_t t = conv.shape[0];
  if (t == 0) return;
  const int64_t key_dim = q_out.Numel() / t, value_dim = v_out.Numel() / t;
  const int64_t n = t * (key_dim > value_dim ? key_dim : value_dim);
  if (n == 0) return;
  VT_CHECK(q_out.dtype == k_out.dtype && q_out.dtype == v_out.dtype,
           "cuda gdn_conv_split: q/k/v out dtypes must match");
  VT_CHECK(q_out.dtype == DType::kF32 || q_out.dtype == DType::kBF16,
           "cuda gdn_conv_split: q/k/v out must be f32 or bf16");
  VT_CHECK(conv.dtype == DType::kF32 || conv.dtype == DType::kBF16,
           "cuda gdn_conv_split: conv must be f32 or bf16");
  cudaStream_t s = AsStream(q);
  auto launch = [&](auto qkv_tag, auto conv_tag) {
    using Tqkv = decltype(qkv_tag);
    using Tconv = decltype(conv_tag);
    GdnConvSplitKernel<Tqkv, Tconv><<<GridFor(n), kBlock, 0, s>>>(
        q_out.Ptr<Tqkv>(), k_out.Ptr<Tqkv>(), v_out.Ptr<Tqkv>(), conv.Ptr<Tconv>(), t, key_dim,
        value_dim);
  };
  const bool qkv_f32 = q_out.dtype == DType::kF32;
  const bool conv_f32 = conv.dtype == DType::kF32;
  if (qkv_f32 && conv_f32) {
    launch(float{}, float{});
  } else if (qkv_f32) {
    launch(float{}, __nv_bfloat16{});
  } else if (conv_f32) {
    launch(__nv_bfloat16{}, float{});
  } else {
    launch(__nv_bfloat16{}, __nv_bfloat16{});
  }
  Check(cudaGetLastError(), "gdn_conv_split launch");
}

// ---------------------------------------------------------------------------
// qkv_split: split a merged QKVParallelLinear projection with INDEPENDENT head
// dims (GQA: q_dim != k_dim). qkv is [T, q_dim+k_dim+v_dim]; q/k/v out are the
// three contiguous shards. Thread per element over the widest shard; each shard
// computes its own (row, col) since the dims differ (unlike GdnConvSplit).
template <typename T>
__global__ void QkvSplitKernel(T* q_out, T* k_out, T* v_out, const T* qkv, int64_t t,
                               int64_t q_dim, int64_t k_dim, int64_t v_dim) {
  const int64_t total = q_dim + k_dim + v_dim;
  const int64_t nq = t * q_dim, nk = t * k_dim, nv = t * v_dim;
  const int64_t n = nq > nk ? (nq > nv ? nq : nv) : (nk > nv ? nk : nv);
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    if (idx < nq) {
      const int64_t i = idx / q_dim, j = idx % q_dim;
      q_out[idx] = qkv[i * total + j];
    }
    if (idx < nk) {
      const int64_t i = idx / k_dim, j = idx % k_dim;
      k_out[idx] = qkv[i * total + q_dim + j];
    }
    if (idx < nv) {
      const int64_t i = idx / v_dim, j = idx % v_dim;
      v_out[idx] = qkv[i * total + q_dim + k_dim + j];
    }
  }
}

void QkvSplitKernelCuda(Queue& q, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  const int64_t t = qkv.shape[0];
  if (t == 0) return;
  const int64_t q_dim = q_out.Numel() / t, k_dim = k_out.Numel() / t, v_dim = v_out.Numel() / t;
  const int64_t nmax = q_dim > k_dim ? (q_dim > v_dim ? q_dim : v_dim) : (k_dim > v_dim ? k_dim : v_dim);
  const int64_t n = t * nmax;
  if (n == 0) return;
  cudaStream_t s = AsStream(q);
  if (q_out.dtype == DType::kF32) {
    QkvSplitKernel<float><<<GridFor(n), kBlock, 0, s>>>(
        q_out.Ptr<float>(), k_out.Ptr<float>(), v_out.Ptr<float>(), qkv.Ptr<float>(), t, q_dim,
        k_dim, v_dim);
  } else {
    QkvSplitKernel<__nv_bfloat16><<<GridFor(n), kBlock, 0, s>>>(
        q_out.Ptr<__nv_bfloat16>(), k_out.Ptr<__nv_bfloat16>(), v_out.Ptr<__nv_bfloat16>(),
        qkv.Ptr<__nv_bfloat16>(), t, q_dim, k_dim, v_dim);
  }
  Check(cudaGetLastError(), "qkv_split launch");
}

// ---------------------------------------------------------------------------
// shared_expert_gate: out[t,c] = bf16(sigmoid(gl[t]) * sd[t*H+c]). Thread per
// output element (flat idx over T*H); the token index t = idx / H picks gl[t].
// sd is read through T: f32, or the bf16 the down-proj GEMM actually produced.
// Widening bf16 in-kernel is EXACT and the store is bf16 either way, so the two
// forms are bit-identical; the bf16 one skips a full [T,H] f32 cast + reread.
template <typename T>
__global__ void SharedExpertGateKernel(__nv_bfloat16* out, const T* sd, const float* gl,
                                       int64_t t, int64_t h) {
  const int64_t n = t * h;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += step) {
    const int64_t row = idx / h;
    Store(out, idx, SigmoidF(gl[row]) * Load(sd, idx));
  }
}

void SharedExpertGateKernelCuda(Queue& q, Tensor& out, const Tensor& sd, const Tensor& gl) {
  const int64_t t = out.shape[0], h = out.shape[1];
  const int64_t n = t * h;
  if (n == 0) return;
  if (sd.dtype == DType::kBF16) {
    SharedExpertGateKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
        out.Ptr<__nv_bfloat16>(), sd.Ptr<__nv_bfloat16>(), gl.Ptr<float>(), t, h);
  } else {
    SharedExpertGateKernel<<<GridFor(n), kBlock, 0, AsStream(q)>>>(
        out.Ptr<__nv_bfloat16>(), sd.Ptr<float>(), gl.Ptr<float>(), t, h);
  }
  Check(cudaGetLastError(), "shared_expert_gate launch");
}

// Registers the CUDA glue kernels during static init (pre-main, like the other
// vt CUDA ops). Harmless on GPU-less machines: the kCUDA backend never
// registers there, so no CUDA queue can dispatch.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kCastBf16, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastBf16KernelCuda)));
    RegisterOp(OpId::kCastF16, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<CastF16Fn>(&CastF16KernelCuda)));
    RegisterOp(OpId::kCastF32, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastF32KernelCuda)));
    RegisterOp(OpId::kPermuteVHeads, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<PermuteVHeadsFn>(&PermuteVHeadsKernelCuda)));
    RegisterOp(OpId::kMulColVecF32, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MulColVecF32Fn>(&MulColVecF32KernelCuda)));
    RegisterOp(OpId::kAttnGateSplit, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<AttnGateSplitFn>(&AttnGateSplitKernelCuda)));
    RegisterOp(
        OpId::kSigmoidGateBf16, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<SigmoidGateBf16Fn>(&SigmoidGateBf16KernelCuda)));
    RegisterOp(OpId::kGdnGBeta, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GdnGBetaFn>(&GdnGBetaKernelCuda)));
    RegisterOp(OpId::kGdnConvSplit, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GdnConvSplitFn>(&GdnConvSplitKernelCuda)));
    RegisterOp(OpId::kQkvSplit, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernelCuda)));
    RegisterOp(
        OpId::kSharedExpertGate, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<SharedExpertGateFn>(&SharedExpertGateKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda

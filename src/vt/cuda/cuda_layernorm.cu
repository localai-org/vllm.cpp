// CUDA LayerNorm / ReLU / Add — the cross-family dense primitives introduced by
// the OPT (`OPTForCausalLM`) bring-up, i.e. the pre-RMSNorm/pre-SwiGLU
// transformer vocabulary that every non-Qwen family needs.
//
// This is a SELF-REGISTERING translation unit in the established additive
// pattern (`src/vt/cuda/cuda_glue.cu` Registrar, `src/vt/backend.cpp`,
// `src/vllm/platforms/platform.cpp`): adding these ops touched no existing
// kernel TU and no shared op array — only the op-table declarations in
// `include/vt/ops.h` + the validating wrappers in `src/vt/ops.cpp`.
//
// Ported FROM (semantics, 1:1):
//   * LayerNorm — ATen `native_layer_norm` / `aten/src/ATen/native/cuda/
//     layer_norm_kernel.cu::vectorized_layer_norm_kernel`, the kernel a CUDA
//     bfloat16 `nn.LayerNorm` dispatches to; this is what vLLM's
//     `vllm/model_executor/models/opt.py:146-148,164-166,248-251` construct.
//     Contract mirrored exactly: BIASED (1/N) variance, `acc_type<bfloat16> ==
//     float` accumulation, affine `y = (x-mean)*rstd*w + b` applied in f32 with
//     a SINGLE rounding on store.
//   * Relu — `vllm/model_executor/layers/activation.py::get_act_fn("relu")` ->
//     `torch.nn.ReLU` (opt.py:156).
//   * Add — `torch.add` with the two shapes OPT needs: the elementwise residual
//     joins (opt.py:178,191,279) and the rank-1 row-broadcast `nn.Linear` bias
//     (opt.py:90-104,149-163, `config.enable_bias`).
//
// Numerics: all math is f32 and every store rounds once via __float2bfloat16
// (round-to-nearest-even, identical to the host F32ToBF16 and to the CPU
// reference in src/vt/cpu/cpu_layernorm.cpp) — the same discipline as
// cuda_glue.cu / cuda_moe.cu.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "vt/ops.h"
#include "vt/op_provider.h"

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

__device__ inline float Load(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline void Store(__half* p, int64_t i, float v) { p[i] = __float2half_rn(v); }
__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16(v); }

// ---------------------------------------------------------------------------
// layer_norm: one BLOCK per row. Two passes over the row in f32 — pass 1 sums to
// the mean, pass 2 sums the squared deviations FROM THAT MEAN. The
// deviation-from-mean form (rather than E[x^2]-E[x]^2) is the numerically stable
// one and matches what ATen's Welford accumulator converges to; on the OPT
// hidden width (D=768) the row fits comfortably in L1 so the second read is
// nearly free.
template <typename OutT, typename InT, typename WT>
__global__ void LayerNormKernel(OutT* out, const InT* x, const WT* weight, const WT* bias,
                                int64_t rows, int64_t d, float eps) {
  __shared__ float red[kBlock / 32];
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const InT* xr = x + row * d;
    OutT* outr = out + row * d;

    // --- pass 1: mean -------------------------------------------------------
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) sum += Load(xr, i);
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
      float v = (threadIdx.x < blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
      for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
      if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float mean = red[0] / static_cast<float>(d);
    __syncthreads();

    // --- pass 2: biased variance about that mean ----------------------------
    float sq = 0.0f;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
      const float dv = Load(xr, i) - mean;
      sq += dv * dv;
    }
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sq;
    __syncthreads();
    if (threadIdx.x < 32) {
      float v = (threadIdx.x < blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
      for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
      if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float rstd = rsqrtf(red[0] / static_cast<float>(d) + eps);
    __syncthreads();

    // --- affine + single rounding on store ----------------------------------
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
      float v = (Load(xr, i) - mean) * rstd;
      if (weight != nullptr) v *= Load(weight, i);
      if (bias != nullptr) v += Load(bias, i);
      Store(outr, i, v);
    }
    __syncthreads();
  }
}

template <typename OutT, typename InT>
void LayerNormDispatchW(Queue& q, Tensor& out, const Tensor& x, const Tensor* weight,
                        const Tensor* bias, float eps, int64_t rows, int64_t d) {
  const DType wdt = weight != nullptr ? weight->dtype : (bias != nullptr ? bias->dtype : x.dtype);
  if (weight != nullptr && bias != nullptr)
    VT_CHECK(weight->dtype == bias->dtype, "cuda layer_norm: weight/bias dtype must match");
  auto* o = static_cast<OutT*>(out.data);
  const auto* i = static_cast<const InT*>(x.data);
  const unsigned grid = static_cast<unsigned>(rows < 65535 ? rows : 65535);
  if (wdt == DType::kBF16) {
    LayerNormKernel<OutT, InT, __nv_bfloat16><<<grid, kBlock, 0, AsStream(q)>>>(
        o, i, weight != nullptr ? static_cast<const __nv_bfloat16*>(weight->data) : nullptr,
        bias != nullptr ? static_cast<const __nv_bfloat16*>(bias->data) : nullptr, rows, d, eps);
  } else {
    VT_CHECK(wdt == DType::kF32, "cuda layer_norm: weight/bias must be f32 or bf16");
    LayerNormKernel<OutT, InT, float><<<grid, kBlock, 0, AsStream(q)>>>(
        o, i, weight != nullptr ? static_cast<const float*>(weight->data) : nullptr,
        bias != nullptr ? static_cast<const float*>(bias->data) : nullptr, rows, d, eps);
  }
  Check(cudaGetLastError(), "layer_norm launch");
}

void LayerNormKernelCuda(Queue& q, Tensor& out, const Tensor& x, const Tensor* weight,
                         const Tensor* bias, const LayerNormArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = x.Numel() / d;
  if (rows == 0 || d == 0) return;
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16,
           "cuda layer_norm: x must be f32 or bf16");
  if (out.dtype == DType::kBF16) {
    if (x.dtype == DType::kBF16)
      LayerNormDispatchW<__nv_bfloat16, __nv_bfloat16>(q, out, x, weight, bias, args.eps, rows, d);
    else
      LayerNormDispatchW<__nv_bfloat16, float>(q, out, x, weight, bias, args.eps, rows, d);
  } else {
    if (x.dtype == DType::kBF16)
      LayerNormDispatchW<float, __nv_bfloat16>(q, out, x, weight, bias, args.eps, rows, d);
    else
      LayerNormDispatchW<float, float>(q, out, x, weight, bias, args.eps, rows, d);
  }
}

// ---------------------------------------------------------------------------
// relu: out[i] = max(x[i], 0). Grid-stride, may alias in-place.
template <typename OutT, typename InT>
__global__ void ReluKernel(OutT* out, const InT* x, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step) {
    const float v = Load(x, i);
    Store(out, i, v > 0.0f ? v : 0.0f);
  }
}

void ReluKernelCuda(Queue& q, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  if (n == 0) return;
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16, "cuda relu: x must be f32 or bf16");
  const unsigned grid = GridFor(n);
  if (out.dtype == DType::kBF16) {
    auto* o = static_cast<__nv_bfloat16*>(out.data);
    if (x.dtype == DType::kBF16)
      ReluKernel<<<grid, kBlock, 0, AsStream(q)>>>(
          o, static_cast<const __nv_bfloat16*>(x.data), n);
    else
      ReluKernel<<<grid, kBlock, 0, AsStream(q)>>>(o, static_cast<const float*>(x.data), n);
  } else {
    auto* o = static_cast<float*>(out.data);
    if (x.dtype == DType::kBF16)
      ReluKernel<<<grid, kBlock, 0, AsStream(q)>>>(
          o, static_cast<const __nv_bfloat16*>(x.data), n);
    else
      ReluKernel<<<grid, kBlock, 0, AsStream(q)>>>(o, static_cast<const float*>(x.data), n);
  }
  Check(cudaGetLastError(), "relu launch");
}

// ---------------------------------------------------------------------------
// gelu (elementwise, Qwen3-VL vision tower). f32 compute, may alias in-place.
//   tanh: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))  — gelu_pytorch_tanh
//         (same constant as cuda_ops.cu GeluAndMulKernel gate).
//   erf : 0.5*x*(1+erf(x/sqrt(2)))                      — exact nn.GELU().
template <typename OutT, typename InT, bool kTanh>
__global__ void GeluKernel(OutT* out, const InT* x, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step) {
    const float v = Load(x, i);
    float g;
    if (kTanh) {
      const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
      g = 0.5f * v * (1.0f + tanhf(inner));
    } else {
      g = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
    }
    Store(out, i, g);
  }
}

template <bool kTanh>
void GeluDispatch(Queue& q, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  if (n == 0) return;
  VT_CHECK(x.dtype == DType::kF32 || x.dtype == DType::kBF16, "cuda gelu: x must be f32 or bf16");
  const unsigned grid = GridFor(n);
  if (out.dtype == DType::kBF16) {
    auto* o = static_cast<__nv_bfloat16*>(out.data);
    if (x.dtype == DType::kBF16)
      GeluKernel<__nv_bfloat16, __nv_bfloat16, kTanh><<<grid, kBlock, 0, AsStream(q)>>>(
          o, static_cast<const __nv_bfloat16*>(x.data), n);
    else
      GeluKernel<__nv_bfloat16, float, kTanh><<<grid, kBlock, 0, AsStream(q)>>>(
          o, static_cast<const float*>(x.data), n);
  } else {
    auto* o = static_cast<float*>(out.data);
    if (x.dtype == DType::kBF16)
      GeluKernel<float, __nv_bfloat16, kTanh><<<grid, kBlock, 0, AsStream(q)>>>(
          o, static_cast<const __nv_bfloat16*>(x.data), n);
    else
      GeluKernel<float, float, kTanh><<<grid, kBlock, 0, AsStream(q)>>>(
          o, static_cast<const float*>(x.data), n);
  }
  Check(cudaGetLastError(), "gelu launch");
}

void GeluTanhKernelCuda(Queue& q, Tensor& out, const Tensor& x) { GeluDispatch<true>(q, out, x); }
void GeluErfKernelCuda(Queue& q, Tensor& out, const Tensor& x) { GeluDispatch<false>(q, out, x); }

// ---------------------------------------------------------------------------
// add: out[i] = a[i] + b[i % row_d] where row_d == d selects the rank-1 bias
// row-broadcast and row_d == 0 selects the elementwise form. Grid-stride, may
// alias in-place.
template <typename OutT, typename AT, typename BT>
__global__ void AddKernel(OutT* out, const AT* a, const BT* b, int64_t n, int64_t bcast_d) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step) {
    const int64_t bi = bcast_d > 0 ? (i % bcast_d) : i;
    Store(out, i, Load(a, i) + Load(b, bi));
  }
}

template <typename OutT, typename AT>
void AddDispatchB(Queue& q, Tensor& out, const Tensor& a, const Tensor& b, int64_t n,
                  int64_t bcast_d) {
  auto* o = static_cast<OutT*>(out.data);
  const auto* ap = static_cast<const AT*>(a.data);
  const unsigned grid = GridFor(n);
  if (b.dtype == DType::kBF16)
    AddKernel<<<grid, kBlock, 0, AsStream(q)>>>(
        o, ap, static_cast<const __nv_bfloat16*>(b.data), n, bcast_d);
  else if (b.dtype == DType::kF16)
    AddKernel<<<grid, kBlock, 0, AsStream(q)>>>(o, ap, b.Ptr<__half>(), n, bcast_d);
  else
    AddKernel<<<grid, kBlock, 0, AsStream(q)>>>(o, ap, static_cast<const float*>(b.data), n,
                                                bcast_d);
  Check(cudaGetLastError(), "add launch");
}

template <typename OutT>
void AddDispatchA(Queue& q, Tensor& out, const Tensor& a, const Tensor& b, int64_t n, int64_t bcast_d) {
  if (a.dtype == DType::kF16) AddDispatchB<OutT, __half>(q, out, a, b, n, bcast_d);
  else if (a.dtype == DType::kBF16) AddDispatchB<OutT, __nv_bfloat16>(q, out, a, b, n, bcast_d);
  else AddDispatchB<OutT, float>(q, out, a, b, n, bcast_d);
}

void AddKernelCuda(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t n = a.Numel();
  if (n == 0) return;
  VT_CHECK((a.dtype == DType::kF32 || a.dtype == DType::kBF16 || a.dtype == DType::kF16) &&
           (b.dtype == DType::kF32 || b.dtype == DType::kBF16 || b.dtype == DType::kF16),
           "cuda add: a/b must be f32, bf16 or f16");
  const int64_t bcast_d = (b.rank == 1 && a.rank != 1) ? a.shape[a.rank - 1] : 0;
  if (out.dtype == DType::kF16) AddDispatchA<__half>(q, out, a, b, n, bcast_d);
  else if (out.dtype == DType::kBF16) AddDispatchA<__nv_bfloat16>(q, out, a, b, n, bcast_d);
  else AddDispatchA<float>(q, out, a, b, n, bcast_d);
}

// Registers the CUDA cross-family dense primitives during static init (pre-main,
// like cuda_glue.cu's Registrar) — this TU is the whole registration surface,
// so no existing kernel file is edited.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kLayerNorm, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernelCuda)));
    RegisterOp(OpId::kRelu, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernelCuda)));
    RegisterOp(OpId::kGeluTanh, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ReluFn>(&GeluTanhKernelCuda)));
    RegisterOp(OpId::kGeluErf, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ReluFn>(&GeluErfKernelCuda)));
    RegisterOp(OpId::kAdd, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernelCuda)));
    RegisterOpScalarType(OpId::kAdd, DeviceType::kCUDA, kNativeProviderName, DType::kF16);
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::cuda

// Ported from: vllm/csrc/libtorch_stable/cache_kernels.cu @ e24d1b24
//   (reshape_and_cache_flash — the "auto"/contiguous-heads write semantics
//    only; the NHD cache layout is FlashAttentionBackend::get_kv_cache_shape's
//    (num_blocks, 2, block_size, num_kv_heads, head_size), NOT the HND cpu_attn
//    layout — see the M1.6 Task-2 layout trap note).
// Correctness-grade (M1.6): one block per token, threads stride over the page
// (num_kv_heads*head_size). The perf kernel (vectorized / fp8) is M2.4.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Word is the raw storage type (uint32_t for f32, uint16_t for f16/bf16): the
// auto cache path is a bit-exact copy, so no dtype-aware conversion is needed.
//
// CRITICAL (M1.6 Task-2): destination is indexed from TENSOR STRIDES, not from
// k_cache.shape. K/V are the two dim-1 unbind slices of one (num_blocks, 2,
// block_size, H, D) allocation, so their block stride is 2*bs*H*D (NOT bs*H*D)
// and they are NON-contiguous rank-4 views. Mirrors pinned cache_kernels.cu @
// e24d1b24 (host ~L797-801 sources key_stride/block_stride/page_stride/
// head_stride from the tensor strides; kernel ~L337-347 does key_dst =
// key_cache + block_idx*block_stride + block_offset*page_stride). The wrapper
// guarantees head_stride == head_size && elem stride == 1 (the NHD unbind
// slice), so the per-token page is one dense run of n_elems inside the block —
// pinned's is_contiguous_heads fast path; here the threads stride that run.
template <typename Word>
__global__ void ReshapeAndCacheKernel(
    const Word* __restrict__ key, const Word* __restrict__ value,
    Word* __restrict__ key_cache, Word* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, int64_t block_size,
    int64_t n_elems_k, int64_t n_elems_v,
    int64_t k_block_stride, int64_t k_page_stride, int64_t v_block_stride,
    int64_t v_page_stride, int64_t k_tok_stride, int64_t v_tok_stride) {
  const int64_t token = blockIdx.x;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) return;  // padded token → skip
  const int64_t block = slot / block_size;
  const int64_t offset = slot % block_size;
  const int64_t kdst = block * k_block_stride + offset * k_page_stride;  // element offset
  const int64_t vdst = block * v_block_stride + offset * v_page_stride;
  const int64_t ksrc = token * k_tok_stride;
  const int64_t vsrc = token * v_tok_stride;
  for (int64_t e = threadIdx.x; e < n_elems_k; e += blockDim.x) {
    key_cache[kdst + e] = key[ksrc + e];
  }
  for (int64_t e = threadIdx.x; e < n_elems_v; e += blockDim.x) {
    value_cache[vdst + e] = value[vsrc + e];
  }
}

void ReshapeAndCacheKernelCuda(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                               Tensor& v_cache, const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t n_elems_k = k_cache.shape[2] * k_cache.shape[3];
  const int64_t n_elems_v = v_cache.shape[2] * v_cache.shape[3];
  const int64_t max_elems = n_elems_k > n_elems_v ? n_elems_k : n_elems_v;
  if (num_slots == 0 || max_elems == 0) return;
  const int64_t k_block_stride = k_cache.stride[0];
  const int64_t k_page_stride = k_cache.stride[1];
  const int64_t v_block_stride = v_cache.stride[0];
  const int64_t v_page_stride = v_cache.stride[1];
  const int64_t k_tok_stride = k.stride[0];
  const int64_t v_tok_stride = v.stride[0];
  const unsigned grid = static_cast<unsigned>(num_slots);
  const unsigned block = static_cast<unsigned>(max_elems < 512 ? max_elems : 512);
  const cudaStream_t s = AsStream(q);
  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  switch (SizeOf(k.dtype)) {
    case 4:
      ReshapeAndCacheKernel<uint32_t><<<grid, block, 0, s>>>(
          k.Ptr<uint32_t>(), v.Ptr<uint32_t>(), k_cache.Ptr<uint32_t>(), v_cache.Ptr<uint32_t>(),
          slots, block_size, n_elems_k, n_elems_v, k_block_stride, k_page_stride, v_block_stride,
          v_page_stride, k_tok_stride, v_tok_stride);
      break;
    case 2:
      ReshapeAndCacheKernel<uint16_t><<<grid, block, 0, s>>>(
          k.Ptr<uint16_t>(), v.Ptr<uint16_t>(), k_cache.Ptr<uint16_t>(), v_cache.Ptr<uint16_t>(),
          slots, block_size, n_elems_k, n_elems_v, k_block_stride, k_page_stride, v_block_stride,
          v_page_stride, k_tok_stride, v_tok_stride);
      break;
    default: VT_CHECK(false, "cuda reshape_and_cache: unsupported dtype element size");
  }
  Check(cudaGetLastError(), "reshape_and_cache launch");
}

// ─── fp8 KV-cache write (KV-FP8 W2, #1593) ─────────────────────────────────
// The CUDA arm of vt::ReshapeAndCacheFp8, and the CUDA sibling of the CPU kernel
// in src/vt/cpu/cpu_cache.cpp that is its ORACLE.
//
// Ported from the fp8 branch of vllm reshape_and_cache_flash_kernel
// (csrc/libtorch_stable/cache_kernels.cu:314-401) + CopyWithScaleOp (:241-252) @
// pin 555967922. Upstream's `is_contiguous_heads && kv_scale_stride == 0` fast
// path (`:352-366`) is the ONLY one this op's wrapper admits: the vt paged cache
// is the NHD unbind slice (head_stride == head_size) and the scales are
// per-TENSOR (`kv_scale_stride == 0`). The HND / per-attention-head arm
// (`:367-400`) is a named later brick (spec W5), and per-head scales cannot
// reach here because ReshapeAndCacheFp8 takes two scalars.
//
// ELEMENTWISE-IDENTICAL, NOT INSTRUCTION-IDENTICAL. Upstream's contiguous-heads
// arm moves the row through `vectorize_with_alignment<VEC_SIZE>` (`:360-363`,
// VEC_SIZE 8 for a 2-byte source and 4 for f32), which converts the same
// elements in the same order under a vectorized load/store. The loop below is a
// SCALAR strided one, so it writes the same bytes and reads the same inputs
// while moving them one at a time. That is a bandwidth difference, not a
// numerical one, and W4 — which owns the memory/throughput measurement — owns
// closing it. Do not read "ported" here as "the same instructions".
//
// THE CONVERTER IS UPSTREAM'S OWN, and its equality to the CPU codec is already
// MEASURED. `fp8::scaled_convert<uint8_t, float, kFp8E4M3>` is
// `__nv_cvt_float_to_fp8(a / scale, __NV_SATFINITE, __NV_E4M3)`
// (quant_utils.cuh:497-503) — a true DIVIDE, unlike the activation-quant path's
// hoisted reciprocal multiply (cuda_quant_fp8.cu:56-63), which matters because
// the two differ by up to one f32 ulp before the round. That same intrinsic is
// gated byte-for-byte at zero tolerance against the CPU software codec
// vt::F32ToF8E4M3 on sm_110 and sm_121a
// (.agents/specs/vt-fp8-quant-arch-gate.md G2), and
// tests/vt/test_cuda_fp8_kv_cache.cpp re-takes that equality on this KV path.
//
// Destination arithmetic is the auto path's, with element size 1: the cache is
// DType::kI8 (the "byte never guesses its semantic type" rule, include/vt/dtype.h)
// and the fp8 INTERPRETATION travels as Fp8KVCacheDataType, exactly as upstream
// carries cache_t = uint8_t plus a KV_DTYPE template parameter.

// Pointer overloads, not by-value ones: __half and __nv_bfloat16 both carry an
// implicit `operator float()`, so a by-value set would put a user conversion in
// the overload resolution for every call. Same shape as cuda_quant_fp8.cu's
// LoadIn and cuda_paged_attn.cu's Load.
__device__ __forceinline__ float Fp8SrcToF32(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ float Fp8SrcToF32(const __nv_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ __forceinline__ float Fp8SrcToF32(const __half* p, int64_t i) {
  return __half2float(p[i]);
}

// fp8 = Quantize(hp / scale) — quant_utils.cuh:296-300 "Convention of the scale".
__device__ __forceinline__ uint8_t StoreKvFp8E4M3Dev(float hp, float scale) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(hp / scale, __NV_SATFINITE, __NV_E4M3));
}

template <typename Tin>
__global__ void ReshapeAndCacheFp8Kernel(
    const Tin* __restrict__ key, const Tin* __restrict__ value,
    uint8_t* __restrict__ key_cache, uint8_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, int64_t block_size,
    int64_t n_elems_k, int64_t n_elems_v,
    int64_t k_block_stride, int64_t k_page_stride, int64_t v_block_stride,
    int64_t v_page_stride, int64_t k_tok_stride, int64_t v_tok_stride,
    float k_scale, float v_scale) {
  const int64_t token = blockIdx.x;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) return;  // padded token → skip (upstream `:328-331`)
  const int64_t block = slot / block_size;
  const int64_t offset = slot % block_size;
  const int64_t kdst = block * k_block_stride + offset * k_page_stride;  // element offset
  const int64_t vdst = block * v_block_stride + offset * v_page_stride;
  const int64_t ksrc = token * k_tok_stride;
  const int64_t vsrc = token * v_tok_stride;
  for (int64_t e = threadIdx.x; e < n_elems_k; e += blockDim.x) {
    key_cache[kdst + e] = StoreKvFp8E4M3Dev(Fp8SrcToF32(key, ksrc + e), k_scale);
  }
  for (int64_t e = threadIdx.x; e < n_elems_v; e += blockDim.x) {
    value_cache[vdst + e] = StoreKvFp8E4M3Dev(Fp8SrcToF32(value, vsrc + e), v_scale);
  }
}

void ReshapeAndCacheFp8KernelCuda(Queue& q, const Tensor& k, const Tensor& v, Tensor& k_cache,
                                  Tensor& v_cache, const Tensor& slot_mapping,
                                  Fp8KVCacheDataType kind, float k_scale, float v_scale) {
  VT_CHECK(kind == Fp8KVCacheDataType::kFp8E4M3,
           "cuda reshape_and_cache_fp8: only fp8_e4m3 is implemented "
           "(fp8_e5m2 is a named later brick, spec W5)");
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t n_elems_k = k_cache.shape[2] * k_cache.shape[3];
  const int64_t n_elems_v = v_cache.shape[2] * v_cache.shape[3];
  const int64_t max_elems = n_elems_k > n_elems_v ? n_elems_k : n_elems_v;
  if (num_slots == 0 || max_elems == 0) return;
  const int64_t k_block_stride = k_cache.stride[0];
  const int64_t k_page_stride = k_cache.stride[1];
  const int64_t v_block_stride = v_cache.stride[0];
  const int64_t v_page_stride = v_cache.stride[1];
  const int64_t k_tok_stride = k.stride[0];
  const int64_t v_tok_stride = v.stride[0];
  const unsigned grid = static_cast<unsigned>(num_slots);
  const unsigned block = static_cast<unsigned>(max_elems < 512 ? max_elems : 512);
  const cudaStream_t s = AsStream(q);
  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  uint8_t* kc = k_cache.Ptr<uint8_t>();
  uint8_t* vc = v_cache.Ptr<uint8_t>();
  // The SOURCE dtype is the model float dtype and is typed here, unlike the auto
  // path's raw-word copy: the fp8 store converts, so it must know what it reads.
  // Same set the CPU LoadSrcF32 serves (cpu_cache.cpp).
  switch (k.dtype) {
    case DType::kF32:
      ReshapeAndCacheFp8Kernel<float><<<grid, block, 0, s>>>(
          k.Ptr<float>(), v.Ptr<float>(), kc, vc, slots, block_size, n_elems_k, n_elems_v, k_block_stride,
          k_page_stride, v_block_stride, v_page_stride, k_tok_stride, v_tok_stride, k_scale,
          v_scale);
      break;
    case DType::kBF16:
      ReshapeAndCacheFp8Kernel<__nv_bfloat16><<<grid, block, 0, s>>>(
          k.Ptr<__nv_bfloat16>(), v.Ptr<__nv_bfloat16>(), kc, vc, slots, block_size, n_elems_k, n_elems_v,
          k_block_stride, k_page_stride, v_block_stride, v_page_stride, k_tok_stride,
          v_tok_stride, k_scale, v_scale);
      break;
    case DType::kF16:
      ReshapeAndCacheFp8Kernel<__half><<<grid, block, 0, s>>>(
          k.Ptr<__half>(), v.Ptr<__half>(), kc, vc, slots, block_size, n_elems_k, n_elems_v, k_block_stride,
          k_page_stride, v_block_stride, v_page_stride, k_tok_stride, v_tok_stride, k_scale,
          v_scale);
      break;
    default:
      VT_CHECK(false, "cuda reshape_and_cache_fp8: unsupported source dtype (f32/f16/bf16)");
  }
  Check(cudaGetLastError(), "reshape_and_cache_fp8 launch");
}

// ─── MLA cache write (W3) ──────────────────────────────────────────────────
// Ported 1:1 from vllm/csrc/libtorch_stable/cache_kernels.cu:401-442
// `concat_and_cache_mla_kernel` @ e24d1b24 — ONE block per token, threads stride
// over the copied run, exactly upstream's `copy` lambda (`:426-438`). Upstream
// launches `grid(num_tokens), block(min(kv_lora_rank, 512))` (`:899-900`); we
// mirror that. The "auto" path is a bit-exact element copy, so Word is the raw
// storage type and no dtype conversion appears — the fp8 `scaled_convert` branch
// is out of scope and refused by the op wrapper.
//
// Destination arithmetic mirrors upstream `:431-433`:
//   dst = block_idx*block_stride + block_offset*entry_stride + i + offset
// with offset 0 for the latent and `kv_lora_rank` for the rope part, i.e. the
// two source tensors are CONCATENATED into one 576-wide cache entry. This
// matches the CPU reference (cpu_cache.cpp ConcatAndCacheMlaKernel) bit-for-bit:
// it is a pure copy, so there is no reduction order to diverge on.
template <typename Word>
__global__ void ConcatAndCacheMlaKernel(
    const Word* __restrict__ kv_c, const Word* __restrict__ k_pe,
    Word* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    int64_t block_size, int64_t kv_lora_rank, int64_t pe_dim, int64_t block_stride,
    int64_t entry_stride, int64_t kv_c_stride, int64_t k_pe_stride) {
  const int64_t token = blockIdx.x;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) return;  // padded token → skip (upstream `:419-422`)
  const int64_t block = slot / block_size;
  const int64_t offset = slot % block_size;
  const int64_t entry = block * block_stride + offset * entry_stride;
  const int64_t csrc = token * kv_c_stride;
  const int64_t psrc = token * k_pe_stride;
  for (int64_t i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    kv_cache[entry + i] = kv_c[csrc + i];
  }
  for (int64_t i = threadIdx.x; i < pe_dim; i += blockDim.x) {
    kv_cache[entry + kv_lora_rank + i] = k_pe[psrc + i];
  }
}

void ConcatAndCacheMlaKernelCuda(Queue& q, const Tensor& kv_c, const Tensor& k_pe,
                                 Tensor& kv_cache, const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = kv_cache.shape[1];
  const int64_t kv_lora_rank = kv_c.shape[1];
  const int64_t pe_dim = k_pe.shape[1];
  if (num_slots == 0) return;
  const int64_t block_stride = kv_cache.stride[0];
  const int64_t entry_stride = kv_cache.stride[1];
  const int64_t kv_c_stride = kv_c.stride[0];
  const int64_t k_pe_stride = k_pe.stride[0];
  const unsigned grid = static_cast<unsigned>(num_slots);
  // Upstream `:899-900`: block(min(kv_lora_rank, 512)).
  const unsigned block = static_cast<unsigned>(kv_lora_rank < 512 ? kv_lora_rank : 512);
  const cudaStream_t s = AsStream(q);
  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  switch (SizeOf(kv_c.dtype)) {
    case 4:
      ConcatAndCacheMlaKernel<uint32_t><<<grid, block, 0, s>>>(
          kv_c.Ptr<uint32_t>(), k_pe.Ptr<uint32_t>(), kv_cache.Ptr<uint32_t>(), slots,
          block_size, kv_lora_rank, pe_dim, block_stride, entry_stride, kv_c_stride,
          k_pe_stride);
      break;
    case 2:
      ConcatAndCacheMlaKernel<uint16_t><<<grid, block, 0, s>>>(
          kv_c.Ptr<uint16_t>(), k_pe.Ptr<uint16_t>(), kv_cache.Ptr<uint16_t>(), slots,
          block_size, kv_lora_rank, pe_dim, block_stride, entry_stride, kv_c_stride,
          k_pe_stride);
      break;
    default: VT_CHECK(false, "cuda concat_and_cache_mla: unsupported dtype element size");
  }
  Check(cudaGetLastError(), "concat_and_cache_mla launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(
        OpId::kReshapeAndCache, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernelCuda)));
    RegisterOp(OpId::kReshapeAndCacheFp8, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<ReshapeAndCacheFp8Fn>(&ReshapeAndCacheFp8KernelCuda)));
    RegisterOp(
        OpId::kConcatAndCacheMla, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<ConcatAndCacheMlaFn>(&ConcatAndCacheMlaKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda

// Legacy CUDA adaptation of the float compute lane selected by pinned llama.cpp
// ggml/src/ggml-cuda/ggml-cuda.cu:1672-1690 @ 62318cd57. cuBLASLt 12.4 rejects
// F32 batched operands with BF16 output on Pascal. Retain storage and output
// types, and use tiled FP32 accumulation without a persistent wider copy.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "vt/tensor.h"

namespace vt::cuda {
namespace legacy {
__global__ void BatchedF32Bf16(Tensor out, Tensor a, Tensor b) {
  __shared__ float as[16][16];  // F32: operands and reduction are F32.
  __shared__ float bs[16][16];  // F32: operands and reduction are F32.
  const int64_t mt = (out.shape[1] + 15) / 16;
  const int64_t nt = (out.shape[2] + 15) / 16;
  const int64_t total = out.shape[0] * mt * nt;
  const int tx = threadIdx.x, ty = threadIdx.y;
  for (int64_t tile = blockIdx.x; tile < total; tile += gridDim.x) {
    const int64_t group = tile / (mt * nt);
    const int64_t row = (tile / nt % mt) * 16 + ty;
    const int64_t col = (tile % nt) * 16 + tx;
    float sum = 0;
    for (int64_t kk = 0; kk < a.shape[2]; kk += 16) {
      as[ty][tx] = row < a.shape[1] && kk + tx < a.shape[2]
          ? static_cast<const float*>(a.data)[group*a.stride[0]+row*a.stride[1]+kk+tx] : 0;
      bs[ty][tx] = col < b.shape[2] && kk + ty < b.shape[1]
          ? static_cast<const float*>(b.data)[group*b.stride[0]+(kk+ty)*b.stride[1]+col] : 0;
      __syncthreads();
      for (int j = 0; j < 16; ++j) sum = fmaf(as[ty][j], bs[j][tx], sum);
      __syncthreads();
    }
    if (row < out.shape[1] && col < out.shape[2])
      static_cast<__nv_bfloat16*>(out.data)[group*out.stride[0]+row*out.stride[1]+col] = __float2bfloat16_rn(sum);
    __syncthreads();
  }
}
inline cudaError_t LaunchBatchedF32Bf16(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t tiles = out.shape[0] * ((out.shape[1]+15)/16) * ((out.shape[2]+15)/16);
  const unsigned blocks = static_cast<unsigned>(tiles < 65535 ? tiles : 65535);
  BatchedF32Bf16<<<blocks,dim3(16,16),0,static_cast<cudaStream_t>(q.handle)>>>(out,a,b);
  return cudaGetLastError();
}
}  // namespace legacy
}  // namespace vt::cuda

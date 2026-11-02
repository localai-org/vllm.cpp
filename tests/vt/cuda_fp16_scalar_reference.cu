// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Ported from vLLM a7c23ac96d7806e7c7e7d862eadbce5a33529b94:
// csrc/libtorch_stable/activation_kernels.cu:30-44,113-155,165-167.
// Independent test reference for tests/kernels/core/test_activation.py:137-221.
// Harness adaptation: CUDA scalar types replace torch scalar wrappers. Launch
// one block per token. The packed primary specialization has the same scalar
// rounding boundaries (packed_compute:60-87, packed_silu_kernel:170-177).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include "vt/ops.h"

namespace {
template <typename T>
__device__ T PrimarySilu(const T& x, const float alpha) {
  return (T)(((float)x) / (1.0f + expf((float)-x * alpha)));
}

template <typename T>
__global__ void PrimaryActAndMul(T* out, const T* input, int d) {
  const int64_t token_idx = blockIdx.x;
  const T* x_ptr = input + token_idx * 2 * d;
  const T* y_ptr = x_ptr + d;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    T gate = x_ptr[idx];
    T up = y_ptr[idx];
    out[token_idx * d + idx] = (T)((float)PrimarySilu(gate, 1.0f) * ((float)up + 0.0f));
  }
}
}

void CudaFp16ScalarReferenceSilu(vt::Queue& q, vt::Tensor& out, const vt::Tensor& input) {
  const unsigned tokens = static_cast<unsigned>(input.shape[0]);
  const int d = static_cast<int>(input.shape[1] / 2);
  const unsigned block = d < 1024 ? d : 1024;
  const auto stream = static_cast<cudaStream_t>(q.handle);
  switch (input.dtype) {
    case vt::DType::kF16:
      PrimaryActAndMul<<<tokens, block, 0, stream>>>(out.Ptr<__half>(), input.Ptr<__half>(), d);
      break;
    case vt::DType::kBF16:
      PrimaryActAndMul<<<tokens, block, 0, stream>>>(out.Ptr<__nv_bfloat16>(), input.Ptr<__nv_bfloat16>(), d);
      break;
    case vt::DType::kF32:
      PrimaryActAndMul<<<tokens, block, 0, stream>>>(out.Ptr<float>(), input.Ptr<float>(), d);
      break;
    default: throw std::runtime_error("scalar reference: float input required");
  }
  if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("scalar reference: launch failed");
}

// Capability-gated integer dot product. Ported from llama.cpp
// ggml/src/ggml-cuda/common.cuh:712-753 @ 62318cd57dc63900ecb153272c0b10f78550855d.
// sm_60 has no DP4A. Keep the four exact byte products there, and retain the
// native instruction on sm_61 and newer. EXL3 needs the unsigned overload.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace vt::cuda {

[[maybe_unused]] __device__ __forceinline__ int LegacyDp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  const auto* av = reinterpret_cast<const int8_t*>(&a);
  const auto* bv = reinterpret_cast<const int8_t*>(&b);
  // Accumulate modulo 2^32, including overflow of the signed accumulator.
  unsigned sum = static_cast<unsigned>(c);
  for (int i = 0; i < 4; ++i) sum += static_cast<unsigned>(av[i] * bv[i]);
  return static_cast<int>(sum);
#endif
}

[[maybe_unused]] __device__ __forceinline__ unsigned LegacyDp4a(unsigned a, unsigned b,
                                                               unsigned c) {
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  const auto* av = reinterpret_cast<const uint8_t*>(&a);
  const auto* bv = reinterpret_cast<const uint8_t*>(&b);
  return c + av[0] * bv[0] + av[1] * bv[1] + av[2] * bv[2] + av[3] * bv[3];
#endif
}

}  // namespace vt::cuda

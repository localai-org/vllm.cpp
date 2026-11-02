// Port of the exact signed-byte fallback in llama.cpp common.cuh:712-753
// @ 62318cd57dc63900ecb153272c0b10f78550855d. The unsigned case preserves
// EXL3's byte-sum arithmetic. Build sm_60 to exercise emulation on sm_61.
#include "vt/cuda/cuda_legacy_intrinsics.cuh"

#include <cstdio>

namespace {
constexpr int kCases = 4096;
__global__ void DotCases(int* signed_out, unsigned* unsigned_out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kCases) return;
  const unsigned a = i * 2654435761u;
  const unsigned b = (i ^ 0x8080ff7fu) * 2246822519u;
  signed_out[i] = vt::cuda::LegacyDp4a(static_cast<int>(a), static_cast<int>(b), i & 1 ? 0x7fffffff : static_cast<int>(0x80000000u));
  unsigned_out[i] = vt::cuda::LegacyDp4a(a, b, 0xfffffff0u);
}
}

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  int* signed_out = nullptr;
  unsigned* unsigned_out = nullptr;
  if (cudaMallocManaged(&signed_out, kCases * sizeof(int)) != cudaSuccess ||
      cudaMallocManaged(&unsigned_out, kCases * sizeof(unsigned)) != cudaSuccess) return 1;
  DotCases<<<kCases / 128, 128>>>(signed_out, unsigned_out);
  if (cudaDeviceSynchronize() != cudaSuccess) return 1;
  for (unsigned i = 0; i < kCases; ++i) {
    const unsigned a = i * 2654435761u;
    const unsigned b = (i ^ 0x8080ff7fu) * 2246822519u;
    unsigned signed_ref = i & 1 ? 0x7fffffffu : 0x80000000u;
    unsigned unsigned_ref = 0xfffffff0u;
    for (unsigned j = 0; j < 4; ++j) {
      const auto av = static_cast<uint8_t>(a >> (j * 8));
      const auto bv = static_cast<uint8_t>(b >> (j * 8));
      signed_ref += static_cast<unsigned>(static_cast<int8_t>(av) * static_cast<int8_t>(bv));
      unsigned_ref += static_cast<unsigned>(av) * bv;
    }
    if (static_cast<unsigned>(signed_out[i]) != signed_ref || unsigned_out[i] != unsigned_ref) {
      std::fprintf(stderr, "DP4A mismatch at case %u\n", i);
      return 1;
    }
  }
  cudaFree(signed_out);
  cudaFree(unsigned_out);
  std::puts("4096 signed and unsigned DP4A cases passed");
}

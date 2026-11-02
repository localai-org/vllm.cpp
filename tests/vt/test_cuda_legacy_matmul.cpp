// Production registered operators on SmolLM2-135M matrix shapes. The pinned
// llama.cpp CUDA dispatch (ggml-cuda.cu:1672-1690 @ 62318cd57) similarly selects
// CUDA-core float arithmetic where native BF16 hardware is unavailable.
#include <cuda_runtime.h>
#include "vt/ops.h"
#include "vt/op_provider.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Allocation {
  void* data = nullptr;
  explicit Allocation(size_t bytes) { Check(cudaMallocManaged(&data, bytes)); }
  ~Allocation() { cudaFree(data); }
};
float Value(int64_t offset, bool weight) {
  return static_cast<float>((offset * (weight ? 7 : 3)) % 17 - 8) / 8;
}
void Run(int64_t m, int64_t k, int64_t n, int op, vt::DType input, vt::DType output) {
  std::printf("m=%lld k=%lld n=%lld op=%d input=%d output=%d\n", static_cast<long long>(m), static_cast<long long>(k), static_cast<long long>(n), op, static_cast<int>(input), static_cast<int>(output));
  std::fflush(stdout);
  const int64_t groups = op == 2 ? 3 : 1;
  const int64_t arow = k + (op == 1 ? 7 : op == 2 ? 3 : 0);
  const int64_t brow = n + (op == 2 ? 5 : 0);
  const int64_t orow = n + (op == 2 ? 9 : 0);
  const int64_t asize = groups * m * arow, bsize = groups * k * brow;
  Allocation a(static_cast<size_t>(asize) * vt::SizeOf(input));
  Allocation b(static_cast<size_t>(bsize) * vt::SizeOf(input));
  Allocation out(static_cast<size_t>(groups * m * orow) * vt::SizeOf(output));
  for (int64_t i = 0; i < asize; ++i) {
    if (input == vt::DType::kBF16) static_cast<uint16_t*>(a.data)[i] = vt::F32ToBF16(Value(i, false));
    else static_cast<float*>(a.data)[i] = Value(i, false);
  }
  for (int64_t i = 0; i < bsize; ++i) {
    if (input == vt::DType::kBF16) static_cast<uint16_t*>(b.data)[i] = vt::F32ToBF16(Value(i, true));
    else static_cast<float*>(b.data)[i] = Value(i, true);
  }
  const vt::Device device{vt::DeviceType::kCUDA, 0};
  vt::Queue queue{device, nullptr};
  auto ta = vt::Tensor::Contiguous(a.data, input, device, {m, arow});
  ta.shape[1] = k;
  auto tb = vt::Tensor::Contiguous(b.data, input, device, op == 1 ? std::initializer_list<int64_t>{n,k} : std::initializer_list<int64_t>{k,n});
  auto to = vt::Tensor::Contiguous(out.data, output, device, {m,n});
  if (op == 0) vt::Matmul(queue, to, ta, tb);
  if (op == 1) vt::MatmulBT(queue, to, ta, tb);
  if (op == 2) {
    ta = vt::Tensor::Contiguous(a.data, input, device, {groups,m,arow});
    ta.shape[2] = k;
    tb = vt::Tensor::Contiguous(b.data, input, device, {groups,k,brow});
    tb.shape[2] = n;
    to = vt::Tensor::Contiguous(out.data, output, device, {groups,m,orow});
    to.shape[2] = n;
    vt::BatchedMatmul(queue, to, ta, tb);
  }
  Check(cudaDeviceSynchronize());
  // Check boundaries and interior cells. Large vocabulary matrices stay a
  // device workload, rather than spending seconds in a full host GEMM.
  for (int64_t g = 0; g < groups; ++g) for (int64_t r : {int64_t{0},m/2,m-1})
    for (int64_t c : {int64_t{0},n/2,n-1}) {
      float expected = 0;
      for (int64_t j = 0; j < k; ++j)
        expected += Value(g*m*arow+r*arow+j, false) * Value(g*k*brow+(op == 1 ? c*k+j : j*brow+c), true);
      if (output == vt::DType::kBF16) expected = vt::BF16ToF32(vt::F32ToBF16(expected));
      const auto index = g*m*orow+r*orow+c;
      const float actual = output == vt::DType::kBF16 ? vt::BF16ToF32(static_cast<uint16_t*>(out.data)[index]) : static_cast<float*>(out.data)[index];
      if (std::abs(actual-expected) > 0.001f)
        throw std::runtime_error("registered CUDA matrix product differs from exact dyadic reference");
    }
}
}

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  try {
    cudaDeviceProp properties{};
    Check(cudaGetDeviceProperties(&properties, 0));
    std::printf("device=%s sm=%d%d\n", properties.name, properties.major, properties.minor);
    const auto reference_hits = vt::GetReferenceTierHits();
    vt::EnableOpProviderCallStats(true);
    for (auto output : {vt::DType::kF32,vt::DType::kBF16}) {
      for (auto m : {1,16,128}) {
        Run(m,576,576,1,vt::DType::kBF16,output);
        Run(m,576,1536,0,vt::DType::kBF16,output);
        Run(m,1536,576,1,vt::DType::kBF16,output);
        Run(m,576,49152,1,vt::DType::kBF16,output);
      }
      Run(16,64,128,2,vt::DType::kBF16,output);
      Run(17,31,13,2,vt::DType::kF32,output);
    }
    if (vt::GetReferenceTierHits() != reference_hits)
      throw std::runtime_error("CUDA matrix workload fell back to the host reference");
    for (auto op : {vt::OpId::kMatmul, vt::OpId::kMatmulBT, vt::OpId::kBatchedMatmul}) {
      const auto stats = vt::GetOpProviderStats(op, vt::DeviceType::kCUDA);
      if (!stats.selections || !stats.last_selected || std::string(stats.last_selected) != vt::kNativeProviderName)
        throw std::runtime_error("CUDA native matrix provider was not selected");
    }
    std::puts("28 registered CUDA matrix workloads passed");
  } catch (const std::exception& error) {
    std::fprintf(stderr,"%s\n",error.what());
    return 1;
  }
}

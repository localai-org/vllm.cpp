// Reach the registered native EXL3 launchers through the shared op surface.
#include <cuda_runtime.h>
#include "vt/ops.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Buffer {
  void* data = nullptr;
  explicit Buffer(size_t bytes) {
    if (cudaMallocManaged(&data, bytes) != cudaSuccess) throw std::runtime_error("CUDA allocation failed");
  }
  ~Buffer() { cudaFree(data); }
};
template <class F> void Refuses(F call) {
  try { call(); } catch (const std::runtime_error& error) {
    const std::string message(error.what());
    if (message.find("sm_80") != std::string::npos && message.find("EXL3") != std::string::npos) return;
    throw;
  }
  throw std::runtime_error("native EXL3 unexpectedly accepted a pre-sm_80 device");
}
}

int main() {
  int count = 0;
  cudaDeviceProp properties{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 1;
  if (properties.major >= 8) return 77;
  try {
    const vt::Device device{vt::DeviceType::kCUDA,0};
    vt::Queue queue{device,nullptr};
    std::vector<std::unique_ptr<Buffer>> buffers;
    auto tensor = [&](vt::DType dtype, std::initializer_list<int64_t> shape) {
      size_t bytes = vt::SizeOf(dtype);
      for (auto dimension : shape) bytes *= static_cast<size_t>(dimension);
      buffers.push_back(std::make_unique<Buffer>(bytes));
      return vt::Tensor::Contiguous(buffers.back()->data,dtype,device,shape);
    };
    auto a = tensor(vt::DType::kF16,{1,128});
    auto c = tensor(vt::DType::kF16,{1,128});
    auto had = tensor(vt::DType::kF16,{1,128});
    auto trellis = tensor(vt::DType::kI8,{8,8,96});
    auto suh = tensor(vt::DType::kF16,{128});
    auto svh = tensor(vt::DType::kF16,{128});
    vt::Exl3GemmArgs gemm;
    gemm.bits = 3;
    gemm.codebook = 0;
    Refuses([&] { vt::Exl3Gemm(queue,c,a,trellis,suh,svh,had,gemm); });
    auto out = tensor(vt::DType::kF32,{1,128});
    auto table = tensor(vt::DType::kI64,{1});
    auto counts = tensor(vt::DType::kI64,{2});
    auto tokens = tensor(vt::DType::kI64,{1});
    auto weights = tensor(vt::DType::kF16,{1});
    auto sg = tensor(vt::DType::kF16,{1,1,128});
    auto su = tensor(vt::DType::kF16,{1,1,128});
    auto ig = tensor(vt::DType::kF16,{1,1,128});
    auto iu = tensor(vt::DType::kF16,{1,1,128});
    const vt::Exl3MoeExpertTables tables{&table,&table,&table,&table,&table,&table,&table,&table,&table};
    const vt::Exl3MoeRouting routing{&counts,&tokens,&weights};
    const vt::Exl3MoeTemps temps{&sg,&su,&ig,&iu};
    vt::Exl3MoeArgs moe;
    moe.bits_gate = moe.bits_up = moe.bits_down = 3;
    moe.num_active = 1;
    Refuses([&] { vt::Exl3MoeMlp(queue,out,a,tables,routing,temps,moe); });
    std::puts("native EXL3 GEMM and MoE refuse pre-sm_80 with the missing operation named");
  } catch (const std::exception& error) {
    std::fprintf(stderr,"%s\n",error.what());
    return 1;
  }
}

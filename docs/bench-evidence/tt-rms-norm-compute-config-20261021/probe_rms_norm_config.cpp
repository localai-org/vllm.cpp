// SCRATCH PROBE (not for commit): measure ttnn::rms_norm against a host-double
// oracle on a captured in-model row, under an explicitly chosen compute-kernel
// config. Mirrors vllm.cpp's device rms path: f32 TILE input, f32 TILE gamma,
// eps=1e-6, one row.
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/device.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#if __has_include(<tt-metalium/tensor/spec/tensor_spec.hpp>)
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#else
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#endif
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <fstream>

namespace {
tt::tt_metal::TensorSpec SpecOf(tt::tt_metal::Shape shape, ttnn::DataType dtype,
                                ttnn::Layout layout) {
  return tt::tt_metal::TensorSpec(
      std::move(shape),
      tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout),
                                 tt::tt_metal::MemoryConfig{}));
}
float bf16_trunc(float f) {  // the harness oracle: keep the top 16 bits
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t v = (uint32_t)(uint16_t)(u >> 16) << 16;
  float o; std::memcpy(&o, &v, 4); return o;
}
float bf16_rne(float f) {  // the device's f32 -> bf16 store
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  uint32_t v = (uint32_t)(uint16_t)(u >> 16) << 16;
  float o; std::memcpy(&o, &v, 4); return o;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: probe <row.bin> <mode>\n"); return 2; }
  const char* bin = argv[1];
  const int mode = std::atoi(argv[2]);
  std::ifstream f(bin, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", bin); return 2; }
  uint32_t hdr[2] = {0, 0};
  f.read(reinterpret_cast<char*>(hdr), 8);
  const uint32_t rows = hdr[0], d = hdr[1];
  const size_t n = static_cast<size_t>(rows) * d;
  std::vector<float> nf(n), xv(n), wf(d);
  f.read(reinterpret_cast<char*>(nf.data()), 4 * n);
  f.read(reinterpret_cast<char*>(xv.data()), 4 * n);
  f.read(reinterpret_cast<char*>(wf.data()), 4 * d);
  double maxdiff = 0.0;
  for (size_t i = 0; i < n; ++i)
    maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(nf[i]) - xv[i]));

  auto device = ttnn::open_mesh_device(0);
  const auto arch = tt::ARCH::BLACKHOLE;  // P150
  // The harness micro uses the FIRST row only ([1, d]).
  std::vector<float> x0(xv.begin(), xv.begin() + d);
  std::vector<float> w0(wf.begin(), wf.end());
  auto x = ttnn::Tensor::from_vector<float>(
      x0, SpecOf(tt::tt_metal::Shape({1, d}), ttnn::DataType::FLOAT32, ttnn::Layout::TILE),
      device.get());
  auto w = ttnn::Tensor::from_vector<float>(
      w0, SpecOf(tt::tt_metal::Shape({1, d}), ttnn::DataType::FLOAT32, ttnn::Layout::TILE),
      device.get());
  std::optional<ttnn::DeviceComputeKernelConfig> cfg;
  if (mode == 1) cfg = ttnn::init_device_compute_kernel_config(arch, std::nullopt, tt::tt_metal::MathFidelity::HiFi4, /*approx_mode=*/false, /*fp32_acc=*/true);
  if (mode == 2) cfg = ttnn::init_device_compute_kernel_config(arch, std::nullopt, tt::tt_metal::MathFidelity::HiFi4, /*approx_mode=*/true,  /*fp32_acc=*/false);
  if (mode == 3) cfg = ttnn::init_device_compute_kernel_config(arch, std::nullopt, tt::tt_metal::MathFidelity::HiFi4, /*approx_mode=*/false, /*fp32_acc=*/false);
  if (mode == 4) cfg = ttnn::init_device_compute_kernel_config(arch, std::nullopt, tt::tt_metal::MathFidelity::HiFi4, /*approx_mode=*/true,  /*fp32_acc=*/true);
  auto y = ttnn::rms_norm(x, 1e-6f, w, std::nullopt, std::nullopt, std::nullopt,
                          std::nullopt, cfg);
  auto yv = y.to_vector<float>();
  if (yv.size() < d) { std::fprintf(stderr, "short output %zu\n", yv.size()); return 3; }
  double sumsq = 0.0;
  for (uint32_t i = 0; i < d; ++i) { double v = x0[i]; sumsq += v * v; }
  const double inv = 1.0 / std::sqrt(sumsq / static_cast<double>(d) + 1e-6);
  double sh = 0.0, shf = 0.0, st = 0.0, stf = 0.0;
  int ndiff = 0;
  for (uint32_t i = 0; i < d; ++i) {
    const double h = static_cast<double>(x0[i]) * inv * static_cast<double>(w0[i]);
    sh += bf16_trunc(static_cast<float>(h));
    shf += h;
    const float t = yv[i];
    st += bf16_rne(t);
    stf += t;
    if (bf16_rne(t) != bf16_trunc(static_cast<float>(h))) ++ndiff;
  }
  std::printf("PROBE mode=%d rows=%u d=%u nf_vs_xv_maxdiff=%g "
              "ratio_bf16=%.6f ratio_f32=%.6f ndiff=%d\n",
              mode, rows, d, maxdiff, st / sh, stf / shf, ndiff);
  return 0;
}

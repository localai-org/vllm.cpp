// SCRATCH W3 diagnostic, plain-ttnn leg: a raw ttnn::matmul through
// tt-metal's own tensor API (Tensor::from_vector -> to_device), no vt
// staging, no vt ops. Spec .agents/specs/tt-matmul-noise.md. If the noise
// reproduces here it lives in the pinned tt-metal kernels, not in our stack.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <doctest/doctest.h>

#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/device.hpp>


namespace {

uint16_t ToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t lsb = (u >> 16) & 1;
  u += 0x7fff + lsb;
  return static_cast<uint16_t>(u >> 16);
}

float FromBf16(uint16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

TEST_CASE("SCRATCH dbg W3 plain ttnn matmul repro") {
  auto mesh = ttnn::open_mesh_device(/*device_id=*/0, /*l1_small_size=*/0,
                                     /*trace_region_size=*/0);
  // (open_mesh_device returns a shared_ptr in this pin; keep it alive.)
  const int64_t M = 1, K = 512, N = 512;
  std::mt19937 rng(31);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  std::vector<uint16_t> xb(static_cast<size_t>(M * K)),
      wb(static_cast<size_t>(K * N));
  std::vector<float> xf(static_cast<size_t>(M * K)),
      wf(static_cast<size_t>(K * N));
  for (size_t i = 0; i < xb.size(); ++i) {
    xb[i] = ToBf16(nd(rng));
    xf[i] = FromBf16(xb[i]);
  }
  for (size_t i = 0; i < wb.size(); ++i) {
    wb[i] = ToBf16(nd(rng) * 0.02f);
    wf[i] = FromBf16(wb[i]);
  }
  std::vector<double> ref(static_cast<size_t>(M * N), 0.0);
  for (int64_t k = 0; k < K; ++k) {
    const double a = static_cast<double>(xf[static_cast<size_t>(k)]);
    for (int64_t n = 0; n < N; ++n)
      ref[static_cast<size_t>(n)] +=
          a * static_cast<double>(wf[static_cast<size_t>(k * N + n)]);
  }

  const auto mk = [&](const std::vector<float>& host, int64_t r,
                      int64_t c) {
    const tt::tt_metal::TensorLayout lay(
        ttnn::DataType::BFLOAT16, tt::tt_metal::PageConfig(ttnn::Layout::TILE),
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                           tt::tt_metal::BufferType::DRAM});
    const tt::tt_metal::TensorSpec spec(ttnn::Shape({static_cast<unsigned int>(r), static_cast<unsigned int>(c)}), lay);
    return ttnn::Tensor::from_vector(host, spec).to_device(
        mesh.get(), std::nullopt, ttnn::QueueId(0));
  };
  ttnn::Tensor a = mk(xf, M, K);
  ttnn::Tensor b = mk(wf, K, N);

  const auto run = [&](const char* tag,
                       std::optional<ttnn::DeviceComputeKernelConfig> cfg) {
    const ttnn::Tensor c = ttnn::operations::matmul::matmul(
        a, b, false, false, std::nullopt, ttnn::DataType::FLOAT32,
        std::nullopt, std::nullopt, cfg);
    const std::vector<float> got = c.to_vector<float>();
    double se = 0.0, sr = 0.0, max_abs = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      const double d = static_cast<double>(got[i]) - ref[i];
      max_abs = std::max(max_abs, std::fabs(d));
      se += d * d;
      sr += ref[i] * ref[i];
      if (std::fabs(ref[i]) > 1e-12 &&
          std::fabs(d) > 0.5 * std::fabs(ref[i]))
        ++bad;
    }
    const double n = static_cast<double>(ref.size());
    MESSAGE(tag << " max_abs=" << max_abs
                << " rel_rms=" << std::sqrt(se / n) / std::sqrt(sr / n)
                << " frac_bad=" << static_cast<double>(bad) / n);
  };

  // arm 1: NO config (the ttnn default: LoFi, approx_mode, no f32 dest).
  run("plainttnn-default", std::nullopt);
  // arm 2: HiFi4, approx off, no f32 dest.
  ttnn::DeviceComputeKernelConfig hifi4{};
  hifi4.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
  hifi4.math_approx_mode = false;
  run("plainttnn-hifi4", hifi4);
  // arm 3: HiFi4 + fp32_dest_acc_en (what vt::Matmul's f32-out arm sets,
  // plus the fidelity fix).
  ttnn::DeviceComputeKernelConfig hifi4f32 = hifi4;
  hifi4f32.fp32_dest_acc_en = true;
  run("plainttnn-hifi4-f32dest", hifi4f32);
  // arm 4: LoFi + f32 dest (isolates whether the 2.5e-2 noise is LoFi's
  // product truncation or the f32-dest path itself).
  ttnn::DeviceComputeKernelConfig lofif32{};
  lofif32.math_fidelity = tt::tt_metal::MathFidelity::LoFi;
  lofif32.fp32_dest_acc_en = true;
  run("plainttnn-lofi-f32dest", lofif32);
  // arm 5: HiFi2 + f32 dest (the compromise tt-metal demos use).
  ttnn::DeviceComputeKernelConfig hifi2f32 = hifi4f32;
  hifi2f32.math_fidelity = tt::tt_metal::MathFidelity::HiFi2;
  run("plainttnn-hifi2-f32dest", hifi2f32);
}

#include "exl3_fixture.h"
#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/xpu/xpu_kernels.h"
#include "vt/xpu/xpu_common.h"
#include <iostream>
#include <array>
#include <future>
#include <map>

namespace {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;

std::vector<float> Run(vt::Queue& q, int m, int k, int n, int bits, DType dtype,
                       bool matrix = true, bool public_gemm = false,
                       const char* matrix_name = nullptr, bool all_rows = false) {
  const auto fixture = exl3_test::MakeFixture(k, n, bits, 0x976315u + bits);
  Buffer a(q, DType::kF16, {m, k}), ah(q, DType::kF16, {m, k});
  Buffer trellis(q, DType::kI8, {k / 16, n / 16, bits * 32});
  Buffer suh(q, DType::kF16, {k}), svh(q, DType::kF16, {n}), out(q, dtype, {m, n});
  exl3_test::Rng rng;
  std::vector<uint16_t> input(m * k);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = vt::F32ToF16(i % 19 == 0 ? 0 : rng.next(0.2f));
  a.upload(input.data()); trellis.upload(fixture.trellis.data());
  suh.upload(fixture.suh.data()); svh.upload(fixture.svh.data());
  if (q.device.type == vt::DeviceType::kXPU && !public_gemm) {
    vt::Exl3HadR128(q, ah.tensor, a.tensor, {&suh.tensor, nullptr, 1.0f});
    REQUIRE(vt::xpu::Exl3PrefillKernel(q, out.tensor, ah.tensor, trellis.tensor, svh.tensor,
                                      bits, matrix, all_rows));
  } else {
    vt::Exl3GemmArgs args{bits, 2};
    args.debug_name = matrix_name;
    vt::Exl3Gemm(q, out.tensor, a.tensor, trellis.tensor, suh.tensor, svh.tensor, ah.tensor, args);
  }
  return out.floats();
}

void Accuracy(const std::vector<float>& got, const std::vector<float>& ref, DType dtype) {
  REQUIRE(got.size() == ref.size());
  double error = 0, norm = 0, peak = 0, peak_error = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) FAIL("Non-finite panel GEMM output");
    const double diff = double(got[i]) - ref[i];
    error += diff * diff; norm += double(ref[i]) * ref[i];
    peak = std::max(peak, std::abs(double(ref[i])));
    peak_error = std::max(peak_error, std::abs(diff));
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-30));
  CAPTURE(relative);
  CAPTURE(peak_error);
  // F32 uses the predeclared matrix-probe budget. The F16 output additionally
  // rounds at its dtype boundary; retain the public GEMM contract's RMS 1e-3.
  CHECK(relative <= (dtype == DType::kF32 ? 3e-5 : 1e-3));
  CHECK(peak_error <= 2e-6 + (dtype == DType::kF32 ? 3e-4 : 1e-3) * peak);
}
}

TEST_CASE("XPU EXL3 all-row prefill reuses K panels across a padded M tail") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const auto reference = Run(cpu.q, 257, 1152, 256, 4, DType::kF32);
  vt::xpu::DrainProfileEvents();
  Accuracy(Run(gpu.q, 257, 1152, 256, 4, DType::kF32,
               true, false, nullptr, true), reference, DType::kF32);
  if (std::getenv("VT_XPU_PROFILE")) {
    std::map<std::string, size_t> counts;
    for (const auto& record : vt::xpu::DrainProfileEvents()) ++counts[record.stage];
    CHECK(counts["exl3_panel_decode"] == 2);
    CHECK(counts["exl3_panel_gemm"] == 2);
    CHECK(counts["exl3_output_hadamard"] == 1);
  }
  const auto column_tail = Run(cpu.q, 129, 128, 4224, 4, DType::kF32);
  Accuracy(Run(gpu.q, 129, 128, 4224, 4, DType::kF32,
               true, false, nullptr, true), column_tail, DType::kF32);
  const auto max_rows = Run(cpu.q, 6656, 128, 128, 4, DType::kF32);
  Accuracy(Run(gpu.q, 6656, 128, 128, 4, DType::kF32,
               true, false, nullptr, true), max_rows, DType::kF32);
  CHECK(vt::xpu::GetMemoryInfo().exl3_workspace_bytes == 32 * 1024 * 1024);
}

TEST_CASE("XPU EXL3 prefill: packed widths, output dtypes, row and column panel tails") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int bits : {3, 4, 5, 6}) for (auto dtype : {DType::kF16, DType::kF32}) {
    CAPTURE(bits);
    CAPTURE(dtype);
    const auto reference = Run(cpu.q, 128, 256, 768, bits, dtype);
    Accuracy(Run(gpu.q, 128, 256, 768, bits, dtype), reference, dtype);
    CHECK(Run(gpu.q, 128, 256, 768, bits, dtype, false) == reference);
  }
  for (int m : {1, 31, 129}) for (int bits : {3, 6}) {
    CAPTURE(m);
    CAPTURE(bits);
    const auto reference = Run(cpu.q, m, 128, 640, bits, DType::kF32);
    Accuracy(Run(gpu.q, m, 128, 640, bits, DType::kF32), reference, DType::kF32);
    CHECK(Run(gpu.q, m, 128, 640, bits, DType::kF32, false) == reference);
  }
  for (const auto shape : {std::array{257, 128, 256}, std::array{31, 1152, 256},
                           std::array{31, 128, 4224}}) {
    CAPTURE(shape[0]);
    CAPTURE(shape[1]);
    CAPTURE(shape[2]);
    const auto reference = Run(cpu.q, shape[0], shape[1], shape[2], 4, DType::kF32);
    Accuracy(Run(gpu.q, shape[0], shape[1], shape[2], 4, DType::kF32), reference, DType::kF32);
    CHECK(Run(gpu.q, shape[0], shape[1], shape[2], 4, DType::kF32, false) == reference);
  }
  const auto info = vt::xpu::GetMemoryInfo();
  CHECK(info.exl3_workspace_bytes == 32 * 1024 * 1024);
  CHECK(info.allocated_bytes == info.exl3_workspace_bytes);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 prefill: profiled stages retain submission timestamps"
          * doctest::skip(!std::getenv("VT_XPU_PROFILE"))) {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  const auto reference = Run(cpu.q, 31, 128, 256, 4, DType::kF32);
  Accuracy(Run(gpu.q, 31, 128, 256, 4, DType::kF32), reference, DType::kF32);
  const auto records = vt::xpu::DrainProfileEvents();
  std::map<std::string, size_t> counts;
  for (const auto& record : records) {
    ++counts[record.stage];
    CHECK(record.queue_id == gpu.q.id);
    CHECK(record.submit_ns > 0);
    CHECK(record.start_ns > 0);
    CHECK(record.end_ns >= record.start_ns);
  }
  CHECK(counts["exl3_input_hadamard"] == 1);
  CHECK(counts["exl3_panel_decode"] == 1);
  CHECK(counts["exl3_panel_gemm"] == 1);
  CHECK(counts["exl3_output_hadamard"] == 1);
  CHECK(vt::xpu::DrainProfileEvents().empty());
  constexpr const char* name = "model.layers.7.mlp.down_proj";
  Accuracy(Run(gpu.q, 31, 128, 256, 4, DType::kF32, true, true, name), reference, DType::kF32);
  const auto named = vt::xpu::DrainProfileEvents();
  REQUIRE_FALSE(named.empty());
  for (const auto& record : named) CHECK(record.matrix == name);
}

TEST_CASE("XPU EXL3 prefill: shared workspace survives queue replacement without growth") {
  Queue cpu(vt::DeviceType::kCPU);
  const auto expected = Run(cpu.q, 129, 128, 640, 4, DType::kF32);
  std::vector<float> first;
  for (int round = 0; round < 3; ++round) {
    Queue gpu(vt::DeviceType::kXPU);
    const auto actual = Run(gpu.q, 129, 128, 640, 4, DType::kF32);
    Accuracy(actual, expected, DType::kF32);
    if (round == 0) first = actual; else CHECK(actual == first);
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 32 * 1024 * 1024);
  }
}

TEST_CASE("XPU EXL3 prefill: two queues serialize access to one workspace") {
  Queue left(vt::DeviceType::kXPU), right(vt::DeviceType::kXPU);
  auto run = [](vt::Queue& q, int value) {
    auto& native = vt::xpu::NativeQueue(q);
    std::vector<int> got(4096);
    for (int round = 0; round < 8; ++round) {
      if (!vt::xpu::WithExl3Workspace(q, 32 * 1024 * 1024, [&](void* storage) {
        native.fill(static_cast<int*>(storage), value + round, got.size());
        native.memcpy(got.data(), storage, got.size() * sizeof(int));
      })) return false;
      for (int x : got) if (x != value + round) return false;
    }
    return true;
  };
  auto a = std::async(std::launch::async, run, std::ref(left.q), 100);
  auto b = std::async(std::launch::async, run, std::ref(right.q), 200);
  CHECK(a.get()); CHECK(b.get());
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == 32 * 1024 * 1024);
}

TEST_CASE("XPU EXL3 prefill: insufficient workspace budget falls back to packed"
          * doctest::skip(!std::getenv("VT_B70_LOW_MEMORY_TEST"))) {
  // Run alone in a fresh process with VT_XPU_MEMORY_BUDGET_BYTES=8388608 and
  // VT_XPU_EXL3_STRATEGY=prefill, since the context/strategy initialize once.
  const char* strategy = std::getenv("VT_XPU_EXL3_STRATEGY");
  REQUIRE(strategy != nullptr);
  REQUIRE(std::string(strategy) == "prefill");
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::xpu::GetMemoryInfo().budget_bytes == 8 * 1024 * 1024);
  const auto expected = Run(cpu.q, 128, 128, 256, 4, DType::kF32);
  CHECK(Run(gpu.q, 128, 128, 256, 4, DType::kF32, true, true) == expected);
  CHECK(vt::xpu::GetMemoryInfo().exl3_workspace_bytes == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}

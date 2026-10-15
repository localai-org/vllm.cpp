#include "xpu_test_helpers.h"
#include "support/exl3_external_artifacts.h"
#include "vt/exl3_grouped.h"
#include "vt/exl3_w8a8_panel_plan.h"
#include "vt/xpu/xpu_common.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>

using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;

TEST_CASE("XPU EXL3 W8A8 F2: direct preparation budget fallback and reuse") {
  const char* fixture = exl3_test::ExternalEnvironment("VT_B70_EXL3_W8A8_FIXTURE");
  exl3_test::ExternalPath(fixture);
  (void)exl3_test::ExternalEnvironment("VT_XPU_MEMORY_BUDGET_BYTES");
  const auto f = vllm::SafetensorsFile::Open(fixture);
  const auto& tr = f.Get("merged_trellis"); const auto& su = f.Get("stacked_suh");
  const auto& sv = f.Get("merged_svh"); const auto& routing = f.Get("source_map");
  const int m = 256, k = int(su.shape[1]), n = int(sv.shape[0]);
  const int groups = int(su.shape[0]), bits = int(tr.shape[2] / 16);
  struct Restore {
    std::array<bool, 2> present;
    std::array<std::string, 2> value;
    const std::array<const char*, 2> names{"VT_XPU_W8A8_PREPARE", "VT_XPU_W8A8_DIRECT_PREPARE"};
    Restore() {
      for (size_t i = 0; i < names.size(); ++i) {
        const char* old = std::getenv(names[i]); present[i] = old != nullptr; value[i] = old ? old : "";
        REQUIRE(setenv(names[i], "0", 1) == 0);
      }
    }
    ~Restore() {
      for (size_t i = 0; i < names.size(); ++i) {
        if (present[i]) setenv(names[i], value[i].c_str(), 1);
        else unsetenv(names[i]);
      }
    }
  } restore;
  Queue gpu(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(gpu.q.device);
  Buffer trellis(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer input(gpu.q, DType::kF16, {m, k}), out(gpu.q, DType::kF16, {m, n});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(sv.data);
  input.upload(f.Get("input_m256").data);
  auto owner = std::shared_ptr<void>(vt::Alloc(gpu.q.device, routing.nbytes),
      [device = gpu.q.device](void* ptr) { vt::Free(device, ptr); });
  auto map = vt::Tensor::Contiguous(owner.get(), DType::kI32, gpu.q.device, {n / 128});
  backend.Copy(gpu.q, map.data, routing.data, routing.nbytes); backend.Synchronize(gpu.q);
  vt::SharedPtrCache<const vt::Exl3W8A8ModelMap> cache;
  const vt::Exl3GroupedLinearArgs args{bits, 2, "F2_DIRECT_BUDGET", 1024};
  auto execute = [&] { vt::detail::Exl3GroupedW8A8Model(gpu.q, out.tensor, input.tensor,
      trellis.tensor, suh.tensor, svh.tensor, map, args, owner, cache); };
  execute(); const auto expected = out.download();
  const auto before = vt::xpu::GetMemoryInfo();
  REQUIRE(before.budget_bytes <= 256 * 1024 * 1024);
  REQUIRE(before.w8a8_preparation_bytes == 0); // Isolated fresh worker.
  const auto plan = vt::PlanExl3W8A8(m, k, n, groups, bits, 1024);
  const size_t needed = plan.intermediate_offset;
  const size_t remaining = before.budget_bytes - before.allocated_bytes - before.graph_device_bytes;
  REQUIRE(remaining > needed);
  REQUIRE(setenv("VT_XPU_W8A8_PREPARE", "1", 1) == 0);
  REQUIRE(setenv("VT_XPU_W8A8_DIRECT_PREPARE", "1", 1) == 0);
  {
    Buffer pressure(gpu.q, DType::kI8, {int64_t(remaining - needed / 2)});
    const auto pressured = vt::xpu::GetMemoryInfo();
    REQUIRE(pressured.budget_bytes - pressured.allocated_bytes - pressured.graph_device_bytes < needed);
    execute(); xpu_test::SameBytes(out.download(), expected);
    const auto refused = vt::xpu::GetMemoryInfo();
    CHECK(refused.w8a8_preparation_bytes == 0);
    CHECK(refused.allocated_bytes == pressured.allocated_bytes);
  }
  execute(); xpu_test::SameBytes(out.download(), expected);
  const auto acquired = vt::xpu::GetMemoryInfo();
  CHECK(acquired.w8a8_preparation_bytes == needed);
  CHECK(acquired.peak_allocated_bytes <= acquired.budget_bytes);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 W8A8 F2: direct preparation complete calls and queued real consumers") {
  const char* fixture = exl3_test::ExternalEnvironment("VT_B70_EXL3_W8A8_FIXTURE");
  exl3_test::ExternalPath(fixture);
  const char* prior = std::getenv("VT_XPU_W8A8_DIRECT_PREPARE");
  struct Restore {
    bool present; std::string value;
    ~Restore() {
      if (present) setenv("VT_XPU_W8A8_DIRECT_PREPARE", value.c_str(), 1);
      else unsetenv("VT_XPU_W8A8_DIRECT_PREPARE");
    }
  } restore{prior != nullptr, prior ? prior : ""};
  const auto f = vllm::SafetensorsFile::Open(fixture);
  const int m = std::getenv("VT_B70_EXL3_PANEL_ROWS") ? std::atoi(std::getenv("VT_B70_EXL3_PANEL_ROWS")) : 129;
  REQUIRE((m == 129 || m == 896 || m == 1600));
  const auto& tr = f.Get("merged_trellis"); const auto& su = f.Get("stacked_suh");
  const auto& sv = f.Get("merged_svh"); const auto& routing = f.Get("source_map");
  const int k = int(su.shape[1]), n = int(sv.shape[0]), groups = int(su.shape[0]);
  const int bits = int(tr.shape[2] / 16);
  Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(first.q.device);
  Buffer trellis(first.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(first.q, DType::kF16, {groups, k}), svh(first.q, DType::kF16, {n});
  Buffer input(first.q, DType::kF16, {m, k}), zero(second.q, DType::kF16, {m, k});
  Buffer out(first.q, DType::kF16, {m, n}), other(second.q, DType::kF16, {m, n});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(sv.data);
  input.upload(f.Get("input_m" + std::to_string(m)).data);
  backend.Memset(second.q, zero.tensor.data, 0, zero.bytes); backend.Synchronize(second.q);
  auto owner = std::shared_ptr<void>(vt::Alloc(first.q.device, routing.nbytes),
      [device = first.q.device](void* ptr) { vt::Free(device, ptr); });
  auto map = vt::Tensor::Contiguous(owner.get(), DType::kI32, first.q.device, {n / 128});
  backend.Copy(first.q, map.data, routing.data, routing.nbytes); backend.Synchronize(first.q);
  const vt::Exl3GroupedLinearArgs args{bits, 2, "F2_DIRECT_PREPARE", 1024};
  const auto plan = vt::PlanExl3W8A8(m, k, n, groups, bits, 1024);
  Buffer scratch(first.q, DType::kI8, {int64_t(plan.workspace_bytes)});
  Buffer panel(first.q, DType::kI8, {k, plan.weight_panel_columns});
  auto select = [](bool on) { REQUIRE(setenv("VT_XPU_W8A8_DIRECT_PREPARE", on ? "1" : "0", 1) == 0); };
  auto explicit_call = [&] {
    vt::Exl3GroupedW8A8(first.q, out.tensor, input.tensor, trellis.tensor,
        suh.tensor, svh.tensor, map, scratch.tensor, panel.tensor, args);
  };
  select(false); backend.Memset(first.q, scratch.tensor.data, 0xa5, scratch.bytes);
  explicit_call(); const auto expected = out.download(), public_bytes = scratch.download();
  const auto panel_bytes = panel.download();
  select(true); explicit_call(); xpu_test::SameBytes(out.download(), expected);
  xpu_test::SameBytes(scratch.download(), public_bytes); xpu_test::SameBytes(panel.download(), panel_bytes);
  const auto& qref = f.Get("xq_m" + std::to_string(m));
  const auto& sref = f.Get("sx_m" + std::to_string(m));
  bool q_exact = true, scale_exact = true;
  for (int g = 0; g < groups; ++g) for (int row = 0; row < m; ++row) {
    q_exact &= std::memcmp(public_bytes.data() + plan.activation_offset + (size_t(g) * plan.padded_rows + row) * k,
                           qref.data + (size_t(g) * m + row) * k, k) == 0;
    scale_exact &= std::memcmp(public_bytes.data() + plan.row_scale_offset + (size_t(g) * plan.padded_rows + row) * 4,
                               sref.data + (size_t(g) * m + row) * 4, 4) == 0;
  }
  CHECK(q_exact); CHECK(scale_exact);
  const auto& yref = f.Get("y_m" + std::to_string(m));
  xpu_test::SameBytes(std::vector<unsigned char>(public_bytes.begin() + plan.intermediate_offset,
      public_bytes.begin() + plan.intermediate_offset + yref.nbytes),
      std::vector<unsigned char>(yref.data, yref.data + yref.nbytes));
  xpu_test::SameBytes(expected, std::vector<unsigned char>(f.Get("output_m" + std::to_string(m)).data,
      f.Get("output_m" + std::to_string(m)).data + f.Get("output_m" + std::to_string(m)).nbytes));
  vt::SharedPtrCache<const vt::Exl3W8A8ModelMap> cache;
  auto model_call = [&](vt::Queue& q, vt::Tensor& target, const vt::Tensor& source) {
    vt::detail::Exl3GroupedW8A8Model(q, target, source, trellis.tensor,
        suh.tensor, svh.tensor, map, args, owner, cache);
  };
  select(false); model_call(second.q, other.tensor, zero.tensor);
  const auto zero_expected = other.download();
  select(true); model_call(first.q, out.tensor, input.tensor);
  xpu_test::SameBytes(out.download(), expected);
  // Read the actual model-private preparation and compact GEMM workspace
  // after completion, before another producer can reuse either shared pool.
  std::vector<unsigned char> prepared_bytes(plan.intermediate_offset);
  const size_t compact_bytes = plan.workspace_bytes - plan.intermediate_offset;
  std::vector<unsigned char> compact(compact_bytes + plan.weight_panel_bytes);
  REQUIRE(vt::xpu::WithExl3W8A8Workspace(first.q, compact.size(), [&](void* storage) {
    REQUIRE(vt::xpu::WithExl3W8A8Preparation(first.q, prepared_bytes.size(), [&](void* prepared) {
      backend.Copy(first.q, prepared_bytes.data(), prepared, prepared_bytes.size());
      backend.Copy(first.q, compact.data(), storage, compact.size());
      backend.Synchronize(first.q);
    }));
  }));
  xpu_test::SameBytes(prepared_bytes, std::vector<unsigned char>(public_bytes.begin(),
      public_bytes.begin() + plan.intermediate_offset));
  xpu_test::SameBytes(std::vector<unsigned char>(compact.begin(), compact.begin() + yref.nbytes),
      std::vector<unsigned char>(yref.data, yref.data + yref.nbytes));
  const auto scale_offset = plan.weight_scale_offset - plan.intermediate_offset;
  xpu_test::SameBytes(std::vector<unsigned char>(compact.begin() + scale_offset,
      compact.begin() + scale_offset + sizeof(float)),
      std::vector<unsigned char>(public_bytes.begin() + plan.weight_scale_offset,
      public_bytes.begin() + plan.weight_scale_offset + sizeof(float)));
  xpu_test::SameBytes(std::vector<unsigned char>(compact.begin() + compact_bytes, compact.end()), panel_bytes);
  // Independent integer witness from the pinned original's final group, not
  // just equality between two native paths. The last panel is compact even
  // when its source group ends with fewer than1024 columns.
  REQUIRE(cache.Load());
  const int last_columns = cache.Load()->Panels(1024).back().columns;
  REQUIRE(last_columns >= 128);
  REQUIRE(last_columns <= plan.weight_panel_columns);
  const auto& last_ref = f.Get("last_weight_panel_m" + std::to_string(m));
  REQUIRE(last_ref.dtype == "I8");
  REQUIRE(last_ref.nbytes == size_t(k) * 128);
  std::vector<unsigned char> last_block(last_ref.nbytes);
  for (int row = 0; row < k; ++row)
    std::memcpy(last_block.data() + size_t(row) * 128,
        compact.data() + compact_bytes + size_t(row) * last_columns + last_columns - 128, 128);
  xpu_test::SameBytes(last_block,
      std::vector<unsigned char>(last_ref.data, last_ref.data + last_ref.nbytes));
  // Both pools are high-water allocations. This mixed-route timing process
  // cannot establish a peak-memory reduction; serving uses fresh workers.
  for (bool cross_queue : {false, true}) {
    model_call(first.q, out.tensor, input.tensor);
    model_call(cross_queue ? second.q : first.q, other.tensor, zero.tensor);
    backend.Synchronize(first.q); backend.Synchronize(second.q);
    xpu_test::SameBytes(out.download(), expected); xpu_test::SameBytes(other.download(), zero_expected);
  }
  // Cancel/retire the caller's result immediately after enqueue. Free must
  // complete its consumers; subsequent pool reuse must not touch retired USM.
  {
    Buffer cancelled(first.q, DType::kF16, {m, n});
    model_call(first.q, cancelled.tensor, input.tensor);
  }
  model_call(second.q, other.tensor, zero.tensor);
  xpu_test::SameBytes(other.download(), zero_expected);
  for (int i = 0; i < 16; ++i) {
    select(i % 2 != 0); model_call(first.q, out.tensor, input.tensor); backend.Synchronize(first.q);
  }
  nlohmann::json trials = nlohmann::json::array();
  for (bool on : {false,true,true,false,true,false,false,true,false,true,true,false,true,false,false,true}) {
    select(on); const auto start = std::chrono::steady_clock::now();
    model_call(first.q, out.tensor, input.tensor); backend.Synchronize(first.q);
    trials.push_back({{"direct_preparation", on}, {"complete_operator_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()}});
    xpu_test::SameBytes(out.download(), expected);
  }
  const auto memory = vt::xpu::GetMemoryInfo();
  std::cout << "F2_DIRECT_PREPARE_CALLS " << nlohmann::json({{"rows",m},{"k",k},{"n",n},{"groups",groups},
      {"panel_columns",1024},{"trials",trials},{"workspace_bytes",memory.w8a8_workspace_bytes},
      {"preparation_bytes",memory.w8a8_preparation_bytes},{"peak_device_bytes",memory.peak_allocated_bytes},
      {"public_scratch_exact",true},{"queued_distinct_consumers_exact",true}}).dump() << std::endl;
  CHECK(vt::GetReferenceTierHits() == 0);
}

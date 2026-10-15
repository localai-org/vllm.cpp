#include "xpu_test_helpers.h"
#include "vt/exl3_grouped.h"
#include "vt/exl3_w8a8_panel_plan.h"
#include "vt/breakable_graph.h"
#include "vt/xpu.h"
#include "vt/unaligned.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <fstream>
#include <future>
#include <string_view>
#include <nlohmann/json.hpp>

namespace {
using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;
void Accuracy(const std::vector<unsigned char>& raw, const vllm::StTensor& ref,
              const char* label) {
  REQUIRE(ref.dtype == "F16");
  REQUIRE(raw.size() == ref.nbytes);
  double error = 0, norm = 0, maximum = 0;
  bool finite = true;
  size_t changed = 0;
  for (size_t i = 0; i < raw.size(); i += 2) {
    const auto a = vt::LoadUnaligned<uint16_t>(raw.data() + i);
    const auto b = vt::LoadUnaligned<uint16_t>(ref.data + i);
    const double got = vt::F16ToF32(a), expected = vt::F16ToF32(b);
    finite &= std::isfinite(got) && std::isfinite(expected);
    error += (got - expected) * (got - expected); norm += expected * expected;
    maximum = std::max(maximum, std::abs(got - expected)); changed += a != b;
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-30));
  CAPTURE(label);
  CAPTURE(relative);
  CAPTURE(maximum);
  std::cout << "W8A8_OPERATOR " << label << " relative=" << relative
            << " max_error=" << maximum << " half_differences=" << changed << '\n';
  CHECK(finite);
  CHECK(relative < 2e-3);
}
}

TEST_CASE("EXL3 W8A8 plan: explicit boundary bounded panel and aligned regions") {
  CHECK(vt::PlanExl3SmallM(128, 5120, 16384, 4).padded_rows == 128);
  CHECK_THROWS(vt::PlanExl3SmallM(129, 5120, 16384, 4));
  CHECK_THROWS(vt::PlanExl3W8A8(128, 5120, 16384, 2, 4));
  for (int m : {129, 256, 4096}) {
    const auto p = vt::PlanExl3W8A8(m, 5120, 16384, 2, 4);
    CHECK(p.padded_rows == (m + 255) / 256 * 256);
    CHECK(p.weight_panel_bytes == 5120 * 128);
    CHECK(p.row_scale_offset == size_t(2) * p.padded_rows * 5120);
    CHECK(p.intermediate_offset >= p.row_scale_offset + size_t(2) * p.padded_rows * 4);
    CHECK(p.weight_scale_offset >= p.intermediate_offset + size_t(p.padded_rows) * 16384 * 2);
    CHECK(p.workspace_bytes == p.weight_scale_offset + 64);
    for (const size_t offset : {p.activation_offset, p.row_scale_offset, p.intermediate_offset,
                                p.weight_scale_offset, p.workspace_bytes}) CHECK(offset % 64 == 0);
  }
  CHECK_THROWS(vt::PlanExl3W8A8(4097, 5120, 16384, 2, 4));
  CHECK_THROWS(vt::PlanExl3W8A8(129, 5130, 16384, 2, 4));
  CHECK_THROWS(vt::PlanExl3W8A8(129, 5120, 16384, 0, 4));
  CHECK_THROWS(vt::PlanExl3W8A8(129, 5120, 16384, 2, 5));
  CHECK_THROWS(vt::PlanExl3W8A8(129, 133248, 128, 1, 4));
  CHECK_THROWS(vt::PlanExl3W8A8(4096, 2147483520LL, 2147483520LL, 32767, 6));
}

TEST_CASE("XPU EXL3 W8A8 P7: model map ownership generations public guards and queues") {
  const char* fixture = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!fixture) std::exit(77);
  const bool present = std::getenv("VT_XPU_W8A8_MODEL_MAP") != nullptr;
  const std::string prior = present ? std::getenv("VT_XPU_W8A8_MODEL_MAP") : "";
  struct Restore {
    bool present; std::string prior;
    ~Restore() {
      if (present) setenv("VT_XPU_W8A8_MODEL_MAP", prior.c_str(), 1);
      else unsetenv("VT_XPU_W8A8_MODEL_MAP");
    }
  } restore{present, prior};
  setenv("VT_XPU_W8A8_MODEL_MAP", "1", 1);
  const auto f = vllm::SafetensorsFile::Open(fixture);
  const auto& tr = f.Get("merged_trellis"); const auto& su = f.Get("stacked_suh");
  const auto& sv = f.Get("merged_svh"); const auto& original_map = f.Get("source_map");
  const int k = int(su.shape[1]), n = int(sv.shape[0]), groups = int(su.shape[0]);
  const int bits = int(tr.shape[2] / 16), m = 129;
  REQUIRE(groups >= 2);
  Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  Buffer trellis(first.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(first.q, DType::kF16, {groups, k}), svh(first.q, DType::kF16, {n});
  Buffer input(first.q, DType::kF16, {m, k}), out(first.q, DType::kF16, {m, n});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(sv.data);
  input.upload(f.Get("input_m129").data);
  auto owner = std::shared_ptr<void>(vt::Alloc(first.q.device, original_map.nbytes),
      [device = first.q.device](void* ptr) { vt::Free(device, ptr); });
  auto map = vt::Tensor::Contiguous(owner.get(), DType::kI32, first.q.device, {n / 128});
  auto upload_map = [&](const void* bytes) {
    vt::GetBackend(first.q.device).Copy(first.q, map.data, bytes, map.Bytes());
    vt::GetBackend(first.q.device).Synchronize(first.q);
  };
  upload_map(original_map.data);
  vt::SharedPtrCache<const vt::Exl3W8A8ModelMap> cache;
  for (int width : {1024, 128, 2048, 1024}) {
    CAPTURE(width);
    const vt::Exl3GroupedLinearArgs args{bits, 2, "P7_MODEL_MAP", width};
    vt::Exl3GroupedW8A8(first.q, out.tensor, input.tensor, trellis.tensor,
        suh.tensor, svh.tensor, map, args);
    const auto expected = out.download();
    const auto old_cache = cache.Load();
    vt::detail::Exl3GroupedW8A8Model(second.q, out.tensor, input.tensor, trellis.tensor,
        suh.tensor, svh.tensor, map, args, owner, cache);
    vt::GetBackend(second.q.device).Synchronize(second.q);
    xpu_test::SameBytes(out.download(), expected);
    REQUIRE(cache.Load()); CHECK(cache.Load()->Matches(map, groups, owner));
    if (old_cache) CHECK(cache.Load() == old_cache);
  }
  // Two cold host callers on distinct queues publish/read the slot atomically.
  // Each call keeps its own immutable certificate until submissions complete.
  {
    const auto expected = out.download();
    Buffer other_out(second.q, DType::kF16, {m, n});
    cache.Reset();
    const vt::Exl3GroupedLinearArgs concurrent_args{bits, 2, "P7_CONCURRENT_MAP", 1024};
    auto first_call = std::async(std::launch::async, [&] {
      vt::detail::Exl3GroupedW8A8Model(first.q, out.tensor, input.tensor, trellis.tensor,
          suh.tensor, svh.tensor, map, concurrent_args, owner, cache);
      vt::GetBackend(first.q.device).Synchronize(first.q);
    });
    auto second_call = std::async(std::launch::async, [&] {
      vt::detail::Exl3GroupedW8A8Model(second.q, other_out.tensor, input.tensor, trellis.tensor,
          suh.tensor, svh.tensor, map, concurrent_args, owner, cache);
      vt::GetBackend(second.q.device).Synchronize(second.q);
    });
    REQUIRE_NOTHROW(first_call.get()); REQUIRE_NOTHROW(second_call.get());
    xpu_test::SameBytes(out.download(), expected);
    xpu_test::SameBytes(other_out.download(), expected);
    REQUIRE(cache.Load()); CHECK(cache.Load()->Matches(map, groups, owner));
  }
  // Same address, distinct ownership control block: this is a new residency
  // generation. An alias sharing the original control block remains valid.
  std::shared_ptr<void> alias(owner, owner.get());
  CHECK(cache.Load()->Matches(map, groups, alias));
  auto generation = std::shared_ptr<void>(owner.get(), [keep = owner](void*) {});
  CHECK_FALSE(cache.Load()->Matches(map, groups, generation));
  auto wrong = map; wrong.shape[0] -= 1;
  CHECK_FALSE(cache.Load()->Matches(wrong, groups, owner));
  wrong = map; wrong.stride[0] = 2;
  CHECK_FALSE(cache.Load()->Matches(wrong, groups, owner));
  wrong = map; wrong.device.type = vt::DeviceType::kCPU;
  CHECK_FALSE(cache.Load()->Matches(wrong, groups, owner));
  CHECK_FALSE(cache.Load()->Matches(map, groups - 1, owner));
  const auto previous_cache = cache.Load();
  const vt::Exl3GroupedLinearArgs args{bits, 2, "P7_MODEL_MAP", 1024};
  vt::detail::Exl3GroupedW8A8Model(first.q, out.tensor, input.tensor, trellis.tensor,
      suh.tensor, svh.tensor, map, args, generation, cache);
  CHECK(cache.Load() != previous_cache); CHECK(cache.Load()->Matches(map, groups, generation));
  const auto preserved = out.download();
  // Public callers cannot suppress validation with a cached payload, even
  // after modifying an allocation in place. Both public seams must discard it.
  std::vector<int32_t> bad(size_t(n / 128));
  std::memcpy(bad.data(), original_map.data, original_map.nbytes); bad[0] = groups;
  upload_map(bad.data());
  const auto injected_map = cache.Load();
  auto injected = args; injected.model_map = injected_map.get();
  CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(first.q, out.tensor, input.tensor,
      trellis.tensor, suh.tensor, svh.tensor, map, injected),
      doctest::Contains("shard_of_nb group out of range"), std::runtime_error);
  xpu_test::SameBytes(out.download(), preserved);
  const auto plan = vt::PlanExl3W8A8(m, k, n, groups, bits, 1024);
  Buffer scratch(first.q, DType::kI8, {int64_t(plan.workspace_bytes)});
  Buffer panel(first.q, DType::kI8, {k, plan.weight_panel_columns});
  const std::vector<unsigned char> poison_scratch(scratch.bytes, 0xa5), poison_panel(panel.bytes, 0xcd);
  scratch.upload(poison_scratch.data()); panel.upload(poison_panel.data());
  CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(first.q, out.tensor, input.tensor,
      trellis.tensor, suh.tensor, svh.tensor, map, scratch.tensor, panel.tensor, injected),
      doctest::Contains("shard_of_nb group out of range"), std::runtime_error);
  xpu_test::SameBytes(out.download(), preserved);
  xpu_test::SameBytes(scratch.download(), poison_scratch);
  xpu_test::SameBytes(panel.download(), poison_panel);
  auto next_generation = std::shared_ptr<void>(owner.get(), [keep = owner](void*) {});
  CHECK_THROWS_WITH_AS(vt::detail::Exl3GroupedW8A8Model(first.q, out.tensor, input.tensor,
      trellis.tensor, suh.tensor, svh.tensor, map, args, next_generation, cache),
      doctest::Contains("shard_of_nb group out of range"), std::runtime_error);
  xpu_test::SameBytes(out.download(), preserved); CHECK(cache.Load()->Matches(map, groups, generation));
  upload_map(original_map.data);
  // Actual new allocation/reload invalidates the certificate and receives its
  // own checked decomposition; the old cache keeps its allocation alive.
  auto replacement = std::shared_ptr<void>(vt::Alloc(first.q.device, map.Bytes()),
      [device = first.q.device](void* ptr) { vt::Free(device, ptr); });
  auto new_map = map; new_map.data = replacement.get();
  vt::GetBackend(first.q.device).Copy(first.q, new_map.data, original_map.data, map.Bytes());
  vt::GetBackend(first.q.device).Synchronize(first.q);
  CHECK_FALSE(cache.Load()->Matches(new_map, groups, replacement));
  vt::detail::Exl3GroupedW8A8Model(first.q, out.tensor, input.tensor, trellis.tensor,
      suh.tensor, svh.tensor, new_map, args, replacement, cache);
  xpu_test::SameBytes(out.download(), preserved);
  CHECK(cache.Load()->Matches(new_map, groups, replacement));
  // A warm routing certificate never removes dynamic input or scale guards.
  auto execute_owned = [&] { vt::detail::Exl3GroupedW8A8Model(first.q, out.tensor,
      input.tensor, trellis.tensor, suh.tensor, svh.tensor, new_map, args, replacement, cache); };
  auto bad_input = input.download(), bad_suh = suh.download(), bad_svh = svh.download();
  const uint16_t nan = 0x7e35, inf = 0x7c00;
  std::memcpy(bad_input.data(), &nan, 2); input.upload(bad_input.data());
  CHECK_THROWS_WITH_AS(execute_owned(), doctest::Contains("finite inputs"), std::runtime_error);
  xpu_test::SameBytes(out.download(), preserved);
  std::memcpy(bad_svh.data(), &inf, 2); svh.upload(bad_svh.data());
  CHECK_THROWS_WITH_AS(execute_owned(), doctest::Contains("finite svh"), std::runtime_error);
  xpu_test::SameBytes(out.download(), preserved);
  svh.upload(sv.data); input.upload(f.Get("input_m129").data);
  std::memcpy(bad_suh.data(), &inf, 2); suh.upload(bad_suh.data());
  CHECK_THROWS_WITH_AS(execute_owned(), doctest::Contains("finite inputs"), std::runtime_error);
  xpu_test::SameBytes(out.download(), preserved);
  suh.upload(su.data); execute_owned(); xpu_test::SameBytes(out.download(), preserved);
  // Managed W8A8 remains eager-only, including a warmed model certificate.
  if (vt::GetBackend(first.q.device).SupportsGraphCapture()) {
    auto& backend = vt::GetBackend(first.q.device);
    backend.BeginCapture(first.q);
    CHECK_THROWS_WITH_AS(execute_owned(), doctest::Contains("eager-only"), std::runtime_error);
    void* graph = backend.EndCaptureGraph(first.q); backend.DestroyGraph(graph);
    xpu_test::SameBytes(out.download(), preserved);
    CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == 0);
  }
  std::weak_ptr<void> lifetime = replacement;
  // The model-only default must exercise the certificate, not silently select
  // the public fallback. Public calls above still ignored an injected payload.
  unsetenv("VT_XPU_W8A8_MODEL_MAP"); cache.Reset();
  execute_owned();
  REQUIRE(cache.Load()); CHECK(cache.Load()->Matches(new_map, groups, replacement));
  xpu_test::SameBytes(out.download(), preserved);
  replacement.reset(); CHECK_FALSE(lifetime.expired());
  cache.Reset(); CHECK(lifetime.expired());
  CHECK(vt::GetReferenceTierHits() == 0);
  std::cout << "P7_MODEL_MAP ownership_generations=1 public_guards=1 cross_queue=1 retired_owner=1" << std::endl;
}

TEST_CASE("XPU EXL3 W8A8 P7: model map complete operator gain") {
  const char* fixture = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!fixture) std::exit(77);
  const bool present = std::getenv("VT_XPU_W8A8_MODEL_MAP") != nullptr;
  const std::string prior = present ? std::getenv("VT_XPU_W8A8_MODEL_MAP") : "";
  struct Restore {
    bool present; std::string prior;
    ~Restore() {
      if (present) setenv("VT_XPU_W8A8_MODEL_MAP", prior.c_str(), 1);
      else unsetenv("VT_XPU_W8A8_MODEL_MAP");
    }
  } restore{present, prior};
  const auto f = vllm::SafetensorsFile::Open(fixture);
  const int m = std::getenv("VT_B70_EXL3_PANEL_ROWS") ? std::atoi(std::getenv("VT_B70_EXL3_PANEL_ROWS")) : 129;
  REQUIRE((m == 129 || m == 1600));
  const auto& tr = f.Get("merged_trellis"); const auto& su = f.Get("stacked_suh");
  const auto& sv = f.Get("merged_svh"); const auto& routing = f.Get("source_map");
  const int k = int(su.shape[1]), n = int(sv.shape[0]), groups = int(su.shape[0]);
  const int bits = int(tr.shape[2] / 16);
  Queue gpu(vt::DeviceType::kXPU);
  Buffer trellis(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer input(gpu.q, DType::kF16, {m, k}), out(gpu.q, DType::kF16, {m, n});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(sv.data);
  input.upload(f.Get("input_m" + std::to_string(m)).data);
  auto owner = std::shared_ptr<void>(vt::Alloc(gpu.q.device, routing.nbytes),
      [device = gpu.q.device](void* ptr) { vt::Free(device, ptr); });
  auto map = vt::Tensor::Contiguous(owner.get(), DType::kI32, gpu.q.device, {n / 128});
  vt::GetBackend(gpu.q.device).Copy(gpu.q, map.data, routing.data, routing.nbytes);
  vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
  vt::SharedPtrCache<const vt::Exl3W8A8ModelMap> cache;
  const vt::Exl3GroupedLinearArgs args{bits, 2, "P7_MODEL_MAP_BENCH", 1024};
  auto execute = [&] { vt::detail::Exl3GroupedW8A8Model(gpu.q, out.tensor, input.tensor,
      trellis.tensor, suh.tensor, svh.tensor, map, args, owner, cache);
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q); };
  const auto before = input.download();
  // Both arms reach steady execution before any sample. The former two-warm,
  // all-off-then-all-on order confounded this small host change with GPU warmup.
  nlohmann::json warm = nlohmann::json::array();
  for (int i = 0; i < 64; ++i) {
    const bool cached = i % 2 != 0;
    setenv("VT_XPU_W8A8_MODEL_MAP", cached ? "1" : "0", 1);
    const auto start = std::chrono::steady_clock::now(); execute();
    warm.push_back({{"cached", cached}, {"complete_operator_wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()}});
    (void)vt::xpu::DrainProfileEvents(); (void)vt::xpu::DrainHostProfileRecords();
  }
  const auto expected = out.download();
  (void)vt::xpu::DrainProfileEvents(); (void)vt::xpu::DrainHostProfileRecords();
  nlohmann::json trials = nlohmann::json::array();
  // ABBA/BAAB puts both routes equally early and late; record actual order and
  // each complete operator, rather than comparing two sequential medians.
  constexpr bool order[] = {false, true, true, false, true, false, false, true,
                           false, true, true, false, true, false, false, true};
  for (bool cached : order) {
    setenv("VT_XPU_W8A8_MODEL_MAP", cached ? "1" : "0", 1);
    const auto start = std::chrono::steady_clock::now(); execute();
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    const auto events = vt::xpu::DrainProfileEvents();
    const auto hosts = vt::xpu::DrainHostProfileRecords();
    nlohmann::json trial = {{"cached", cached}, {"complete_operator_wall_ms", elapsed},
        {"device_events", nlohmann::json::array()}, {"host_events", nlohmann::json::array()}};
    for (const auto& e : events)
      trial["device_events"].push_back({{"stage", e.stage}, {"ms", (e.end_ns - e.start_ns) / 1e6}});
    for (const auto& e : hosts)
      trial["host_events"].push_back({{"stage", e.stage}, {"ms", (e.end_steady_ns - e.start_steady_ns) / 1e6}});
    trials.push_back(std::move(trial));
    xpu_test::SameBytes(out.download(), expected);
    xpu_test::SameBytes(input.download(), before);
    REQUIRE(cache.Load()); CHECK(cache.Load()->Matches(map, groups, owner));
    (void)vt::xpu::DrainProfileEvents(); (void)vt::xpu::DrainHostProfileRecords();
  }
  nlohmann::json result = {{"rows", m}, {"k", k}, {"n", n}, {"groups", groups},
      {"bits", bits}, {"warm_complete_operators", warm}, {"trials", trials},
      {"measurement_order", "64 alternating warmups; ABBA BAAB ABBA BAAB"},
      {"device_profile", std::getenv("VT_XPU_PROFILE") && std::string_view(std::getenv("VT_XPU_PROFILE")) == "1"},
      {"host_profile", std::getenv("VT_XPU_HOST_PROFILE") && std::string_view(std::getenv("VT_XPU_HOST_PROFILE")) == "1"},
      {"exact_native_bytes", true}};
  std::cout << "P7_MODEL_MAP_BALANCED " << result.dump() << std::endl;
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 W8A8 P2: real rows wider panels preserve rounded intermediates") {
  const char* env = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!env) std::exit(77);
  const auto f = vllm::SafetensorsFile::Open(env);
  const auto& tr = f.Get("merged_trellis");
  const auto& su = f.Get("stacked_suh");
  const auto& sv = f.Get("merged_svh");
  const auto& map = f.Get("source_map");
  int m = 256;
  if (const char* rows = std::getenv("VT_B70_EXL3_PANEL_ROWS")) {
    REQUIRE((std::string_view(rows) == "256" || std::string_view(rows) == "896" ||
             std::string_view(rows) == "1600"));
    m = std::atoi(rows);
  }
  const int k = int(su.shape[1]), n = int(sv.shape[0]);
  const int bits = int(tr.shape[2] / 16), groups = int(su.shape[0]);
  REQUIRE(groups >= 1);
  if (groups > 1) REQUIRE(std::memcmp(su.data, su.data + k * 2, k * 2) != 0);
  REQUIRE(map.nbytes == size_t(n / 128) * sizeof(int32_t));
  const auto suffix = "_m" + std::to_string(m);
  const auto& x = f.Get("input" + suffix);
  REQUIRE(x.dtype == "F16");
  REQUIRE((x.shape == std::vector<int64_t>{m, k}));
  std::vector<int32_t> source_map(n / 128);
  std::memcpy(source_map.data(), map.data, map.nbytes);
  Queue gpu(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(gpu.q.device);
  Buffer trellis(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer mapping(gpu.q, DType::kI32, {n / 128}), input(gpu.q, DType::kF16, {m, k});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(sv.data);
  mapping.upload(map.data); input.upload(x.data);
  std::vector<unsigned char> baseline_output, baseline_workspace;
  nlohmann::json reports = nlohmann::json::array();
  for (int width : {128, 1024, 2048}) {
    CAPTURE(width);
    const auto p = vt::PlanExl3W8A8(m, k, n, groups, bits, width);
    Buffer out(gpu.q, DType::kF16, {m, n});
    Buffer workspace(gpu.q, DType::kI8, {int64_t(p.workspace_bytes)});
    Buffer panel_storage(gpu.q, DType::kI8, {int64_t(p.weight_panel_bytes + 64)});
    auto panel = vt::Tensor::Contiguous(panel_storage.tensor.data, DType::kI8, gpu.q.device,
                                      {k, p.weight_panel_columns});
    const vt::Exl3GroupedLinearArgs args{bits, 2, "P2_REAL_ROWS", width};
    std::vector<unsigned char> poison(panel_storage.bytes, 0xcd);
    panel_storage.upload(poison.data());
    auto execute = [&] {
      const auto start = std::chrono::steady_clock::now();
      vt::Exl3GroupedW8A8(gpu.q, out.tensor, input.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, workspace.tensor, panel, args);
      backend.Synchronize(gpu.q);
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
    };
    (void)vt::xpu::DrainProfileEvents();
    const double cold = execute();
    const std::vector<double> warm{execute(), execute(), execute()};
    // Separate device cost, cold then three warm samples; empty without profiling.
    std::vector<double> validation_device_ms;
    std::vector<double> preparation_device_ms, quantize_device_ms, prepared_commit_device_ms;
    for (const auto& event : vt::xpu::DrainProfileEvents()) {
      if (event.stage == "exl3_w8a8_validate_rows")
        validation_device_ms.push_back((event.end_ns - event.start_ns) / 1e6);
      if (event.stage == "exl3_w8a8_prepare_checked")
        preparation_device_ms.push_back((event.end_ns - event.start_ns) / 1e6);
      if (event.stage == "exl3_w8a8_input_quantize")
        quantize_device_ms.push_back((event.end_ns - event.start_ns) / 1e6);
      if (event.stage == "exl3_w8a8_prepared_commit")
        prepared_commit_device_ms.push_back((event.end_ns - event.start_ns) / 1e6);
    }
    const auto result = out.download(), scratch = workspace.download(), weights = panel_storage.download();
    const auto parts = vt::PlanExl3W8A8Panels(source_map, groups, width);
    // Check the last128 columns of the actual reconstructed tail in its compact
    // layout, and confirm no store crosses the planned capacity into its guard.
    const int tail = parts.back().columns;
    const auto& last = f.Get("last_weight_panel" + suffix);
    REQUIRE(last.nbytes == size_t(k) * 128);
    bool reconstructed_tail_exact = true;
    for (int row = 0; row < k; ++row)
      reconstructed_tail_exact &= std::memcmp(weights.data() + size_t(row) * tail + tail - 128,
                                               last.data + size_t(row) * 128, 128) == 0;
    CHECK(reconstructed_tail_exact);
    CHECK(std::all_of(weights.begin() + p.weight_panel_bytes, weights.end(),
                      [](unsigned char v) { return v == 0xcd; }));
    bool preparation_exact = true, y_exact = true, output_exact = true;
    if (width == 128) {
      baseline_output = result; baseline_workspace = scratch;
      const auto& qref = f.Get("xq" + suffix);
      const auto& sref = f.Get("sx" + suffix);
      const auto& yref = f.Get("y" + suffix);
      REQUIRE(qref.nbytes == size_t(groups) * m * k);
      REQUIRE(sref.nbytes == size_t(groups) * m * sizeof(float));
      REQUIRE(yref.nbytes == size_t(m) * n * 2);
      bool original_quantized_exact = true, original_scales_exact = true, padding_exact = true;
      for (int group = 0; group < groups; ++group) for (int row = 0; row < p.padded_rows; ++row) {
        const auto* quantized = scratch.data() + p.activation_offset + (size_t(group) * p.padded_rows + row) * k;
        const auto* scale = scratch.data() + p.row_scale_offset + (size_t(group) * p.padded_rows + row) * sizeof(float);
        if (row < m) {
          original_quantized_exact &= std::memcmp(quantized, qref.data + (size_t(group) * m + row) * k, k) == 0;
          original_scales_exact &= std::memcmp(scale, sref.data + (size_t(group) * m + row) * sizeof(float), sizeof(float)) == 0;
        } else {
          padding_exact &= std::all_of(quantized, quantized + k, [](unsigned char v) { return v == 0; });
          const float one = 1.f;
          padding_exact &= std::memcmp(scale, &one, sizeof(one)) == 0;
        }
      }
      CHECK(original_quantized_exact);
      CHECK(original_scales_exact);
      CHECK(padding_exact);
      CHECK(std::memcmp(scratch.data() + p.intermediate_offset, yref.data, yref.nbytes) == 0);
      const auto* pad_y = scratch.data() + p.intermediate_offset + yref.nbytes;
      CHECK(std::all_of(pad_y, scratch.data() + p.weight_scale_offset,
                        [](unsigned char v) { return v == 0; }));
      const auto& original_output = f.Get("output" + suffix);
      REQUIRE(original_output.nbytes == result.size());
      CHECK(std::memcmp(result.data(), original_output.data, result.size()) == 0);
      Accuracy(result, f.Get("output" + suffix), "real_rows_128_control");
    } else {
      preparation_exact = std::equal(scratch.begin(), scratch.begin() + p.intermediate_offset,
                                    baseline_workspace.begin());
      y_exact = std::equal(scratch.begin() + p.intermediate_offset,
                          scratch.begin() + p.weight_scale_offset,
                          baseline_workspace.begin() + p.intermediate_offset);
      output_exact = result == baseline_output;
      CHECK(preparation_exact);
      CHECK(y_exact);
      CHECK(output_exact);
    }
    reports.push_back({{"m", m}, {"padded_m", p.padded_rows}, {"width", width}, {"panel_count", parts.size()},
        {"panel_bytes", p.weight_panel_bytes}, {"workspace_bytes", p.workspace_bytes},
        {"cold_operator_ms", cold}, {"warm_operator_ms", warm},
        {"validation_device_ms", validation_device_ms},
        {"preparation_device_ms", preparation_device_ms}, {"quantize_device_ms", quantize_device_ms},
        {"prepared_commit_device_ms", prepared_commit_device_ms},
        {"private_preparation_bytes", vt::xpu::GetMemoryInfo().w8a8_preparation_bytes},
        {"preparation_exact", preparation_exact}, {"Y_exact", y_exact},
        {"output_exact", output_exact}, {"reconstructed_tail_exact", reconstructed_tail_exact}});
    std::cout << "P2_PANEL_OPERATOR " << reports.back().dump() << '\n';
  }
  if (const char* output = std::getenv("VT_B70_EXL3_PANEL_REPORT")) {
    REQUIRE_FALSE(std::filesystem::exists(output));
    std::ofstream file(output); file << reports.dump(2) << '\n';
    REQUIRE(file.good());
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 W8A8 P2: non-monotonic source runs and128 tails preserve original columns") {
  const char* env = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!env) std::exit(77);
  const auto f = vllm::SafetensorsFile::Open(env);
  const auto& tr = f.Get("merged_trellis");
  const auto& su = f.Get("stacked_suh");
  const auto& sv = f.Get("merged_svh");
  const auto& map = f.Get("source_map");
  const int m = 256, k = int(su.shape[1]), n = int(sv.shape[0]);
  const int bits = int(tr.shape[2] / 16), groups = int(su.shape[0]);
  REQUIRE(groups >= 2);
  REQUIRE(map.nbytes == size_t(n / 128) * sizeof(int32_t));
  std::vector<int32_t> source_map(n / 128);
  std::memcpy(source_map.data(), map.data, map.nbytes);
  REQUIRE(source_map.front() != source_map.back());
  // H128 is block-local: exchanging whole trellis/SV/source blocks must
  // exchange exactly the corresponding original Y and output columns.
  std::vector<unsigned char> packed(tr.data, tr.data + tr.nbytes);
  std::vector<unsigned char> scales(sv.data, sv.data + sv.nbytes);
  const size_t block_bytes = size_t(8) * 32 * bits;
  for (int tile = 0; tile < k / 16; ++tile) {
    const size_t first = size_t(tile) * (n / 16) * 32 * bits;
    const size_t last = first + size_t(n / 128 - 1) * block_bytes;
    std::swap_ranges(packed.begin() + first, packed.begin() + first + block_bytes,
                     packed.begin() + last);
  }
  std::swap_ranges(scales.begin(), scales.begin() + 256, scales.end() - 256);
  std::swap(source_map.front(), source_map.back());
  auto permute = [&](const vllm::StTensor& t) {
    REQUIRE(t.nbytes == size_t(m) * n * 2);
    std::vector<unsigned char> raw(t.data, t.data + t.nbytes);
    for (int row = 0; row < m; ++row) {
      const size_t first = size_t(row) * n * 2, last = first + size_t(n - 128) * 2;
      std::swap_ranges(raw.begin() + first, raw.begin() + first + 256, raw.begin() + last);
    }
    return raw;
  };
  const auto original_output = permute(f.Get("output_m256"));
  const auto original_y = permute(f.Get("y_m256"));
  Queue gpu(vt::DeviceType::kXPU);
  Buffer trellis(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer mapping(gpu.q, DType::kI32, {n / 128}), input(gpu.q, DType::kF16, {m, k});
  trellis.upload(packed.data()); suh.upload(su.data); svh.upload(scales.data());
  mapping.upload(source_map.data()); input.upload(f.Get("input_m256").data);
  std::vector<unsigned char> control_preparation, control_tail;
  for (const int width : {128, 1024, 2048}) {
    CAPTURE(width);
    const auto parts = vt::PlanExl3W8A8Panels(source_map, groups, width);
    REQUIRE(parts.back().columns == 128);
    REQUIRE(parts.front().columns == 128);
    const auto p = vt::PlanExl3W8A8(m, k, n, groups, bits, width);
    Buffer out(gpu.q, DType::kF16, {m, n});
    Buffer scratch(gpu.q, DType::kI8, {int64_t(p.workspace_bytes)});
    Buffer weights(gpu.q, DType::kI8, {int64_t(p.weight_panel_bytes + 64)});
    auto panel = vt::Tensor::Contiguous(weights.tensor.data, DType::kI8, gpu.q.device,
                                      {k, p.weight_panel_columns});
    std::vector<unsigned char> poison(weights.bytes, 0xcd); weights.upload(poison.data());
    const vt::Exl3GroupedLinearArgs args{bits, 2, "P2_NON_MONOTONIC", width};
    auto execute = [&] { vt::Exl3GroupedW8A8(gpu.q, out.tensor, input.tensor, trellis.tensor,
        suh.tensor, svh.tensor, mapping.tensor, scratch.tensor, panel, args); };
    execute();
    const auto result = out.download(), raw = scratch.download(), reconstructed = weights.download();
    xpu_test::SameBytes(result, original_output);
    xpu_test::SameBytes(std::vector<unsigned char>(raw.begin() + p.intermediate_offset,
        raw.begin() + p.weight_scale_offset), original_y);
    CHECK(std::all_of(reconstructed.begin() + p.weight_panel_bytes, reconstructed.end(),
                      [](unsigned char v) { return v == 0xcd; }));
    const std::vector<unsigned char> preparation(raw.begin(), raw.begin() + p.intermediate_offset);
    const std::vector<unsigned char> tail(reconstructed.begin(), reconstructed.begin() + size_t(k) * 128);
    if (width == 128) { control_preparation = preparation; control_tail = tail; }
    else { xpu_test::SameBytes(preparation, control_preparation); xpu_test::SameBytes(tail, control_tail); }
    auto unchanged = [&] {
      xpu_test::SameBytes(out.download(), result);
      xpu_test::SameBytes(scratch.download(), raw);
      xpu_test::SameBytes(weights.download(), reconstructed);
    };
    auto short_scratch = scratch.tensor; --short_scratch.shape[0];
    CHECK_THROWS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, input.tensor, trellis.tensor,
        suh.tensor, svh.tensor, mapping.tensor, short_scratch, panel, args)); unchanged();
    auto alias = panel; alias.data = out.tensor.data;
    CHECK_THROWS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, input.tensor, trellis.tensor,
        suh.tensor, svh.tensor, mapping.tensor, scratch.tensor, alias, args)); unchanged();
    auto invalid_map = source_map; invalid_map.back() = groups; mapping.upload(invalid_map.data());
    CHECK_THROWS_WITH_AS(execute(), doctest::Contains("group out of range"), std::runtime_error);
    unchanged(); mapping.upload(source_map.data());
    execute(); xpu_test::SameBytes(out.download(), original_output);
    std::cout << "P2_NON_MONOTONIC width=" << width << " panels=" << parts.size()
              << " first_tail=128 last_tail=128 exact_original_Y_output=1\n";
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 W8A8 P2: budget refusal preserves completed shared scratch and output") {
  const char* env = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!env || !std::getenv("VT_XPU_MEMORY_BUDGET_BYTES")) std::exit(77);
  const auto f = vllm::SafetensorsFile::Open(env);
  const auto& tr = f.Get("merged_trellis"); const auto& su = f.Get("stacked_suh");
  const int k = int(su.shape[1]), n = int(f.Get("merged_svh").shape[0]);
  const int bits = int(tr.shape[2] / 16), groups = int(su.shape[0]);
  Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::xpu::GetMemoryInfo().budget_bytes <= 256 * 1024 * 1024);
  Buffer trellis(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer map(gpu.q, DType::kI32, {n / 128}), input(gpu.q, DType::kF16, {256, k});
  Buffer out(gpu.q, DType::kF16, {256, n});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(f.Get("merged_svh").data);
  map.upload(f.Get("source_map").data); input.upload(f.Get("input_m256").data);
  auto execute = [&](int width) { vt::Exl3GroupedW8A8(gpu.q, out.tensor, input.tensor,
      trellis.tensor, suh.tensor, svh.tensor, map.tensor, {bits, 2, "P2_BUDGET", width}); };
  execute(128);
  const auto before = out.download();
  xpu_test::SameBytes(before, std::vector<unsigned char>(f.Get("output_m256").data,
      f.Get("output_m256").data + f.Get("output_m256").nbytes));
  const auto narrow = vt::xpu::GetMemoryInfo();
  const auto wide = vt::PlanExl3W8A8(256, k, n, groups, bits, 2048);
  const size_t capacity = wide.workspace_bytes + wide.weight_panel_bytes;
  REQUIRE(capacity > narrow.w8a8_workspace_bytes);
  const size_t growth = capacity - narrow.w8a8_workspace_bytes;
  const size_t remaining = narrow.budget_bytes - narrow.allocated_bytes - narrow.graph_device_bytes;
  REQUIRE(remaining > growth);
  {
    // Exhaust only the deliberately small test budget, never the real card.
    Buffer pressure(gpu.q, DType::kI8, {int64_t(remaining - growth / 2)});
    const auto pressured = vt::xpu::GetMemoryInfo();
    CHECK_THROWS_WITH_AS(execute(2048), doctest::Contains("exceeds device memory budget"), std::runtime_error);
    const auto rejected = vt::xpu::GetMemoryInfo();
    CHECK(rejected.w8a8_workspace_bytes == narrow.w8a8_workspace_bytes);
    CHECK(rejected.allocated_bytes == pressured.allocated_bytes);
    xpu_test::SameBytes(out.download(), before);
    execute(128); xpu_test::SameBytes(out.download(), before);
  }
  execute(2048); xpu_test::SameBytes(out.download(), before);
  const auto grown = vt::xpu::GetMemoryInfo();
  CHECK(grown.w8a8_workspace_bytes == capacity);
  CHECK(grown.peak_allocated_bytes <= grown.budget_bytes);
  execute(128); xpu_test::SameBytes(out.download(), before);
  std::cout << "P2_BUDGET budget=" << grown.budget_bytes << " narrow=" << narrow.w8a8_workspace_bytes
            << " grown=" << capacity << " peak=" << grown.peak_allocated_bytes << '\n';
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 W8A8 P2: shared scratch completes growth reuse and cross-queue consumers") {
  const char* env = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!env) std::exit(77);
  const auto f = vllm::SafetensorsFile::Open(env);
  const auto& tr = f.Get("merged_trellis");
  const auto& su = f.Get("stacked_suh");
  const int k = int(su.shape[1]), n = int(f.Get("merged_svh").shape[0]);
  const int bits = int(tr.shape[2] / 16), groups = int(su.shape[0]);
  Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  Buffer trellis(first.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(first.q, DType::kF16, {groups, k}), svh(first.q, DType::kF16, {n});
  Buffer map(first.q, DType::kI32, {n / 128});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(f.Get("merged_svh").data);
  map.upload(f.Get("source_map").data);
  std::vector<unsigned char> baseline;
  size_t capacity = vt::xpu::GetMemoryInfo().w8a8_workspace_bytes;
  std::cout << "P2_SHARED_INITIAL retained_bytes=" << capacity << '\n';
  for (const int width : {128, 1024, 2048, 1024, 128}) {
    CAPTURE(width);
    Buffer input(first.q, DType::kF16, {256, k}), out(first.q, DType::kF16, {256, n});
    input.upload(f.Get("input_m256").data);
    const auto p = vt::PlanExl3W8A8(256, k, n, groups, bits, width);
    const vt::Exl3GroupedLinearArgs args{bits, 2, "P2_SHARED", width};
    const auto start = std::chrono::steady_clock::now();
    vt::Exl3GroupedW8A8(first.q, out.tensor, input.tensor, trellis.tensor, suh.tensor,
                        svh.tensor, map.tensor, args);
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    // The owned call itself has completed all consumers. A download or a
    // second queue may now reuse the one pool, with no caller retirement fence.
    const auto raw = out.download();
    if (baseline.empty()) { baseline = raw; Accuracy(raw, f.Get("output_m256"), "shared128"); }
    else xpu_test::SameBytes(raw, baseline);
    capacity = std::max(capacity, p.workspace_bytes + p.weight_panel_bytes);
    const auto info = vt::xpu::GetMemoryInfo();
    CHECK(info.w8a8_workspace_bytes == capacity);
    CHECK(info.allocated_bytes >= capacity);
    std::cout << "P2_SHARED_OPERATOR width=" << width << " cold_or_reuse_ms=" << elapsed
              << " retained_bytes=" << capacity << " device_peak_bytes="
              << info.peak_allocated_bytes << '\n';
  }
  {
    Buffer input(second.q, DType::kF16, {129, k}), out(second.q, DType::kF16, {129, n});
    input.upload(f.Get("input_m129").data);
    const vt::Exl3GroupedLinearArgs args{bits, 2, "P2_SECOND_QUEUE", 2048};
    vt::Exl3GroupedW8A8(second.q, out.tensor, input.tensor, trellis.tensor, suh.tensor,
                        svh.tensor, map.tensor, args);
    const auto before = out.download();
    Accuracy(before, f.Get("output_m129"), "shared_second_queue");
    CHECK(vt::xpu::GetMemoryInfo().w8a8_workspace_bytes == capacity);
    auto invalid = map.download(); const int32_t bad = groups;
    std::memcpy(invalid.data(), &bad, sizeof(bad)); map.upload(invalid.data());
    CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(second.q, out.tensor, input.tensor, trellis.tensor,
        suh.tensor, svh.tensor, map.tensor, args), doctest::Contains("group out of range"),
        std::runtime_error);
    xpu_test::SameBytes(out.download(), before);
    map.upload(f.Get("source_map").data);
    // A rejected lease must leave the pool safe for a later valid consumer.
    vt::Exl3GroupedW8A8(second.q, out.tensor, input.tensor, trellis.tensor, suh.tensor,
                        svh.tensor, map.tensor, args);
    xpu_test::SameBytes(out.download(), before);
    auto& backend = vt::GetBackend(second.q.device);
    if (backend.SupportsGraphCapture()) {
      Buffer witness(second.q, DType::kI8, {64});
      const auto memory_before = vt::xpu::GetMemoryInfo();
      backend.BeginCapture(second.q);
      backend.Memset(second.q, witness.tensor.data, 0, witness.bytes);
      CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(second.q, out.tensor, input.tensor, trellis.tensor,
          suh.tensor, svh.tensor, map.tensor, args), doctest::Contains("eager-only"), std::runtime_error);
      void* graph = backend.EndCaptureGraph(second.q);
      backend.DestroyGraph(graph);
      xpu_test::SameBytes(out.download(), before);
      const auto memory_after = vt::xpu::GetMemoryInfo();
      CHECK(memory_after.w8a8_workspace_bytes == capacity);
      CHECK(memory_after.graph_device_bytes == memory_before.graph_device_bytes);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}


namespace {
struct P4PrepareEnv {
  bool present = std::getenv("VT_XPU_W8A8_PREPARE") != nullptr;
  std::string value = present ? std::getenv("VT_XPU_W8A8_PREPARE") : "";
  ~P4PrepareEnv() {
    if (present) setenv("VT_XPU_W8A8_PREPARE", value.c_str(), 1);
    else unsetenv("VT_XPU_W8A8_PREPARE");
  }
  void Select(bool prepared) { REQUIRE(setenv("VT_XPU_W8A8_PREPARE", prepared ? "1" : "0", 1) == 0); }
};
}

TEST_CASE("XPU EXL3 W8A8 P4: private preparation preserves failure boundaries and zero rows") {
  const char* fixture = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!fixture) std::exit(77);
  const auto f = vllm::SafetensorsFile::Open(fixture);
  int m = 129;
  if (const char* rows = std::getenv("VT_B70_EXL3_PANEL_ROWS")) {
    REQUIRE((std::string_view(rows) == "129" || std::string_view(rows) == "1600"));
    m = std::atoi(rows);
  }
  const auto& tr = f.Get("merged_trellis"); const auto& su = f.Get("stacked_suh");
  const auto& sv = f.Get("merged_svh"); const auto& map = f.Get("source_map");
  const auto& x = f.Get("input_m" + std::to_string(m));
  const int k = int(su.shape[1]), n = int(sv.shape[0]);
  const int bits = int(tr.shape[2] / 16), groups = int(su.shape[0]);
  REQUIRE(groups >= 1);
  Queue gpu(vt::DeviceType::kXPU); P4PrepareEnv route;
  const auto plan = vt::PlanExl3W8A8(m, k, n, groups, bits, 1024);
  Buffer trellis(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer mapping(gpu.q, DType::kI32, {n / 128}), input(gpu.q, DType::kF16, {m, k});
  Buffer out(gpu.q, DType::kF16, {m, n});
  Buffer scratch(gpu.q, DType::kI8, {int64_t(plan.workspace_bytes)});
  Buffer panel(gpu.q, DType::kI8, {k, plan.weight_panel_columns});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(sv.data); mapping.upload(map.data); input.upload(x.data);
  const auto original_input = input.download(), original_suh = suh.download();
  const auto original_svh = svh.download(), original_map = mapping.download();
  const vt::Exl3GroupedLinearArgs args{bits, 2, "P4_BOUNDARY", 1024};
  auto execute = [&] { vt::Exl3GroupedW8A8(gpu.q, out.tensor, input.tensor, trellis.tensor,
      suh.tensor, svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args); };
  const std::vector<unsigned char> scratch_poison(scratch.bytes, 0xa5), panel_poison(panel.bytes, 0xcd),
                                   out_poison(out.bytes, 0x5a);
  auto poison = [&] { scratch.upload(scratch_poison.data()); panel.upload(panel_poison.data()); out.upload(out_poison.data()); };
  auto reject = [&](const char* message) {
    poison(); const auto before_input = input.download(), before_suh = suh.download();
    const auto before_svh = svh.download(), before_map = mapping.download();
    CHECK_THROWS_WITH_AS(execute(), doctest::Contains(message), std::runtime_error);
    xpu_test::SameBytes(out.download(), out_poison); xpu_test::SameBytes(panel.download(), panel_poison);
    auto actual = scratch.download();
    // The existing API writes only its two reserved status words on failure.
    // Every activation, scale, intermediate and other public byte stays poisoned.
    std::copy_n(scratch_poison.begin() + plan.weight_scale_offset + 4, 8,
                actual.begin() + plan.weight_scale_offset + 4);
    xpu_test::SameBytes(actual, scratch_poison);
    xpu_test::SameBytes(input.download(), before_input); xpu_test::SameBytes(suh.download(), before_suh);
    xpu_test::SameBytes(svh.download(), before_svh); xpu_test::SameBytes(mapping.download(), before_map);
  };
  auto replace_half = [](std::vector<unsigned char>& bytes, size_t offset, uint16_t bits) {
    std::memcpy(bytes.data() + offset, &bits, 2);
  };
  std::vector<unsigned char> real_control;
  std::vector<unsigned char> prefix_control;
  std::vector<std::vector<unsigned char>> safe_controls;
  for (bool prepared : {false, true}) {
    CAPTURE(m);
    CAPTURE(prepared);
    route.Select(prepared);
    input.upload(original_input.data()); suh.upload(original_suh.data()); svh.upload(original_svh.data()); mapping.upload(original_map.data());
    if (m == 1600) {
      // Warm the private owner at129, then grow on the same queue at1600.
      auto prefix_in = input.tensor.Slice(0, 0, 129);
      auto prefix_out = out.tensor.Slice(0, 0, 129);
      vt::Exl3GroupedW8A8(gpu.q, prefix_out, prefix_in, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args);
      auto raw = out.download(); raw.resize(size_t(129) * n * 2);
      if (!prepared) prefix_control = raw;
      else {
        xpu_test::SameBytes(raw, prefix_control);
        const auto small = vt::PlanExl3W8A8(129, k, n, groups, bits, 1024);
        CHECK(vt::xpu::GetMemoryInfo().w8a8_preparation_bytes ==
              size_t(groups) * small.padded_rows * (k + sizeof(float)));
      }
    }
    execute();
    if (!prepared) real_control = out.download(); else xpu_test::SameBytes(out.download(), real_control);
    for (uint16_t bad : {uint16_t{0x7e00}, uint16_t{0x7c00}, uint16_t{0xfc00}}) {
      CAPTURE(bad);
      auto bad_input = original_input; replace_half(bad_input, bad_input.size() - 2, bad);
      input.upload(bad_input.data()); reject("finite inputs");
      auto bad_svh = original_svh; replace_half(bad_svh, bad_svh.size() - 2, bad);
      svh.upload(bad_svh.data()); reject("finite svh"); // Both fail: preserve SV error priority.
      input.upload(original_input.data()); reject("finite svh"); svh.upload(original_svh.data());
      auto bad_suh = original_suh; replace_half(bad_suh, bad_suh.size() - 2, bad);
      suh.upload(bad_suh.data()); reject("finite inputs"); suh.upload(original_suh.data());
    }
    auto bad_map = original_map; const int32_t invalid = groups;
    std::memcpy(bad_map.data() + bad_map.size() - 4, &invalid, 4);
    mapping.upload(bad_map.data()); reject("group out of range"); mapping.upload(original_map.data());
    std::vector<uint16_t> zero(size_t(m) * k, 0), ones(size_t(groups) * k, 0x3c00);
    // Valid zero, a finite product below65520, and a finite transform just below
    // overflow. Compare all public bytes against the existing native expression.
    for (int safe = 0; safe < 3; ++safe) {
      auto values = zero, scales = ones;
      if (safe == 1) { values[0] = vt::F32ToF16(21824.f); std::fill(scales.begin(), scales.end(), vt::F32ToF16(3.f)); }
      if (safe == 2) std::fill_n(values.begin(), 128, vt::F32ToF16(5788.f));
      input.upload(values.data()); suh.upload(scales.data()); poison(); execute();
      auto result = out.download(), workspace = scratch.download();
      if (!prepared) { safe_controls.push_back(result); safe_controls.push_back(workspace); }
      else { xpu_test::SameBytes(result, safe_controls[safe * 2]); xpu_test::SameBytes(workspace, safe_controls[safe * 2 + 1]); }
      if (safe == 0) {
        CHECK(std::all_of(workspace.begin() + plan.activation_offset,
            workspace.begin() + plan.row_scale_offset, [](unsigned char v) { return v == 0; }));
        for (size_t offset = plan.row_scale_offset; offset < plan.intermediate_offset; offset += 4)
          CHECK(vt::LoadUnaligned<float>(workspace.data() + offset) == 1.f);
        bool all_zero = true;
        for (size_t i = 0; i < result.size(); i += 2) all_zero &= (vt::LoadUnaligned<uint16_t>(result.data() + i) & 0x7fff) == 0;
        CHECK(all_zero);
      }
    }
    auto values = zero, scales = ones;
    // Exact F32 product65520 at the round-to-nearest half overflow midpoint.
    values[0] = vt::F32ToF16(21840.f); std::fill(scales.begin(), scales.end(), vt::F32ToF16(3.f));
    input.upload(values.data()); suh.upload(scales.data()); reject("finite inputs");
    // Every product is finite, but this block's FP16 Hadamard boundary overflows.
    std::fill_n(values.begin(), 128, vt::F32ToF16(5792.f));
    suh.upload(ones.data()); input.upload(values.data()); reject("finite inputs");
    values = zero; values[0] = 0x7bff; scales = ones; scales[0] = 0x7bff;
    input.upload(values.data()); suh.upload(scales.data()); reject("finite inputs");
    poison(); const auto before_alias = input.download(); auto alias = out.tensor; alias.data = input.tensor.data;
    CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(gpu.q, alias, input.tensor, trellis.tensor,
        suh.tensor, svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args),
        doctest::Contains("overlap"), std::runtime_error);
    xpu_test::SameBytes(input.download(), before_alias); xpu_test::SameBytes(out.download(), out_poison);
    xpu_test::SameBytes(scratch.download(), scratch_poison); xpu_test::SameBytes(panel.download(), panel_poison);
    input.upload(original_input.data()); suh.upload(original_suh.data()); execute();
    xpu_test::SameBytes(out.download(), real_control); // Failed private rows cannot poison a later valid lease.
    if (prepared) {
      const auto memory = vt::xpu::GetMemoryInfo();
      CHECK(memory.w8a8_preparation_bytes >= size_t(groups) * plan.padded_rows * (k + sizeof(float)));
      CHECK(memory.w8a8_preparation_bytes <= 64 * 1024 * 1024);
      CHECK(memory.allocated_bytes >= memory.w8a8_preparation_bytes);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  std::cout << "P4_PREPARE_BOUNDARY m=" << m << " k=" << k << " groups=" << groups
            << " original_native_exact=1 safe_rows=3 failure_contracts=1 private_bytes="
            << vt::xpu::GetMemoryInfo().w8a8_preparation_bytes << std::endl;
}

TEST_CASE("XPU EXL3 W8A8 P4: private budget refusal retains checked public preparation") {
  const char* fixture = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!fixture || !std::getenv("VT_XPU_MEMORY_BUDGET_BYTES")) std::exit(77);
  const auto f = vllm::SafetensorsFile::Open(fixture);
  const auto& tr = f.Get("merged_trellis"); const auto& su = f.Get("stacked_suh");
  const auto& sv = f.Get("merged_svh"); const auto& map = f.Get("source_map");
  const int m = 256, k = int(su.shape[1]), n = int(sv.shape[0]);
  const int groups = int(su.shape[0]), bits = int(tr.shape[2] / 16);
  Queue gpu(vt::DeviceType::kXPU); P4PrepareEnv route;
  const auto plan = vt::PlanExl3W8A8(m, k, n, groups, bits, 1024);
  Buffer trellis(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer mapping(gpu.q, DType::kI32, {n / 128}), input(gpu.q, DType::kF16, {m, k});
  Buffer out(gpu.q, DType::kF16, {m, n}), scratch(gpu.q, DType::kI8, {int64_t(plan.workspace_bytes)});
  Buffer panel(gpu.q, DType::kI8, {k, plan.weight_panel_columns});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(sv.data); mapping.upload(map.data);
  input.upload(f.Get("input_m256").data);
  const vt::Exl3GroupedLinearArgs args{bits, 2, "P4_PRIVATE_BUDGET", 1024};
  auto execute = [&] { vt::Exl3GroupedW8A8(gpu.q, out.tensor, input.tensor, trellis.tensor,
      suh.tensor, svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args); };
  route.Select(false); execute(); const auto expected = out.download();
  const auto before = vt::xpu::GetMemoryInfo();
  REQUIRE(before.budget_bytes <= 256 * 1024 * 1024);
  REQUIRE(before.w8a8_preparation_bytes == 0); // This case runs alone in a fresh worker.
  const size_t needed = size_t(groups) * plan.padded_rows * (k + sizeof(float));
  const size_t remaining = before.budget_bytes - before.allocated_bytes - before.graph_device_bytes;
  REQUIRE(remaining > needed);
  route.Select(true);
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
  std::cout << "P4_PRIVATE_BUDGET fallback_exact=1 acquired_bytes=" << needed
            << " peak_bytes=" << acquired.peak_allocated_bytes << std::endl;
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 W8A8: real grouped operator boundary large-M and reuse") {
  const char* env = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!env) { std::cerr << "SKIP: VT_B70_EXL3_W8A8_FIXTURE required\n"; std::exit(77); }
  auto f = vllm::SafetensorsFile::Open(env);
  const auto& tr = f.Get("merged_trellis");
  const auto& su = f.Get("stacked_suh"); const auto& sv = f.Get("merged_svh");
  const auto& map = f.Get("source_map");
  const int k = int(tr.shape[0] * 16), n = int(tr.shape[1] * 16);
  const int bits = int(tr.shape[2] / 16), groups = int(su.shape[0]);
  REQUIRE(tr.dtype == "I16"); REQUIRE(su.dtype == "F16"); REQUIRE(sv.dtype == "F16");
  REQUIRE(map.dtype == "I32");
  Queue gpu(vt::DeviceType::kXPU);
  Buffer trellis(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
  Buffer suh(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
  Buffer mapping(gpu.q, DType::kI32, {n / 128});
  trellis.upload(tr.data); suh.upload(su.data); svh.upload(sv.data); mapping.upload(map.data);
  const std::string matrix = std::filesystem::path(env).stem().string();
  const vt::Exl3GroupedLinearArgs args{bits, 2, matrix.c_str()};
  const bool four_k = std::find(f.Names().begin(), f.Names().end(), "input_m4096") != f.Names().end();
  const std::vector<int> sequence = four_k ? std::vector<int>{4096} : std::vector<int>{129, 128, 256, 128, 129};
  const auto maximum = vt::PlanExl3W8A8(four_k ? 4096 : 256, k, n, groups, bits);
  Buffer scratch(gpu.q, DType::kI8, {int64_t(maximum.workspace_bytes)});
  Buffer panel(gpu.q, DType::kI8, {k, 128});
  // Explicit large/small/large transitions exercise both arithmetic routes,
  // reusing the W8A8 storage instead of relying on fresh allocator contents.
  for (int m : sequence) {
    CAPTURE(m);
    Buffer in(gpu.q, DType::kF16, {m, k}), out(gpu.q, DType::kF16, {m, n});
    const auto& x = f.Get("input_m" + std::to_string(m)); in.upload(x.data);
    if (m == 128) {
      const auto p = vt::PlanExl3SmallM(m, k, n, bits);
      Buffer had(gpu.q, DType::kF16, {groups, k / 16, p.padded_rows, 16});
      Buffer parts(gpu.q, DType::kF32, {p.splits, m, n});
      vt::Exl3GroupedLinear(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, had.tensor, parts.tensor, args);
      xpu_test::SameBytes(out.download(), std::vector<unsigned char>(
          f.Get("output_m128").data, f.Get("output_m128").data + f.Get("output_m128").nbytes));
      continue;
    }
    std::vector<unsigned char> poison(scratch.bytes, 0xff); scratch.upload(poison.data());
    vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
        svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args);
    Accuracy(out.download(), f.Get("output_m" + std::to_string(m)), "output");
    const auto& wref = f.Get("last_weight_panel_m" + std::to_string(m));
    xpu_test::SameBytes(panel.download(), std::vector<unsigned char>(wref.data, wref.data + wref.nbytes));
    const auto p = vt::PlanExl3W8A8(m, k, n, groups, bits);
    const auto raw = scratch.download();
    const auto& qref = f.Get("xq_m" + std::to_string(m));
    const auto& sref = f.Get("sx_m" + std::to_string(m));
    bool quantized_exact = true, scales_exact = true, padding_zero = true;
    for (int g = 0; g < groups; ++g) for (int row = 0; row < p.padded_rows; ++row) {
      const auto* a = raw.data() + p.activation_offset + (size_t(g) * p.padded_rows + row) * k;
      const auto* scale = raw.data() + p.row_scale_offset + (size_t(g) * p.padded_rows + row) * 4;
      if (row < m) {
        quantized_exact &= std::memcmp(a, qref.data + (size_t(g) * m + row) * k, k) == 0;
        scales_exact &= std::memcmp(scale, sref.data + (size_t(g) * m + row) * 4, 4) == 0;
      } else {
        padding_zero &= std::all_of(a, a + k, [](unsigned char v) { return v == 0; });
        padding_zero &= vt::LoadUnaligned<float>(scale) == 1.0f;
      }
    }
    CHECK(quantized_exact); CHECK(scales_exact); CHECK(padding_zero);
    const auto ybytes = size_t(m) * n * 2;
    Accuracy(std::vector<unsigned char>(raw.begin() + p.intermediate_offset,
        raw.begin() + p.intermediate_offset + ybytes), f.Get("y_m" + std::to_string(m)), "F16_intermediate");
    if (m == 129) {
      if (groups > 1) {
        // H128 is block-local. Permuting complete packed output blocks, SV
        // and their source IDs must give exactly the same permutation of the
        // original W8A8 output, including a noncontiguous source-group map.
        auto packed_bytes = trellis.download(), sv_bytes = svh.download();
        auto map_bytes = mapping.download();
        const size_t block_bytes = size_t(8) * 32 * bits;
        for (int tile = 0; tile < k / 16; ++tile) {
          const size_t first = size_t(tile) * (n / 16) * 32 * bits;
          const size_t last = first + size_t(n / 128 - 1) * block_bytes;
          std::swap_ranges(packed_bytes.begin() + first, packed_bytes.begin() + first + block_bytes,
                           packed_bytes.begin() + last);
        }
        std::swap_ranges(sv_bytes.begin(), sv_bytes.begin() + 256, sv_bytes.end() - 256);
        for (int i = 0; i < 4; ++i) std::swap(map_bytes[i], map_bytes[map_bytes.size() - 4 + i]);
        REQUIRE(vt::LoadUnaligned<int32_t>(map_bytes.data()) !=
                vt::LoadUnaligned<int32_t>(map_bytes.data() + map_bytes.size() - 4));
        trellis.upload(packed_bytes.data()); svh.upload(sv_bytes.data()); mapping.upload(map_bytes.data());
        const auto& expected = f.Get("output_m129");
        std::vector<unsigned char> permuted(expected.data, expected.data + expected.nbytes);
        for (int row = 0; row < m; ++row) {
          const size_t start = size_t(row) * n * 2;
          std::swap_ranges(permuted.begin() + start, permuted.begin() + start + 256,
                           permuted.begin() + start + size_t(n - 128) * 2);
        }
        vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
            svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args);
        xpu_test::SameBytes(out.download(), permuted);
        trellis.upload(tr.data); svh.upload(sv.data); mapping.upload(map.data);
      }
      const auto before = out.download();
      auto bad = scratch.tensor; bad.shape[0] = int64_t(p.workspace_bytes) - 1;
      CHECK_THROWS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, bad, panel.tensor, args));
      auto alias = panel.tensor; alias.data = out.tensor.data;
      CHECK_THROWS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, scratch.tensor, alias, args));
      auto map_raw = mapping.download();
      const int32_t invalid = groups; std::memcpy(map_raw.data(), &invalid, sizeof(invalid));
      mapping.upload(map_raw.data());
      CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args),
          doctest::Contains("group out of range"), std::runtime_error);
      xpu_test::SameBytes(out.download(), before);
      mapping.upload(map.data);
      const auto original_input = in.download();
      const auto original_svh = svh.download();
      const auto original_suh = suh.download();
      auto reject = [&](const char* message, const std::string& boundary = "nonfinite operand") {
        CAPTURE(boundary);
        CHECK_THROWS_WITH_AS(vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor,
            suh.tensor, svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args),
            doctest::Contains(message), std::runtime_error);
        xpu_test::SameBytes(out.download(), before);
      };
      for (const uint16_t value : {uint16_t{0x7e00}, uint16_t{0x7c00}, uint16_t{0xfc00}}) {
        CAPTURE(value);
        // First/last blocks exercise the whole parallel SV check, including
        // its tail. The input and metadata failures must not write output.
        for (const size_t offset : {size_t{0}, svh.bytes - 2}) {
          auto bad_svh = original_svh;
          std::memcpy(bad_svh.data() + offset, &value, 2); svh.upload(bad_svh.data());
          reject("finite svh");
        }
        svh.upload(original_svh.data());
        auto bad_input = original_input;
        std::memcpy(bad_input.data() + bad_input.size() - 2, &value, 2);
        in.upload(bad_input.data()); reject("finite inputs");
        auto bad_svh = original_svh;
        std::memcpy(bad_svh.data(), &value, 2); svh.upload(bad_svh.data());
        reject("finite svh");  // Prior error priority when both checks fail.
        in.upload(original_input.data()); svh.upload(original_svh.data());
      }
      // Finite operands can overflow at either FP16 rounding boundary.
      auto bad_input = original_input, bad_suh = original_suh;
      const uint16_t maximum_half = 0x7bff, one_half = 0x3c00;
      std::memcpy(bad_input.data(), &maximum_half, 2);
      std::memcpy(bad_suh.data(), &maximum_half, 2);
      in.upload(bad_input.data()); suh.upload(bad_suh.data()); reject("finite inputs", "FP16 product");
      for (int col = 0; col < 128; ++col) {
        std::memcpy(bad_input.data() + col * 2, &maximum_half, 2);
        std::memcpy(bad_suh.data() + col * 2, &one_half, 2);
      }
      in.upload(bad_input.data()); suh.upload(bad_suh.data()); reject("finite inputs", "FP16 Hadamard");
      in.upload(original_input.data()); suh.upload(original_suh.data());
      // Reuse the same flag storage after failed calls: neither zero flag may
      // leak into the next valid projection.
      vt::Exl3GroupedW8A8(gpu.q, out.tensor, in.tensor, trellis.tensor, suh.tensor,
          svh.tensor, mapping.tensor, scratch.tensor, panel.tensor, args);
      Accuracy(out.download(), f.Get("output_m129"), "after_rejected_operands");
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 W8A8 model seam: grouped and single projection own scratch retirement") {
  const char* env = std::getenv("VT_B70_EXL3_W8A8_FIXTURE");
  if (!env) std::exit(77);
  auto f = vllm::SafetensorsFile::Open(env);
  const auto& tr = f.Get("merged_trellis");
  const auto& su = f.Get("stacked_suh");
  const int64_t k = tr.shape[0] * 16, n = tr.shape[1] * 16;
  const int bits = int(tr.shape[2] / 16), groups = int(su.shape[0]);
  auto owned = [](const vllm::StTensor& source, DType dtype,
                   std::initializer_list<int64_t> shape) {
    vllm::OwnedTensor t; t.dtype = dtype; t.rank = int(shape.size());
    std::copy(shape.begin(), shape.end(), t.shape);
    t.bytes.resize(source.nbytes); std::memcpy(t.bytes.data(), source.data, source.nbytes);
    return t;
  };
  vllm::Exl3GroupedWeight w; w.name = "model_seam_grouped"; w.codebook = 2;
  w.trellis = owned(tr, DType::kI8, {k / 16, n / 16, 32 * bits});
  w.suh = owned(su, DType::kF16, {groups, k});
  w.svh = owned(f.Get("merged_svh"), DType::kF16, {n});
  w.source_map = owned(f.Get("source_map"), DType::kI32, {n / 128});
  Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  size_t capacity = vt::xpu::GetMemoryInfo().w8a8_workspace_bytes;
  for (int m : {129, 256, 128}) {
    Buffer input(gpu.q, DType::kF16, {m, k}); input.upload(f.Get("input_m" + std::to_string(m)).data);
    auto out = vllm::dense_attn::Exl3GroupedMatmulD(d, input.tensor, w);
    std::vector<unsigned char> raw(size_t(m) * n * 2);
    d.b.Copy(d.q, raw.data(), out.t().data, raw.size()); d.b.Synchronize(d.q);
    Accuracy(raw, f.Get("output_m" + std::to_string(m)), "grouped_model_seam");
    CHECK(w.trellis.bytes.empty()); CHECK(w.suh.bytes.empty());
    if (m > 128) {
      const auto p = vt::PlanExl3W8A8(m, k, n, groups, bits, vt::Exl3W8A8ModelPanelColumns());
      capacity = std::max(capacity, p.workspace_bytes + p.weight_panel_bytes);
      CHECK(vt::xpu::GetMemoryInfo().w8a8_workspace_bytes == capacity);
    }
  }
  if (groups == 1) {
    vllm::Exl3Weight single; single.name = "model_seam_single"; single.codebook = 2;
    single.trellis = owned(tr, DType::kI8, {k / 16, n / 16, 32 * bits});
    single.suh = owned(su, DType::kF16, {k});
    single.svh = owned(f.Get("merged_svh"), DType::kF16, {n});
    Buffer input(gpu.q, DType::kF16, {129, k}); input.upload(f.Get("input_m129").data);
    auto out = vllm::dense_attn::Exl3MatmulD(d, input.tensor, single, DType::kF16);
    std::vector<unsigned char> raw(129 * n * 2);
    d.b.Copy(d.q, raw.data(), out.t().data, raw.size()); d.b.Synchronize(d.q);
    Accuracy(raw, f.Get("output_m129"), "single_model_seam");
    const auto p = vt::PlanExl3W8A8(129, k, n, 1, bits, vt::Exl3W8A8ModelPanelColumns());
    capacity = std::max(capacity, p.workspace_bytes + p.weight_panel_bytes);
    CHECK(vt::xpu::GetMemoryInfo().w8a8_workspace_bytes == capacity);
    // Exercise the projection's generated routing metadata under real XPU
    // capture. A per-call zero-filled map must fail backend preflight.
    Buffer decode_input(gpu.q, DType::kF16, {128, k});
    decode_input.upload(f.Get("input_m128").data);
    auto warm = vllm::dense_attn::Exl3MatmulD(d, decode_input.tensor, single, DType::kF16);
    std::vector<unsigned char> eager_bytes(128 * n * 2);
    d.b.Copy(gpu.q, eager_bytes.data(), warm.t().data, eager_bytes.size());
    d.b.Synchronize(d.q);
    warm = {};
    const auto* map_address = single.single_source_map.d_dev.get();
    vt::BreakableGraph graph;
    vllm::dense_attn::DBuf captured;
    {
      vt::GraphCaptureScope scope(d.b, gpu.q, graph, vt::GraphCaptureMode::kFull);
      captured = vllm::dense_attn::Exl3MatmulD(d, decode_input.tensor, single, DType::kF16);
    }
    REQUIRE(graph.captured());
    graph.Replay(gpu.q);
    raw.resize(128 * n * 2);
    d.b.Copy(gpu.q, raw.data(), captured.t().data, raw.size());
    d.b.Synchronize(gpu.q);
    Accuracy(raw, f.Get("output_m128"), "single_model_graph");
    CHECK(raw == eager_bytes);
    CHECK(single.single_source_map.d_dev.get() == map_address);
    // Changed input contents reach the same graph and persistent map.
    d.b.Memset(gpu.q, decode_input.tensor.data, 0, decode_input.tensor.Bytes());
    graph.Replay(gpu.q);
    d.b.Copy(gpu.q, raw.data(), captured.t().data, raw.size());
    d.b.Synchronize(gpu.q);
    bool all_zero = true;
    for (size_t i = 0; i < raw.size(); i += 2)
      all_zero &= vt::F16ToF32(vt::LoadUnaligned<uint16_t>(raw.data() + i)) == 0.0f;
    CHECK(all_zero);
    graph.Reset();
  }
  const auto info = vt::xpu::GetMemoryInfo();
  std::cout << "P2_MODEL_POOL retained_bytes=" << info.w8a8_workspace_bytes
            << " device_peak_bytes=" << info.peak_allocated_bytes << '\n';
  CHECK(vt::GetReferenceTierHits() == 0);
}

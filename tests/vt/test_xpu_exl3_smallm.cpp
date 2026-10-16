#include "exl3_smallm_test_helpers.h"
#include "vt/xpu.h"
#include "vt/xpu_graph_metadata.h"
#include "exl3_fixture.h"
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string_view>

using namespace exl3_smallm_test;

TEST_CASE("XPU EXL3 SmallM P7: owner retirement excludes a concurrent new capture") {
  Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(first.q.device); REQUIRE(backend.SupportsGraphCapture());
  const auto graph_bytes = vt::xpu::GetMemoryInfo().graph_device_bytes;
  std::promise<void> deleting, release, requesting;
  auto deleting_future = deleting.get_future(), requesting_future = requesting.get_future();
  const auto release_future = release.get_future().share();
  Buffer destination(first.q, DType::kI32, {16});
  auto owner = std::shared_ptr<void>(vt::Alloc(first.q.device, 64),
      [&, device = first.q.device](void* p) {
        deleting.set_value(); release_future.wait(); vt::Free(device, p);
      });
  backend.Memset(first.q, owner.get(), 0x35, 64); backend.Synchronize(first.q);
  backend.BeginCapture(first.q);
  vt::xpu::RecordGraphImmutableRead(first.q, owner.get(), 64, owner, "concurrent retirement source");
  backend.Copy(first.q, destination.tensor.data, owner.get(), 64);
  void* graph = backend.EndCaptureGraph(first.q); owner.reset();
  auto retire = std::async(std::launch::async, [&] { backend.DestroyGraph(graph); });
  deleting_future.get();
  auto capture = std::async(std::launch::async, [&] {
    requesting.set_value(); backend.BeginCapture(second.q);
  });
  requesting_future.get();
  const bool excluded = capture.wait_for(std::chrono::milliseconds(30)) == std::future_status::timeout;
  // Test-only coordination, no product polling or timed readiness loop.
  release.set_value(); retire.get(); capture.get(); CHECK(excluded);
  void* empty = backend.EndCaptureGraph(second.q); backend.DestroyGraph(empty);
  CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == graph_bytes);
}

TEST_CASE("XPU EXL3 SmallM P7: immutable model maps preserve guards graph ownership and reload") {
  const char* name = "VT_XPU_SMALLM_MODEL_MAP";
  const bool present = std::getenv(name) != nullptr;
  const std::string prior = present ? std::getenv(name) : "";
  struct Restore {
    const char* name; bool present; std::string prior;
    ~Restore() {
      if (present) setenv(name, prior.c_str(), 1); else unsetenv(name);
    }
  } restore{name, present, prior};
  setenv(name, "1", 1);
  constexpr int k = 256, n = 384, groups = 2;
  const std::vector<int32_t> values{1, 0, 1};
  Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(first.q.device);
  auto resident_map = [&] {
    auto owner = std::shared_ptr<void>(vt::Alloc(first.q.device, values.size() * sizeof(int32_t)),
        [device = first.q.device](void* p) { vt::Free(device, p); });
    backend.Copy(first.q, owner.get(), values.data(), values.size() * sizeof(int32_t));
    backend.Synchronize(first.q); return owner;
  };
  for (int bits : {4, 6}) for (int m : {1, 4, 16}) {
    CAPTURE(bits);
    CAPTURE(m);
    auto f = exl3_test::MakeFixture(k, n, bits, 0x93adu + bits);
    auto su = f.suh;
    for (int i = 0; i < k; ++i)
      su.push_back(vt::F32ToF16((i % 3 ? .31f : -.62f) * vt::F16ToF32(f.suh[i])));
    Buffer input(first.q, DType::kF16, {m, k});
    Buffer packed(first.q, DType::kI8, {k / 16, n / 16, 32 * bits});
    Buffer scales(first.q, DType::kF16, {groups, k}), svh(first.q, DType::kF16, {n});
    Buffer output(first.q, DType::kF16, {m, n}); Scratch scratch(first.q, m, k, n, bits, groups);
    input.put(xpu_test::Values(m * k, 11, .009f)); packed.upload(f.trellis.data());
    scales.upload(su.data()); svh.upload(f.svh.data());
    auto owner = resident_map();
    auto map = vt::Tensor::Contiguous(owner.get(), DType::kI32, first.q.device, {n / 128});
    vt::SharedPtrCache<const vt::Exl3W8A8ModelMap> cache;
    const vt::Exl3GroupedLinearArgs args{bits, 2, "P7_SMALLM_MODEL_MAP"};
    auto run_public = [&](const vt::Tensor& routing, const vt::Exl3GroupedLinearArgs& a) {
      vt::Exl3GroupedLinear(first.q, output.tensor, input.tensor, packed.tensor, scales.tensor,
          svh.tensor, routing, scratch.had.tensor, scratch.parts.tensor, a);
    };
    auto run_owned = [&](vt::Queue& q) {
      vt::detail::Exl3GroupedLinearModel(q, output.tensor, input.tensor, packed.tensor, scales.tensor,
          svh.tensor, map, scratch.had.tensor, scratch.parts.tensor, args, owner, cache);
    };
    run_public(map, args);
    const auto expected = output.download(), had = scratch.had.download(), parts = scratch.parts.download();
    run_owned(first.q); REQUIRE(cache.Load()); CHECK(cache.Load()->Matches(map, groups, owner));
    xpu_test::SameBytes(output.download(), expected); xpu_test::SameBytes(scratch.had.download(), had);
    xpu_test::SameBytes(scratch.parts.download(), parts);
    run_owned(second.q); backend.Synchronize(second.q);
    xpu_test::SameBytes(output.download(), expected); xpu_test::SameBytes(scratch.parts.download(), parts);
    Buffer bad_map(first.q, DType::kI32, {n / 128});
    const int32_t bad[] = {1, groups, 0}; bad_map.upload(bad);
    {
      const auto injected_map = cache.Load();
      auto injected = args; injected.model_map = injected_map.get();
      CHECK_THROWS_WITH_AS(run_public(bad_map.tensor, injected), doctest::Contains("group out of range"), std::runtime_error);
    }
    xpu_test::SameBytes(output.download(), expected); xpu_test::SameBytes(scratch.had.download(), had);
    xpu_test::SameBytes(scratch.parts.download(), parts);
    auto alias = scratch.parts.tensor; alias.data = output.tensor.data;
    CHECK_THROWS_AS(vt::detail::Exl3GroupedLinearModel(first.q, output.tensor, input.tensor,
        packed.tensor, scales.tensor, svh.tensor, map, scratch.had.tensor, alias, args, owner, cache), std::runtime_error);
    xpu_test::SameBytes(output.download(), expected);
    if (bits != 4 || m != 4) continue;
    REQUIRE(backend.SupportsGraphCapture());
    const auto initial = vt::xpu::GetMemoryInfo();
    CHECK_FALSE(vt::xpu::IsGraphCapturing(first.q)); backend.BeginCapture(first.q);
    CHECK(vt::xpu::IsGraphCapturing(first.q)); run_owned(first.q);
    void* warm_graph = backend.EndCaptureGraph(first.q);
    CHECK_FALSE(vt::xpu::IsGraphCapturing(first.q));
    const auto warm_nodes = vt::xpu::GetMemoryInfo().graph_nodes - initial.graph_nodes;
    backend.ReplayGraph(second.q, warm_graph); backend.Synchronize(second.q);
    xpu_test::SameBytes(output.download(), expected); xpu_test::SameBytes(scratch.parts.download(), parts);
    // Even a checked model map must stay read-only in the recorded graph.
    backend.BeginCapture(first.q); run_owned(first.q); backend.Memset(first.q, map.data, 0, map.Bytes());
    CHECK_THROWS_WITH_AS(backend.EndCaptureGraph(first.q), doctest::Contains("immutable metadata must remain read-only"), std::runtime_error);
    xpu_test::SameBytes(output.download(), expected);
    CHECK(vt::xpu::GetMemoryInfo().graph_count == initial.graph_count + 1);
    // A rejected capture may contain the last allocation reference. Releasing
    // it under the context lock would deadlock its ordinary VT Free deleter.
    auto temporary = resident_map(); std::weak_ptr<void> failed_lifetime = temporary;
    auto temporary_map = map; temporary_map.data = temporary.get();
    vt::SharedPtrCache<const vt::Exl3W8A8ModelMap> temporary_cache;
    auto temporary_run = [&] { vt::detail::Exl3GroupedLinearModel(first.q, output.tensor,
        input.tensor, packed.tensor, scales.tensor, svh.tensor, temporary_map,
        scratch.had.tensor, scratch.parts.tensor, args, temporary, temporary_cache); };
    temporary_run(); backend.Synchronize(first.q);
    backend.BeginCapture(second.q); backend.BeginCapture(first.q); temporary_run();
    backend.Memset(first.q, temporary_map.data, 0, temporary_map.Bytes());
    temporary.reset(); temporary_cache.Reset(); CHECK_FALSE(failed_lifetime.expired());
    CHECK_THROWS_AS(backend.EndCaptureGraph(first.q), std::runtime_error);
    CHECK_FALSE(failed_lifetime.expired());
    void* other_capture = backend.EndCaptureGraph(second.q);
    CHECK(failed_lifetime.expired()); backend.DestroyGraph(other_capture);
    // A cold model capture uses the actual replay guard, while still pinning
    // its supplied map owner. No CPU/device readback is attempted in capture.
    auto cold_owner = resident_map(); auto cold_map = map; cold_map.data = cold_owner.get();
    vt::SharedPtrCache<const vt::Exl3W8A8ModelMap> cold_cache;
    backend.BeginCapture(first.q);
    vt::detail::Exl3GroupedLinearModel(first.q, output.tensor, input.tensor, packed.tensor,
        scales.tensor, svh.tensor, cold_map, scratch.had.tensor, scratch.parts.tensor,
        args, cold_owner, cold_cache);
    void* cold_graph = backend.EndCaptureGraph(first.q); CHECK_FALSE(cold_cache.Load());
    const auto cold_nodes = vt::xpu::GetMemoryInfo().graph_nodes - initial.graph_nodes - warm_nodes;
    CHECK(cold_nodes == warm_nodes + 1);
    backend.ReplayGraph(first.q, cold_graph); xpu_test::SameBytes(output.download(), expected);
    backend.Copy(first.q, cold_map.data, bad, cold_map.Bytes()); backend.Synchronize(first.q);
    CHECK_THROWS_WITH_AS(backend.ReplayGraph(first.q, cold_graph), doctest::Contains("group out of range"), std::runtime_error);
    xpu_test::SameBytes(output.download(), expected); xpu_test::SameBytes(scratch.had.download(), had);
    xpu_test::SameBytes(scratch.parts.download(), parts);
    backend.Copy(first.q, cold_map.data, values.data(), cold_map.Bytes()); backend.Synchronize(first.q);
    std::weak_ptr<void> cold_lifetime = cold_owner; cold_owner.reset();
    CHECK_FALSE(cold_lifetime.expired()); backend.ReplayGraph(first.q, cold_graph);
    xpu_test::SameBytes(output.download(), expected); backend.DestroyGraph(cold_graph);
    CHECK(cold_lifetime.expired());
    // A new valid generation may coexist with an old captured graph. The old
    // graph keeps its exact map allocation despite the model cache being reset.
    std::weak_ptr<void> old_lifetime = owner; cache.Reset(); owner.reset();
    CHECK_FALSE(old_lifetime.expired());
    owner = resident_map(); map.data = owner.get();
    const int32_t changed[] = {0, 1, 0}; backend.Copy(first.q, map.data, changed, map.Bytes());
    backend.Synchronize(first.q); run_public(map, args); const auto changed_expected = output.download();
    run_owned(first.q); REQUIRE(cache.Load()); CHECK(cache.Load()->Matches(map, groups, owner));
    xpu_test::SameBytes(output.download(), changed_expected);
    backend.ReplayGraph(first.q, warm_graph); xpu_test::SameBytes(output.download(), expected);
    backend.BeginCapture(second.q); backend.DestroyGraph(warm_graph);
    CHECK_FALSE(old_lifetime.expired());
    other_capture = backend.EndCaptureGraph(second.q);
    CHECK(old_lifetime.expired()); backend.DestroyGraph(other_capture);
    CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == initial.graph_device_bytes);
    unsetenv(name); cache.Reset(); run_owned(first.q);
    REQUIRE(cache.Load()); CHECK(cache.Load()->Matches(map, groups, owner));
    xpu_test::SameBytes(output.download(), changed_expected);
    setenv(name, "1", 1);
    std::cout << "P7_SMALLM_MODEL_MAP warm_nodes=" << warm_nodes << " cold_nodes=" << cold_nodes
              << " public_replay_guards=1 owner_retirement=1 generation_reload=1" << std::endl;
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 producer SmallM: separate group transforms, padding and refusals") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int k = 256, n = 384, groups = 2;
  const std::vector<int32_t> mapping{1, 0, 1};
  for (int bits : {4, 6}) for (int m : {1, 4, 12, 16, 128}) {
    CAPTURE(bits);
    CAPTURE(m);
    auto f = exl3_test::MakeFixture(k, n, bits, 0x9827u + bits);
    auto u = f.suh;
    // Group1 is not merely the same transform under another index.
    for (int i = 0; i < k; ++i)
      u.push_back(vt::F32ToF16((i % 3 ? 0.31f : -0.62f) * vt::F16ToF32(f.suh[i])));
    Buffer input(gpu.q, DType::kF16, {m, k}), packed(gpu.q, DType::kI8, {k / 16, n / 16, 32 * bits});
    Buffer scales(gpu.q, DType::kF16, {groups, k}), svh(gpu.q, DType::kF16, {n});
    Buffer shard(gpu.q, DType::kI32, {n / 128}), output(gpu.q, DType::kF16, {m, n});
    input.put(xpu_test::Values(m * k, 13, 0.007f));
    packed.upload(f.trellis.data()); scales.upload(u.data()); svh.upload(f.svh.data());
    shard.upload(mapping.data());
    Scratch scratch(gpu.q, m, k, n, bits, groups);
    const vt::Exl3GroupedLinearArgs args{bits, 2, "synthetic_grouped"};
    vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
                          scales.tensor, svh.tensor, shard.tensor,
                          scratch.had.tensor, scratch.parts.tensor, args);
    const auto got = output.floats();
    CheckPadding(scratch, m, k, groups);
    std::vector<float> expected(m * n);
    const auto input_bytes = input.download();
    // Independent CPU-order reference: one 128-output block at a time with
    // the input transform selected by its source group. Use F32 output before
    // the final F16 rounding, as in the producer's FP32 HadOut/scale tail.
    for (int nb = 0; nb < n / 128; ++nb) {
      Buffer ca(cpu.q, DType::kF16, {m, k}), cb(cpu.q, DType::kI8, {k / 16, 8, 32 * bits});
      Buffer cu(cpu.q, DType::kF16, {k}), cv(cpu.q, DType::kF16, {128});
      Buffer ch(cpu.q, DType::kF16, {m, k}), co(cpu.q, DType::kF32, {m, 128});
      std::vector<uint8_t> panel(cb.bytes);
      for (int tile = 0; tile < k / 16; ++tile)
        std::memcpy(panel.data() + size_t(tile) * 8 * 32 * bits,
                    reinterpret_cast<const uint8_t*>(f.trellis.data()) +
                        (size_t(tile) * (n / 16) + nb * 8) * 32 * bits,
                    8 * 32 * bits);
      ca.upload(input_bytes.data()); cb.upload(panel.data());
      cu.upload(u.data() + mapping[nb] * k); cv.upload(f.svh.data() + nb * 128);
      vt::Exl3Gemm(cpu.q, co.tensor, ca.tensor, cb.tensor, cu.tensor, cv.tensor, ch.tensor,
                   vt::Exl3GemmArgs{bits, 2});
      const auto panel_out = co.floats();
      for (int row = 0; row < m; ++row) for (int col = 0; col < 128; ++col)
        expected[row * n + nb * 128 + col] =
            vt::F16ToF32(vt::F32ToF16(panel_out[row * 128 + col]));
    }
    const double relative = Relative(got, expected);
    std::cout << "GROUPED_SMALLM bits=" << bits << " M=" << m << " relative=" << relative << '\n';
    CHECK(relative < 2e-3);
    if (m == 4 && bits == 4) {
      const auto before = output.download();
      auto bad = args; bad.codebook = 1;
      CHECK_THROWS(vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
          scales.tensor, svh.tensor, shard.tensor, scratch.had.tensor, scratch.parts.tensor, bad));
      auto alias = scratch.parts.tensor; alias.data = output.tensor.data;
      CHECK_THROWS(vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
          scales.tensor, svh.tensor, shard.tensor, scratch.had.tensor, alias, args));
      auto wrong = scratch.had.tensor; --wrong.shape[2];
      CHECK_THROWS(vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
          scales.tensor, svh.tensor, shard.tensor, wrong, scratch.parts.tensor, args));
      const int32_t invalid[] = {1, 0, groups}; shard.upload(invalid);
      CHECK_THROWS_WITH_AS(vt::Exl3GroupedLinear(gpu.q, output.tensor, input.tensor, packed.tensor,
          scales.tensor, svh.tensor, shard.tensor, scratch.had.tensor, scratch.parts.tensor, args),
          doctest::Contains("group out of range"), std::runtime_error);
      xpu_test::SameBytes(output.download(), before);
    }
  }
  CHECK_THROWS(vt::PlanExl3SmallM(129, k, n, 4));
  CHECK_THROWS(vt::PlanExl3SmallM(1, k, n, 3));
  CHECK(vt::GetReferenceTierHits() == 0);
}

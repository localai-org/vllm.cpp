#include "xpu_test_helpers.h"
#include "vt/xpu_sampling.h"
#include "vt/xpu_argmax_plan.h"
#include "vt/xpu.h"
#include "vt/breakable_graph.h"
#include "vllm/v1/sample/sampler.h"
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <string>

namespace {
using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;
constexpr auto Inf = std::numeric_limits<float>::infinity();
constexpr auto NaN = std::numeric_limits<float>::quiet_NaN();
bool Profiling() {
  return std::getenv("VT_XPU_PROFILE") && std::string(std::getenv("VT_XPU_PROFILE")) == "1";
}
std::vector<float> Values(int rows, int vocab, int mode) {
  std::vector<float> values(size_t(rows) * vocab, -Inf);
  for (int r = 0; r < rows; ++r) {
    auto* row = values.data() + size_t(r) * vocab;
    switch ((mode + r) % 8) {
      case 0: row[0] = 2; row[vocab - 1] = 2; break;
      case 1: row[0] = NaN; row[vocab - 1] = Inf; if (vocab == 1) row[0] = NaN; break;
      case 2: for (int c = 1; c < vocab; ++c) row[c] = NaN; break;
      case 3: std::fill(row, row + vocab, 0.f); row[0] = -0.f; break;
      case 4: if (vocab > 1) { row[vocab / 2] = Inf; row[vocab - 1] = Inf; } break;
      case 5: row[vocab - 1] = 9; break;
      case 6: row[vocab / 2] = -0.f; row[vocab - 1] = +0.f; break;
      case 7: break; // All -infinity, including nonwinning tiles.
    }
  }
  return values;
}
void RouteEvents() {
  const auto events = vt::xpu::DrainProfileEvents();
  if (!Profiling()) return;
  for (const char* name : {"greedy_argmax_tiled_partials", "greedy_argmax_tiled_merge"})
    CHECK(std::count_if(events.begin(), events.end(), [&](const auto& event) { return event.stage == name; }) == 1);
}
}

TEST_CASE("XPU tiled argmax: exact target IDs tails ties NaNs and untouched capacity") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int rows : {1, 4}) for (int vocab : {1, 4095, 4096, 4097, 8191, 8192, 8193, 248320})
  for (int tile : {4096, 8192}) {
    CAPTURE(rows);
    CAPTURE(vocab);
    CAPTURE(tile);
    Buffer host_input(cpu.q, DType::kF32, {rows, vocab}), host_ids(cpu.q, DType::kI64, {rows});
    Buffer input(gpu.q, DType::kF32, {rows + 1, vocab}), ids(gpu.q, DType::kI64, {rows + 4});
    const auto bytes = vt::XpuTiledArgmaxWorkspaceBytes(rows, vocab, tile);
    Buffer workspace(gpu.q, DType::kI8, {int64_t(bytes + 32)});
    auto logical_input = input.tensor; logical_input.shape[0] = rows;
    auto logical_ids = vt::Tensor::Contiguous(static_cast<int64_t*>(ids.tensor.data) + 2, DType::kI64, gpu.q.device, {rows});
    auto scratch = vt::Tensor::Contiguous(static_cast<unsigned char*>(workspace.tensor.data) + 16,
        DType::kI8, gpu.q.device, {int64_t(bytes)});
    for (int mode : {0, 1, 2, 3, 4, 5, 6, 7}) {
      auto data = Values(rows, vocab, mode); host_input.put(data);
      data.resize(size_t(rows + 1) * vocab, NaN); input.put(data);
      workspace.upload(std::vector<unsigned char>(bytes + 32, 0xa5).data());
      ids.upload(std::vector<uint64_t>(rows + 4, 0x9d9d9d9d9d9d9d9dULL).data());
      const auto before = input.download();
      vt::GreedyArgmax(cpu.q, host_ids.tensor, host_input.tensor); const auto expected = host_ids.download();
      (void)vt::xpu::DrainProfileEvents();
      vt::xpu::TiledGreedyArgmax(gpu.q, logical_ids, logical_input, scratch, tile);
      const auto output = ids.download(); RouteEvents();
      xpu_test::SameBytes(std::vector<unsigned char>(output.begin() + 16, output.begin() + 16 + expected.size()), expected);
      CHECK(std::all_of(output.begin(), output.begin() + 16, [](auto x) { return x == 0x9d; }));
      CHECK(std::all_of(output.end() - 16, output.end(), [](auto x) { return x == 0x9d; }));
      const auto work = workspace.download();
      CHECK(std::all_of(work.begin(), work.begin() + 16, [](auto x) { return x == 0xa5; }));
      CHECK(std::all_of(work.end() - 16, work.end(), [](auto x) { return x == 0xa5; }));
      xpu_test::SameBytes(input.download(), before);
    }
  }
}

TEST_CASE("XPU tiled argmax: invalid geometry scratch and alias fail before writes") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int rows = 4, vocab = 8193, tile = 4096;
  Buffer input(gpu.q, DType::kF32, {rows, vocab}), ids(gpu.q, DType::kI64, {rows});
  const auto bytes = vt::XpuTiledArgmaxWorkspaceBytes(rows, vocab, tile);
  Buffer workspace(gpu.q, DType::kI8, {int64_t(bytes + 32)});
  input.put(Values(rows, vocab, 5));
  ids.upload(std::vector<uint64_t>(rows, 0x9d9d9d9d9d9d9d9dULL).data());
  workspace.upload(std::vector<unsigned char>(bytes + 32, 0xa5).data());
  const auto initial_input = input.download(), initial_ids = ids.download(), initial_work = workspace.download();
  auto scratch = workspace.tensor; scratch.shape[0] = bytes;
  const auto refuse = [&](vt::Tensor in, vt::Tensor out, vt::Tensor work, int selected_tile = tile) {
    CHECK_THROWS(vt::xpu::TiledGreedyArgmax(gpu.q, out, in, work, selected_tile));
    xpu_test::SameBytes(input.download(), initial_input);
    xpu_test::SameBytes(ids.download(), initial_ids);
    xpu_test::SameBytes(workspace.download(), initial_work);
  };
  auto bad = scratch; bad.shape[0] = bytes - 1; refuse(input.tensor, ids.tensor, bad);
  bad = scratch; bad.data = static_cast<unsigned char*>(bad.data) + 1; refuse(input.tensor, ids.tensor, bad);
  bad = scratch; bad.dtype = DType::kI32; refuse(input.tensor, ids.tensor, bad);
  bad = scratch; bad.stride[0] = 2; refuse(input.tensor, ids.tensor, bad);
  bad = scratch; bad.shape[0] = int64_t(vt::kXpuArgmaxMaxWorkspaceBytes) + 1; refuse(input.tensor, ids.tensor, bad);
  bad = scratch; bad.data = input.tensor.data; refuse(input.tensor, ids.tensor, bad);
  bad = scratch; bad.data = ids.tensor.data; refuse(input.tensor, ids.tensor, bad);
  bad = scratch; bad.data = nullptr; refuse(input.tensor, ids.tensor, bad);
  refuse(input.tensor, ids.tensor, scratch, 16384);
  bad = ids.tensor; bad.data = input.tensor.data; refuse(input.tensor, bad, scratch);
  bad = ids.tensor; bad.shape[0] = rows - 1; refuse(input.tensor, bad, scratch);
  bad = ids.tensor; bad.dtype = DType::kF32; refuse(input.tensor, bad, scratch);
  bad = input.tensor; bad.dtype = DType::kF16; refuse(bad, ids.tensor, scratch);
  bad = input.tensor; bad.stride[0] += 1; refuse(bad, ids.tensor, scratch);
  bad = input.tensor; bad.shape[0] = std::numeric_limits<int64_t>::max();
  auto huge_ids = ids.tensor; huge_ids.shape[0] = bad.shape[0]; refuse(bad, huge_ids, scratch);
  bad = input.tensor; bad.shape[0] = -1; huge_ids.shape[0] = -1; refuse(bad, huge_ids, scratch);
  bad = input.tensor; bad.device.index = 1; refuse(bad, ids.tensor, scratch);
  auto empty_input = input.tensor; empty_input.shape[0] = 0;
  auto empty_output = ids.tensor; empty_output.shape[0] = 0;
  auto empty_scratch = scratch; empty_scratch.shape[0] = 0; empty_scratch.data = nullptr;
  (void)vt::xpu::DrainProfileEvents();
  CHECK_NOTHROW(vt::xpu::TiledGreedyArgmax(gpu.q, empty_output, empty_input, empty_scratch, tile));
  CHECK(vt::xpu::DrainProfileEvents().empty());
  xpu_test::SameBytes(input.download(), initial_input);
  xpu_test::SameBytes(ids.download(), initial_ids);
  xpu_test::SameBytes(workspace.download(), initial_work);
}

TEST_CASE("XPU tiled argmax: caller-owned two-queue graphs see new data and retire before growth") {
  Queue cpu(vt::DeviceType::kCPU), first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(first.q.device);
  REQUIRE(backend.SupportsGraphCapture());
  for (int tile : {4096, 8192}) for (int vocab : {8193, 248320}) {
    CAPTURE(tile);
    CAPTURE(vocab);
    constexpr int rows = 4;
    Buffer h_in(cpu.q, DType::kF32, {rows, vocab}), h_ids(cpu.q, DType::kI64, {rows});
    Buffer a(first.q, DType::kF32, {rows, vocab}), b(second.q, DType::kF32, {rows, vocab});
    Buffer out_a(first.q, DType::kI64, {rows}), out_b(second.q, DType::kI32, {rows});
    const auto bytes = vt::XpuTiledArgmaxWorkspaceBytes(rows, vocab, tile);
    Buffer work_a(first.q, DType::kI8, {int64_t(bytes)}), work_b(second.q, DType::kI8, {int64_t(bytes)});
    a.put(Values(rows, vocab, 0)); b.put(Values(rows, vocab, 5));
    vt::BreakableGraph graphs[2];
    for (int i = 0; i < 2; ++i) {
      auto& q = i ? second.q : first.q;
      vt::GraphCaptureScope scope(backend, q, graphs[i], vt::GraphCaptureMode::kFull);
      vt::xpu::TiledGreedyArgmax(q, i ? out_b.tensor : out_a.tensor,
          i ? b.tensor : a.tensor, i ? work_b.tensor : work_a.tensor, tile);
    }
    REQUIRE(graphs[0].captured()); REQUIRE(graphs[1].captured());
    for (int mode : {5, 1, 6, 7, 2, 4, 3, 0}) {
      // Submit both independent replays before downloading either result.
      a.put(Values(rows, vocab, mode)); b.put(Values(rows, vocab, mode + 3));
      graphs[1].Replay(second.q); graphs[0].Replay(first.q);
      const auto got_a = out_a.download(), got_b = out_b.download();
      for (int i = 0; i < 2; ++i) {
        h_in.put(Values(rows, vocab, mode + (i ? 3 : 0)));
        vt::GreedyArgmax(cpu.q, h_ids.tensor, h_in.tensor);
        const auto expected = h_ids.download();
        if (!i) xpu_test::SameBytes(got_a, expected);
        else for (int r = 0; r < rows; ++r) {
          int32_t actual; int64_t target;
          std::memcpy(&actual, got_b.data() + r * sizeof(actual), sizeof(actual));
          std::memcpy(&target, expected.data() + r * sizeof(target), sizeof(target));
          CHECK(actual == target);
        }
      }
    }
    backend.Synchronize(first.q); backend.Synchronize(second.q);
    for (auto& graph : graphs) graph.Reset(); // Before buffers shrink/grow on the next geometry.
    CHECK_FALSE(graphs[0].captured()); CHECK_FALSE(graphs[1].captured());
    const auto memory = vt::xpu::GetMemoryInfo();
    (void)vt::xpu::DrainHostProfileRecords();
    for (int i = 0; i < 8; ++i)
      vt::xpu::TiledGreedyArgmax(first.q, out_a.tensor, a.tensor, work_a.tensor, tile);
    // Inspect instrumented allocation/wait records before the test's own sync.
    const auto host = vt::xpu::DrainHostProfileRecords();
    CHECK(host.empty());
    backend.Synchronize(first.q);
    const auto after = vt::xpu::GetMemoryInfo();
    CHECK(after.allocated_bytes == memory.allocated_bytes);
    CHECK(after.pinned_bytes == memory.pinned_bytes);
    CHECK(after.sampling_workspace_bytes == memory.sampling_workspace_bytes);
    CHECK(after.graph_count == memory.graph_count);
    CHECK(after.graph_device_bytes == memory.graph_device_bytes);
  }
}

TEST_CASE("XPU tiled target sampler: exact IDs raw logprob bytes ranks and bounded owner reuse") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int vocab = 248320;
  struct Restore {
    std::optional<std::string> old;
    Restore() { if (const auto* value = std::getenv("VT_XPU_TARGET_ARGMAX_TILE")) old = value; }
    ~Restore() {
      if (old) setenv("VT_XPU_TARGET_ARGMAX_TILE", old->c_str(), 1);
      else unsetenv("VT_XPU_TARGET_ARGMAX_TILE");
    }
  } restore;
  const auto initial_memory = vt::xpu::GetMemoryInfo();
  {
    vllm::v1::Sampler sampler;
    for (int rows : {1, 4, 1}) for (int lp : {-2, 0, 1}) {
      CAPTURE(rows);
      CAPTURE(lp);
      Buffer input(gpu.q, DType::kF32, {rows, vocab});
      std::vector<float> values(size_t(rows) * vocab);
      for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 43) - 21) * 0.125f;
      for (int r = 0; r < rows; ++r) values[size_t(r + 1) * vocab - 1] = 9;
      vllm::v1::SamplingMetadata sm;
      if (lp != -2) sm.max_num_logprobs = lp;
      REQUIRE(setenv("VT_XPU_TARGET_ARGMAX_TILE", "0", 1) == 0);
      input.put(values);
      const auto control = sampler.forward(gpu.q, input.tensor, sm);
      for (const char* tile : {"4096", "8192"}) {
        REQUIRE(setenv("VT_XPU_TARGET_ARGMAX_TILE", tile, 1) == 0);
        input.put(values); (void)vt::xpu::DrainProfileEvents();
        const auto selected = sampler.forward(gpu.q, input.tensor, sm);
        const auto events = vt::xpu::DrainProfileEvents();
        if (Profiling()) {
          CHECK(std::count_if(events.begin(), events.end(), [](const auto& e) { return e.stage == "greedy_argmax_tiled_partials"; }) == 1);
          CHECK(std::count_if(events.begin(), events.end(), [](const auto& e) { return e.stage == "greedy_argmax_tiled_merge"; }) == 1);
          CHECK(std::none_of(events.begin(), events.end(), [](const auto& e) { return e.stage == "greedy_argmax"; }));
        }
        CHECK(selected.sampled_token_ids == control.sampled_token_ids);
        REQUIRE(selected.logprobs_tensors.has_value() == control.logprobs_tensors.has_value());
        if (control.logprobs_tensors) {
          const auto& a = *control.logprobs_tensors;
          const auto& b = *selected.logprobs_tensors;
          CHECK(a.num_positions == b.num_positions);
          CHECK(a.num_tokens_per_position == b.num_tokens_per_position);
          CHECK(a.logprob_token_ids == b.logprob_token_ids);
          CHECK(a.selected_token_ranks == b.selected_token_ranks);
          REQUIRE(a.logprobs.size() == b.logprobs.size());
          CHECK(std::memcmp(a.logprobs.data(), b.logprobs.data(), a.logprobs.size() * sizeof(float)) == 0);
        }
      }
    }
    // Warmed repeated geometry neither grows device/pinned buffers nor graphs.
    Buffer input(gpu.q, DType::kF32, {4, vocab});
    input.put(std::vector<float>(size_t(4) * vocab, 0));
    vllm::v1::SamplingMetadata sm;
    const auto before = vt::xpu::GetMemoryInfo();
    for (int i = 0; i < 4; ++i) sampler.forward(gpu.q, input.tensor, sm);
    const auto after = vt::xpu::GetMemoryInfo();
    CHECK(after.allocated_bytes == before.allocated_bytes);
    CHECK(after.pinned_bytes == before.pinned_bytes);
    CHECK(after.graph_count == before.graph_count);
  }
  const auto retired = vt::xpu::GetMemoryInfo();
  CHECK(retired.allocated_bytes == initial_memory.allocated_bytes);
  CHECK(retired.pinned_bytes == initial_memory.pinned_bytes);
}

TEST_CASE("XPU tiled target sampler: qualified default and unchanged asynchronous device output") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int vocab = 248320;
  const char* flag = std::getenv("VT_XPU_TARGET_ARGMAX_TILE");
  const std::optional<std::string> saved = flag ? std::optional<std::string>(flag) : std::nullopt;
  struct Restore {
    std::optional<std::string> saved;
    ~Restore() {
      if (saved) setenv("VT_XPU_TARGET_ARGMAX_TILE", saved->c_str(), 1);
      else unsetenv("VT_XPU_TARGET_ARGMAX_TILE");
    }
  } restore{saved};
  REQUIRE(unsetenv("VT_XPU_TARGET_ARGMAX_TILE") == 0);
  Buffer input(gpu.q, DType::kF32, {1, vocab}), ids(gpu.q, DType::kI64, {1});
  auto values = std::vector<float>(vocab, -5); values.back() = 9;
  input.put(values);
  vllm::v1::Sampler sampler;
  vllm::v1::SamplingMetadata sm; sm.max_num_logprobs = 0;
  (void)vt::xpu::DrainProfileEvents();
  const auto selected = sampler.forward(gpu.q, input.tensor, sm);
  REQUIRE(selected.sampled_token_ids == std::vector<std::vector<int32_t>>{{vocab - 1}});
  REQUIRE(selected.logprobs_tensors.has_value());
  RouteEvents();
  // With no logprobs and caller-owned output, the original asynchronous path
  // stays allocation/wait-free and retains no new comparison scratch.
  sm.max_num_logprobs.reset();
  const auto before = vt::xpu::GetMemoryInfo();
  (void)vt::xpu::DrainHostProfileRecords(); (void)vt::xpu::DrainProfileEvents();
  const auto asynchronous = sampler.forward(gpu.q, input.tensor, sm, &ids.tensor);
  CHECK(asynchronous.sampled_on_device);
  CHECK(asynchronous.sampled_token_ids.empty());
  CHECK_FALSE(asynchronous.logprobs_tensors.has_value());
  CHECK(vt::xpu::DrainHostProfileRecords().empty());
  const auto events = vt::xpu::DrainProfileEvents();
  if (Profiling()) {
    CHECK(std::count_if(events.begin(), events.end(), [](const auto& e) { return e.stage == "greedy_argmax"; }) == 1);
    CHECK(std::none_of(events.begin(), events.end(), [](const auto& e) { return e.stage == "greedy_argmax_tiled_partials" || e.stage == "greedy_argmax_tiled_merge"; }));
  }
  const auto output = ids.download(); int64_t id;
  std::memcpy(&id, output.data(), sizeof(id)); CHECK(id == vocab - 1);
  const auto after = vt::xpu::GetMemoryInfo();
  CHECK(after.allocated_bytes == before.allocated_bytes);
  CHECK(after.pinned_bytes == before.pinned_bytes);
  CHECK(after.graph_count == before.graph_count);
}

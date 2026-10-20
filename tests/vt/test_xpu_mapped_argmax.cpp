#include "xpu_test_helpers.h"
#include "vt/xpu_argmax_plan.h"
#include "vt/xpu_sampling.h"
#include "vt/xpu.h"
#include "vt/breakable_graph.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>

namespace {
using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;
constexpr int TargetVocab = 248320;
std::vector<int32_t> Mapping(int vocab, int mode) {
  std::vector<int32_t> result(vocab);
  for (int c = 0; c < vocab; ++c)
    result[c] = (mode % 2 ? (int64_t(c) * 127 + 12345) % vocab : vocab - 1 - c) + 100;
  if (mode == 2) result.front() = result[vocab / 2] = result.back() = 5;
  if (mode == 8) result.back() = -1;
  if (mode == 9) result.back() = TargetVocab;
  if (mode == 10) result.back() = std::numeric_limits<int32_t>::max();
  if (mode == 13) result.back() = 0;
  return result;
}
std::vector<float> Logits(int rows, int vocab, int mode) {
  std::vector<float> values(size_t(rows) * vocab, -5);
  for (int r = 0; r < rows; ++r) {
    auto* row = values.data() + size_t(r) * vocab;
    row[0] = row[vocab - 1] = 9;
    switch ((mode + r) % 14) {
      case 0: break;
      case 1: std::fill(row, row + vocab, 0); break;
      case 2: row[vocab / 2] = 9; break;
      case 3: row[0] = -5; row[vocab - 1] = 10; break;
      case 4: row[vocab - 1] = std::numeric_limits<float>::quiet_NaN(); break;
      case 5: row[vocab - 1] = std::numeric_limits<float>::infinity(); break;
      case 6: row[vocab - 1] = -std::numeric_limits<float>::infinity(); break;
      case 7: row[0] = std::numeric_limits<float>::quiet_NaN(); break;
      case 8: case 9: case 10: row[vocab - 1] = -10; break;
      case 11: row[0] = -0.f; row[vocab - 1] = +0.f; break;
      case 12: std::fill(row, row + vocab, -std::numeric_limits<float>::infinity()); break;
      case 13: break;
    }
  }
  return values;
}
std::vector<int32_t> Expected(const std::vector<float>& logits, const std::vector<int32_t>& map, int rows) {
  const auto vocab = map.size();
  std::vector<int32_t> result(rows, -1);
  // Independent scalar contract: reject any invalid operand, then scan by
  // global token ID on equal scores. No tiled reduction in this oracle.
  for (int r = 0; r < rows; ++r) {
    const auto* row = logits.data() + size_t(r) * vocab;
    bool bad = false;
    for (size_t c = 0; c < vocab; ++c)
      bad |= !std::isfinite(row[c]) || map[c] < 0 || map[c] >= TargetVocab;
    if (bad) continue;
    auto chosen = size_t{0};
    for (size_t c = 1; c < vocab; ++c)
      if (row[c] > row[chosen] || (row[c] == row[chosen] && map[c] < map[chosen])) chosen = c;
    result[r] = map[chosen];
  }
  return result;
}
std::vector<unsigned char> Bytes(const std::vector<int32_t>& ids) {
  std::vector<unsigned char> bytes(ids.size() * sizeof(int32_t));
  std::memcpy(bytes.data(), ids.data(), bytes.size()); return bytes;
}
void CheckMappedEvents() {
  const auto events = vt::xpu::DrainProfileEvents();
  if (!std::getenv("VT_XPU_PROFILE") || std::string(std::getenv("VT_XPU_PROFILE")) != "1") return;
  for (const char* name : {"mapped_greedy_argmax_tiled_partials", "mapped_greedy_argmax_tiled_merge"})
    CHECK(std::count_if(events.begin(), events.end(), [&](const auto& e) { return e.stage == name; }) == 1);
}
}

TEST_CASE("XPU tiled mapped argmax: exact global IDs and invalidity in every tile") {
  Queue gpu(vt::DeviceType::kXPU);
  for (int rows : {1, 4}) for (int vocab : {1, 4095, 4096, 4097, 8191, 8192, 8193, 65535, 65536, 65537})
  for (int tile : {4096, 8192}) {
    CAPTURE(rows);
    CAPTURE(vocab);
    CAPTURE(tile);
    Buffer input(gpu.q, DType::kF32, {rows + 1, vocab}), map(gpu.q, DType::kI32, {vocab + 1});
    Buffer ids(gpu.q, DType::kI32, {rows + 8}), control(gpu.q, DType::kI32, {rows});
    const auto required = vt::XpuTiledArgmaxWorkspaceBytes(rows, vocab, tile);
    Buffer work(gpu.q, DType::kI8, {int64_t(required + 32)});
    auto in = input.tensor; in.shape[0] = rows;
    auto mapping = map.tensor; mapping.shape[0] = vocab;
    auto out = vt::Tensor::Contiguous(static_cast<int32_t*>(ids.tensor.data) + 4, DType::kI32, gpu.q.device, {rows});
    auto scratch = vt::Tensor::Contiguous(static_cast<unsigned char*>(work.tensor.data) + 16, DType::kI8, gpu.q.device, {int64_t(required)});
    for (int mode = 0; mode < 14; ++mode) {
      CAPTURE(mode);
      auto data = Logits(rows, vocab, mode); auto indices = Mapping(vocab, mode);
      const auto expected = Bytes(Expected(data, indices, rows));
      data.resize(size_t(rows + 1) * vocab, std::numeric_limits<float>::quiet_NaN());
      indices.push_back(-1); input.put(data); map.upload(indices.data());
      ids.upload(std::vector<unsigned char>((rows + 8) * 4, 0x9d).data());
      work.upload(std::vector<unsigned char>(required + 32, 0xa5).data());
      const auto before_input = input.download(), before_map = map.download();
      vt::MappedGreedyArgmax(gpu.q, control.tensor, in, mapping, TargetVocab);
      xpu_test::SameBytes(control.download(), expected);
      (void)vt::xpu::DrainProfileEvents();
      vt::xpu::TiledMappedGreedyArgmax(gpu.q, out, in, mapping, TargetVocab, scratch, tile);
      const auto result = ids.download(); CheckMappedEvents();
      xpu_test::SameBytes(std::vector<unsigned char>(result.begin() + 16, result.begin() + 16 + expected.size()), expected);
      CHECK(std::all_of(result.begin(), result.begin() + 16, [](auto x) { return x == 0x9d; }));
      CHECK(std::all_of(result.end() - 16, result.end(), [](auto x) { return x == 0x9d; }));
      const auto workspace = work.download();
      CHECK(std::all_of(workspace.begin(), workspace.begin() + 16, [](auto x) { return x == 0xa5; }));
      CHECK(std::all_of(workspace.end() - 16, workspace.end(), [](auto x) { return x == 0xa5; }));
      xpu_test::SameBytes(input.download(), before_input); xpu_test::SameBytes(map.download(), before_map);
    }
  }
}

TEST_CASE("XPU tiled mapped argmax: invalid geometry scratch and aliases leave operands untouched") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int rows = 4, vocab = 8193, tile = 4096;
  const auto bytes = vt::XpuTiledArgmaxWorkspaceBytes(rows, vocab, tile);
  Buffer input(gpu.q, DType::kF32, {rows, vocab}), map(gpu.q, DType::kI32, {vocab});
  Buffer ids(gpu.q, DType::kI32, {rows}), work(gpu.q, DType::kI8, {int64_t(bytes)});
  input.put(Logits(rows, vocab, 3)); map.upload(Mapping(vocab, 3).data());
  ids.upload(std::vector<unsigned char>(rows * 4, 0x9d).data());
  work.upload(std::vector<unsigned char>(bytes, 0xa5).data());
  const auto before_input = input.download(), before_map = map.download(), before_ids = ids.download(), before_work = work.download();
  const auto refuse = [&](vt::Tensor in, vt::Tensor mapping, vt::Tensor out, vt::Tensor scratch,
                           int64_t target = TargetVocab, int selected_tile = tile) {
    CHECK_THROWS(vt::xpu::TiledMappedGreedyArgmax(gpu.q, out, in, mapping, target, scratch, selected_tile));
    xpu_test::SameBytes(input.download(), before_input); xpu_test::SameBytes(map.download(), before_map);
    xpu_test::SameBytes(ids.download(), before_ids); xpu_test::SameBytes(work.download(), before_work);
  };
  auto bad = work.tensor; bad.shape[0] -= 1; refuse(input.tensor, map.tensor, ids.tensor, bad);
  bad = work.tensor; bad.data = static_cast<unsigned char*>(bad.data) + 1; refuse(input.tensor, map.tensor, ids.tensor, bad);
  bad = work.tensor; bad.dtype = DType::kI32; refuse(input.tensor, map.tensor, ids.tensor, bad);
  bad = work.tensor; bad.stride[0] = 2; refuse(input.tensor, map.tensor, ids.tensor, bad);
  bad = work.tensor; bad.data = nullptr; refuse(input.tensor, map.tensor, ids.tensor, bad);
  bad = work.tensor; bad.shape[0] = int64_t(vt::kXpuArgmaxMaxWorkspaceBytes) + 1; refuse(input.tensor, map.tensor, ids.tensor, bad);
  for (auto* alias : {input.tensor.data, map.tensor.data, ids.tensor.data}) {
    bad = work.tensor; bad.data = alias; refuse(input.tensor, map.tensor, ids.tensor, bad);
  }
  for (auto* alias : {input.tensor.data, map.tensor.data}) {
    bad = ids.tensor; bad.data = alias; refuse(input.tensor, map.tensor, bad, work.tensor);
  }
  bad = map.tensor; bad.shape[0] -= 1; refuse(input.tensor, bad, ids.tensor, work.tensor);
  bad = map.tensor; bad.stride[0] = 2; refuse(input.tensor, bad, ids.tensor, work.tensor);
  bad = map.tensor; bad.dtype = DType::kF32; refuse(input.tensor, bad, ids.tensor, work.tensor);
  bad = map.tensor; bad.data = static_cast<unsigned char*>(bad.data) + 1; refuse(input.tensor, bad, ids.tensor, work.tensor);
  bad = map.tensor; bad.device.index = 1; refuse(input.tensor, bad, ids.tensor, work.tensor);
  bad = input.tensor; bad.stride[0] += 1; refuse(bad, map.tensor, ids.tensor, work.tensor);
  bad = input.tensor; bad.dtype = DType::kF16; refuse(bad, map.tensor, ids.tensor, work.tensor);
  bad = ids.tensor; bad.dtype = DType::kI64; refuse(input.tensor, map.tensor, bad, work.tensor);
  auto huge = input.tensor; huge.shape[0] = std::numeric_limits<int64_t>::max();
  auto huge_ids = ids.tensor; huge_ids.shape[0] = huge.shape[0]; refuse(huge, map.tensor, huge_ids, work.tensor);
  refuse(input.tensor, map.tensor, ids.tensor, work.tensor, 0);
  refuse(input.tensor, map.tensor, ids.tensor, work.tensor, int64_t(std::numeric_limits<int32_t>::max()) + 1);
  refuse(input.tensor, map.tensor, ids.tensor, work.tensor, TargetVocab, 2048);
  auto empty = input.tensor; empty.shape[0] = 0;
  auto empty_ids = ids.tensor; empty_ids.shape[0] = 0;
  auto empty_work = work.tensor; empty_work.shape[0] = 0; empty_work.data = nullptr;
  (void)vt::xpu::DrainProfileEvents();
  CHECK_NOTHROW(vt::xpu::TiledMappedGreedyArgmax(gpu.q, empty_ids, empty, map.tensor, TargetVocab, empty_work, tile));
  CHECK(vt::xpu::DrainProfileEvents().empty());
  xpu_test::SameBytes(ids.download(), before_ids); xpu_test::SameBytes(work.download(), before_work);
}

TEST_CASE("XPU tiled mapped argmax: two queue-owned graphs see changed maps and logits") {
  Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(first.q.device);
  REQUIRE(backend.SupportsGraphCapture());
  for (int rows : {1, 4}) for (int vocab : {8193, 65536}) for (int tile : {4096, 8192}) {
    CAPTURE(rows);
    CAPTURE(vocab);
    CAPTURE(tile);
    const auto bytes = vt::XpuTiledArgmaxWorkspaceBytes(rows, vocab, tile);
    struct Packet {
      Buffer in, map, out, work;
      std::vector<unsigned char> expected, input_before, map_before;
      Packet(vt::Queue& q, int r, int v, size_t n) : in(q, DType::kF32, {r, v}),
          map(q, DType::kI32, {v}), out(q, DType::kI32, {r}), work(q, DType::kI8, {int64_t(n)}) {}
      void Update(int r, int v, int mode) {
        const auto data = Logits(r, v, mode); const auto indices = Mapping(v, mode);
        expected = Bytes(Expected(data, indices, r));
        in.put(data); map.upload(indices.data());
        input_before = in.download(); map_before = map.download();
      }
    } a(first.q, rows, vocab, bytes), b(second.q, rows, vocab, bytes);
    a.Update(rows, vocab, 0); b.Update(rows, vocab, 3);
    vt::BreakableGraph graphs[2];
    for (int i = 0; i < 2; ++i) {
      auto& q = i ? second.q : first.q; auto& p = i ? b : a;
      vt::GraphCaptureScope scope(backend, q, graphs[i], vt::GraphCaptureMode::kFull);
      vt::xpu::TiledMappedGreedyArgmax(q, p.out.tensor, p.in.tensor, p.map.tensor, TargetVocab, p.work.tensor, tile);
    }
    REQUIRE(graphs[0].captured()); REQUIRE(graphs[1].captured());
    for (int mode : {0, 3, 4, 6, 8, 9, 11, 13}) {
      a.Update(rows, vocab, mode); b.Update(rows, vocab, (mode + 2) % 14);
      graphs[1].Replay(second.q); graphs[0].Replay(first.q);
      for (auto* p : {&a, &b}) {
        xpu_test::SameBytes(p->out.download(), p->expected);
        xpu_test::SameBytes(p->in.download(), p->input_before);
        xpu_test::SameBytes(p->map.download(), p->map_before);
      }
    }
    for (auto& graph : graphs) graph.Reset(); // Retire before next geometry's allocation.
    const auto memory = vt::xpu::GetMemoryInfo();
    (void)vt::xpu::DrainHostProfileRecords();
    for (int i = 0; i < 8; ++i)
      vt::xpu::TiledMappedGreedyArgmax(first.q, a.out.tensor, a.in.tensor, a.map.tensor, TargetVocab, a.work.tensor, tile);
    CHECK(vt::xpu::DrainHostProfileRecords().empty());
    backend.Synchronize(first.q);
    const auto after = vt::xpu::GetMemoryInfo();
    CHECK(after.allocated_bytes == memory.allocated_bytes);
    CHECK(after.pinned_bytes == memory.pinned_bytes);
    CHECK(after.graph_count == memory.graph_count);
    CHECK(after.graph_device_bytes == memory.graph_device_bytes);
    CHECK(after.sampling_workspace_bytes == memory.sampling_workspace_bytes);
  }
}

TEST_CASE("XPU tiled mapped argmax: existing pooled owner retains aligned partials until ID copy") {
  Queue gpu(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(gpu.q.device);
  vllm::dense_attn::Dev device(backend, gpu.q);
  constexpr int vocab = 65536;
  for (int rows : {1, 4}) for (int tile : {4096, 8192}) {
    const auto layout = vt::XpuMappedSelectionBuffers(rows, vocab, tile);
    Buffer input(gpu.q, DType::kF32, {rows, vocab}), map(gpu.q, DType::kI32, {vocab});
    auto data = Logits(rows, vocab, 3); const auto indices = Mapping(vocab, 3);
    const auto expected = Expected(data, indices, rows);
    input.put(data); map.upload(indices.data());
    const auto round_trip = [&] {
      vllm::dense_attn::DBuf owner(device, DType::kI8, {int64_t(layout.total_bytes)});
      std::vector<unsigned char> poison(layout.total_bytes, 0xa5);
      backend.Copy(gpu.q, owner.ptr(), poison.data(), poison.size());
      auto ids = vt::Tensor::Contiguous(owner.ptr(), DType::kI32, gpu.q.device, {rows});
      auto work = vt::Tensor::Contiguous(static_cast<unsigned char*>(owner.ptr()) + layout.scratch_offset,
          DType::kI8, gpu.q.device, {int64_t(layout.scratch_bytes)});
      vt::xpu::TiledMappedGreedyArgmax(gpu.q, ids, input.tensor, map.tensor, TargetVocab, work, tile);
      std::vector<int32_t> selected(rows);
      backend.Copy(gpu.q, selected.data(), ids.data, layout.id_bytes); backend.Synchronize(gpu.q);
      CHECK(selected == expected);
      std::vector<unsigned char> full(layout.total_bytes);
      owner.Download(device, full.data());
      CHECK(std::all_of(full.begin() + layout.id_bytes, full.begin() + layout.scratch_offset,
                        [](auto x) { return x == 0xa5; }));
      // The owner returns to its original pool only after both kernels/copy.
    };
    round_trip(); round_trip();
    const auto before = vllm::Pool(backend).stats(); const auto memory = vt::xpu::GetMemoryInfo();
    for (int i = 0; i < 8; ++i) round_trip();
    const auto after = vllm::Pool(backend).stats(); const auto end = vt::xpu::GetMemoryInfo();
    CHECK(after.misses == before.misses);
    CHECK(after.live_blocks == before.live_blocks);
    CHECK(after.retained_bytes == before.retained_bytes);
    CHECK(end.allocated_bytes == memory.allocated_bytes);
    CHECK(end.pinned_bytes == memory.pinned_bytes);
  }
}

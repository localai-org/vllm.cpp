#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include <array>
#include <cstdlib>
#include <memory>

TEST_CASE("XPU graphs: two slots replay changing inputs without capture-time execution") {
  xpu_test::Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  auto& b = vt::GetBackend(first.q.device);
  REQUIRE(b.SupportsGraphCapture());
  const auto before = vt::xpu::GetMemoryInfo();
  const auto captures = b.GraphsCaptured(), replays = b.GraphReplays();
  {
    constexpr int N = 257;
    std::array<vt::Queue*, 2> queues{&first.q, &second.q};
    std::array<std::unique_ptr<xpu_test::Buffer>, 2> input, bias, sum, out;
    std::array<void*, 2> graphs{};
    for (int slot = 0; slot < 2; ++slot) {
      auto& q = *queues[slot];
      for (auto* buffers : {&input, &bias, &sum, &out})
        (*buffers)[slot] = std::make_unique<xpu_test::Buffer>(q, vt::DType::kF32, std::initializer_list<int64_t>{N});
      input[slot]->put(std::vector<float>(N, 1)); bias[slot]->put(std::vector<float>(N, .5f));
      // Warm up precisely the kernels that will be recorded.
      vt::Add(q, sum[slot]->tensor, input[slot]->tensor, bias[slot]->tensor);
      vt::MoeSiluMul(q, out[slot]->tensor, sum[slot]->tensor, bias[slot]->tensor); b.Synchronize(q);
      out[slot]->put(std::vector<float>(N, -99));
      b.BeginCapture(q);
      CHECK_THROWS(b.BeginCapture(q)); CHECK_THROWS(b.Synchronize(q));
      CHECK_THROWS(b.Alloc(64));
      CHECK_THROWS(b.AllocPinned(64));
      float ordinary_host = 7;
      CHECK_THROWS(b.Copy(q, input[slot]->tensor.data, &ordinary_host, sizeof(float)));
      vt::Add(q, sum[slot]->tensor, input[slot]->tensor, bias[slot]->tensor);
      vt::MoeSiluMul(q, out[slot]->tensor, sum[slot]->tensor, bias[slot]->tensor);
      graphs[slot] = b.EndCaptureGraph(q);
      xpu_test::Close(out[slot]->floats(), std::vector<float>(N, -99), 0);
      CHECK_THROWS(b.EndCaptureGraph(q));
    }
    const auto stable = vt::xpu::GetMemoryInfo();
    CHECK(stable.graph_count == before.graph_count + 2);
    CHECK(stable.graph_nodes == before.graph_nodes + 4);
    std::array<std::vector<float>, 2> expected;
    for (int step = 0; step < 16; ++step) {
      for (int slot = 0; slot < 2; ++slot) {
        auto values = xpu_test::Values(N, step + 3 * slot, .15f);
        expected[slot].resize(N);
        for (int j = 0; j < N; ++j) {
          const float value = values[j] + .5f;
          expected[slot][j] = .5f * value / (1 + std::exp(-value));
        }
        input[slot]->put(values);
      }
      // Both graphs are in flight before either output is gathered.
      for (int slot = 0; slot < 2; ++slot) b.ReplayGraph(*queues[slot], graphs[slot]);
      for (int slot = 0; slot < 2; ++slot) xpu_test::Close(out[slot]->floats(), expected[slot], 2e-6f);
      CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == stable.allocated_bytes);
      CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == stable.graph_device_bytes);
      CHECK(b.GraphsCaptured() == captures + 2);
    }
    CHECK(b.GraphReplays() == replays + 32);
    // A graph keeps its final submission alive through destruction.
    for (int slot = 0; slot < 2; ++slot) {
      b.ReplayGraph(*queues[1 - slot], graphs[slot]);
      b.DestroyGraph(graphs[slot]);
      CHECK_THROWS(b.ReplayGraph(*queues[slot], graphs[slot]));
    }
  }
  const auto after = vt::xpu::GetMemoryInfo();
  if (std::getenv("VT_XPU_GRAPH_PROFILE")) {
    const auto records = vt::xpu::DrainProfileEvents();
    const auto host = vt::xpu::DrainHostProfileRecords();
    REQUIRE(records.size() == 34);
    REQUIRE(host.size() == 34);
    for (const auto& record : host) {
      CHECK(record.stage == "graph_compute_submit");
      CHECK(record.end_steady_ns >= record.start_steady_ns);
    }
    for (const auto& record : records) {
      CHECK(record.stage == "graph_compute");
      CHECK(record.start_ns > 0);
      CHECK(record.end_ns >= record.start_ns);
    }
  }
  CHECK(after.graph_count == before.graph_count); CHECK(after.graph_nodes == before.graph_nodes);
  CHECK(after.allocated_bytes == before.allocated_bytes);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU graphs: live executable count is bounded before another capture begins") {
  xpu_test::Queue queue(vt::DeviceType::kXPU);
  auto& b = vt::GetBackend(queue.q.device);
  xpu_test::Buffer out(queue.q, vt::DType::kI32, {4});
  std::vector<void*> graphs;
  for (int i = 0; i < 8; ++i) {
    b.BeginCapture(queue.q); b.Memset(queue.q, out.tensor.data, 0, out.bytes);
    graphs.push_back(b.EndCaptureGraph(queue.q));
  }
  CHECK_THROWS(b.BeginCapture(queue.q));
  for (auto graph : graphs) b.DestroyGraph(graph);
  CHECK(vt::xpu::GetMemoryInfo().graph_count == 0);
  // Failed admission leaves the queue usable for ordinary eager work.
  b.Memset(queue.q, out.tensor.data, 0, out.bytes);
  CHECK(out.download() == std::vector<unsigned char>(out.bytes, 0));
}

TEST_CASE("XPU graphs: default graph replacement and queue teardown release resources") {
  auto q = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  auto& b = vt::GetBackend(q.device);
  xpu_test::Buffer out(q, vt::DType::kI32, {4});
  for (int step = 0; step < 3; ++step) {
    b.BeginCapture(q); b.Memset(q, out.tensor.data, step, out.bytes); b.EndCapture(q);
    b.Replay(q); CHECK(out.download() == std::vector<unsigned char>(out.bytes, step));
    CHECK(vt::xpu::GetMemoryInfo().graph_count == 1);
  }
  b.Replay(q); vt::DestroyQueue(q);
  CHECK(vt::xpu::GetMemoryInfo().graph_count == 0);
}

TEST_CASE("XPU graphs: changed indices are checked before gather or state mutation on every replay") {
  xpu_test::Queue queue(vt::DeviceType::kXPU); auto& q = queue.q;
  auto& b = vt::GetBackend(q.device);
  xpu_test::Buffer base(q, vt::DType::kF32, {4, 3}), out(q, vt::DType::kF32, {3, 3});
  xpu_test::Buffer cache(q, vt::DType::kF32, {4, 3}), updates(q, vt::DType::kF32, {3, 3});
  xpu_test::Buffer indices(q, vt::DType::kI32, {3});
  const auto source = xpu_test::Values(12, 5), values = xpu_test::Values(9, 9);
  base.put(source); updates.put(values); cache.put(std::vector<float>(12, 0));
  int32_t ids[3] = {2, 0, 2}; indices.upload(ids);
  vt::IndexSelect(q, out.tensor, base.tensor, indices.tensor); b.Synchronize(q);
  b.BeginCapture(q);
  vt::IndexSelect(q, out.tensor, base.tensor, indices.tensor);
  vt::IndexCopy(q, cache.tensor, updates.tensor, indices.tensor);
  void* graph = b.EndCaptureGraph(q);
  const auto captured = b.GraphsCaptured();
  std::vector<float> expected_cache(12, 0);
  for (int step = 0; step < 12; ++step) {
    for (int i = 0; i < 3; ++i) ids[i] = (step + 2 * i) % 4;
    indices.upload(ids); b.ReplayGraph(q, graph);
    std::vector<float> expected(9);
    for (int row = 0; row < 3; ++row) for (int col = 0; col < 3; ++col) {
      expected[row * 3 + col] = source[ids[row] * 3 + col];
      expected_cache[ids[row] * 3 + col] = values[row * 3 + col];
    }
    CHECK(out.floats() == expected); CHECK(cache.floats() == expected_cache);
    CHECK(b.GraphsCaptured() == captured);
  }
  const auto prior = out.download();
  ids[1] = 4; indices.upload(ids);
  CHECK_THROWS(b.ReplayGraph(q, graph));
  CHECK(out.download() == prior); CHECK(cache.floats() == expected_cache);
  ids[1] = 0; indices.upload(ids); CHECK_NOTHROW(b.ReplayGraph(q, graph));
  b.DestroyGraph(graph);
}

TEST_CASE("XPU graphs: metadata produced by the graph is refused instead of prevalidated stale") {
  xpu_test::Queue queue(vt::DeviceType::kXPU); auto& q = queue.q;
  auto& b = vt::GetBackend(q.device);
  xpu_test::Buffer base(q, vt::DType::kF32, {4, 3}), out(q, vt::DType::kF32, {3, 3});
  xpu_test::Buffer indices(q, vt::DType::kI32, {3}), replacement(q, vt::DType::kI32, {3});
  const int32_t valid[3] = {0, 1, 2}, invalid[3] = {0, 4, 2};
  indices.upload(valid); replacement.upload(invalid);
  const auto before = vt::xpu::GetMemoryInfo();
  for (int mode = 0; mode < 3; ++mode) {
    b.BeginCapture(q);
    if (mode == 0) vt::Copy(q, indices.tensor, replacement.tensor);
    if (mode == 1) b.Copy(q, indices.tensor.data, replacement.tensor.data, indices.bytes);
    if (mode == 2) b.Memset(q, indices.tensor.data, 0xff, indices.bytes);
    vt::IndexSelect(q, out.tensor, base.tensor, indices.tensor);
    CHECK_THROWS(b.EndCaptureGraph(q));
    CHECK_NOTHROW(b.Synchronize(q));
    CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == before.graph_device_bytes);
    CHECK(vt::xpu::GetMemoryInfo().pinned_bytes == before.pinned_bytes);
  }
}

TEST_CASE("XPU graphs: attention lengths change and two queues share one bounded workspace safely") {
  xpu_test::Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU), eager(vt::DeviceType::kXPU);
  auto& b = vt::GetBackend(first.q.device);
  constexpr int Blocks = 256, Page = 16, KvHeads = 4, Heads = 24, D = 256;
  xpu_test::Buffer keys(eager.q, vt::DType::kBF16, {Blocks, Page, KvHeads, D});
  xpu_test::Buffer values(eager.q, vt::DType::kBF16, {Blocks, Page, KvHeads, D});
  xpu_test::Buffer table(eager.q, vt::DType::kI32, {1, Blocks}), qsl(eager.q, vt::DType::kI32, {2});
  keys.put(xpu_test::Values(Blocks * Page * KvHeads * D, 3, .02f));
  values.put(xpu_test::Values(Blocks * Page * KvHeads * D, 7, .03f));
  std::vector<int32_t> blocks(Blocks); for (int i = 0; i < Blocks; ++i) blocks[i] = Blocks - 1 - i;
  table.upload(blocks.data()); const int32_t offsets[] = {0, 1}; qsl.upload(offsets);
  std::array<vt::Queue*, 2> queues{&first.q, &second.q};
  std::array<std::unique_ptr<xpu_test::Buffer>, 2> queries, output, reference, lengths;
  std::array<void*, 2> graphs{};
  vt::PagedAttentionArgs args; args.scale = 1.f / 16;
  auto run = [&](vt::Queue& q, int slot, vt::Tensor& out) {
    vt::PagedAttention(q, out, queries[slot]->tensor, keys.tensor, values.tensor,
                       table.tensor, lengths[slot]->tensor, qsl.tensor, args);
  };
  for (int slot = 0; slot < 2; ++slot) {
    for (auto* buffers : {&queries, &output, &reference})
      (*buffers)[slot] = std::make_unique<xpu_test::Buffer>(*queues[slot], vt::DType::kF32,
                                                           std::initializer_list<int64_t>{1, Heads, D});
    lengths[slot] = std::make_unique<xpu_test::Buffer>(*queues[slot], vt::DType::kI32, std::initializer_list<int64_t>{1});
    queries[slot]->put(xpu_test::Values(Heads * D, slot, .03f));
    int32_t length = 4096; lengths[slot]->upload(&length);
    run(*queues[slot], slot, output[slot]->tensor); // warm the shared scratch before capture
    REQUIRE(vt::xpu::GetMemoryInfo().attention_workspace_bytes == 16 * 1024 * 1024);
    b.BeginCapture(*queues[slot]); run(*queues[slot], slot, output[slot]->tensor);
    graphs[slot] = b.EndCaptureGraph(*queues[slot]);
  }
  const auto stable = vt::xpu::GetMemoryInfo(); const auto captured = b.GraphsCaptured();
  for (int step = 0; step < 16; ++step) {
    for (int slot = 0; slot < 2; ++slot) {
      int32_t length = 512 + step * 197 + 31 * slot; lengths[slot]->upload(&length);
      queries[slot]->put(xpu_test::Values(Heads * D, step + slot * 7, .03f));
      run(eager.q, slot, reference[slot]->tensor);
    }
    b.ReplayGraph(first.q, graphs[0]); b.ReplayGraph(second.q, graphs[1]);
    // Eager attention also participates in the same scratch ordering.
    run(eager.q, 0, reference[0]->tensor);
    for (int slot = 0; slot < 2; ++slot) CHECK(output[slot]->download() == reference[slot]->download());
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == stable.allocated_bytes);
    CHECK(vt::xpu::GetMemoryInfo().attention_workspace_bytes == 16 * 1024 * 1024);
    CHECK(b.GraphsCaptured() == captured);
  }
  int32_t invalid = 4097; lengths[0]->upload(&invalid);
  const auto previous = output[0]->download(); CHECK_THROWS(b.ReplayGraph(first.q, graphs[0]));
  CHECK(output[0]->download() == previous);
  for (auto graph : graphs) b.DestroyGraph(graph);
}

TEST_CASE("XPU graphs: Conv and GDN states follow changing and padded slots across two executions") {
  using xpu_test::Buffer;
  using vt::DType;
  constexpr int Batch = 4, Slots = 6, Channels = 10240, Hk = 16, Hv = 48, D = 128;
  xpu_test::Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU), eager(vt::DeviceType::kXPU);
  auto& b = vt::GetBackend(first.q.device);
  struct State {
    Buffer x, weight, conv, history, ids, query, key, value, decay, beta, out, ssm;
    explicit State(vt::Queue& q)
        : x(q, DType::kBF16, {Batch, Channels}), weight(q, DType::kBF16, {Channels, 4}),
          conv(q, DType::kBF16, {Batch, Channels}), history(q, DType::kBF16, {Slots, Channels, 3}),
          ids(q, DType::kI32, {Batch}), query(q, DType::kBF16, {Batch, Hk, D}),
          key(q, DType::kBF16, {Batch, Hk, D}), value(q, DType::kBF16, {Batch, Hv, D}),
          decay(q, DType::kF32, {Batch, Hv}), beta(q, DType::kF32, {Batch, Hv}),
          out(q, DType::kBF16, {Batch, Hv, D}), ssm(q, DType::kF32, {Slots, Hv, D, D}) {
      weight.put(xpu_test::Values(Channels * 4, 1, .04f));
      key.put(xpu_test::Values(Batch * Hk * D, 2, .03f));
      value.put(xpu_test::Values(Batch * Hv * D, 3, .04f));
      decay.put(std::vector<float>(Batch * Hv, -.13f));
      beta.put(std::vector<float>(Batch * Hv, .61f));
      reset(); stage(0);
    }
    void reset() {
      conv.put(std::vector<float>(Batch * Channels, -1));
      out.put(std::vector<float>(Batch * Hv * D, -1));
      history.put(xpu_test::Values(Slots * Channels * 3, 4, .02f));
      ssm.put(xpu_test::Values(Slots * Hv * D * D, 5, .01f));
    }
    void stage(int step) {
      int32_t indices[Batch];
      for (int row = 0; row < Batch; ++row) indices[row] = row == step % Batch ? -1 : (step + row) % Slots;
      ids.upload(indices);
      x.put(xpu_test::Values(Batch * Channels, step, .05f));
      query.put(xpu_test::Values(Batch * Hk * D, step + 2, .03f));
    }
    void run(vt::Queue& q) {
      vt::CausalConv1dUpdate(q, conv.tensor, x.tensor, weight.tensor, nullptr, history.tensor, {}, &ids.tensor);
      vt::GdnDecode(q, out.tensor, query.tensor, key.tensor, value.tensor, decay.tensor, beta.tensor,
                    ssm.tensor, {1.f / std::sqrt(float(D))}, &ids.tensor);
    }
  };
  std::array<vt::Queue*, 2> queues{&first.q, &second.q};
  std::array<std::unique_ptr<State>, 2> states, reference;
  std::array<void*, 2> graphs{};
  for (int slot = 0; slot < 2; ++slot) {
    states[slot] = std::make_unique<State>(*queues[slot]);
    reference[slot] = std::make_unique<State>(eager.q);
    states[slot]->run(*queues[slot]); b.Synchronize(*queues[slot]); states[slot]->reset();
    b.BeginCapture(*queues[slot]); states[slot]->run(*queues[slot]);
    graphs[slot] = b.EndCaptureGraph(*queues[slot]);
  }
  const auto captured = b.GraphsCaptured(); const auto stable = vt::xpu::GetMemoryInfo();
  for (int step = 0; step < 8; ++step) {
    for (int slot = 0; slot < 2; ++slot) {
      states[slot]->stage(step + slot); reference[slot]->stage(step + slot);
      reference[slot]->run(eager.q);
    }
    for (int slot = 0; slot < 2; ++slot) b.ReplayGraph(*queues[slot], graphs[slot]);
    for (int slot = 0; slot < 2; ++slot) {
      CHECK(states[slot]->conv.download() == reference[slot]->conv.download());
      CHECK(states[slot]->history.download() == reference[slot]->history.download());
      CHECK(states[slot]->out.download() == reference[slot]->out.download());
      CHECK(states[slot]->ssm.download() == reference[slot]->ssm.download());
    }
    CHECK(b.GraphsCaptured() == captured);
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == stable.allocated_bytes);
    CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == stable.graph_device_bytes);
  }
  // Every metadata check precedes both state mutations, including a later
  // operator in the same graph. An invalid slot cannot partially advance Conv.
  for (const auto& bad : {std::array<int32_t, Batch>{0, 1, 2, Slots},
                          std::array<int32_t, Batch>{0, 1, 2, 2}}) {
    states[0]->ids.upload(bad.data()); CHECK_THROWS(b.ReplayGraph(first.q, graphs[0]));
    CHECK(states[0]->history.download() == reference[0]->history.download());
    CHECK(states[0]->ssm.download() == reference[0]->ssm.download());
  }
  for (auto graph : graphs) b.DestroyGraph(graph);
}

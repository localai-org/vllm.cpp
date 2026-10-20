#include "xpu_test_helpers.h"
#include "vt/xpu_sampling.h"
#include "vt/xpu_argmax_plan.h"
#include "vt/xpu.h"
#include "vt/breakable_graph.h"
#include "vllm/v1/spec_decode/rejection_sampler.h"
#include "vllm/v1/sample/device_scratch.h"
#include <limits>
#include <cstdlib>

namespace {
using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;
constexpr float Inf = std::numeric_limits<float>::infinity();
constexpr float NaN = std::numeric_limits<float>::quiet_NaN();
std::vector<int32_t> Integers(const Buffer& b) {
  const auto bytes = b.download();
  std::vector<int32_t> result(bytes.size() / sizeof(int32_t));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}
struct Fixture {
  static constexpr int rows = 10, requests = 4, width = 4;
  vt::Queue& q;
  int vocab;
  Buffer input, draft, offsets, ids, sampled, counts, work;
  Fixture(vt::Queue& queue, int v) : q(queue), vocab(v),
    input(q, DType::kF32, {rows + 1, v}), draft(q, DType::kI32, {rows + 1}),
    offsets(q, DType::kI32, {requests + 1}), ids(q, DType::kI32, {rows + 2}),
    sampled(q, DType::kI32, {requests + 1, width}), counts(q, DType::kI32, {requests + 1}),
    work(q, DType::kI8, {int64_t(vt::XpuTiledArgmaxWorkspaceBytes(rows, v, 4096) + 32)}) {}
  vt::Tensor Input() { auto t = input.tensor; t.shape[0] = rows; return t; }
  vt::Tensor Draft() { auto t = draft.tensor; t.shape[0] = rows; return t; }
  vt::Tensor Ids() { return vt::Tensor::Contiguous(static_cast<int32_t*>(ids.tensor.data) + 1, DType::kI32, q.device, {rows}); }
  vt::Tensor Sampled() { auto t = sampled.tensor; t.shape[0] = requests; return t; }
  vt::Tensor Counts() { auto t = counts.tensor; t.shape[0] = requests; return t; }
  vt::Tensor Scratch(int tile) { return vt::Tensor::Contiguous(static_cast<unsigned char*>(work.tensor.data) + 16,
    DType::kI8, q.device, {int64_t(vt::XpuTiledArgmaxWorkspaceBytes(rows, vocab, tile))}); }
  void Load(int mode) {
    std::vector<float> values(size_t(rows + 1) * vocab, -Inf);
    // All accepted, first rejected, last rejected, and a bonus-only request.
    for (int row = 0; row < rows; ++row) {
      float* data = values.data() + row * vocab;
      const int winner = (row + mode) % vocab;
      data[winner] = 2;
      if (mode == 1) { data[winner] = -Inf; data[vocab - 1] = 9; }
      if (mode == 2) { data[0] = NaN; data[vocab - 1] = Inf; }
      if (mode == 3) { std::fill(data, data + vocab, -Inf); }
      if (mode == 4) { std::fill(data, data + vocab, 0.f); data[0] = -0.f; }
      if (mode == 5) { data[vocab - 1] = 2; if (vocab > 4096) data[4096] = 2; }
      if (mode == 6) { data[vocab - 1] = NaN; }
      if (mode == 7) { data[0] = Inf; data[vocab - 1] = Inf; }
    }
    std::fill(values.begin() + size_t(rows) * vocab, values.end(), NaN);
    input.put(values);
    const std::vector<int32_t> cu{0, 4, 6, 9, 10}; offsets.upload(cu.data());
    std::vector<int32_t> proposals{99, 0, 1, 2, 99, -1, 99, 6, -1, 99, -987};
    draft.upload(proposals.data());
    ids.upload(std::vector<int32_t>(rows + 2, -987).data());
    sampled.upload(std::vector<int32_t>((requests + 1) * width, -987).data());
    counts.upload(std::vector<int32_t>(requests + 1, -987).data());
    work.upload(std::vector<unsigned char>(work.bytes, 0xa5).data());
  }
  void Apply(bool tiled, int tile) {
    auto in = Input(), d = Draft(), id = Ids(), s = Sampled(), n = Counts(), w = Scratch(tile);
    if (tiled) vt::xpu::TiledGreedyRejectionSample(q, s, n, id, in, d, offsets.tensor, w, tile);
    else vt::GreedyRejectionSample(q, s, n, id, in, d, offsets.tensor);
  }
};
}

TEST_CASE("XPU tiled rejection: exact ragged accept replace bonus and poisoned tails") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int vocab : {17, 4095, 4096, 4097, 8193, 248320}) for (int tile : {4096, 8192}) {
    CAPTURE(vocab);
    CAPTURE(tile);
    Fixture reference(cpu.q, vocab), legacy(gpu.q, vocab), candidate(gpu.q, vocab);
    for (int mode = 0; mode < 8; ++mode) {
      CAPTURE(mode);
      reference.Load(mode); legacy.Load(mode); candidate.Load(mode);
      const auto in = candidate.input.download(), draft = candidate.draft.download(), offsets = candidate.offsets.download();
      reference.Apply(false, tile); legacy.Apply(false, tile);
      (void)vt::xpu::DrainProfileEvents();
      candidate.Apply(true, tile);
      for (auto pair : {std::make_pair(&reference.ids, &candidate.ids),
                        std::make_pair(&reference.sampled, &candidate.sampled),
                        std::make_pair(&reference.counts, &candidate.counts)})
        xpu_test::SameBytes(pair.first->download(), pair.second->download());
      xpu_test::SameBytes(legacy.ids.download(), candidate.ids.download());
      xpu_test::SameBytes(legacy.sampled.download(), candidate.sampled.download());
      xpu_test::SameBytes(legacy.counts.download(), candidate.counts.download());
      if (std::getenv("VT_XPU_PROFILE")) {
        const auto events = vt::xpu::DrainProfileEvents();
        for (const char* stage : {"greedy_argmax_tiled_partials", "greedy_argmax_tiled_merge", "greedy_rejection_accept"})
          CHECK(std::count_if(events.begin(), events.end(), [&](const auto& e) { return e.stage == stage; }) == 1);
        CHECK(std::none_of(events.begin(), events.end(), [](const auto& e) { return e.stage == "greedy_rejection_argmax"; }));
      }
      const auto scratch = candidate.work.download();
      const auto used = vt::XpuTiledArgmaxWorkspaceBytes(Fixture::rows, vocab, tile);
      CHECK(std::all_of(scratch.begin(), scratch.begin() + 16, [](auto b) { return b == 0xa5; }));
      CHECK(std::all_of(scratch.begin() + 16 + used, scratch.end(), [](auto b) { return b == 0xa5; }));
      xpu_test::SameBytes(candidate.input.download(), in);
      xpu_test::SameBytes(candidate.draft.download(), draft);
      xpu_test::SameBytes(candidate.offsets.download(), offsets);
      if (mode == 0) {
        CHECK(Integers(candidate.counts) == std::vector<int32_t>{4, 1, 2, 1, -987});
        CHECK(Integers(candidate.sampled) == std::vector<int32_t>{0, 1, 2, 3, 4, -1, -1, -1, 6, 7, -1, -1, 9, -1, -1, -1, -987, -987, -987, -987});
      }
    }
  }
}

TEST_CASE("XPU tiled rejection: host metadata aliases and live offsets fail before writes") {
  Queue gpu(vt::DeviceType::kXPU);
  Fixture f(gpu.q, 8193); f.Load(0);
  const auto input = f.input.download(), draft = f.draft.download();
  const auto ids = f.ids.download(), sampled = f.sampled.download(), counts = f.counts.download(), work = f.work.download();
  const auto unchanged = [&] {
    xpu_test::SameBytes(f.input.download(), input);
    xpu_test::SameBytes(f.draft.download(), draft);
    xpu_test::SameBytes(f.ids.download(), ids);
    xpu_test::SameBytes(f.sampled.download(), sampled);
    xpu_test::SameBytes(f.counts.download(), counts);
    xpu_test::SameBytes(f.work.download(), work);
  };
  auto in = f.Input(), d = f.Draft(), id = f.Ids(), s = f.Sampled(), n = f.Counts(), w = f.Scratch(4096), cu = f.offsets.tensor;
  const auto refuse = [&](vt::Tensor a, vt::Tensor b, vt::Tensor c, vt::Tensor e, vt::Tensor g, vt::Tensor h, vt::Tensor i, int tile = 4096) {
    CHECK_THROWS(vt::xpu::TiledGreedyRejectionSample(gpu.q, e, g, c, a, b, h, i, tile));
    unchanged();
  };
  auto bad = w; --bad.shape[0]; refuse(in, d, id, s, n, cu, bad);
  bad = w; bad.data = static_cast<unsigned char*>(w.data) + 1; refuse(in, d, id, s, n, cu, bad);
  refuse(in, d, id, s, n, cu, w, 128);
  bad = cu; bad.shape[0] = std::numeric_limits<int64_t>::max(); refuse(in, d, id, s, n, bad, w);
  bad = cu; bad.dtype = DType::kI64; refuse(in, d, id, s, n, bad, w);
  bad = d; bad.shape[0] = 9; refuse(in, bad, id, s, n, cu, w);
  bad = s; bad.shape[1] = 0; refuse(in, d, id, bad, n, cu, w);
  bad = s; bad.shape[1] = std::numeric_limits<int64_t>::max(); refuse(in, d, id, bad, n, cu, w);
  bad = n; bad.stride[0] = 2; refuse(in, d, id, s, bad, cu, w);
  bad = id; bad.dtype = DType::kI64; refuse(in, d, bad, s, n, cu, w);
  for (auto* source : {&in, &d, &cu, &id, &s, &n}) {
    bad = w; bad.data = source->data; refuse(in, d, id, s, n, cu, bad);
  }
  for (auto* source : {&in, &d, &cu, &s, &n}) {
    bad = id; bad.data = source->data; refuse(in, d, bad, s, n, cu, w);
  }
  bad = s; bad.data = cu.data; refuse(in, d, id, bad, n, cu, w);
  bad = n; bad.data = d.data; refuse(in, d, id, s, bad, cu, w);
  for (const auto& invalid : {std::vector<int32_t>{1,4,6,9,10}, {0,4,4,9,10},
                              {0,4,6,9,11}, {0,5,6,9,10}, {0,4,3,9,10}}) {
    f.offsets.upload(invalid.data());
    refuse(in, d, id, s, n, cu, w);
  }
  // A genuinely empty verification remains a no-op with no queued work.
  in.shape[0] = d.shape[0] = id.shape[0] = s.shape[0] = n.shape[0] = w.shape[0] = 0;
  cu.shape[0] = 1;
  (void)vt::xpu::DrainProfileEvents();
  CHECK_NOTHROW(vt::xpu::TiledGreedyRejectionSample(gpu.q, s, n, id, in, d, cu, w, 4096));
  CHECK(vt::xpu::DrainProfileEvents().empty());
  unchanged();
}

TEST_CASE("XPU tiled rejection: independent graph queues see changed rows drafts and offsets") {
  Queue cpu(vt::DeviceType::kCPU), first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(first.q.device);
  REQUIRE(backend.SupportsGraphCapture());
  for (int tile : {4096,8192}) {
    Fixture reference(cpu.q, 8193), a(first.q, 8193), b(second.q, 8193);
    a.Load(0); b.Load(1);
    vt::BreakableGraph graphs[2];
    for (int i = 0; i < 2; ++i) {
      auto& f = i ? b : a;
      f.Apply(true, tile); backend.Synchronize(f.q);
      vt::GraphCaptureScope capture(backend, f.q, graphs[i], vt::GraphCaptureMode::kFull);
      f.Apply(true, tile);
    }
    REQUIRE(graphs[0].captured()); REQUIRE(graphs[1].captured());
    for (int mode : {0,1,2,3,5,7}) {
      a.Load(mode); b.Load((mode + 1) % 8);
      const std::vector<int32_t> offsets{0,1,3,6,10};
      if (mode % 2) { a.offsets.upload(offsets.data()); b.offsets.upload(offsets.data()); }
      graphs[1].Replay(second.q); graphs[0].Replay(first.q);
      for (int i = 0; i < 2; ++i) {
        auto& f = i ? b : a;
        reference.Load(i ? (mode + 1) % 8 : mode);
        if (mode % 2) reference.offsets.upload(offsets.data());
        reference.Apply(false, tile);
        xpu_test::SameBytes(reference.ids.download(), f.ids.download());
        xpu_test::SameBytes(reference.sampled.download(), f.sampled.download());
        xpu_test::SameBytes(reference.counts.download(), f.counts.download());
      }
    }
    // Validation must consume changed live offsets before any graph output write.
    a.Load(0);
    const auto ids = a.ids.download(), sampled = a.sampled.download(), counts = a.counts.download(), work = a.work.download();
    const std::vector<int32_t> bad{0,4,4,9,10}; a.offsets.upload(bad.data());
    CHECK_THROWS(graphs[0].Replay(first.q));
    xpu_test::SameBytes(a.ids.download(), ids);
    xpu_test::SameBytes(a.sampled.download(), sampled);
    xpu_test::SameBytes(a.counts.download(), counts);
    xpu_test::SameBytes(a.work.download(), work);
    backend.Synchronize(first.q); backend.Synchronize(second.q);
    for (auto& graph : graphs) graph.Reset();
  }
}

TEST_CASE("XPU tiled rejection: verifier owns partials across moves and later larger calls") {
  Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  struct Restore {
    bool had = std::getenv("VT_XPU_REJECTION_ARGMAX_TILE") != nullptr;
    std::string old = had ? std::getenv("VT_XPU_REJECTION_ARGMAX_TILE") : "";
    ~Restore() { if (had) setenv("VT_XPU_REJECTION_ARGMAX_TILE", old.c_str(), 1); else unsetenv("VT_XPU_REJECTION_ARGMAX_TILE"); }
  } restore;
  for (int tile : {4096,8192}) {
    setenv("VT_XPU_REJECTION_ARGMAX_TILE", std::to_string(tile).c_str(), 1);
    Buffer small(first.q, DType::kF32, {1,248320}), large(first.q, DType::kF32, {4,248320}), other(second.q, DType::kF32, {4,248320});
    std::vector<float> a(248320, -1), b(4*248320, -1);
    a.back() = 9;
    for (int r = 0; r < 4; ++r) b[size_t(r)*248320 + r] = 7;
    small.put(a); large.put(b); other.put(b);
    vllm::v1::RejectionSampler sampler(3);
    (void)vt::xpu::DrainProfileEvents();
    auto first_owner = sampler.verify(first.q, small.tensor, {99}, {0,1});
    auto large_owner = sampler.verify(first.q, large.tensor, {99,0,1,2}, {0,4});
    auto independent = sampler.verify(second.q, other.tensor, {99,0,-1,2}, {0,4});
    // A later geometry must never resize/free a process-global earlier window.
    CHECK(first_owner.target_argmax_scratch() != large_owner.target_argmax_scratch());
    CHECK(large_owner.target_argmax_scratch() != independent.target_argmax_scratch());
    auto moved = std::move(first_owner);
    first_owner = std::move(large_owner);
    std::vector<int32_t> small_out(4,-987), large_out(4,-987), independent_out(4,-987), n(1), m(1), k(1);
    moved.CopyToHost(first.q, small_out.data(), n.data());
    first_owner.CopyToHost(first.q, large_out.data(), m.data());
    independent.CopyToHost(second.q, independent_out.data(), k.data());
    // Destroy before an explicit caller wait: existing owner drains its own
    // verification/copy consumers before freeing IDs and their partial tail.
    moved = {}; first_owner = {}; independent = {};
    CHECK(small_out == std::vector<int32_t>{248319,-1,-1,-1});
    CHECK(large_out == std::vector<int32_t>{0,1,2,3});
    CHECK(independent_out == std::vector<int32_t>{0,1,-1,-1});
    CHECK(n == std::vector<int32_t>{1});
    CHECK(m == std::vector<int32_t>{4});
    CHECK(k == std::vector<int32_t>{2});
    if (std::getenv("VT_XPU_PROFILE")) {
      const auto events = vt::xpu::DrainProfileEvents();
      for (const char* stage : {"greedy_argmax_tiled_partials","greedy_argmax_tiled_merge","greedy_rejection_accept"})
        CHECK(std::count_if(events.begin(), events.end(), [&](const auto& e) { return e.stage == stage; }) == 3);
      CHECK(std::none_of(events.begin(), events.end(), [](const auto& e) { return e.stage == "greedy_rejection_argmax"; }));
    }
  }
}

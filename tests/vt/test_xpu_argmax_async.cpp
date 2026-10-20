#include "xpu_test_helpers.h"
#include "vt/xpu_argmax_plan.h"
#include "vt/xpu_sampling.h"
#include "vllm/v1/spec_decode/rejection_sampler.h"
#include "vllm/v1/sample/device_scratch.h"
#include <sycl/sycl.hpp>
#include <future>
#include <cstdlib>
#include <string>

// Test-only sequencing through the existing native queue. No vendor type is
// added to the public runtime interface or to the production selection path.
namespace vt::xpu { sycl::queue& NativeQueue(Queue&); }
namespace {
struct Gate {
  std::promise<void> entered, release;
  std::shared_future<void> released = release.get_future().share();
  bool open = false;
  void Open() { if (!open) { release.set_value(); open = true; } }
  ~Gate() { Open(); }
};
}

TEST_CASE("XPU tiled argmax: submits behind unfinished work without a host fence") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto& q = gpu.q;
  auto& backend = vt::GetBackend(q.device);
  for (bool mapped : {false, true}) for (int rows : {1, 4}) for (int tile : {0, 4096, 8192})
  for (bool inject_wait : {false, true}) {
    if (inject_wait && tile != 4096) continue;
    CAPTURE(mapped);
    CAPTURE(rows);
    CAPTURE(tile);
    CAPTURE(inject_wait);
    const int vocab = mapped ? 65536 : 248320;
    xpu_test::Buffer input(q, vt::DType::kF32, {rows, vocab}),
        ids(q, mapped ? vt::DType::kI32 : vt::DType::kI64, {rows}), map(q, vt::DType::kI32, {vocab});
    xpu_test::Buffer scratch(q, vt::DType::kI8,
        {int64_t(vt::XpuTiledArgmaxWorkspaceBytes(rows, vocab, tile ? tile : 4096))});
    std::vector<float> data(size_t(rows) * vocab, -1);
    for (int r = 0; r < rows; ++r) data[size_t(r + 1) * vocab - 1] = 9;
    input.put(data);
    std::vector<int32_t> mapping(vocab);
    for (int c = 0; c < vocab; ++c) mapping[c] = vocab - 1 - c;
    map.upload(mapping.data());
    const auto apply = [&] {
      if (mapped) {
        if (tile) vt::xpu::TiledMappedGreedyArgmax(q, ids.tensor, input.tensor, map.tensor, 248320, scratch.tensor, tile);
        else vt::MappedGreedyArgmax(q, ids.tensor, input.tensor, map.tensor, 248320);
      } else if (tile) vt::xpu::TiledGreedyArgmax(q, ids.tensor, input.tensor, scratch.tensor, tile);
      else vt::GreedyArgmax(q, ids.tensor, input.tensor);
    };
    // Warm compilation before testing only the submission boundary.
    for (int i = 0; i < 8; ++i) apply();
    backend.Synchronize(q);
    Gate gate;
    auto entered = gate.entered.get_future();
    vt::xpu::NativeQueue(q).submit([&](sycl::handler& h) {
      h.host_task([&] { gate.entered.set_value(); gate.released.wait(); });
    });
    const auto started = entered.wait_for(std::chrono::seconds(2));
    if (started != std::future_status::ready) {
      gate.Open(); backend.Synchronize(q);
      FAIL("host gate did not start: no nonblocking conclusion");
    }
    auto call = std::async(std::launch::async, [&] {
      apply();
      if (inject_wait) backend.Synchronize(q); // Detecting negative control.
    });
    const bool returned = call.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    // Always release before any assertion or future destruction, even on fail.
    gate.Open(); call.get(); backend.Synchronize(q);
    CHECK(returned == !inject_wait);
    const auto output = ids.download();
    for (int r = 0; r < rows; ++r) {
      int64_t id;
      if (mapped) { int32_t narrow; std::memcpy(&narrow, output.data() + r * sizeof(narrow), sizeof(narrow)); id = narrow; }
      else std::memcpy(&id, output.data() + r * sizeof(id), sizeof(id));
      CHECK(id == (mapped ? 0 : vocab - 1));
    }
  }
}

TEST_CASE("XPU tiled rejection: owner drains an explicitly delayed final copy") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto& q = gpu.q;
  struct Restore {
    bool had = std::getenv("VT_XPU_REJECTION_ARGMAX_TILE") != nullptr;
    std::string old = had ? std::getenv("VT_XPU_REJECTION_ARGMAX_TILE") : "";
    ~Restore() { if (had) setenv("VT_XPU_REJECTION_ARGMAX_TILE",old.c_str(),1); else unsetenv("VT_XPU_REJECTION_ARGMAX_TILE"); }
  } restore;
  for (int tile : {0,4096,8192}) {
    setenv("VT_XPU_REJECTION_ARGMAX_TILE",std::to_string(tile).c_str(),1);
    xpu_test::Buffer input(q,vt::DType::kF32,{4,248320});
    std::vector<float> data(4*248320,-1);
    for (int row=0; row<4; ++row) data[size_t(row)*248320+row]=7;
    input.put(data);
    vllm::v1::RejectionSampler sampler(3);
    auto owner = sampler.verify(q,input.tensor,{99,0,1,2},{0,4});
    struct Pinned {
      vt::Backend& backend;
      int32_t* data;
      explicit Pinned(vt::Backend& b) : backend(b), data(static_cast<int32_t*>(b.AllocPinned(5*sizeof(int32_t)))) {}
      ~Pinned() { backend.FreePinned(data); }
    } pinned(vt::GetBackend(q.device));
    std::fill(pinned.data,pinned.data+5,-987);
    Gate gate; auto entered = gate.entered.get_future();
    vt::xpu::NativeQueue(q).submit([&](sycl::handler& h) {
      h.host_task([&] { gate.entered.set_value(); gate.released.wait(); });
    });
    if (entered.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
      gate.Open(); vt::GetBackend(q.device).Synchronize(q);
      FAIL("delayed-copy gate did not start");
    }
    // Pageable D2H intentionally waits in this backend. Persistent USM-host
    // endpoints exercise the actual asynchronous copy/owner boundary instead.
    owner.CopyToHost(q,pinned.data,pinned.data+4);
    auto release = std::async(std::launch::async,[owner=std::move(owner)]() mutable { owner = {}; });
    const bool drained_early = release.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready;
    gate.Open(); release.get();
    CHECK_FALSE(drained_early);
    CHECK(std::vector<int32_t>(pinned.data,pinned.data+4) == std::vector<int32_t>{0,1,2,3});
    CHECK(pinned.data[4] == 4);
  }
}

TEST_CASE("XPU compact logprobs0: submits behind unfinished work without a host fence") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto& q=gpu.q;auto& backend=vt::GetBackend(q.device);
  for(int rows:{1,4}) for(bool forced:{false,true}) {
    xpu_test::Buffer lp(q,vt::DType::kF32,{rows,248320}), ids(q,vt::DType::kI64,{rows}),
      output(q,vt::DType::kI8,{int64_t(rows*sizeof(vt::xpu::SelectedLogprobRecord))}),
      work(q,vt::DType::kI8,{int64_t(vt::XpuCompactLogprobsWorkspaceBytes(rows,248320))});
    lp.put(std::vector<float>(size_t(rows)*248320,-100));ids.upload(std::vector<int64_t>(rows,248319).data());
    const auto apply=[&]{vt::xpu::CompactRawLogprobs0(q,output.tensor,lp.tensor,ids.tensor,work.tensor);};
    for(int i=0;i<8;++i)apply();backend.Synchronize(q);
    Gate gate;auto entered=gate.entered.get_future();
    vt::xpu::NativeQueue(q).submit([&](sycl::handler& h){h.host_task([&]{gate.entered.set_value();gate.released.wait();});});
    if(entered.wait_for(std::chrono::seconds(2))!=std::future_status::ready){gate.Open();backend.Synchronize(q);FAIL("compact helper predecessor gate did not start");}
    auto call=std::async(std::launch::async,[&]{apply();if(forced)backend.Synchronize(q);});
    const bool ready=call.wait_for(std::chrono::seconds(1))==std::future_status::ready;
    gate.Open();call.get();backend.Synchronize(q);
    CHECK(ready==!forced);
    const auto raw=output.download();
    for(int row=0;row<rows;++row){
      vt::xpu::SelectedLogprobRecord record;
      std::memcpy(&record,raw.data()+row*sizeof(record),sizeof(record));
      CHECK(record.token_id==248319);CHECK(record.logprob==-100);CHECK(record.rank==248320);
    }
  }
}

#include "exl3_smallm_test_helpers.h"
#include "vt/xpu.h"
#include "vt/xpu_graph_metadata.h"
#include <iostream>

using namespace exl3_smallm_test;

TEST_CASE("XPU EXL3 SmallM P7: immutable owner survives context teardown") {
  Queue gpu(vt::DeviceType::kXPU); auto& backend = vt::GetBackend(gpu.q.device);
  REQUIRE(backend.SupportsGraphCapture());
  auto owner = std::shared_ptr<void>(vt::Alloc(gpu.q.device, 64),
      [device = gpu.q.device](void* p) { vt::Free(device, p); });
  std::weak_ptr<void> lifetime = owner;
  // Destination is context-owned USM, deliberately live until context teardown.
  // No freed input/output storage is left referenced by this test graph.
  void* destination = vt::Alloc(gpu.q.device, 64);
  backend.Memset(gpu.q, owner.get(), 0x35, 64); backend.Synchronize(gpu.q);
  backend.BeginCapture(gpu.q);
  vt::xpu::RecordGraphImmutableRead(gpu.q, owner.get(), 64, owner, "teardown source map");
  backend.Copy(gpu.q, destination, owner.get(), 64);
  (void)backend.EndCaptureGraph(gpu.q); owner.reset(); CHECK_FALSE(lifetime.expired());
  // The actual worker exit is part of this sentinel: static context teardown
  // destroys its graph before releasing the final owner and the destination.
  std::cout << "P7_SMALLM_MAP_CONTEXT_TEARDOWN pending_at_worker_exit=1" << std::endl;
}

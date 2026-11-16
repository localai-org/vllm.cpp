// vllm.cpp original (BACKEND-TENSTORRENT W0); no upstream mirror.
#include "vt/tenstorrent/tenstorrent_device.h"

#include <cstdlib>
#include <cstring>
#include <memory>

#include <ttnn/device.hpp>
#include <tt-metalium/host_api.hpp>

namespace vt::tenstorrent {

bool DeviceAvailable() { return tt::tt_metal::GetNumAvailableDevices() > 0; }

namespace {
// ITEM 5 trace-region policy: 0 (dynamic) unless VT_TT_TRACE_REGION_MB opts
// into a fixed region; an unusable value falls back to 50 MB, the vLLM
// plugin's generic fixed size.
constexpr size_t kFallbackTraceRegionBytes = 50 * 1024 * 1024;

size_t TraceRegionSizeFromEnv() {
  const char* mb = std::getenv("VT_TT_TRACE_REGION_MB");
  if (mb == nullptr || *mb == '\0') return 0;  // DYNAMIC
  char* end = nullptr;
  const long parsed = std::strtol(mb, &end, 10);
  if (end == mb || *end != '\0' || parsed <= 0) return kFallbackTraceRegionBytes;
  return static_cast<size_t>(parsed) * 1024 * 1024;
}
}  // namespace

MeshDevice& SharedMeshDevice() {
  // DELIBERATE LEAK, not an oversight: a plain `static std::shared_ptr`
  // reproducibly SEGFAULTS on process exit (tests/vt/test_tenstorrent_backend.cpp,
  // 2026-08-09) inside MeshDevice's own destructor chain
  // (GraphTracker::track_deallocate_cb <- ProgramImpl::deallocate_circular_buffers
  // <- ~MeshWorkloadImpl), i.e. C++ static destruction order across the
  // libtt_metal.so boundary tears down some tt_metal-internal global this
  // teardown path depends on before our function-local static's destructor
  // runs. The standalone spike this backend is modeled on (spec's "Resolved:
  // hands-on spike result") did NOT hit this: it held the device in a
  // `main()`-local variable, destroyed deterministically before any static
  // teardown begins, not during it. Allocating on the heap and never
  // `delete`-ing means MeshDevice's destructor never runs at process exit at
  // all, sidestepping the ordering problem entirely — safe in practice (the
  // OS/kernel driver reclaim the PCIe device's file descriptors and hardware
  // state on process exit regardless of a userspace close() call, the same
  // assumption CUDA processes routinely rely on), if not textbook-clean.
  // ITEM 5 (amended, wave-3b-1c): open the device with a DYNAMIC trace
  // region (`trace_region_size=0`), mirroring the pinned tt-metal's own
  // practice (models/demos/utils/trace_region_sizes.py): unconfigured
  // (model, SKU) pairs resolve to TRACE_REGION_SIZE_DYNAMIC (0) — the
  // runtime then carves trace buffers from the general DRAM pool at
  // capture time — and deepseek-v3 sets 0 explicitly. Our previous fixed
  // 50 MB was the vLLM plugin's generic value (its worker.py:710), and it
  // sized the region as a constant rather than from demand: the chunked
  // E=1 keep-quant arm's captured command stream alone measures ~0.44 GB
  // on the 0.8B vehicle (444,424,192 B vs the 52,428,800 B region,
  // mesh_trace.cpp:81), a per-chain cost no fixed constant tracks. The old
  // reason for a fixed region was the 2026-08-09 capture fatal "Trace
  // buffer ... overlaps with DRAM activity" on the then-current runtime;
  // that hazard still exists in dynamic mode (mesh_trace.cpp:113 detects
  // it), so VT_TT_TRACE_REGION_MB=<MB> opts back into a fixed region —
  // the fallback's default is 50, the ITEM 5 generic value — when a
  // capture overlaps live DRAM allocations.
  static std::shared_ptr<MeshDevice>* device = new std::shared_ptr<MeshDevice>(
      ttnn::open_mesh_device(
          /*device_id=*/0, /*l1_small_size=*/DEFAULT_L1_SMALL_SIZE,
          /*trace_region_size=*/TraceRegionSizeFromEnv()));
  return **device;
}

void ProgramCacheToggleForTest(bool enable) {
  MeshDevice& device = SharedMeshDevice();
  if (enable) {
    device.enable_program_cache();
  } else {
    device.disable_and_clear_program_cache();
  }
}

}  // namespace vt::tenstorrent

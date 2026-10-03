// Tenstorrent backend — the `vt::Backend` implementation + its static
// registrar. BACKEND-TENSTORRENT, W0 skeleton
// (.agents/specs/tenstorrent-backend.md). vllm.cpp original: vLLM has no
// Tenstorrent platform anywhere, so there is no upstream mirror; the SHAPE is
// the CPU reference `src/vt/cpu/cpu_backend.cpp` (the 6 pure virtuals).
//
// SCOPE / STUBS — stated plainly so nothing here reads as more than it is:
//   * Blackhole is a DISCRETE PCIe device (no shared host/device address
//     space), unlike Vulkan's W0 here (unified on its GB10 target, so its
//     Alloc returns a directly host-dereferenceable mapped pointer).
//     vt::Tensor.data for this backend is still a HOST pointer (aligned_alloc)
//     so host-staged ops (PA/RoPE) and weight load via Backend::Copy keep
//     working. Device-resident ttnn::Tensor shadows are keyed by that host
//     pointer in tenstorrent_ops.cpp (RegisterHostBuffer / EnsureDevice /
//     CommitDevice) so the matmul/norm/silu chain can skip host
//     download+reupload between ops. UnifiedMemory() remains false — the real
//     hardware property.
//   * `SupportsGraphCapture()` is TRUE: maps onto ttnn mesh-trace capture
//     (begin_trace_capture / end_trace_capture / execute_trace) via free
//     functions in tenstorrent_ops.cpp. Capture still requires fixed device
//     buffers and a warmed program cache — same class of contract as CUDA
//     graphs (see BeginCapture notes on the CUDA backend).
//   * `UnifiedMemory()` is `false`: this is the real hardware property (a
//     discrete card over PCIe), independent of this W0's host-staging
//     implementation detail above. It also means op_provider.h's portable CPU
//     reference tier stays gated off for this device — no free correctness
//     net for an unregistered op.
#include "vt/backend.h"
#include "vt/tenstorrent/tenstorrent_device.h"

#include <cstdlib>
#include <cstring>

namespace vt::tenstorrent {
namespace {

class TenstorrentBackend final : public Backend {
 public:
  void* Alloc(size_t bytes) override {
    VT_CHECK(bytes <= SIZE_MAX - 63, "tenstorrent alloc size overflow");
    const size_t n = ((bytes + 63) / 64) * 64;
    void* p = std::aligned_alloc(64, n);
    VT_CHECK(p != nullptr, "tenstorrent alloc failed");
    RegisterHostBuffer(p, n);
    return p;
  }
  void Free(void* p) override {
    if (p == nullptr) return;
    UnregisterHostBuffer(p);
    std::free(p);
  }
  // DevicePool::Get hands a retained block to a NEW tensor without passing
  // through Alloc, so the slot at that address still describes the previous
  // tenant (device shadow committed, host stale). W7 (#2282): register the
  // acquisition as a RESERVATION — the block's bytes are garbage for the new
  // tenant on BOTH sides, so its first device use is served without staging
  // the garbage; the pending op establishes the content by writing the buffer
  // it is served (#1715).
  void OnScratchBlockAcquired(void* p) override { MarkScratchAcquired(p); }
  void Memset(Queue&, void* p, int value, size_t bytes) override {
    // HOST-FREE-FORWARD R3: on-device zero-fill when capturing.
    if (MemsetDeviceIfCapture(p, value, bytes)) return;
    // W7 (#2282): an eager FULL-slot zero of a device-resident slot fills the
    // shadow and keeps it — the next device read serves the shadow instead of
    // restaging the zeros. The host bytes are still memset here.
    if (MemsetDeviceFill(p, value, bytes)) {
      std::memset(p, value, bytes);
      return;
    }
    std::memset(p, value, bytes);
    MarkHostWritten(p);
  }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    // HOST-FREE-FORWARD R2: when capturing, prefer a device->device copy so the
    // captured region has no host readback (which ttnn trace prohibits).
    if (CopyDeviceDeviceIfCapture(dst, src, bytes)) return;
    // W7 (#2282): outside capture, a device-resident source copied into a
    // slot that already owns a persistent buffer goes device->device — the
    // host path would download, memcpy, drop the shadow, and re-upload the
    // same bytes on the next device read.
    if (CopyDeviceDeviceIfResident(dst, src, bytes)) return;
    // Device-resident results leave host stale until read; materialize first.
    EnsureHostBytes(const_cast<void*>(src));
    std::memcpy(dst, src, bytes);
    MarkHostWritten(dst);
  }
  Queue CreateQueue() override { return Queue{Device{DeviceType::kTENSTORRENT, 0}, nullptr}; }

  // Discrete PCIe card — never host-dereference device pages. Host pointer +
  // optional device shadow (ops TU) is the residency model; the CPU reference
  // tier stays gated off.
  bool UnifiedMemory() const override { return false; }

  // ttnn mesh-trace capture — see Trace* in tenstorrent_device.h / ops.cpp.
  bool SupportsGraphCapture() const override { return true; }

  // Production mamba_cache_dtype is bf16 (qwen3_5_common.cpp conv_dtype
  // default), so the GDN decode conv-update addresses a bf16 conv_state. The
  // TT conv kernel computes through its f32 transposed shadow and honors that
  // STORAGE semantics at the host boundary — LoadElemF32 widens on the way in,
  // StoreElemF32 narrows on the way out, and (the bf16-state arm's test in
  // tests/vt/test_tenstorrent_backend.cpp) the shadow itself re-rounds
  // through bf16 on every commit, mirroring CUDA's "read/written in f32
  // registers" (cuda_backend.cu:119, cuda_gdn.cu's conv kernels). This states
  // as a CAPABILITY what the shared CheckConvCommon gate asks, the same
  // device-agnostic query CUDA/ROCm/Vulkan answer.
  bool SupportsCompressedConvState() const override { return true; }

  // The GDN decode recurrence passes the persistent ssm_state (production
  // mamba_ssm_dtype = bf16) straight to kGdnDecode. The TT kernel computes in
  // its f32 shadow but honors bf16 STORAGE semantics — the committed shadow
  // re-rounds through bf16 every step (mirroring CUDA's "read/written in f32
  // registers", cuda_backend.cu:123 / rocm_backend.hip:353), pinned by the
  // bf16-state arm of the kGdnDecode oracle test.
  bool SupportsCompressedGdnState() const override { return true; }
  void BeginCapture(Queue&) override { TraceBeginCapture(); }
  void EndCapture(Queue&) override { TraceEndCapture(); }
  void Replay(Queue&) override { TraceReplay(); }
  void* EndCaptureGraph(Queue&) override { return TraceEndCaptureGraph(); }
  void ReplayGraph(Queue&, void* graph) override { TraceReplayGraph(graph); }
  void DestroyGraph(void* graph) override { TraceDestroyGraph(graph); }
};

struct Registrar {
  Registrar() noexcept {
    // Same runtime-probe-before-registering shape as Vulkan's
    // VulkanDeviceAvailable() gate: a Tenstorrent-enabled build on a host
    // with no Blackhole card registers nothing instead of throwing during
    // static init (unspecified TU order rules out trusting another TU's
    // initializer to have probed already).
    if (!DeviceAvailable()) return;
    static TenstorrentBackend backend;
    RegisterBackend(DeviceType::kTENSTORRENT, &backend);
  }
} registrar;

}  // namespace
}  // namespace vt::tenstorrent

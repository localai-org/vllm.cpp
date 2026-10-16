// Native SYCL/Level Zero resource queries. No vendor types cross this interface.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vt::xpu {
struct MemoryInfo {
  size_t total_bytes = 0;       // device capacity, never process allocation count
  size_t budget_bytes = 0;      // process allocation ceiling, <= total
  size_t allocated_bytes = 0;   // live device USM owned by this backend
  size_t pinned_bytes = 0;      // host USM; excluded from device usage
  size_t free_bytes = 0;        // live driver report, valid only if free_known
  bool free_known = false;
  size_t exl3_workspace_bytes = 0;  // included in allocated_bytes, shared by all queues
  size_t gdn_workspace_bytes = 0;   // included in allocated_bytes, chunk-64 scratch
  size_t native_gdn_workspace_bytes = 0; // included in allocated_bytes, Xe2 A/W/U scratch
  size_t sampling_workspace_bytes = 0; // included in allocated_bytes, top-k/top-p scratch
  size_t attention_workspace_bytes = 0; // included in allocated_bytes, split-KV scratch
  size_t peak_allocated_bytes = 0; // backend-tracked device high-water mark (excludes driver allocations)
  size_t graph_count = 0, graph_nodes = 0;
  size_t graph_device_bytes = 0; // validation buffers plus SYCL-reported graph memory; excludes driver command lists
  size_t w8a8_workspace_bytes = 0; // included in allocated_bytes; eager shared data+panel pool
  size_t w8a8_preparation_bytes = 0; // included in allocated_bytes; private checked INT8 rows/scales
  size_t fp16_silu_table_bytes = 0; // included in allocated_bytes; immutable context-owned table
  size_t gated_silu_table_bytes = 0; // F16 input / F32 result, included in allocated_bytes
};
int DeviceCount() noexcept;
MemoryInfo GetMemoryInfo(int index = 0);
std::string DeviceDescription(int index = 0);
// Eager SYCL submission timestamps. Drain only at a request boundary; collection
// is disabled unless VT_XPU_PROFILE=1 was set before queue creation.
struct ProfileRecord {
  std::string stage;
  std::string matrix;
  uint64_t queue_id = 0;
  // On the pinned Level Zero stack, submit and device start can straddle
  // different clock calibrations. Do not infer a delay from their raw difference.
  uint64_t submit_ns = 0, start_ns = 0, end_ns = 0;
  // A stream span brackets a library call with two commands on the same
  // in-order queue; it can include gaps between its internal kernels.
  bool stream_span = false;
  uint64_t host_submit_ns = 0;
};
// Brackets one profiled device command between two steady-clock reads. Device
// and host clocks have different epochs on the pinned stack; use the bracket
// to align intervals and report its width as calibration uncertainty.
struct ProfileClockAnchor {
  uint64_t host_before_ns = 0, host_after_ns = 0;
  uint64_t device_start_ns = 0, device_end_ns = 0;
};
struct HostProfileRecord {
  std::string stage;
  uint64_t queue_id = 0;
  uint64_t start_steady_ns = 0, end_steady_ns = 0;
  // Staged-copy waits carry the bytes in this chunk and the calling PC.
  // Other spans leave both zero. Resolve PCs against the captured process maps.
  uint64_t copy_bytes = 0, caller_address = 0;
};
ProfileClockAnchor CaptureProfileClockAnchor(int index = 0);
std::vector<ProfileRecord> DrainProfileEvents(int index = 0);
// VT_XPU_HOST_PROFILE=1 records eager waits/allocation calls without enabling
// SYCL event profiling. Nested spans (for example metadata readback and its
// staged D2H copy) overlap and must not be summed as disjoint time.
std::vector<HostProfileRecord> DrainHostProfileRecords(int index = 0);
size_t PendingProfileEventCount(int index = 0);
// Host-only labels copied into XPU profile records during a synchronous
// submission. Nested projection labels preserve their caller's layer label.
class ProfileLayerScope {
 public:
  explicit ProfileLayerScope(int64_t layer) noexcept;
  ~ProfileLayerScope() noexcept;
  ProfileLayerScope(const ProfileLayerScope&) = delete;
  ProfileLayerScope& operator=(const ProfileLayerScope&) = delete;
 private:
  int64_t previous_;
};
class ProfileMatrixScope {
 public:
  explicit ProfileMatrixScope(const char* matrix) noexcept;
  ~ProfileMatrixScope() noexcept;
  ProfileMatrixScope(const ProfileMatrixScope&) = delete;
  ProfileMatrixScope& operator=(const ProfileMatrixScope&) = delete;
 private:
  const char* previous_;
};
}  // namespace vt::xpu

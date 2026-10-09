#pragma once
#include <sycl/sycl.hpp>
#include "vt/backend.h"
#include "vt/tensor.h"
#include "vt/ops.h"
#include "vt/xpu.h"
#include <algorithm>
#include <initializer_list>
#include <functional>
#include <string>

namespace vt::xpu {
// All kernels use the queue's owning context. No ambient/default SYCL queue.
sycl::queue& NativeQueue(Queue& q);
void RecordProfileEvent(Queue& q, const char* stage, const sycl::event& event);
bool ProfileQueueEventsEnabled();
bool HostProfileSpansEnabled();
uint64_t HostProfileClockNs();
void RecordHostProfileSpan(Queue& q, const char* stage,
                           uint64_t start_ns, uint64_t end_ns);
// Bracket all commands submitted by a library call on this in-order queue.
// The resulting duration is a stream span, including submission gaps, rather
// than the duration of one kernel or of the two marker commands.
void RecordProfileSpan(Queue& q, const char* stage, const sycl::event& begin,
                       const sycl::event& end, uint64_t host_submit_ns,
                       const std::string& detail);
// One persistent, serialized EXL3 workspace per device context. The callback's
// GPU work is completed before another queue can reuse it. False means budget
// was insufficient; callers can retain their native non-panel path.
bool WithExl3Workspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch);
// Eager-only growable W8A8 workspace plus panel, accounted once per device.
// Completes every consumer before reuse/growth or another queue's lease.
bool WithExl3W8A8Workspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch);
// Separate eager-only private preparation, at most64 MiB per context. False
// means the caller should retain its original checked preparation route.
bool WithExl3W8A8Preparation(Queue& q, size_t bytes, const std::function<void(void*)>& launch);
bool WithGdnWorkspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch);
bool WithGdnNativeWorkspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch);
bool WithAttentionWorkspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch);
bool WithSamplingWorkspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch);
// Immutable finite-domain SiLU table, owned/accounted by this device context.
// Initialization/readiness stays on the GPU; capture must be warmed on its
// queue. Readers may overlap and the context drains all queues before freeing.
bool WithFp16SiluTable(Queue& q,
    const std::function<sycl::event(void*)>& initialize,
    const std::function<void(const void*)>& launch);
bool WithGatedSiluTable(Queue& q,
    const std::function<sycl::event(void*)>& initialize,
    const std::function<void(const void*)>& launch);
// Captured metadata checks run in a separate graph before the mutating graph.
// False means eager execution; the caller performs its ordinary checked readback.
bool CaptureMetadataCheck(Queue& q, const std::function<void(sycl::handler&, int*)>& submit, const char* message,
                          std::initializer_list<const Tensor*> inputs);
void RecordGraphWrite(Queue& q, const void* data, size_t bytes);
void TraceXpuOp(OpId op, Queue& q, std::initializer_list<const Tensor*> tensors);

// Trivially copyable kernel argument; never capture Tensor's optional metadata.
struct View {
  void* data;
  DType dtype;
  int rank;
  int64_t shape[kMaxRank], stride[kMaxRank];
  explicit View(const Tensor& t) : data(t.data), dtype(t.dtype), rank(t.rank) {
    for (int i = 0; i < kMaxRank; ++i) { shape[i] = t.shape[i]; stride[i] = t.stride[i]; }
  }
  int64_t offset(int64_t linear) const {
    int64_t result = 0;
    for (int d = rank - 1; d >= 0; --d) { result += (linear % shape[d]) * stride[d]; linear /= shape[d]; }
    return result;
  }
};
inline uint16_t Bf16(float f) {
  uint32_t u = sycl::bit_cast<uint32_t>(f);
  if ((u & 0x7f800000) == 0x7f800000 && (u & 0x7fffff)) return (u >> 16) | 0x40;
  return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
inline float Load(View t, int64_t offset) {
  if (t.dtype == DType::kF32) return static_cast<const float*>(t.data)[offset];
  if (t.dtype == DType::kF16) return static_cast<float>(static_cast<const sycl::half*>(t.data)[offset]);
  return sycl::bit_cast<float>(uint32_t(static_cast<const uint16_t*>(t.data)[offset]) << 16);
}
inline void Store(View t, int64_t offset, float f) {
  if (t.dtype == DType::kF32) static_cast<float*>(t.data)[offset] = f;
  else if (t.dtype == DType::kF16) static_cast<sycl::half*>(t.data)[offset] = sycl::half(f);
  else static_cast<uint16_t*>(t.data)[offset] = Bf16(f);
}
inline float Round(DType d, float f) {
  if (d == DType::kF16) return static_cast<float>(sycl::half(f));
  if (d == DType::kBF16) return sycl::bit_cast<float>(uint32_t(Bf16(f)) << 16);
  return f;
}
inline void FloatTensor(const Tensor& t) {
  VT_CHECK(t.dtype == DType::kF32 || t.dtype == DType::kF16 || t.dtype == DType::kBF16,
           "XPU operator requires F16/BF16/F32 storage");
}
inline size_t Span(const Tensor& t) {
  if (t.Numel() == 0) return 0;
  size_t elements = 1;
  for (int i = 0; i < t.rank; ++i) {
    VT_CHECK(t.shape[i] >= 0 && t.stride[i] >= 0, "XPU requires nonnegative shape/stride");
    VT_CHECK(t.shape[i] <= 1 || static_cast<uint64_t>(t.stride[i]) <=
                 (SIZE_MAX - elements) / static_cast<uint64_t>(t.shape[i] - 1), "XPU span overflow");
    elements += (t.shape[i] - 1) * t.stride[i];
  }
  VT_CHECK(elements <= SIZE_MAX / SizeOf(t.dtype), "XPU byte span overflow");
  return elements * SizeOf(t.dtype);
}
inline bool Overlap(const Tensor& a, const Tensor& b) {
  const auto ap = reinterpret_cast<uintptr_t>(a.data), bp = reinterpret_cast<uintptr_t>(b.data);
  const auto as = Span(a), bs = Span(b);
  return as && bs && (ap <= bp ? bp - ap < as : ap - bp < bs);
}
class Scratch {
  Device device_;
 public:
  void* data;
  Scratch(Device d, size_t bytes) : device_(d), data(vt::Alloc(d, std::max(bytes, size_t{1}))) {}
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() { try { vt::Free(device_, data); } catch (...) { /* backend retains failed-wait allocations */ } }
};
template<class Check>
void CheckDeviceMetadata(Queue& q, Check check, const char* message,
                          std::initializer_list<const Tensor*> inputs) {
  const auto check_start = HostProfileSpansEnabled() ? HostProfileClockNs() : 0;
  if (CaptureMetadataCheck(q, [check](sycl::handler& h, int* result) {
        h.single_task([=] { *result = check() ? 1 : 0; });
      }, message, inputs)) {
    if (check_start) {
      const auto check_end = HostProfileClockNs();
      RecordHostProfileSpan(q, "metadata_validation_host", check_start,
                            check_end);
      RecordHostProfileSpan(q, message, check_start, check_end);
    }
    return;
  }
  Scratch scratch(q.device, sizeof(int));
  auto* result = static_cast<int*>(scratch.data);
  NativeQueue(q).single_task([=] { *result = check() ? 1 : 0; });
  int valid = 0;
  auto& backend = GetBackend(q.device);
  const auto wait_start = HostProfileSpansEnabled() ? HostProfileClockNs() : 0;
  backend.Copy(q, &valid, result, sizeof(valid));
  backend.Synchronize(q);
  if (wait_start)
    RecordHostProfileSpan(q, "metadata_d2h_wait", wait_start,
                          HostProfileClockNs());
  if (check_start) {
    const auto check_end = HostProfileClockNs();
    RecordHostProfileSpan(q, "metadata_validation_host", check_start,
                          check_end);
    RecordHostProfileSpan(q, message, check_start, check_end);
  }
  VT_CHECK(valid, message);
}
// The correctness path snapshots only when the output could clobber an input.
// Temporary releases drain their final kernel/copy use through Backend::Free.
template<class Launch>
void WithOutput(Queue& q, Tensor& out, std::initializer_list<const Tensor*> inputs, Launch launch,
                bool preserve = false) {
  bool alias = false;
  for (const auto* in : inputs) if (in && Overlap(out, *in)) alias = true;
  if (!alias) { launch(out); return; }
  VT_CHECK(static_cast<uint64_t>(out.Numel()) <= SIZE_MAX / SizeOf(out.dtype), "XPU scratch size overflow");
  Scratch scratch(q.device, static_cast<size_t>(out.Numel()) * SizeOf(out.dtype));
  Tensor temp = out;
  temp.data = scratch.data;
  int64_t stride = 1;
  for (int d = temp.rank - 1; d >= 0; --d) { temp.stride[d] = stride; stride *= temp.shape[d]; }
  if (preserve) vt::Copy(q, temp, out);
  launch(temp);
  vt::Copy(q, out, temp);
  // Surface asynchronous failures before the nonthrowing cleanup.
  GetBackend(q.device).Synchronize(q);
}
}  // namespace vt::xpu

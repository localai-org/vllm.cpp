// Initial correctness backend: in-order queues, device USM, conservative frees.
#include "xpu_common.h"
#ifdef VLLM_CPP_XPU_ONEDNN
#include "xpu_gptq4.h"
#endif
#include "vt/xpu.h"
#include "vt/xpu_graph_metadata.h"
#include "vt/xpu_profile_span.h"
#include "vt/gdn_fp16_plan.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <nlohmann/json.hpp>

namespace vt::xpu {
namespace {
namespace graph_api = sycl::ext::oneapi::experimental;
using RecordingGraph = graph_api::command_graph<graph_api::graph_state::modifiable>;
using ExecutableGraph = graph_api::command_graph<graph_api::graph_state::executable>;
constexpr size_t MaxGraphs = 8, MaxGraphNodes = 65536, MaxGraphDeviceBytes = 64 * 1024 * 1024;
constexpr size_t MaxGraphChecks = 1024, GraphCheckBytes = MaxGraphChecks * sizeof(int);
struct GraphChecks {
  sycl::context context;
  int* device = nullptr;
  int* host = nullptr;
  std::vector<std::string> messages;
  GraphChecks(const sycl::context& c, const sycl::device& d) : context(c) {
    device = static_cast<int*>(sycl::aligned_alloc_device(64, GraphCheckBytes, d, c));
    try {
      VT_CHECK(device != nullptr, "XPU graph validation allocation failed");
      host = static_cast<int*>(sycl::aligned_alloc_host(64, GraphCheckBytes, c));
      VT_CHECK(host != nullptr, "XPU graph validation readback allocation failed");
    } catch (...) { if (device) sycl::free(device, context); throw; }
  }
  ~GraphChecks() { if (device) sycl::free(device, context); if (host) sycl::free(host, context); }
};
struct Recording {
  struct ImmutableRead {
    uintptr_t start, end;
    std::shared_ptr<void> owner;
    std::string message;
  };
  // Declared first so owners outlive the recorded command-graph objects.
  std::vector<ImmutableRead> immutable_reads;
  RecordingGraph compute, validation;
  std::unique_ptr<GraphChecks> checks;
  struct Span { uintptr_t start, end; size_t check = 0; };
  std::vector<Span> writes, metadata;
  unsigned workspace_mask = 0;
  Recording(const sycl::context& c, const sycl::device& d)
      : compute(c, d), validation(c, d), checks(std::make_unique<GraphChecks>(c, d)) {}
};
struct Graph {
  // Also keep allocations alive through executable destruction at shutdown.
  std::vector<std::shared_ptr<void>> immutable_owners;
  ExecutableGraph executable;
  std::optional<ExecutableGraph> validation;
  std::unique_ptr<GraphChecks> checks;
  size_t nodes, bytes;
  unsigned workspace_mask;
  std::optional<sycl::event> last;
};
struct Devices {
  std::vector<sycl::device> gpu;
  std::string error;
  Devices() noexcept {
    try {
      for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        // OpenCL exposes the same card again. One Level Zero device = one context.
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpu.push_back(d);
      }
    } catch (const std::exception& e) { error = e.what(); }
  }
};
Devices& AllDevices() { static Devices d; return d; }
const sycl::device& DeviceAt(int index) {
  const auto& all = AllDevices();
  VT_CHECK(index >= 0 && static_cast<size_t>(index) < all.gpu.size(),
           "XPU: no Level Zero GPU at index " + std::to_string(index) +
           (all.error.empty() ? " (CPU fallback is disabled)" : ": " + all.error));
  return all.gpu[index];
}
size_t Budget(size_t total) {
  const char* env = std::getenv("VT_XPU_MEMORY_BUDGET_BYTES");
  if (env == nullptr) return total;
  char* end = nullptr;
  errno = 0;
  const auto bytes = std::strtoull(env, &end, 10);
  VT_CHECK(*env >= '0' && *env <= '9' && end != env && *end == '\0' && errno == 0
               && bytes > 0 && bytes <= total,
           "VT_XPU_MEMORY_BUDGET_BYTES must be positive and no larger than device memory");
  return static_cast<size_t>(bytes);
}
bool ProfileQueuesEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VT_XPU_PROFILE");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}
bool GraphProfileEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VT_XPU_GRAPH_PROFILE");
    return value && std::strcmp(value, "1") == 0;
  }();
  VT_CHECK(!enabled || ProfileQueuesEnabled(), "XPU graph profiling requires VT_XPU_PROFILE=1");
  return enabled;
}
bool HostProfileEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VT_XPU_HOST_PROFILE");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}
thread_local const char* current_profile_matrix = nullptr;
thread_local int64_t current_profile_layer = -1;
struct ProfileAnchorKernel { void operator()() const {} };
struct Workspace {
  std::mutex mutex;
  void* data = nullptr;
  size_t bytes = 0;
  std::optional<sycl::event> last;
};
struct ImmutableTable {
  std::mutex mutex;
  void* data = nullptr;
  std::optional<sycl::event> ready;
  std::unordered_set<sycl::queue*> joined;
};
struct PendingProfileEvent {
  const char* stage;
  std::string matrix;
  uint64_t queue_id;
  sycl::event event;
  std::optional<sycl::event> begin;
  uint64_t host_submit_ns = 0;
};
struct Context {
  sycl::device device;
  sycl::context context;
  std::mutex mutex;
  // Captures must not start between unlocking mutex and releasing the last
  // retired owner (whose Free deleter re-enters mutex). Replay is unaffected.
  std::mutex graph_owner_lifetime;
  Workspace exl3, gdn, attention, sampling, native_gdn, w8a8, w8a8_preparation;
  ImmutableTable fp16_silu, gated_silu;
  std::unordered_map<sycl::queue*, std::unique_ptr<sycl::queue>> queues;
  std::vector<PendingProfileEvent> profile_events;
  std::vector<HostProfileRecord> host_profile_records;
  std::unordered_map<sycl::queue*, std::unique_ptr<Recording>> recordings;
  std::unordered_map<void*, std::unique_ptr<Graph>> graphs;
  // Free is forbidden while any queue records. A retired immutable map can
  // therefore release its last owner only after the last capture has ended.
  std::vector<std::shared_ptr<void>> retired_immutable_owners;
  std::unordered_map<sycl::queue*, void*> default_graphs;
  size_t graph_nodes = 0, graph_bytes = 0;
  int64_t captures = 0, replays = 0;
  std::unordered_map<void*, size_t> allocations, pinned;
  size_t total, budget, allocated = 0, pinned_bytes = 0, peak_allocated = 0;
  explicit Context(int index) : device(DeviceAt(index)), context(device),
      total(device.get_info<sycl::info::device::global_mem_size>()), budget(Budget(total)) {
    VT_CHECK(device.has(sycl::aspect::usm_device_allocations)
                 && device.has(sycl::aspect::usm_host_allocations),
             "XPU requires device and host USM allocations");
  }
  // Callers hold mutex. Waiting every owned queue also covers kernel submissions,
  // which do not carry allocation lifetime metadata in VT's current ABI.
  void Drain() { for (auto& [_, q] : queues) q->wait_and_throw(); }
  ~Context() {
    for (auto& [_, graph] : recordings) {
      try { graph->compute.end_recording(); } catch (...) {}
    }
    // Owner deleters can call Free/GetContext. Empty the live maps before
    // releasing their references, with the queues/context still available.
    auto retired_recordings = std::move(recordings); recordings.clear();
    try { Drain(); } catch (const std::exception& e) {
      std::fprintf(stderr, "[vt xpu] shutdown wait failed: %s\n", e.what());
    }
    auto retired_graphs = std::move(graphs); graphs.clear();
    retired_recordings.clear(); retired_graphs.clear();
    retired_immutable_owners.clear();
    for (const auto& [p, _] : allocations) sycl::free(p, context);
    for (const auto& [p, _] : pinned) sycl::free(p, context);
  }
};
void AppendProfileEvent(Context& c, Queue& q, const char* stage,
                        const char* matrix, const sycl::event& event) {
  constexpr size_t kMaxProfileEvents = 1000000;
  VT_CHECK(c.profile_events.size() < kMaxProfileEvents,
           "XPU profile event limit exceeded; narrow or drain the diagnostic window");
  c.profile_events.push_back({stage, matrix ? matrix : "", q.id, event,
                              std::nullopt, 0});
}
uint64_t SteadyNs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
void AppendHostProfileRecord(Context& c, uint64_t queue_id, const char* stage,
                             uint64_t start, uint64_t end,
                             uint64_t copy_bytes = 0, uint64_t caller = 0) {
  constexpr size_t kMaxHostProfileRecords = 100000;
  VT_CHECK(c.host_profile_records.size() < kMaxHostProfileRecords,
           "XPU host profile record limit exceeded; narrow or drain the diagnostic window");
  c.host_profile_records.push_back({stage, queue_id, start, end, copy_bytes, caller});
}
Context& GetContext(int index) {
  // Stable per-device storage; construction is lazy and failures reach the caller.
  static std::mutex mutex;
  static std::unique_ptr<Context> contexts[kMaxDevicesPerType];
  VT_CHECK(index >= 0 && static_cast<size_t>(index) < kMaxDevicesPerType, "XPU device index out of range");
  std::lock_guard<std::mutex> lock(mutex);
  if (!contexts[index]) contexts[index] = std::make_unique<Context>(index);
  return *contexts[index];
}
struct EventState {
  int index;
  sycl::event event;
  bool recorded = false;
};
EventState& CheckEvent(Event& e, int index) {
  VT_CHECK(e.device == (Device{DeviceType::kXPU, index}) && e.handle != nullptr,
           "XPU event owner mismatch or destroyed event");
  auto& state = *static_cast<EventState*>(e.handle);
  VT_CHECK(state.index == index, "XPU event context mismatch");
  return state;
}
class XpuBackend final : public Backend {
  int index_;
  Context& ctx() const { return GetContext(index_); }
  sycl::queue& queue(Queue& q) const {
    VT_CHECK(q.device == (Device{DeviceType::kXPU, index_}), "XPU queue owner mismatch");
    return NativeQueue(q);
  }
 public:
  explicit XpuBackend(int index) : index_(index) {}
  void* Alloc(size_t bytes) override {
    auto& c = ctx();
    bytes = std::max(bytes, size_t{1});
    std::lock_guard<std::mutex> lock(c.mutex);
    VT_CHECK(c.recordings.empty(), "XPU graph capture requires preallocated device buffers");
    VT_CHECK(bytes <= c.budget - c.allocated - c.graph_bytes, "XPU device allocation exceeds memory budget");
    const auto alloc_start = HostProfileEnabled() ? SteadyNs() : 0;
    void* p = sycl::aligned_alloc_device(64, bytes, c.device, c.context);
    VT_CHECK(p != nullptr, "XPU device USM allocation failed");
    if (alloc_start)
      AppendHostProfileRecord(c, 0, "alloc_device", alloc_start, SteadyNs());
    try { c.allocations.emplace(p, bytes); } catch (...) { sycl::free(p, c.context); throw; }
    c.allocated += bytes;
    c.peak_allocated = std::max(c.peak_allocated, c.allocated);
    return p;
  }
  void Free(void* p) override {
    if (p == nullptr) return;
    auto& c = ctx();
    std::lock_guard<std::mutex> lock(c.mutex);
    VT_CHECK(c.recordings.empty(), "XPU graph capture cannot free device buffers");
    const auto it = c.allocations.find(p);
    VT_CHECK(it != c.allocations.end(), "XPU free: pointer not owned by this device");
    const auto drain_start = HostProfileEnabled() ? SteadyNs() : 0;
    c.Drain();
    if (drain_start)
      AppendHostProfileRecord(c, 0, "free_device_drain", drain_start, SteadyNs());
    sycl::free(p, c.context);
    c.allocated -= it->second;
    c.allocations.erase(it);
  }
  void* AllocPinned(size_t bytes) override {
    auto& c = ctx();
    bytes = std::max(bytes, size_t{1});
    std::lock_guard<std::mutex> lock(c.mutex);
    VT_CHECK(c.recordings.empty(), "XPU graph capture requires preallocated pinned buffers");
    const auto alloc_start = HostProfileEnabled() ? SteadyNs() : 0;
    void* p = sycl::aligned_alloc_host(64, bytes, c.context);
    VT_CHECK(p != nullptr, "XPU pinned host USM allocation failed");
    if (alloc_start)
      AppendHostProfileRecord(c, 0, "alloc_pinned", alloc_start, SteadyNs());
    try { c.pinned.emplace(p, bytes); } catch (...) { sycl::free(p, c.context); throw; }
    c.pinned_bytes += bytes;
    return p;
  }
  void FreePinned(void* p) override {
    if (p == nullptr) return;
    auto& c = ctx();
    std::lock_guard<std::mutex> lock(c.mutex);
    VT_CHECK(c.recordings.empty(), "XPU graph capture cannot free pinned buffers");
    const auto it = c.pinned.find(p);
    VT_CHECK(it != c.pinned.end(), "XPU pinned free: pointer not owned by this device");
    const auto drain_start = HostProfileEnabled() ? SteadyNs() : 0;
    c.Drain();
    if (drain_start)
      AppendHostProfileRecord(c, 0, "free_pinned_drain", drain_start, SteadyNs());
    sycl::free(p, c.context);
    c.pinned_bytes -= it->second;
    c.pinned.erase(it);
  }
  void Memset(Queue& q, void* p, int value, size_t bytes) override {
    RecordGraphWrite(q, p, bytes);
    auto& native = queue(q);
    if (bytes) native.memset(p, value, bytes);
  }
  void Copy(Queue& q, void* dst, const void* src, size_t bytes) override {
    RecordGraphWrite(q, dst, bytes);
    auto& native = queue(q);
    if (!bytes) return;
    const auto context = native.get_context();
    const bool host_src = sycl::get_pointer_type(src, context) == sycl::usm::alloc::unknown;
    const bool host_dst = sycl::get_pointer_type(dst, context) == sycl::usm::alloc::unknown;
    if (!host_src && !host_dst) { native.memcpy(dst, src, bytes); return; }
    {
      auto& c = ctx(); std::lock_guard lock(c.mutex);
      VT_CHECK(!c.recordings.count(&native), "XPU graph capture requires persistent USM copy endpoints");
    }
    if (host_src && host_dst) {
      const auto wait_start = HostProfileEnabled() ? SteadyNs() : 0;
      native.wait_and_throw();
      if (wait_start)
        RecordHostProfileSpan(q, "copy_host_wait", wait_start, SteadyNs());
      std::memcpy(dst, src, bytes);
      return;
    }
    const bool host_profile = HostProfileEnabled();
    const auto caller = host_profile ? reinterpret_cast<uintptr_t>(
        __builtin_extract_return_addr(__builtin_return_address(0))) : 0;
    const auto record_wait = [&](const char* stage, uint64_t start,
                                 uint64_t end, size_t count) {
      if (!start) return;
      auto& c = ctx();
      std::lock_guard lock(c.mutex);
      AppendHostProfileRecord(c, q.id, stage, start, end, count, caller);
    };
    // Ordinary host pointers include read-only, unaligned safetensors mmaps.
    // Level Zero's direct import of those mappings can fault in the copy engine.
    // Only known USM pointers reach DMA; bound staging independently of weights.
    const size_t chunk = std::min(bytes, size_t{4 * 1024 * 1024});
    void* staging = AllocPinned(chunk);
    try {
      for (size_t offset = 0; offset < bytes; offset += chunk) {
        const size_t count = std::min(chunk, bytes - offset);
        if (host_src) {
          std::memcpy(staging, static_cast<const char*>(src) + offset, count);
          const auto wait_start = host_profile ? SteadyNs() : 0;
          auto copy = native.memcpy(static_cast<char*>(dst) + offset, staging, count);
          copy.wait_and_throw();
          if (wait_start)
            record_wait("staged_h2d_wait", wait_start, SteadyNs(), count);
          RecordProfileEvent(q, "staged_h2d_copy", copy);
        } else {
          const auto wait_start = host_profile ? SteadyNs() : 0;
          auto copy = native.memcpy(staging, static_cast<const char*>(src) + offset, count);
          copy.wait_and_throw();
          if (wait_start)
            record_wait("staged_d2h_wait", wait_start, SteadyNs(), count);
          RecordProfileEvent(q, "staged_d2h_copy", copy);
          std::memcpy(static_cast<char*>(dst) + offset, staging, count);
        }
      }
    } catch (...) {
      try { FreePinned(staging); } catch (...) { /* retained until context shutdown */ }
      throw;
    }
    FreePinned(staging);
  }
  Queue CreateQueue() override {
    auto& c = ctx();
    const auto errors = [](sycl::exception_list errors) {
      for (auto e : errors) std::rethrow_exception(e);
    };
    auto q = ProfileQueuesEnabled()
        ? std::make_unique<sycl::queue>(c.context, c.device, errors,
              sycl::property_list{sycl::property::queue::in_order{},
                                  sycl::property::queue::enable_profiling{}})
        : std::make_unique<sycl::queue>(c.context, c.device, errors,
              sycl::property_list{sycl::property::queue::in_order{}});
    auto* handle = q.get();
    std::lock_guard<std::mutex> lock(c.mutex);
    c.queues.emplace(handle, std::move(q));
    return Queue{Device{DeviceType::kXPU, index_}, handle};
  }
  void DestroyQueue(Queue& q) override {
    if (q.handle == nullptr) return;
    auto* native = &queue(q);
    void* default_graph = nullptr;
    {
      auto& c = ctx(); std::lock_guard lock(c.mutex);
      VT_CHECK(!c.recordings.count(native), "XPU queue is still recording a graph");
      if (auto it = c.default_graphs.find(native); it != c.default_graphs.end()) default_graph = it->second;
    }
    queue(q).wait_and_throw();
#ifdef VLLM_CPP_XPU_ONEDNN
    ReleaseGptq4Queue(q);
#endif
    if (default_graph) DestroyGraph(default_graph);
    auto& c = ctx();
    std::scoped_lock table_lock(c.fp16_silu.mutex, c.gated_silu.mutex);
    std::lock_guard<std::mutex> lock(c.mutex);
    c.fp16_silu.joined.erase(native);
    c.gated_silu.joined.erase(native);
    c.queues.erase(static_cast<sycl::queue*>(q.handle));
    q.handle = nullptr;
  }
  void Synchronize(Queue& q) override {
    auto& native = queue(q);
    { auto& c = ctx(); std::lock_guard lock(c.mutex);
      VT_CHECK(!c.recordings.count(&native), "XPU graph capture cannot synchronize its recording queue"); }
    native.wait_and_throw();
  }
  bool SupportsGraphCapture() const override {
    return ctx().device.has(sycl::aspect::ext_oneapi_graph) ||
           ctx().device.has(sycl::aspect::ext_oneapi_limited_graph);
  }
  bool SupportsCompressedConvState() const override { return true; }
  void BeginCapture(Queue& q) override {
    auto& native = queue(q); auto& c = ctx();
    std::lock_guard lifetime_lock(c.graph_owner_lifetime);
    std::lock_guard lock(c.mutex);
    VT_CHECK(SupportsGraphCapture(), "XPU device does not support SYCL command graphs");
    VT_CHECK(!c.recordings.count(&native), "XPU queue is already recording a graph");
    VT_CHECK(c.graphs.size() + c.recordings.size() < MaxGraphs, "XPU live graph limit exceeded");
    // A pre-existing owned graph may be retired by another queue during this
    // capture, even if this recording itself never reads immutable metadata.
    for (const auto& [_, graph] : c.graphs) if (!graph->immutable_owners.empty()) {
      c.retired_immutable_owners.reserve(MaxGraphNodes); break;
    }
    VT_CHECK(GraphCheckBytes <= c.budget - c.allocated - c.graph_bytes &&
             GraphCheckBytes <= MaxGraphDeviceBytes - c.graph_bytes, "XPU graph validation exceeds memory budget");
    native.wait_and_throw();
    auto [it, inserted] = c.recordings.emplace(&native, std::make_unique<Recording>(c.context, c.device));
    (void)inserted;
    try { it->second->compute.begin_recording(native); }
    catch (...) { c.recordings.erase(it); throw; }
    c.graph_bytes += GraphCheckBytes; c.pinned_bytes += GraphCheckBytes;
  }
  void* EndCaptureGraph(Queue& q) override {
    auto& native = queue(q); auto& c = ctx();
    std::lock_guard lifetime_lock(c.graph_owner_lifetime);
    // Last owner references must be released after the context lock, including
    // a rejected capture. Their deleters may re-enter the allocation backend.
    std::unique_ptr<Recording> recording;
    std::unique_ptr<Graph> graph;
    std::vector<std::shared_ptr<void>> immutable_owners;
    std::vector<std::shared_ptr<void>> retired_owners;
    std::lock_guard lock(c.mutex);
    auto it = c.recordings.find(&native);
    VT_CHECK(it != c.recordings.end(), "XPU queue has no active graph capture");
    recording = std::move(it->second); c.recordings.erase(it);
    if (c.recordings.empty() && !c.retired_immutable_owners.empty())
      retired_owners.swap(c.retired_immutable_owners);
    try {
      recording->compute.end_recording(native);
      // Preflight is valid only for externally staged metadata. A graph that
      // produces or overwrites its own indices must use an eager boundary; it
      // cannot validate yesterday's indices and then execute on today's ones.
      for (const auto& metadata : recording->metadata)
        for (const auto& write : recording->writes)
          VT_CHECK(metadata.start >= write.end || write.start >= metadata.end,
                   std::string("XPU graph metadata must be staged outside capture and remain read-only during replay: ") +
                   recording->checks->messages[metadata.check]);
      for (const auto& read : recording->immutable_reads) {
        for (const auto& write : recording->writes)
          VT_CHECK(read.start >= write.end || write.start >= read.end,
                   std::string("XPU graph immutable metadata must remain read-only: ") + read.message);
        immutable_owners.push_back(read.owner);
      }
      const size_t nodes = recording->compute.get_nodes().size() + recording->validation.get_nodes().size();
      VT_CHECK(nodes <= MaxGraphNodes - c.graph_nodes, "XPU graph node budget exceeded");
      const bool graph_profile = GraphProfileEnabled();
      const sycl::property_list profile_properties{graph_api::property::graph::enable_profiling{}};
      auto executable = graph_profile ? recording->compute.finalize(profile_properties)
                                      : recording->compute.finalize();
      std::optional<ExecutableGraph> validation;
      if (!recording->checks->messages.empty())
        validation = graph_profile ? recording->validation.finalize(profile_properties)
                                   : recording->validation.finalize();
      const size_t bytes = executable.get_required_mem_size() + (validation ? validation->get_required_mem_size() : 0);
      VT_CHECK(bytes <= MaxGraphDeviceBytes - c.graph_bytes && bytes <= c.budget - c.allocated - c.graph_bytes,
               "XPU graph device-memory budget exceeded");
      graph = std::make_unique<Graph>(Graph{std::move(immutable_owners), std::move(executable),
          std::move(validation), std::move(recording->checks), nodes,
          bytes + GraphCheckBytes, recording->workspace_mask, std::nullopt});
      void* handle = graph.get();
      c.graphs.emplace(handle, std::move(graph));
      c.graph_nodes += nodes; c.graph_bytes += bytes; ++c.captures;
      return handle;
    } catch (...) {
      try { recording->compute.end_recording(native); } catch (...) {}
      if (!c.recordings.empty()) {
        // Registration reserved the bounded retirement capacity. Keep failed
        // capture owners until other queues stop recording; no allocation here.
        for (auto& read : recording->immutable_reads)
          c.retired_immutable_owners.push_back(std::move(read.owner));
      }
      c.graph_bytes -= GraphCheckBytes; c.pinned_bytes -= GraphCheckBytes;
      throw;
    }
  }
  void ReplayGraph(Queue& q, void* handle) override {
    auto& native = queue(q); auto& c = ctx();
    // The bounded workspaces belong to the device, not to each captured slot.
    // Share their ordering with eager callers; a second queue must wait for
    // the prior user's GPU event before overwriting the same scratch.
    std::scoped_lock workspaces(c.exl3.mutex, c.gdn.mutex, c.attention.mutex,
                                c.sampling.mutex, c.native_gdn.mutex);
    std::lock_guard lock(c.mutex);
    auto it = c.graphs.find(handle);
    VT_CHECK(it != c.graphs.end(), "XPU graph handle is not owned by this device");
    VT_CHECK(!c.recordings.count(&native), "XPU graph replay cannot be nested in a capture");
    auto& graph = *it->second;
    const bool graph_profile = GraphProfileEnabled();
    const bool host_profile = graph_profile || HostProfileEnabled();
    Workspace* scratch[] = {&c.exl3, &c.gdn, &c.attention, &c.sampling, &c.native_gdn};
    if (graph.validation) {
      // All check kernels share one small D2H. They read the freshly staged
      // metadata on this queue; invalid indices fail before compute can write
      // KV or recurrent state. Never reuse capture-time validation results.
      const auto validation_start = host_profile ? SteadyNs() : 0;
      const auto validation_event = native.submit([&](sycl::handler& h) {
        if (graph.last) h.depends_on(*graph.last);
        h.ext_oneapi_graph(*graph.validation);
      });
      if (graph_profile)
        AppendProfileEvent(c, q, "graph_validation", nullptr, validation_event);
      if (host_profile)
        AppendHostProfileRecord(c, q.id, "graph_validation_submit", validation_start, SteadyNs());
      const auto d2h_start = host_profile ? SteadyNs() : 0;
      native.memcpy(graph.checks->host, graph.checks->device,
                    graph.checks->messages.size() * sizeof(int)).wait_and_throw();
      if (host_profile) AppendHostProfileRecord(c, q.id, "graph_validation_d2h_wait", d2h_start, SteadyNs());
      const auto scan_start = host_profile ? SteadyNs() : 0;
      for (size_t i = 0; i < graph.checks->messages.size(); ++i)
        VT_CHECK(graph.checks->host[i] != 0, graph.checks->messages[i]);
      if (host_profile) AppendHostProfileRecord(c, q.id, "graph_validation_scan", scan_start, SteadyNs());
    }
    // Replays of the same executable serialize even when callers switch queues;
    // distinct executables may overlap when their buffers are independent.
    const auto compute_submit_start = host_profile ? SteadyNs() : 0;
    graph.last = native.submit([&](sycl::handler& h) {
      if (graph.last) h.depends_on(*graph.last);
      for (unsigned i = 0; i < 5; ++i)
        if ((graph.workspace_mask & (1u << i)) && scratch[i]->last) h.depends_on(*scratch[i]->last);
      h.ext_oneapi_graph(graph.executable);
    });
    if (graph_profile)
      AppendProfileEvent(c, q, "graph_compute", nullptr, *graph.last);
    if (host_profile)
      AppendHostProfileRecord(c, q.id, "graph_compute_submit", compute_submit_start, SteadyNs());
    for (unsigned i = 0; i < 5; ++i) if (graph.workspace_mask & (1u << i)) scratch[i]->last = graph.last;
    ++c.replays;
  }
  void DestroyGraph(void* handle) override {
    if (!handle) return;
    auto& c = ctx();
    std::lock_guard lifetime_lock(c.graph_owner_lifetime);
    std::vector<std::shared_ptr<void>> retired_owners;
    std::lock_guard lock(c.mutex);
    auto it = c.graphs.find(handle);
    VT_CHECK(it != c.graphs.end(), "XPU graph handle is not owned by this device");
    if (it->second->last) it->second->last->wait_and_throw();
    if (c.recordings.empty()) retired_owners = std::move(it->second->immutable_owners);
    else {
      for (auto& owner : it->second->immutable_owners)
        c.retired_immutable_owners.push_back(std::move(owner));
      it->second->immutable_owners.clear();
    }
    c.graph_nodes -= it->second->nodes; c.graph_bytes -= it->second->bytes;
    c.pinned_bytes -= GraphCheckBytes;
    for (auto p = c.default_graphs.begin(); p != c.default_graphs.end();) {
      if (p->second == handle) p = c.default_graphs.erase(p); else ++p;
    }
    c.graphs.erase(it);  // Destroy GPU executable before releasing map owners.
  }
  void EndCapture(Queue& q) override {
    void* old = nullptr;
    { auto& c = ctx(); std::lock_guard lock(c.mutex);
      auto it = c.default_graphs.find(static_cast<sycl::queue*>(q.handle));
      if (it != c.default_graphs.end()) old = it->second; }
    void* handle = EndCaptureGraph(q);
    if (old) DestroyGraph(old);
    auto& c = ctx(); std::lock_guard lock(c.mutex);
    c.default_graphs[static_cast<sycl::queue*>(q.handle)] = handle;
  }
  void Replay(Queue& q) override {
    void* handle = nullptr;
    { auto& c = ctx(); std::lock_guard lock(c.mutex);
      auto it = c.default_graphs.find(static_cast<sycl::queue*>(q.handle));
      VT_CHECK(it != c.default_graphs.end(), "XPU queue has no default graph"); handle = it->second; }
    ReplayGraph(q, handle);
  }
  int64_t GraphsCaptured() const override { auto& c = ctx(); std::lock_guard lock(c.mutex); return c.captures; }
  int64_t GraphReplays() const override { auto& c = ctx(); std::lock_guard lock(c.mutex); return c.replays; }
  bool UnifiedMemory() const override { return false; }
  bool DeviceMemoryIsHostAddressable() const override { return false; }
  bool DeviceMemoryInfo(size_t* free, size_t* total) const override {
    const auto info = GetMemoryInfo(index_);
    if (!info.free_known) return false;
    if (free) *free = std::min(info.free_bytes, info.budget_bytes - info.allocated_bytes - info.graph_device_bytes);
    if (total) *total = info.total_bytes;
    return true;
  }
  Event CreateEvent(bool = false) override {
    (void)ctx();
    return Event{Device{DeviceType::kXPU, index_}, new EventState{index_, {}, false}};
  }
  void DestroyEvent(Event& e) override {
    if (!e.handle) return;
    auto& state = CheckEvent(e, index_);
    if (state.recorded) state.event.wait_and_throw();
    delete &state;
    e.handle = nullptr;
  }
  void RecordEvent(Event& e, Queue& q) override {
    auto& state = CheckEvent(e, index_);
    state.event = queue(q).ext_oneapi_submit_barrier();
    state.recorded = true;
  }
  void SynchronizeEvent(Event& e) override {
    auto& state = CheckEvent(e, index_);
    if (state.recorded) state.event.wait_and_throw();
  }
  bool QueryEvent(Event& e) override {
    auto& state = CheckEvent(e, index_);
    return !state.recorded || state.event.get_info<sycl::info::event::command_execution_status>()
                                 == sycl::info::event_command_status::complete;
  }
  void QueueWaitEvent(Queue& q, Event& e) override {
    auto& state = CheckEvent(e, index_);
    if (state.recorded) queue(q).ext_oneapi_submit_barrier({state.event});
  }
};
struct Registrar {
  Registrar() {
    // Register a refusing index-0 backend even without hardware: explicit XPU
    // resource requests get a diagnostic and can never resolve to a CPU queue.
    static std::vector<std::unique_ptr<XpuBackend>> backends;
    const int count = std::max(1, std::min(DeviceCount(), static_cast<int>(kMaxDevicesPerType)));
    for (int i = 0; i < count; ++i) {
      backends.push_back(std::make_unique<XpuBackend>(i));
      RegisterBackend(Device{DeviceType::kXPU, i}, backends.back().get());
    }
  }
};
[[maybe_unused]] Registrar registrar;
}  // namespace

int DeviceCount() noexcept { return static_cast<int>(AllDevices().gpu.size()); }
sycl::queue& NativeQueue(Queue& q) {
  VT_CHECK(q.device.type == DeviceType::kXPU && q.handle != nullptr, "XPU requires a live native queue");
  auto& c = GetContext(q.device.index);
  std::lock_guard<std::mutex> lock(c.mutex);
  auto it = c.queues.find(static_cast<sycl::queue*>(q.handle));
  VT_CHECK(it != c.queues.end(), "XPU queue does not belong to this context");
  return *it->second;
}
ProfileMatrixScope::ProfileMatrixScope(const char* matrix) noexcept
    : previous_(current_profile_matrix) { current_profile_matrix = matrix; }
ProfileMatrixScope::~ProfileMatrixScope() noexcept { current_profile_matrix = previous_; }
ProfileLayerScope::ProfileLayerScope(int64_t layer) noexcept
    : previous_(current_profile_layer) { current_profile_layer = layer; }
ProfileLayerScope::~ProfileLayerScope() noexcept { current_profile_layer = previous_; }
void RecordProfileEvent(Queue& q, const char* stage, const sycl::event& event) {
  if (!ProfileQueuesEnabled()) return;
  auto& c = GetContext(q.device.index);
  std::lock_guard<std::mutex> lock(c.mutex);
  auto* native = static_cast<sycl::queue*>(q.handle);
  VT_CHECK(c.queues.count(native) != 0, "XPU profiling requires a live queue");
  if (c.recordings.count(native)) return;  // Graph nodes need separate profiling.
  AppendProfileEvent(c, q, stage, current_profile_matrix, event);
}
bool ProfileQueueEventsEnabled() { return ProfileQueuesEnabled(); }
bool HostProfileSpansEnabled() { return HostProfileEnabled(); }
uint64_t HostProfileClockNs() { return SteadyNs(); }
struct ProfileSpanState {
  uint64_t queue_id = 0, host_start_ns = 0;
  std::optional<sycl::event> begin;
};
ProfileSpan::ProfileSpan() = default;
ProfileSpan::~ProfileSpan() = default;
ProfileSpan::ProfileSpan(ProfileSpan&&) noexcept = default;
ProfileSpan& ProfileSpan::operator=(ProfileSpan&&) noexcept = default;
ProfileSpan BeginProfileSpan(Queue& q) {
  ProfileSpan span;
  if (q.device.type != DeviceType::kXPU ||
      (!ProfileQueuesEnabled() && !HostProfileEnabled())) return span;
  span.state = std::make_unique<ProfileSpanState>();
  span.state->queue_id = q.id;
  span.state->host_start_ns = SteadyNs();
  if (ProfileQueuesEnabled()) {
    span.state->begin = NativeQueue(q).single_task(ProfileAnchorKernel{});
  }
  return span;
}
void EndProfileSpan(Queue& q, const char* stage, ProfileSpan span) {
  if (!span.state) return;
  VT_CHECK(span.state->queue_id == q.id,
           "XPU profile span must end on its original queue");
  if (span.state->begin) {
    const auto end = NativeQueue(q).single_task(ProfileAnchorKernel{});
    RecordProfileSpan(q, stage, *span.state->begin, end,
                      span.state->host_start_ns, {});
  }
  RecordHostProfileSpan(q, stage, span.state->host_start_ns, SteadyNs());
}
void RecordHostProfileSpan(Queue& q, const char* stage,
                           uint64_t start_ns, uint64_t end_ns) {
  if (!HostProfileEnabled()) return;
  auto& c = GetContext(q.device.index);
  std::lock_guard<std::mutex> lock(c.mutex);
  AppendHostProfileRecord(c, q.id, stage, start_ns, end_ns);
}
void RecordProfileSpan(Queue& q, const char* stage, const sycl::event& begin,
                       const sycl::event& end, uint64_t host_submit_ns,
                       const std::string& detail) {
  if (!ProfileQueuesEnabled()) return;
  auto& c = GetContext(q.device.index);
  std::lock_guard<std::mutex> lock(c.mutex);
  auto* native = static_cast<sycl::queue*>(q.handle);
  VT_CHECK(c.queues.count(native) != 0, "XPU profiling requires a live queue");
  if (c.recordings.count(native)) return;
  AppendProfileEvent(c, q, stage, current_profile_matrix, end);
  auto& pending = c.profile_events.back();
  pending.begin = begin;
  pending.host_submit_ns = host_submit_ns;
  if (current_profile_layer >= 0)
    pending.matrix = "layer=" + std::to_string(current_profile_layer) +
                     (pending.matrix.empty() ? "" : " " + pending.matrix);
  if (!detail.empty()) {
    if (!pending.matrix.empty()) pending.matrix += ' ';
    pending.matrix += detail;
  }
}
ProfileClockAnchor CaptureProfileClockAnchor(int index) {
  VT_CHECK(ProfileQueuesEnabled(), "XPU profile clock anchor requires VT_XPU_PROFILE=1");
  auto& c = GetContext(index);
  sycl::queue queue(c.context, c.device,
      sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
  auto warm = queue.single_task(ProfileAnchorKernel{});
  warm.wait_and_throw();
  const auto host_ns = [](auto point) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        point.time_since_epoch()).count());
  };
  const auto before = std::chrono::steady_clock::now();
  auto event = queue.single_task(ProfileAnchorKernel{});
  event.wait_and_throw();
  const auto after = std::chrono::steady_clock::now();
  return {host_ns(before), host_ns(after),
      event.get_profiling_info<sycl::info::event_profiling::command_start>(),
      event.get_profiling_info<sycl::info::event_profiling::command_end>()};
}
std::vector<ProfileRecord> DrainProfileEvents(int index) {
  if (!ProfileQueuesEnabled()) return {};
  auto& c = GetContext(index);
  std::vector<PendingProfileEvent> pending;
  {
    std::lock_guard<std::mutex> lock(c.mutex);
    pending.swap(c.profile_events);
  }
  std::vector<ProfileRecord> records;
  records.reserve(pending.size());
  for (auto& item : pending) {
    if (item.begin) item.begin->wait_and_throw();
    item.event.wait_and_throw();
    const auto start = item.begin
        ? item.begin->get_profiling_info<sycl::info::event_profiling::command_end>()
        : item.event.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end = item.begin
        ? item.event.get_profiling_info<sycl::info::event_profiling::command_start>()
        : item.event.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto submit = item.begin
        ? item.begin->get_profiling_info<sycl::info::event_profiling::command_submit>()
        : item.event.get_profiling_info<sycl::info::event_profiling::command_submit>();
    records.push_back({item.stage, std::move(item.matrix), item.queue_id,
        submit, start, end, item.begin.has_value(), item.host_submit_ns});
  }
  return records;
}
std::vector<HostProfileRecord> DrainHostProfileRecords(int index) {
  if (!GraphProfileEnabled() && !HostProfileEnabled()) return {};
  auto& c = GetContext(index);
  std::vector<HostProfileRecord> records;
  {
    std::lock_guard<std::mutex> lock(c.mutex);
    records.swap(c.host_profile_records);
  }
  return records;
}
size_t PendingProfileEventCount(int index) {
  if (!ProfileQueuesEnabled()) return 0;
  auto& c = GetContext(index);
  std::lock_guard<std::mutex> lock(c.mutex);
  return c.profile_events.size();
}
bool IsGraphCapturing(Queue& q) {
  auto& native = NativeQueue(q); auto& c = GetContext(q.device.index);
  std::lock_guard lock(c.mutex);
  return c.recordings.count(&native) != 0;
}
void RecordGraphImmutableRead(Queue& q, const void* data, size_t bytes,
    const std::shared_ptr<void>& owner, const char* message) {
  auto& native = NativeQueue(q); auto& c = GetContext(q.device.index);
  std::lock_guard lock(c.mutex);
  auto it = c.recordings.find(&native);
  if (it == c.recordings.end()) return;
  VT_CHECK(owner && owner.get() == data, "XPU immutable metadata requires its allocation owner");
  const auto start = reinterpret_cast<uintptr_t>(data);
  VT_CHECK(bytes > 0 && bytes <= UINTPTR_MAX - start, "XPU immutable metadata span overflow");
  size_t owners = c.retired_immutable_owners.size();
  for (const auto& [_, graph] : c.graphs) owners += graph->immutable_owners.size();
  for (const auto& [_, recording] : c.recordings) owners += recording->immutable_reads.size();
  VT_CHECK(owners < MaxGraphNodes, "XPU immutable metadata owner limit exceeded");
  // One bounded host array makes retirement non-allocating, including a
  // rejected capture while another queue is recording. No GPU scratch/polling.
  c.retired_immutable_owners.reserve(MaxGraphNodes);
  it->second->immutable_reads.push_back({start, start + bytes, owner, message});
}
bool CaptureMetadataCheck(Queue& q, const std::function<void(sycl::handler&, int*)>& submit, const char* message,
                          std::initializer_list<const Tensor*> inputs) {
  auto& native = NativeQueue(q); auto& c = GetContext(q.device.index);
  std::lock_guard lock(c.mutex);
  auto it = c.recordings.find(&native);
  if (it == c.recordings.end()) return false;
  auto& recording = *it->second;
  const size_t index = recording.checks->messages.size();
  VT_CHECK(index < MaxGraphChecks, "XPU graph metadata validation budget exceeded");
  for (const auto* input : inputs) if (input && input->Numel()) {
    const auto start = reinterpret_cast<uintptr_t>(input->data);
    recording.metadata.push_back({start, start + Span(*input), index});
  }
  recording.validation.add([&](sycl::handler& h) { submit(h, recording.checks->device + index); });
  recording.checks->messages.emplace_back(message);
  return true;
}
void RecordGraphWrite(Queue& q, const void* data, size_t bytes) {
  if (!bytes) return;
  auto& native = NativeQueue(q); auto& c = GetContext(q.device.index);
  std::lock_guard lock(c.mutex);
  if (auto it = c.recordings.find(&native); it != c.recordings.end()) {
    const auto start = reinterpret_cast<uintptr_t>(data);
    VT_CHECK(bytes <= UINTPTR_MAX - start, "XPU graph write span overflow");
    it->second->writes.push_back({start, start + bytes});
  }
}
MemoryInfo GetMemoryInfo(int index) {
  auto& c = GetContext(index);
  std::lock_guard<std::mutex> lock(c.mutex);
  MemoryInfo info{c.total, c.budget, c.allocated, c.pinned_bytes, 0, false};
  info.exl3_workspace_bytes = c.exl3.bytes;
  info.gdn_workspace_bytes = c.gdn.bytes;
  info.native_gdn_workspace_bytes = c.native_gdn.bytes;
  info.attention_workspace_bytes = c.attention.bytes;
  info.sampling_workspace_bytes = c.sampling.bytes;
  info.w8a8_workspace_bytes = c.w8a8.bytes;
  info.w8a8_preparation_bytes = c.w8a8_preparation.bytes;
  info.fp16_silu_table_bytes = c.fp16_silu.data ? 65536 * sizeof(uint16_t) : 0;
  info.gated_silu_table_bytes = c.gated_silu.data ? 65536 * sizeof(float) : 0;
  info.peak_allocated_bytes = c.peak_allocated;
  info.graph_count = c.graphs.size(); info.graph_nodes = c.graph_nodes;
  info.graph_device_bytes = c.graph_bytes;
  if (c.device.has(sycl::aspect::ext_intel_free_memory)) {
    info.free_bytes = c.device.get_info<sycl::ext::intel::info::device::free_memory>();
    info.free_known = true;
  }
  return info;
}
namespace {
bool WithWorkspace(Queue& q, Workspace& workspace, unsigned mask,
                   const char* wait_stage, size_t bytes,
                   const std::function<void(void*)>& launch) {
  auto& native = NativeQueue(q);
  auto& c = GetContext(q.device.index);
  std::lock_guard<std::mutex> execution(workspace.mutex);
  bool capturing = false;
  {
    std::lock_guard lock(c.mutex);
    if (auto it = c.recordings.find(&native); it != c.recordings.end()) {
      VT_CHECK(workspace.data != nullptr, "XPU graph workspace must be warmed before capture");
      it->second->workspace_mask |= mask;
      capturing = true;
    }
  }
  if (!workspace.data) {
    std::lock_guard<std::mutex> lock(c.mutex);
    // Reserve and account under the same lock as ordinary allocations: another
    // queue must not consume this budget between the check and allocation.
    if (bytes > c.budget - c.allocated - c.graph_bytes) return false;
    void* storage = sycl::aligned_alloc_device(64, bytes, c.device, c.context);
    VT_CHECK(storage != nullptr, "XPU persistent workspace allocation failed");
    try { c.allocations.emplace(storage, bytes); }
    catch (...) { sycl::free(storage, c.context); throw; }
    c.allocated += bytes;
    c.peak_allocated = std::max(c.peak_allocated, c.allocated);
    workspace.data = storage;
    workspace.bytes = bytes;
  }
  VT_CHECK(bytes <= workspace.bytes, "XPU persistent workspace cannot grow");
  if (!capturing && workspace.last) native.ext_oneapi_submit_barrier({*workspace.last});
  try {
    launch(workspace.data);
    if (!capturing) {
      const auto wait_start = HostProfileEnabled() ? SteadyNs() : 0;
      native.wait_and_throw();
      if (wait_start)
        RecordHostProfileSpan(q, wait_stage, wait_start, SteadyNs());
      workspace.last.reset();
    }
  } catch (...) {
    // Even when host-side submission throws, complete earlier kernels before
    // releasing this workspace to another queue.
    if (!capturing) { try { native.wait_and_throw(); } catch (...) {} }
    throw;
  }
  return true;
}
}
bool WithExl3Workspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
  VT_CHECK(bytes > 0 && bytes <= 32 * 1024 * 1024, "XPU EXL3 workspace exceeds 32 MiB budget");
  return WithWorkspace(q, GetContext(q.device.index).exl3, 1,
                       "workspace_wait_exl3", bytes, launch);
}
namespace {
bool WithEagerW8A8Workspace(Queue& q, Workspace& workspace, size_t bytes,
                           const std::function<void(void*)>& launch, const char* wait_stage) {
  VT_CHECK(bytes > 0, "XPU W8A8 workspace must be nonempty");
  auto& native = NativeQueue(q);
  auto& c = GetContext(q.device.index);
  std::lock_guard execution(workspace.mutex);
  {
    std::lock_guard lock(c.mutex);
    // No W8A8 pool address enters a graph. Existing SmallM graph ownership is
    // independent, so eager growth cannot invalidate a captured panel.
    VT_CHECK(!c.recordings.count(&native), "XPU shared W8A8 workspace is eager-only");
    if (bytes > workspace.bytes) {
      if (bytes - workspace.bytes > c.budget - c.allocated - c.graph_bytes) return false;
      // Every previous lease completed before returning, including exceptions.
      // Free first to avoid retaining both the old and new high-water capacities.
      if (workspace.data) {
        sycl::free(workspace.data, c.context);
        c.allocations.erase(workspace.data);
        c.allocated -= workspace.bytes;
        workspace.data = nullptr; workspace.bytes = 0;
      }
      void* storage = sycl::aligned_alloc_device(64, bytes, c.device, c.context);
      VT_CHECK(storage != nullptr, "XPU shared W8A8 workspace allocation failed");
      try { c.allocations.emplace(storage, bytes); }
      catch (...) { sycl::free(storage, c.context); throw; }
      c.allocated += bytes;
      c.peak_allocated = std::max(c.peak_allocated, c.allocated);
      workspace.data = storage; workspace.bytes = bytes;
    }
  }
  try {
    launch(workspace.data);
    // Retain this completion fence until an asynchronous panel-lease protocol
    // can safely retire all validation, oneDNN and output-Hadamard consumers.
    const auto start = HostProfileEnabled() ? SteadyNs() : 0;
    native.wait_and_throw();
    if (start) RecordHostProfileSpan(q, wait_stage, start, SteadyNs());
  } catch (...) {
    try { native.wait_and_throw(); } catch (...) {}
    throw;
  }
  return true;
}
} // namespace
bool WithExl3W8A8Workspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
  return WithEagerW8A8Workspace(q, GetContext(q.device.index).w8a8, bytes, launch,
                                "workspace_wait_w8a8");
}
bool WithExl3W8A8Preparation(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
  if (bytes > 64 * 1024 * 1024) return false;
  return WithEagerW8A8Workspace(q, GetContext(q.device.index).w8a8_preparation, bytes, launch,
                                "workspace_wait_w8a8_preparation");
}
bool WithGdnWorkspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
  VT_CHECK(bytes > 0 && bytes <= 32 * 1024 * 1024, "XPU GDN workspace exceeds 32 MiB budget");
  return WithWorkspace(q, GetContext(q.device.index).gdn, 2,
                       "workspace_wait_gdn", bytes, launch);
}
bool WithGdnNativeWorkspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
  VT_CHECK(bytes > 0 && bytes <= kGdnFp16ReservationBytes,
           "XPU native GDN workspace exceeds the stable 220 MiB reservation");
  return WithWorkspace(q, GetContext(q.device.index).native_gdn, 16,
                       "workspace_wait_gdn_native", kGdnFp16ReservationBytes, launch);
}
bool WithAttentionWorkspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
  VT_CHECK(bytes > 0 && bytes <= 16 * 1024 * 1024, "XPU attention workspace exceeds 16 MiB budget");
  return WithWorkspace(q, GetContext(q.device.index).attention, 4,
                       "workspace_wait_attention", bytes, launch);
}
namespace {
bool WithImmutableSiluTable(Queue& q, ImmutableTable& table, size_t bytes,
    const char* cold_message, const char* allocation_message,
    const std::function<sycl::event(void*)>& initialize,
    const std::function<void(const void*)>& launch) {
  auto& native = NativeQueue(q);
  auto& c = GetContext(q.device.index);
  std::lock_guard execution(table.mutex);
  bool capturing;
  {
    std::lock_guard lock(c.mutex);
    capturing = c.recordings.count(&native) != 0;
  }
  // BeginCapture completes this queue. A prior eager join guarantees that the
  // immutable producer has completed too, including when it used another queue.
  VT_CHECK(!capturing || (table.data && table.joined.count(&native)),
           cold_message);
  if (!table.data) {
    {
      std::lock_guard lock(c.mutex);
      if (bytes > c.budget - c.allocated - c.graph_bytes) return false;
      void* storage = sycl::aligned_alloc_device(64, bytes, c.device, c.context);
      VT_CHECK(storage, allocation_message);
      try { c.allocations.emplace(storage, bytes); }
      catch (...) { sycl::free(storage, c.context); throw; }
      c.allocated += bytes;
      c.peak_allocated = std::max(c.peak_allocated, c.allocated);
      table.data = storage;
    }
    try { table.ready = initialize(table.data); }
    catch (...) {
      // Finish any submitted initializer work before reclaiming its storage.
      try { native.wait_and_throw(); } catch (...) {}
      std::lock_guard lock(c.mutex);
      c.allocations.erase(table.data); c.allocated -= bytes;
      sycl::free(table.data, c.context); table.data = nullptr;
      throw;
    }
    table.joined.insert(&native);  // initialization is ordered on this queue
  } else if (!capturing && !table.joined.count(&native)) {
    native.ext_oneapi_submit_barrier({*table.ready});
    table.joined.insert(&native);
  }
  // No reader serialization: the table is never overwritten. Context teardown
  // completes every owned queue, destroys graphs, then frees accounted storage.
  launch(table.data);
  return true;
}
}  // namespace
bool WithFp16SiluTable(Queue& q,
    const std::function<sycl::event(void*)>& initialize,
    const std::function<void(const void*)>& launch) {
  return WithImmutableSiluTable(q, GetContext(q.device.index).fp16_silu,
      65536 * sizeof(uint16_t), "XPU FP16 SiLU table must be warmed on the capture queue",
      "XPU FP16 SiLU table allocation failed",
      initialize, launch);
}
bool WithGatedSiluTable(Queue& q,
    const std::function<sycl::event(void*)>& initialize,
    const std::function<void(const void*)>& launch) {
  return WithImmutableSiluTable(q, GetContext(q.device.index).gated_silu,
      65536 * sizeof(float), "XPU gated SiLU table must be warmed on the capture queue",
      "XPU gated SiLU table allocation failed",
      initialize, launch);
}
bool WithSamplingWorkspace(Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
  VT_CHECK(bytes > 0 && bytes <= 16 * 1024 * 1024, "XPU sampling workspace exceeds 16 MiB budget");
  return WithWorkspace(q, GetContext(q.device.index).sampling, 8,
                       "workspace_wait_sampling", bytes, launch);
}
std::string DeviceDescription(int index) {
  const auto& d = DeviceAt(index);
  nlohmann::json info = {
      {"index", index}, {"name", d.get_info<sycl::info::device::name>()},
      {"driver", d.get_info<sycl::info::device::driver_version>()},
      {"runtime", d.get_platform().get_info<sycl::info::platform::version>()},
      {"compiler", __VERSION__}, {"sycl_language", SYCL_LANGUAGE_VERSION},
      {"queue_profiling", ProfileQueuesEnabled()},
      {"subgroups", d.get_info<sycl::info::device::sub_group_sizes>()},
      {"total_bytes", d.get_info<sycl::info::device::global_mem_size>()}};
  if (d.has(sycl::aspect::ext_intel_device_id))
    info["device_id"] = d.get_info<sycl::ext::intel::info::device::device_id>();
  info["matrix_combinations"] = nlohmann::json::array();
  if (d.has(sycl::aspect::ext_intel_matrix)) {
    for (const auto& m : d.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
      info["matrix_combinations"].push_back({{"m", m.msize}, {"n", m.nsize}, {"k", m.ksize},
          {"max_m", m.max_msize}, {"max_n", m.max_nsize}, {"max_k", m.max_ksize},
          {"a_type", static_cast<int>(m.atype)}, {"b_type", static_cast<int>(m.btype)},
          {"c_type", static_cast<int>(m.ctype)}, {"d_type", static_cast<int>(m.dtype)}});
  }
  return info.dump();
}
}  // namespace vt::xpu

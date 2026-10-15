#include "xpu_gptq4.h"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#include "xpu_common.h"

namespace vt::xpu {
namespace {

constexpr char kExpectedOneDnnHash[] =
    "0e2a5bfeef1bfbffc3137464606540233086ce9b";

enum class Kind : uint8_t { kGptq4, kDenseF16, kExl3W8A8 };

struct PrimitiveKey {
  Kind kind;
  int64_t m, k, n;
  int group_size;
  bool bias;
  int64_t dst_stride;

  friend bool operator<(const PrimitiveKey& a, const PrimitiveKey& b) {
    return std::tie(a.kind, a.m, a.k, a.n, a.group_size, a.bias, a.dst_stride) <
           std::tie(b.kind, b.m, b.k, b.n, b.group_size, b.bias, b.dst_stride);
  }
};

struct ExecutionBinding;

struct PrimitiveEntry {
  dnnl::memory::desc src;
  dnnl::memory::desc weights;
  dnnl::memory::desc dst;
  dnnl::memory::desc bias;
  dnnl::memory::desc scales;
  dnnl::memory::desc row_scales;
  dnnl::memory::desc zero_points;
  dnnl::memory::desc scratchpad;
  std::unique_ptr<dnnl::matmul> primitive;
  std::mutex bindings_mutex;
  std::unordered_map<uint64_t, std::unique_ptr<ExecutionBinding>> bindings;
  size_t scratchpad_bytes = 0;
  std::string implementation;
};

struct OneDnnProfileAnchor { void operator()() const {} };

struct ExecutionBinding {
  std::mutex mutex;
  std::unordered_map<int, dnnl::memory> args;
};

struct QueueScratchpad {
  void* data = nullptr;
  size_t bytes = 0;
};

struct DeviceRuntime {
  explicit DeviceRuntime(sycl::queue& queue)
      : engine(dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context())) {
    const auto* version = dnnl_version();
    if (version == nullptr || version->hash == nullptr ||
        std::string(version->hash) != kExpectedOneDnnHash) {
      throw std::runtime_error(
          "GPTQ oneDNN path requires the pinned v3.13 source hash "
          "0e2a5bfeef1bfbffc3137464606540233086ce9b");
    }
    stats.engine_count = 1;
  }

  dnnl::engine engine;
  std::mutex mutex;
  std::map<PrimitiveKey, std::shared_ptr<PrimitiveEntry>> primitives;
  std::unordered_map<uint64_t, std::unique_ptr<dnnl::stream>> streams;
  std::unordered_map<uint64_t, QueueScratchpad> scratchpads;
  Gptq4RuntimeStats stats;
};

std::mutex runtimes_mutex;
std::map<int, std::unique_ptr<DeviceRuntime>> runtimes;

dnnl::memory UsmMemory(const dnnl::memory::desc& descriptor,
                       const dnnl::engine& engine, void* pointer) {
  return dnnl::sycl_interop::make_memory(
      descriptor, engine, dnnl::sycl_interop::memory_kind::usm,
      pointer == nullptr ? DNNL_MEMORY_ALLOCATE : pointer);
}

DeviceRuntime& Runtime(Queue& queue) {
  auto& native = NativeQueue(queue);
  std::lock_guard<std::mutex> lock(runtimes_mutex);
  auto& runtime = runtimes[queue.device.index];
  if (!runtime) runtime = std::make_unique<DeviceRuntime>(native);
  return *runtime;
}

dnnl::stream& Stream(DeviceRuntime& runtime, Queue& queue) {
  std::lock_guard<std::mutex> lock(runtime.mutex);
  auto& stream = runtime.streams[queue.id];
  if (!stream)
    stream = std::make_unique<dnnl::stream>(
        dnnl::sycl_interop::make_stream(runtime.engine, NativeQueue(queue)));
  return *stream;
}

std::shared_ptr<PrimitiveEntry> GetPrimitive(DeviceRuntime& runtime,
                                             const PrimitiveKey& key) {
  std::lock_guard<std::mutex> lock(runtime.mutex);
  if (auto it = runtime.primitives.find(key); it != runtime.primitives.end())
    return it->second;

  using dt = dnnl::memory::data_type;
  auto entry = std::make_shared<PrimitiveEntry>();
  entry->src = dnnl::memory::desc({key.m, key.k},
      key.kind == Kind::kExl3W8A8 ? dt::s8 : dt::f16, {key.k, 1});
  entry->dst = dnnl::memory::desc({key.m, key.n}, dt::f16, {key.dst_stride, 1});
  entry->weights = key.kind == Kind::kGptq4
      ? dnnl::memory::desc({key.k, key.n}, dt::u4, {1, key.k})
      : dnnl::memory::desc({key.k, key.n}, dt::f16, {1, key.k});
  if (key.kind == Kind::kExl3W8A8)
    entry->weights = dnnl::memory::desc({key.k, key.n}, dt::s8, {key.n, 1});

  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  // Fixed-input dense FP16 BA projections must not vary across native
  // request resets or graph/eager controls. Ask oneDNN for deterministic
  // reductions; quantized GPTQ4 and EXL3 W8A8 descriptors retain their policy.
  if (key.kind == Kind::kDenseF16) attr.set_deterministic(true);
  if (key.kind == Kind::kGptq4) {
    attr.set_scales(DNNL_ARG_WEIGHTS, 3, {key.group_size, 1}, dt::f16);
    attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, dt::s8);
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    entry->scales = dnnl::memory::desc(
        {key.k / key.group_size, key.n}, dt::f16,
        {key.n, 1});
    entry->zero_points = dnnl::memory::desc({1}, dt::s8, {1});
  }
  if (key.kind == Kind::kExl3W8A8) {
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
    entry->scales = dnnl::memory::desc({1}, dt::f32, {1});
    entry->row_scales = dnnl::memory::desc({key.m, 1}, dt::f32, {1, 1});
    dnnl::post_ops post;
    post.append_binary(dnnl::algorithm::binary_mul, entry->row_scales);
    attr.set_post_ops(post);
  }
  if (key.bias) entry->bias = dnnl::memory::desc({1, key.n}, dt::f16,
                                                 {key.n, 1});

  dnnl::matmul::primitive_desc pd = key.bias
      ? dnnl::matmul::primitive_desc(runtime.engine, entry->src, entry->weights,
                                     entry->bias, entry->dst, attr)
      : dnnl::matmul::primitive_desc(runtime.engine, entry->src, entry->weights,
                                     entry->dst, attr);
  entry->scratchpad = pd.scratchpad_desc();
  entry->scratchpad_bytes = entry->scratchpad.get_size();
  entry->implementation = pd.impl_info_str();
  entry->primitive = std::make_unique<dnnl::matmul>(pd);
  runtime.primitives.emplace(key, entry);
  ++runtime.stats.primitive_count;
  return entry;
}

ExecutionBinding& Binding(PrimitiveEntry& entry, uint64_t queue_id) {
  std::lock_guard<std::mutex> lock(entry.bindings_mutex);
  auto& binding = entry.bindings[queue_id];
  if (!binding) binding = std::make_unique<ExecutionBinding>();
  return *binding;
}

void* Scratchpad(DeviceRuntime& runtime, Queue& queue, size_t bytes) {
  if (bytes == 0) return nullptr;
  std::lock_guard<std::mutex> lock(runtime.mutex);
  auto& slot = runtime.scratchpads[queue.id];
  if (slot.bytes < bytes) {
    NativeQueue(queue).wait_and_throw();
    if (slot.data != nullptr) {
      vt::Free(queue.device, slot.data);
      runtime.stats.scratchpad_capacity_bytes -= slot.bytes;
      slot.data = nullptr;
      slot.bytes = 0;
    }
    slot.data = vt::Alloc(queue.device, bytes);
    slot.bytes = bytes;
    ++runtime.stats.scratchpad_allocation_count;
    runtime.stats.scratchpad_capacity_bytes += bytes;
  }
  return slot.data;
}

void Execute(Queue& queue, Tensor& out, const Tensor& activation,
             const Tensor& weights, const Tensor* bias, Kind kind,
             int group_size, const Tensor* scales, const Tensor* zero_points,
             const Tensor* row_scales = nullptr) {
  auto& runtime = Runtime(queue);
  const PrimitiveKey key{kind, activation.shape[0], activation.shape[1],
                         out.shape[1], group_size, bias != nullptr, out.stride[0]};
  const auto entry = GetPrimitive(runtime, key);
  auto& binding = Binding(*entry, queue.id);
  std::lock_guard<std::mutex> binding_lock(binding.mutex);
  auto bind = [&](int arg, const dnnl::memory::desc& descriptor, void* pointer) {
    auto it = binding.args.find(arg);
    if (it == binding.args.end())
      binding.args.emplace(arg, UsmMemory(descriptor, runtime.engine, pointer));
    else
      it->second.set_data_handle(pointer);
  };
  bind(DNNL_ARG_SRC, entry->src, activation.data);
  bind(DNNL_ARG_WEIGHTS, entry->weights, weights.data);
  bind(DNNL_ARG_DST, entry->dst, out.data);
  if (bias != nullptr) bind(DNNL_ARG_BIAS, entry->bias, bias->data);
  if (kind == Kind::kGptq4) {
    bind(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, entry->scales, scales->data);
    bind(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
         entry->zero_points, zero_points->data);
  }
  if (kind == Kind::kExl3W8A8) {
    bind(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, entry->scales, scales->data);
    bind(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
         entry->row_scales, row_scales->data);
  }
  if (entry->scratchpad_bytes != 0) {
    auto* scratch = Scratchpad(runtime, queue, entry->scratchpad_bytes);
    bind(DNNL_ARG_SCRATCHPAD, entry->scratchpad, scratch);
  }
  // oneDNN 3.13's SYCL execute returns void. Two device markers on the same
  // in-order queue bracket all commands it submits to this stream. Keep this
  // diagnostic out of ordinary throughput runs.
  const bool profile = ProfileQueueEventsEnabled();
  std::optional<sycl::event> profile_begin;
  if (profile)
    profile_begin = NativeQueue(queue).single_task(OneDnnProfileAnchor{});
  const auto host_begin = std::chrono::steady_clock::now();
  dnnl::sycl_interop::execute(*entry->primitive, Stream(runtime, queue),
                              binding.args);
  const auto host_end = std::chrono::steady_clock::now();
  if (profile) {
    const auto profile_end =
        NativeQueue(queue).single_task(OneDnnProfileAnchor{});
    const auto host_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        host_end - host_begin).count();
    const std::string detail =
        "M=" + std::to_string(key.m) + " K=" + std::to_string(key.k) +
        " N=" + std::to_string(key.n) + " impl=" + entry->implementation +
        " scratch=" + std::to_string(entry->scratchpad_bytes);
    RecordProfileSpan(queue,
                      kind == Kind::kGptq4 ? "onednn_gptq4_stream" :
                      kind == Kind::kExl3W8A8 ? "onednn_exl3_w8a8_stream" :
                                              "onednn_dense_f16_stream",
                      *profile_begin, profile_end,
                      static_cast<uint64_t>(host_ns), detail);
  }
}

void MatmulGptq4Kernel(Queue& queue, Tensor& out, const Tensor& activation,
                       const Tensor& qweight, const Tensor& scales,
                       const Tensor& zero_points, int group_size,
                       const Tensor* bias) {
  Execute(queue, out, activation, qweight, bias, Kind::kGptq4, group_size,
          &scales, &zero_points);
}

void MatmulDenseF16Kernel(Queue& queue, Tensor& out, const Tensor& activation,
                          const Tensor& weight, const Tensor* bias) {
  Execute(queue, out, activation, weight, bias, Kind::kDenseF16, 0, nullptr,
          nullptr);
}

}  // namespace

void Exl3W8A8Matmul(Queue& q, Tensor& out, const Tensor& in,
                   const Tensor& weights, const Tensor& sx, const Tensor& sw) {
  Execute(q, out, in, weights, nullptr, Kind::kExl3W8A8, 0, &sw, nullptr, &sx);
}

Gptq4RuntimeStats GetGptq4RuntimeStats(int device_index) {
  std::lock_guard<std::mutex> runtimes_lock(runtimes_mutex);
  const auto it = runtimes.find(device_index);
  if (it == runtimes.end()) return {};
  std::lock_guard<std::mutex> runtime_lock(it->second->mutex);
  return it->second->stats;
}

dnnl::engine& Exl3OneDnnEngine(Queue& queue) { return Runtime(queue).engine; }
dnnl::stream& Exl3OneDnnStream(Queue& queue) { return Stream(Runtime(queue), queue); }
void* Exl3OneDnnScratchpad(Queue& queue, size_t bytes) {
  return Scratchpad(Runtime(queue), queue, bytes);
}

void ReleaseGptq4Queue(const Queue& queue) {
  std::lock_guard<std::mutex> runtimes_lock(runtimes_mutex);
  const auto it = runtimes.find(queue.device.index);
  if (it == runtimes.end()) return;
  auto& runtime = *it->second;
  std::lock_guard<std::mutex> runtime_lock(runtime.mutex);
  runtime.streams.erase(queue.id);
  for (auto& [_, entry] : runtime.primitives) {
    std::lock_guard<std::mutex> binding_lock(entry->bindings_mutex);
    entry->bindings.erase(queue.id);
  }
  const auto scratch = runtime.scratchpads.find(queue.id);
  if (scratch == runtime.scratchpads.end()) return;
  if (scratch->second.data != nullptr) vt::Free(queue.device, scratch->second.data);
  runtime.stats.scratchpad_capacity_bytes -= scratch->second.bytes;
  runtime.scratchpads.erase(scratch);
}

}  // namespace vt::xpu

namespace vt::xpu {
namespace {
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulGptq4W4A16, DeviceType::kXPU,
               reinterpret_cast<void*>(static_cast<MatmulGptq4W4A16Fn>(
                   &MatmulGptq4Kernel)));
    RegisterOp(OpId::kMatmulDenseF16, DeviceType::kXPU,
               reinterpret_cast<void*>(static_cast<MatmulDenseF16Fn>(
                   &MatmulDenseF16Kernel)));
  }
};
[[maybe_unused]] Registrar registrar;
}  // namespace
}  // namespace vt::xpu

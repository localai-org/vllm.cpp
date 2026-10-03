// Tenstorrent host/device residency (BACKEND-TENSTORRENT-SPLIT stage 1).
// Definitions moved verbatim from tenstorrent_ops.cpp; declarations live
// in tenstorrent_internal.h.
#include "vt/tenstorrent/tenstorrent_internal.h"

#include <cstdlib>

#include "vllm/config/tt_weight_residency.h"

namespace vt::tenstorrent {
namespace {
void DropEmbedTableShadow(void* host) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(EmbedTableMutex());
  EmbedTableShadows().erase(reinterpret_cast<uintptr_t>(host));
}

}  // namespace

ttnn::Tensor EnsureEmbedTableDevice(const Tensor& table, MeshDevice& device) {
  VT_CHECK(table.rank == 2 && table.IsContiguous(), "EnsureEmbedTable: rank-2 contiguous");
  const uint32_t vocab = static_cast<uint32_t>(table.shape[0]);
  const uint32_t h = static_cast<uint32_t>(table.shape[1]);
  {
    std::lock_guard<std::mutex> g(EmbedTableMutex());
    auto it = EmbedTableShadows().find(reinterpret_cast<uintptr_t>(table.data));
    if (it != EmbedTableShadows().end() && it->second.device.has_value() &&
        it->second.vocab == vocab && it->second.h == h) {
      return *it->second.device;
    }
  }
  EnsureHost(table);
  std::vector<float> host_table = ToHostF32(table);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] EnsureEmbedTableDevice from_vector WRITE during capture\n");
  AllocTraceSnapshot(device, "EnsureEmbedTableDevice/pre");
  ttnn::Tensor dev_table = ttnn::Tensor::from_vector<float>(
      host_table,
      SpecOf(tt::tt_metal::Shape({vocab, h}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR),
      &device);
  std::lock_guard<std::mutex> g(EmbedTableMutex());
  EmbedTableShadow& s = EmbedTableShadows()[reinterpret_cast<uintptr_t>(table.data)];
  s.device = dev_table;
  s.vocab = vocab;
  s.h = h;
  AllocTraceSnapshot(device, "EnsureEmbedTableDevice/post");
  return dev_table;
}


// Upload a rank-1 affine vector as TILE BFLOAT16 [1, d], caching on the weight's
// host buffer slot so RmsNorm/LayerNorm do not re-upload every layer call.
// ttnn's TILE-gamma path requires padded height == tile_height (32); from_vector
// with TILE pads a [1,d] tensor to that.
ttnn::Tensor EnsureAffine1D(const Tensor& t, uint32_t d, MeshDevice& device) {
  VT_CHECK(t.rank == 1 && t.IsContiguous() && t.shape[0] == static_cast<int64_t>(d),
           "tenstorrent EnsureAffine1D: rank-1 [d] contiguous");
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    // Hit only for the BASE pointer (same interior-view hazard as
    // EnsureDevice2D — a differently-offset rank-1 view must not consume the
    // base's staged affine).
    if (s != nullptr && t.data == s->host && s->device_current &&
        s->device.has_value() && s->dev_rows == 1 && s->dev_cols == d) {
      return *s->device;
    }
  }
  EnsureHost(t);
  std::vector<float> host(d);
  for (uint32_t i = 0; i < d; ++i) host[i] = LoadElemF32(t, i);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] EnsureAffine1D from_vector WRITE during capture\n");
  ttnn::Tensor dev = ttnn::Tensor::from_vector<float>(
      host, SpecOf(tt::tt_metal::Shape({1, d}), ttnn::DataType::BFLOAT16, ttnn::Layout::TILE),
      &device);
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s != nullptr && t.data == s->host) {
    s->device = dev;
    s->dev_rows = 1;
    s->dev_cols = d;
    s->device_current = true;
    s->host_current = true;
    s->device_reserved = false;  // real bytes staged — the reservation is spent
  }
  return dev;
}

ttnn::Tensor ZeroCacheGet(const ttnn::Shape& shape, ttnn::DataType dt,
                          ttnn::Layout lt, MeshDevice& device) {
  const std::string key = ZeroCacheKey(shape, dt, lt);
  std::lock_guard<std::mutex> g(ZeroCacheMutex());
  auto& c = ZeroCache();
  auto it = c.find(key);
  if (it == c.end()) {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent: zero-cache miss during capture — warm the "
             "host-free path eagerly (VT_TT_HOST_FREE_DECODE warmup) first");
    it = c.emplace(key, ttnn::zeros(shape, dt, lt, std::ref(device))).first;
  }
  return it->second;
}

ttnn::Tensor ZeroCacheGet(const ttnn::Tensor& like, MeshDevice& device) {
  return ZeroCacheGet(like.logical_shape(), like.dtype(), like.layout(),
                      device);
}

void ZeroCachePrime(const ttnn::Shape& shape, ttnn::DataType dt,
                    ttnn::Layout lt, MeshDevice& device) {
  const std::string key = ZeroCacheKey(shape, dt, lt);
  std::lock_guard<std::mutex> g(ZeroCacheMutex());
  auto& c = ZeroCache();
  if (c.find(key) == c.end()) {
    c.emplace(key, ttnn::zeros(shape, dt, lt, std::ref(device)));
  }
}



std::mutex& SlotMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, BufferSlot>& Slots() {
  static std::map<uintptr_t, BufferSlot>* m = new std::map<uintptr_t, BufferSlot>(); // never destroyed (#1486)
  return *m;
}

// Base slot for `p` or any interior pointer into a registered allocation.
BufferSlot* FindSlot(void* p) {
  if (p == nullptr) return nullptr;
  auto& m = Slots();
  const uintptr_t key = reinterpret_cast<uintptr_t>(p);
  auto it = m.upper_bound(key);
  if (it == m.begin()) return nullptr;
  --it;
  BufferSlot& s = it->second;
  const uintptr_t base = reinterpret_cast<uintptr_t>(s.host);
  if (key < base || key >= base + s.bytes) return nullptr;
  return &s;
}

tt::tt_metal::TensorSpec SpecOf(tt::tt_metal::Shape shape, ttnn::DataType dtype,
                                ttnn::Layout layout) {
  return tt::tt_metal::TensorSpec(
      std::move(shape),
      tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout),
                                 tt::tt_metal::MemoryConfig{}));
}

tt::tt_metal::TensorSpec TileSpecOf(uint32_t rows, uint32_t cols) {
  return SpecOf(tt::tt_metal::Shape({rows, cols}), ttnn::DataType::BFLOAT16,
                ttnn::Layout::TILE);
}

// OPT-125m (and the rest of the dense path) runs BF16 weights/activations with
// F32 logits. Host-stage every float dtype to f32 for from_vector, then round
// back on download — ttnn already computes in BFLOAT16 tiles.
float LoadElemF32(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[i]);
    default: VT_CHECK(false, "tenstorrent: unsupported float dtype"); return 0.0f;
  }
}

void StoreElemF32(Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = F32ToBF16(v); break;
    default: VT_CHECK(false, "tenstorrent: unsupported out dtype (f32/bf16)");
  }
}

bool IsFloatDType(DType d) {
  return d == DType::kF32 || d == DType::kBF16 || d == DType::kF16;
}

void DownloadToHost(ttnn::Tensor& dev, Tensor& out, const char* ctx) {
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] to_vector readback DURING CAPTURE\n");
  std::vector<float> result = dev.to_vector<float>();
  if (result.size() != static_cast<size_t>(out.Numel())) {
    void* bt[8];
    const int nbt = ::backtrace(bt, 8);
    char** sym = ::backtrace_symbols(bt, nbt);
    std::fprintf(stderr, "[TT-SLOT] MISMATCH self=%p frames=%d\n",
                 reinterpret_cast<const void*>(&DownloadToHost), nbt);
    for (int i = 0; sym != nullptr && i < nbt; ++i)
      std::fprintf(stderr, "[TT-SLOT]   bt[%d]=%p %s\n", i, bt[i], sym[i]);
    std::fflush(stderr);
    std::free(sym);
  }
  VT_CHECK(static_cast<int64_t>(result.size()) == out.Numel(),
           std::string("tenstorrent: unexpected result size: got ") +
               std::to_string(result.size()) + " want " +
               std::to_string(out.Numel()) + " out_shape=" +
               std::to_string(out.shape[0]) + "x" + std::to_string(out.shape[1]) +
               "x" + std::to_string(out.shape[2]) + "x" + std::to_string(out.shape[3]) +
               " rank=" + std::to_string(out.rank) +
               " ctx=" + std::string(ctx) +
               " dev[" + DevShapeStr(dev) + "]");
  for (int64_t i = 0; i < out.Numel(); ++i)
    StoreElemF32(out, i, result[static_cast<size_t>(i)]);
}

// Pull device → host if the host view is stale (required before host kernels).
void EnsureHost(Tensor& t) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s == nullptr || s->host_current) return;
  VT_CHECK(s->device_current && s->device.has_value(),
           "tenstorrent: EnsureHost with no current device or host copy");
  if (std::getenv("VT_TT_SLOT_TRACE") != nullptr)
    std::fprintf(stderr,
                 "[TT-SLOT] ensure t=%p numel=%" PRId64
                 " slot=%p bytes=%zu dev=%ux%u dt=%d ht=%d dc=%d ct=%d\n",
                 static_cast<const void*>(t.data), t.Numel(),
                 static_cast<void*>(s->host), s->bytes, s->dev_rows,
                 s->dev_cols, static_cast<int>(s->device->dtype()),
                 s->host_current, s->device_current, s->conv_transposed);
  if (s->conv_transposed) {
    std::vector<float> v = s->device->to_vector<float>();
    for (int64_t i = 0; i < t.Numel(); ++i) {
      // host[s*C*sl + c*sl + j] == device[j*R + s*C + c]
      const int64_t j = i % s->conv_sl, rc = i / s->conv_sl;
      const int64_t col = rc % s->conv_c, slot = rc / s->conv_c;
      StoreElemF32(t, i, v[static_cast<size_t>(j * (s->conv_slots * s->conv_c) +
                                               slot * s->conv_c + col)]);
    }
    s->host_current = true;
    return;
  }
  // Interior contiguous view of the owner slot: the host tensor is a window
  // (a strided row view, a rank-3 activation) whose bytes live at one flat
  // span of the owner's device plane. Download the owner once and copy the
  // span — the owner plane is what the device holds, the span is what this
  // view owns. Without this branch EnsureHost compared the OWNER's numel
  // against the WINDOW's and refused by name (the vllm-bench chunked-prefill
  // GdnPrefillKernel readback, ISSUE-LOCAL-01M2E5F69CMWDERKXG32YY9P8N). A
  // non-contiguous view is not served — its bytes are not one span.
  const int64_t total_dev =
      static_cast<int64_t>(s->dev_rows) * s->dev_cols;
  // A slot whose registered allocation does not COVER this view is not this
  // view's owner: the slot map is never erased (#1486) and a freed tensor's
  // range can be re-allocated to an unrelated later tensor (the vllm-bench
  // prefill chunk buffers). Treat it as untracked host memory — the bytes
  // are whoever the current producer wrote.
  const int64_t t_elem =
      t.dtype == DType::kF32 ? 4 : 2;  // host-side float storages
  const uintptr_t slot_base = reinterpret_cast<uintptr_t>(s->host);
  if (reinterpret_cast<uintptr_t>(t.data) + t.Numel() * t_elem >
      slot_base + s->bytes)
    return;
  if (total_dev != t.Numel()) {
    // A device plane SMALLER than the view cannot back the view's bytes at
    // all — the slot's shadow is a stale per-step commit (the captured
    // decode web commits one token's plane into a chunk-sized arena slot;
    // the prefill chunk then reuses the arena and its producer writes the
    // host bytes directly). The host bytes are the truth here: read them,
    // touch nothing on the slot (ISSUE-LOCAL-01M2E5F69CMWDERKXG32YY9P8N).
    if (total_dev < t.Numel()) return;
    VT_CHECK(t.IsContiguous(),
             "tenstorrent: EnsureHost on a non-contiguous window of a "
             "device-authoritative owner is not served");
    const int64_t elem_bytes =
        s->device->dtype() == ttnn::DataType::FLOAT32 ? 4 : 2;
    const int64_t delta = static_cast<const char*>(t.data) -
                          static_cast<const char*>(s->host);
    if (!(delta >= 0 && delta % elem_bytes == 0 &&
          delta / elem_bytes + t.Numel() <= total_dev)) {
      std::fprintf(stderr,
                   "[TT-WINDOW] reader t=%p rank=%d shape=%" PRId64 "x%" PRId64
                   "x%" PRId64 "x%" PRId64 " strides=%" PRId64 "x%" PRId64
                   "x%" PRId64 "x%" PRId64 " dt=%d delta=%" PRId64
                   " numel=%" PRId64 " dev=%s total_dev=%" PRId64 "\n",
                   static_cast<const void*>(t.data), t.rank, t.shape[0],
                   t.shape[1], t.shape[2], t.shape[3], t.stride[0], t.stride[1],
                   t.stride[2], t.stride[3], static_cast<int>(t.dtype), delta,
                   t.Numel(), DevShapeStr(*s->device).c_str(), total_dev);
      void* bt[10];
      const int nbt = ::backtrace(bt, 10);
      char** sym = ::backtrace_symbols(bt, nbt);
      for (int i = 0; sym != nullptr && i < nbt; ++i)
        std::fprintf(stderr, "[TT-WINDOW]   bt[%d]=%p %s\n", i, bt[i], sym[i]);
      std::fflush(stderr);
      std::free(sym);
    }
    VT_CHECK(delta >= 0 && delta % elem_bytes == 0 &&
                 delta / elem_bytes + t.Numel() <= total_dev,
             "tenstorrent: EnsureHost window is not an in-bounds flat span "
             "of the owner slot (delta=" + std::to_string(delta) +
                 " elem_bytes=" + std::to_string(elem_bytes) +
                 " off=" + std::to_string(delta / elem_bytes) +
                 " numel=" + std::to_string(t.Numel()) +
                 " total_dev=" + std::to_string(total_dev) +
                 " dev=" + DevShapeStr(*s->device) + ")");
    const int64_t off = delta / elem_bytes;
    std::vector<float> v = s->device->to_vector<float>();
    for (int64_t i = 0; i < t.Numel(); ++i)
      StoreElemF32(t, i, v[static_cast<size_t>(off + i)]);
    // The OWNER slot's host buffer is NOT filled — only this window's bytes
    // are. Leave the slot device-authoritative (host_current stays false) so
    // no consumer reads stale owner bytes outside the span.
    return;
  }
  DownloadToHost(*s->device, t, "EnsureHost");
  s->host_current = true;
}

void EnsureHost(const Tensor& t) {
  // const_cast: host bytes are filled in place; logical tensor is unchanged.
  EnsureHost(const_cast<Tensor&>(t));
}

std::vector<float> ToHostF32(const Tensor& t) {
  EnsureHost(t);
  const int64_t n = t.Numel();
  std::vector<float> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) host[static_cast<size_t>(i)] = LoadElemF32(t, i);
  return host;
}

ttnn::Tensor UploadRows(const float* data, uint32_t rows, uint32_t cols, MeshDevice& device) {
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] UploadRows ptr=%p rows=%u cols=%u\n",
                 static_cast<const void*>(data), rows, cols);
  std::vector<float> host(data, data + static_cast<size_t>(rows) * cols);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] UploadRows from_vector WRITE during capture\n");
  return ttnn::Tensor::from_vector<float>(host, TileSpecOf(rows, cols), &device);
}

// ---- BACKEND-TENSTORRENT-QWEN35 W4 (#2107): bulk staging counters ----
// Incremented by EnsureDevice2D's two staging routes; read through the
// tenstorrent_device.h API (GetStagingStats below) so the W4 route pin is
// evidence, not assumption. At vt::tenstorrent scope so both the staging
// paths above and the readers below see them.
std::atomic<uint64_t>& StagingBulkUploads() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingBulkBytes() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingF32Elems() {
  static std::atomic<uint64_t> v{0};
  return v;
}
// W5 (#2244): the persistent-route counters — in-place mesh CQ writes into
// the per-slot buffer, the (re)allocations of that buffer, and the bytes
// pushed through it. Read through GetStagingStats below.
std::atomic<uint64_t>& StagingPersistentWrites() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingPersistentAllocs() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingPersistentBytes() {
  static std::atomic<uint64_t> v{0};
  return v;
}
// BACKEND-TENSTORRENT-QWEN35 W7 (#2282): staging writes the precise-residency
// arms eliminate, one counter per class — reservation (EnsureDevice2D serving
// a pool-acquired block from its resident allocation), device_memset
// (MemsetDeviceFill keeping the shadow across an eager full-slot zero-fill),
// device_copy (CopyDeviceDeviceIfResident skipping the download+restage pair).
std::atomic<uint64_t>& StagingAvoidedReservation() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingAvoidedMemset() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingAvoidedDeviceCopy() {
  static std::atomic<uint64_t> v{0};
  return v;
}

// W4 lever 1 (#2107): bulk upload of a contiguous bf16 master. The host
// bytes ARE the payload: one from_span over the tensor's own memory — no f32
// intermediate, no per-element dtype dispatch, no extra vector copy, and no
// bfloat16::from_float round-trip on the ttnn side. Bit-identical to the f32
// path: bf16→f32 widening is exact, and packing a value whose low 16 mantissa
// bits are zero back to bf16 returns the same bits under any rounding rule.
//
// W5 (#2244): the upload no longer pays tt-metal's per-upload creation path
// on every step. A tracked base slot stages through its PERSISTENT device
// buffer: the first staging for a geometry runs the full from_span creation
// (and the buffer stays resident in the slot); every later staging packs the
// host bytes with the SAME function from_span calls (tt-metal
// host_tensor_from_span_with_pad_value, ttnn/core/tensor/tensor.cpp:170) and
// writes them through tt-metal's in-place H2D — ttnn::copy_to_device into the
// resident MeshTensor, which reaches MeshCommandQueue::enqueue_write /
// enqueue_write_shards against the existing buffer. No fresh MeshBuffer, no
// cluster/chip rediscovery, no new tensor attributes. The bytes on the device
// are the same packed bytes from_span writes, into a buffer of the same
// geometry, fully overwritten each time: bit-identical. Untracked pointers
// and interior views keep the anonymous from_span arm (W2c: a view must
// never store against the base slot).
ttnn::Tensor UploadRowsBf16(const Tensor& t, uint32_t rows, uint32_t cols,
                            MeshDevice& device) {
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] UploadRowsBf16 from_span WRITE during capture ptr=%p rows=%u cols=%u\n",
                 (const void*)t.data, rows, cols);
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  // The bytes at t.Ptr are the window's own bf16 bits (bfloat16 is a 2-byte
  // class wrapping the same uint16 pattern).
  const bfloat16* src = reinterpret_cast<const bfloat16*>(t.Ptr<uint16_t>());
  // Short locked probe: resolve the persistent buffer once. The CQ write runs
  // OUTSIDE the lock — the same probe/upload/re-lock discipline EnsureDevice2D
  // uses — and the copied handle (shared TensorAttributes) keeps the resident
  // MeshBuffer alive even if the slot is unregistered mid-upload.
  std::optional<ttnn::Tensor> persistent;
  bool tracked_base = false;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    tracked_base = (s != nullptr && t.data == s->host);
    if (tracked_base && s->persistent.has_value() &&
        s->persist_rows == rows && s->persist_cols == cols) {
      persistent = s->persistent;
    }
  }
  if (!tracked_base) {
    // Untracked pointer or interior view (W2c): anonymous staging, no
    // persistent buffer, no W5 counters — the caller's bulk counters still see it.
    return ttnn::Tensor::from_span(ttsl::Span<const bfloat16>(src, n),
                                   TileSpecOf(rows, cols), &device);
  }
  if (persistent.has_value()) {
    // In-place arm. Host half: exactly the packing from_span performs
    // (tt-metal ttnn/core/tensor/tensor.cpp:170 — same function, same spec,
    // same pad), with no device argument so no device work happens. Device
    // half: tt-metal's own in-place H2D (ttnn/core/tensor/tensor_ops.cpp:161
    // copy_to_device → enqueue_write_tensor into the EXISTING MeshTensor,
    // tt_metal/impl/tensor/tensor_apis.cpp:149) — the same write path
    // from_span's to_device takes, minus the fresh MeshBuffer allocation and
    // tensor creation. Bytes on the device are the same packed bytes, into a
    // buffer of the same geometry, fully overwritten: bit-identical.
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-UP] UploadRowsBf16 persistent enqueue_write during capture ptr=%p rows=%u cols=%u\n",
                   (const void*)t.data, rows, cols);
    ttnn::Tensor host = ttnn::Tensor::from_span(ttsl::Span<const bfloat16>(src, n),
                                                TileSpecOf(rows, cols),
                                                /*device=*/nullptr);
    ttnn::copy_to_device(host, *persistent);
    StagingPersistentWrites().fetch_add(1, std::memory_order_relaxed);
    StagingPersistentBytes().fetch_add(static_cast<uint64_t>(n) * 2,
                                       std::memory_order_relaxed);
    return *persistent;
  }
  // Allocating arm (cold slot or staging-geometry change): the full W4
  // creation path, and the returned tensor becomes the slot's persistent
  // buffer. The stale resident buffer, if any, is released here.
  ttnn::Tensor dev = ttnn::Tensor::from_span(ttsl::Span<const bfloat16>(src, n),
                                             TileSpecOf(rows, cols), &device);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && t.data == s->host) {
      s->persistent = dev;
      s->persist_rows = rows;
      s->persist_cols = cols;
    }
  }
  StagingPersistentWrites().fetch_add(1, std::memory_order_relaxed);
  StagingPersistentAllocs().fetch_add(1, std::memory_order_relaxed);
  StagingPersistentBytes().fetch_add(static_cast<uint64_t>(n) * 2,
                                     std::memory_order_relaxed);
  return dev;
}

std::mutex& WeightViewMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, WeightViewShadow>& WeightViewShadows() {
  static std::map<uintptr_t, WeightViewShadow>* m =
      new std::map<uintptr_t, WeightViewShadow>(); // never destroyed (#1486)
  return *m;
}

// True iff `t` is a rank-2 contiguous tensor whose pointer is a tracked BASE
// (EnsureDevice2D's fast-path condition). Views and untracked pointers are
// the ones EnsureDevice2D stages anonymously.
bool IsTrackedBase2D(const Tensor& t) {
  if (t.rank != 2 || !t.IsContiguous()) return false;
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  return s != nullptr && t.data == s->host;
}

// Persistent device staging for an interior/untracked BF16 weight view.
// Stages the window's own bytes once (the same from_span the anonymous arm
// runs, so the packed device bytes are identical) and reuses the resident
// tensor from then on — capture-safe and eager-cheap.
ttnn::Tensor EnsureWeightViewDevice(const Tensor& t, MeshDevice& device) {
  VT_CHECK(t.rank == 2 && t.IsContiguous() && t.dtype == DType::kBF16,
           "tenstorrent EnsureWeightViewDevice: contiguous rank-2 bf16");
  const uint32_t rows = static_cast<uint32_t>(t.shape[0]);
  const uint32_t cols = static_cast<uint32_t>(t.shape[1]);
  const uintptr_t key = reinterpret_cast<uintptr_t>(t.data);
  {
    std::lock_guard<std::mutex> g(WeightViewMutex());
    auto it = WeightViewShadows().find(key);
    if (it != WeightViewShadows().end() && it->second.device.has_value() &&
        it->second.rows == rows && it->second.cols == cols) {
      return *it->second.device;
    }
  }
  EnsureHost(t);
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr,
                 "[TT-UP] EnsureWeightViewDevice from_span WRITE during "
                 "capture ptr=%p rows=%u cols=%u\n",
                 t.data, rows, cols);
  ttnn::Tensor dev = ttnn::Tensor::from_span(
      ttsl::Span<const bfloat16>(static_cast<const bfloat16*>(t.data), n),
      TileSpecOf(rows, cols), &device);
  // The view stage IS a bulk bf16 staging: report it through the same
  // counters EnsureDevice2D's bf16 arm bumps, so the W4 contract ("each view
  // stage is its own bulk upload") holds for the shadow arm too. Later calls
  // serve the resident shadow and, like a resident serve there, count nothing.
  StagingBulkUploads().fetch_add(1, std::memory_order_relaxed);
  StagingBulkBytes().fetch_add(static_cast<uint64_t>(n) * 2,
                               std::memory_order_relaxed);
  std::lock_guard<std::mutex> g(WeightViewMutex());
  WeightViewShadow& s = WeightViewShadows()[key];
  s.device = dev;
  s.rows = rows;
  s.cols = cols;
  return dev;
}

// MatmulBT's weight staging: tracked bases keep EnsureDevice2D's fast path;
// interior views get the persistent shadow instead of the per-call anonymous
// staging.
ttnn::Tensor EnsureMatmulWeightDevice(const Tensor& b, MeshDevice& device) {
  // One lever selects the residency; a future variant is a new VALUE of
  // VT_TT_WEIGHT_RESIDENCY, not a new flag.
  switch (vllm::ParseTtWeightResidency(
      std::getenv("VT_TT_WEIGHT_RESIDENCY"))) {
    case vllm::TtWeightResidency::kBfp8:
      return EnsureBfp8WeightDevice(b, device);
    case vllm::TtWeightResidency::kBfp4:
      NoteBfp8Refusal("VT_TT_WEIGHT_RESIDENCY=bfp4: not implemented; staged "
                      "bf16");
      break;
    case vllm::TtWeightResidency::kOff:
      break;
  }
  if (b.dtype == DType::kBF16 && !IsTrackedBase2D(b))
    return EnsureWeightViewDevice(b, device);
  return EnsureDevice2D(b, device);
}

// ---- Weight residency ----------------------------------------------------
// spec .agents/specs/tenstorrent-bfp-weight-residency.md. DEFAULT OFF
// (VT_TT_WEIGHT_RESIDENCY=off / unset leaves the byte-identical bf16 staging
// path —
// the inertness gate pins that). When bfp8, a bf16 matmul weight uploads once in
// bf16, is TYPECAST on device to BFLOAT8_B (BFP8: 1 sign + 7 shared-group
// mantissa bits per element, one 8-bit exponent per 16-element group —
// tt_metal/impl/data_format/blockfloat_common.cpp `convert_bfp_to_u32`,
// Bfp8_b arm), and the bf16 staging is dropped: exactly the expand-bf16
// conversion shape, no second full-precision device copy. The native
// ttnn::matmul consumes it (bf16 activation x bfloat8_b weight is a supported
// operand pair — ttnn/operations/matmul/matmul.cpp:521-532).

std::atomic<uint64_t> g_bfp8_resident{0};
std::atomic<uint64_t> g_bfp8_uses{0};
std::atomic<uint64_t> g_bfp8_refusals{0};
std::mutex& Bfp8RefusalMutex() {
  static std::mutex m;
  return m;
}
std::string& Bfp8RefusalMsg() {
  static std::string* s = new std::string();  // never destroyed (#1486)
  return *s;
}

bool Bfp8WeightsEnabled() {
  return vllm::ParseTtWeightResidency(std::getenv("VT_TT_WEIGHT_RESIDENCY")) ==
         vllm::TtWeightResidency::kBfp8;
}
uint64_t Bfp8ResidentWeights() { return g_bfp8_resident.load(std::memory_order_relaxed); }
uint64_t Bfp8MatmulUses() { return g_bfp8_uses.load(std::memory_order_relaxed); }
uint64_t Bfp8Refusals() { return g_bfp8_refusals.load(std::memory_order_relaxed); }
void Bfp8MatmulUse() { g_bfp8_uses.fetch_add(1, std::memory_order_relaxed); }
const char* Bfp8LastRefusal() {
  std::lock_guard<std::mutex> g(Bfp8RefusalMutex());
  return Bfp8RefusalMsg().c_str();
}
void NoteBfp8Refusal(std::string reason) {
  g_bfp8_refusals.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> g(Bfp8RefusalMutex());
  Bfp8RefusalMsg() = std::move(reason);
}

// Keyed by the weight's host base pointer — the EnsureWeightViewShadow
// persistent-shadow pattern. An interior view carries its own pointer, so a
// differently-offset view stages its own BFP8 tensor and never consumes
// another slice's bytes (the Qwen3.5 BA interior-view fatality).
struct Bfp8WeightShadow {
  std::optional<ttnn::Tensor> device;
  uint32_t rows = 0, cols = 0;
};
std::mutex& Bfp8WeightMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, Bfp8WeightShadow>& Bfp8WeightShadows() {
  static std::map<uintptr_t, Bfp8WeightShadow>* m =
      new std::map<uintptr_t, Bfp8WeightShadow>();  // never destroyed (#1486)
  return *m;
}

ttnn::Tensor EnsureBfp8WeightDevice(const Tensor& b, MeshDevice& device) {
  const uint32_t rows = static_cast<uint32_t>(b.shape[0]);
  const uint32_t cols = static_cast<uint32_t>(b.shape[1]);
  const auto refuse = [&](std::string why) {
    NoteBfp8Refusal(std::move(why));
    if (b.dtype == DType::kBF16 && !IsTrackedBase2D(b))
      return EnsureWeightViewDevice(b, device);
    return EnsureDevice2D(b, device);
  };
  if (b.dtype != DType::kBF16)
    return refuse("tenstorrent BFP8 weight residency: weight is not bf16 (" +
                  std::to_string(static_cast<int>(b.dtype)) + ")");
  if (b.rank != 2 || !b.IsContiguous())
    return refuse("tenstorrent BFP8 weight residency: non-contiguous or "
                  "non-rank-2 weight");
  if (rows % 32 != 0 || cols % 32 != 0)
    return refuse("tenstorrent BFP8 weight residency: shape " +
                  std::to_string(rows) + "x" + std::to_string(cols) +
                  " is not TILE-aligned");
  const uintptr_t key = reinterpret_cast<uintptr_t>(b.data);
  {
    std::lock_guard<std::mutex> g(Bfp8WeightMutex());
    auto it = Bfp8WeightShadows().find(key);
    if (it != Bfp8WeightShadows().end() && it->second.device.has_value() &&
        it->second.rows == rows && it->second.cols == cols) {
      return *it->second.device;
    }
  }
  EnsureHost(b);
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  ttnn::Tensor bf16_dev = ttnn::Tensor::from_span(
      ttsl::Span<const bfloat16>(static_cast<const bfloat16*>(b.data), n),
      TileSpecOf(rows, cols), &device);
  // The device typecast is the SAME single RNE BFP pack the tt-metal host
  // packer performs (typecast_sharded_program_factory.cpp:48 — BFLOAT16 ->
  // BFLOAT8_B is a supported TILE conversion), executed on-device so the
  // packed operand never exists on the host and the bf16 staging is dropped
  // with the temporary.
  ttnn::Tensor bfp8_dev = ttnn::typecast(bf16_dev, ttnn::DataType::BFLOAT8_B);
  g_bfp8_resident.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> g(Bfp8WeightMutex());
  Bfp8WeightShadow& s = Bfp8WeightShadows()[key];
  s.device = bfp8_dev;
  s.rows = rows;
  s.cols = cols;
  return bfp8_dev;
}


// Return a TILE BFLOAT16 device tensor for rank-2 `t`, uploading only when the
// device shadow is missing or stale. Same-numel reshape reuses a resident
// shadow without host round-trip — needed so qk-RmsNorm on a [T*H, Dh] view
// can consume QkvSplit's [T, H*Dh] device result.
ttnn::Tensor EnsureDevice2D(const Tensor& t, MeshDevice& device) {
  VT_CHECK(t.rank == 2 && t.IsContiguous(),
           "tenstorrent: EnsureDevice2D expects contiguous rank-2");
  const uint32_t rows = static_cast<uint32_t>(t.shape[0]);
  const uint32_t cols = static_cast<uint32_t>(t.shape[1]);
  // The per-slot cache describes the BASE allocation. An interior view
  // (e.g. packed_weight.Slice(0, Hv, 2*Hv) fed to a matmul) resolves to the
  // base slot but must NEVER hit nor store against it: the cached tensor is
  // the BASE's staging, and returning it for a differently-offset view makes
  // the consumer read another slice's bytes (Qwen3.5 BA: TT computed the `a`
  // projection with the `b` weight rows — BACKEND-TENSTORRENT-QWEN35 W2c).
  bool tracked_base = false;
  bool need_host_refresh = false;
  bool reserved_base = false;
  {
    // W4 lever 2 (#2107): ONE locked probe resolves everything this call
    // needs — the tracked fast paths (exact-shape hit and same-numel
    // reshape), the interior-view refusal, and whether the base's host
    // master must be pulled back from the device before its bytes are read.
    // The pre-W4 shape re-probed the slot map under the mutex up to four
    // times per call.
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    tracked_base = (s != nullptr && t.data == s->host);
    reserved_base = tracked_base && s->device_reserved;
    if (tracked_base && s->device_current && s->device.has_value()) {
      if (s->dev_rows == rows && s->dev_cols == cols) {
        const auto ls = s->device->logical_shape();
        if (ls.rank() == 2 && ls[0] == rows && ls[1] == cols)
          return *s->device;
        if (tt_capture_active()) {
          s->device = CaptureSafeReshape(*s->device, ttnn::Shape({rows, cols}));
          return *s->device;
        }
        ttnn::Tensor reshaped = ttnn::to_layout(
            ttnn::reshape(ttnn::to_layout(*s->device, ttnn::Layout::ROW_MAJOR),
                          ttnn::Shape({rows, cols})),
            s->device->layout());
        s->device = reshaped;
        return *s->device;
      }
      const uint64_t have =
          static_cast<uint64_t>(s->dev_rows) * static_cast<uint64_t>(s->dev_cols);
      const uint64_t want = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
      if (have == want) {
        if (tt_capture_active()) {
          ttnn::Tensor reshaped =
              CaptureSafeReshape(*s->device, ttnn::Shape({rows, cols}));
          s->device = reshaped;
          s->dev_rows = rows;
          s->dev_cols = cols;
          return *s->device;
        }
        ttnn::Tensor reshaped =
            ttnn::reshape(*s->device, ttnn::Shape({rows, cols}));
        s->device = reshaped;
        s->dev_rows = rows;
        s->dev_cols = cols;
        return *s->device;
      }
    }
    // Interior view over a base whose host master is neither current nor
    // device-refreshable would read stale bytes — refuse loudly, the same
    // state the post-refresh check below the old probe observed.
    need_host_refresh = (s != nullptr && s->device_current && !s->host_current);
    VT_CHECK(tracked_base || s == nullptr || need_host_refresh || s->host_current,
             "tenstorrent: EnsureDevice2D interior view of a device-current "
             "base is unsupported (would read stale bytes); stage via the "
             "base tensor");
  }
  // The reservation arm is bf16-only: it serves the slot's W5 persistent
  // staging buffer (BFLOAT16/TILE) or a fresh BFLOAT16 empty, so a master of
  // any other dtype taking the arm would receive a bf16 device tensor for its
  // declared dtype. A non-bf16 master whose first device use lands on an
  // acquired block falls through to the normal staging path below, whose f32
  // arm keeps the declared dtype (the W7 invariant).
  if (reserved_base && t.dtype == DType::kBF16) {
    // W7 (#2282) reservation: see BufferSlot::device_reserved. The previous
    // tenant's bytes sit on BOTH sides and the new tenant's pending device op
    // overwrites the buffer it is served — stage NOTHING. Serve the resident
    // persistent allocation at the same geometry, or hand out an empty one at
    // a new geometry (no bytes move in either direction). Restage semantics:
    // the service ALIASES the slot's persistent buffer in place (W5), it does
    // not create a fresh snapshot. The re-check under the lock covers a host
    // write that raced the probe: once the bytes are real, the upload
    // contract applies again.
    bool served = false;
    ttnn::Tensor dev;
    {
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(t.data);
      // The reservation describes a slot whose bytes are still the previous
      // tenant's on both sides. A live shadow means some producer already
      // established the content — serving the resident persistent buffer here
      // would discard it and hand the consumer stale bytes (the W7 drift,
      // #2282). Refuse: the normal paths below serve or restage the truth.
      if (s != nullptr && t.data == s->host && s->device_reserved &&
          !s->device_current) {
        if (s->persistent.has_value() && s->persist_rows == rows &&
            s->persist_cols == cols) {
          dev = *s->persistent;
        } else {
          dev = ttnn::empty(ttnn::Shape({rows, cols}), ttnn::DataType::BFLOAT16,
                            ttnn::Layout::TILE, &device, ttnn::MemoryConfig{});
        }
        s->device = dev;
        s->dev_rows = rows;
        s->dev_cols = cols;
        s->device_current = true;
        s->host_current = false;
        s->device_reserved = false;
        StagingAvoidedReservation().fetch_add(1, std::memory_order_relaxed);
        served = true;
      }
    }
    if (served) return dev;
  }
  if (need_host_refresh) EnsureHostBytes(t.data);
  // HOST-FREE-FORWARD defect: this loop consumed HOST bytes without checking
  // slot residency. Under host-free decode a producer commits device-only
  // (host_current=false), so the next consumer staging through here uploaded
  // pool-fresh zeros and then marked the slot host_current=true — corrupting
  // both the value and the record. Refresh the host bytes from the device
  // shadow before reading them.
  EnsureHostBytes(t.data);
  ttnn::Tensor dev;
  if (t.dtype == DType::kBF16) {
    // W4 lever 1 (#2107): contiguous bf16 masters skip the f32 intermediate
    // entirely — the raw window bytes go to the device in one from_span.
    dev = UploadRowsBf16(t, rows, cols, device);
    StagingBulkUploads().fetch_add(1, std::memory_order_relaxed);
    StagingBulkBytes().fetch_add(static_cast<uint64_t>(rows) * cols * 2,
                                 std::memory_order_relaxed);
  } else {
    // f16/f32 masters keep the f32 reference arm (genuine conversion); the
    // element count is hoisted — the old loop evaluated Numel() per element.
    const int64_t n = t.Numel();
    std::vector<float> host(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
      host[static_cast<size_t>(i)] = LoadElemF32(t, i);
    dev = UploadRows(host.data(), rows, cols, device);
    StagingF32Elems().fetch_add(static_cast<uint64_t>(n),
                                std::memory_order_relaxed);
  }
  if (HostFreeDecodeEnabled()) {
    // Prime the persistent-zero cache for this spec during the eager warmup
    // (capture-safe zeroing replays ttnn::copy(zero, dst) — see MemsetDevice).
    ZeroCachePrime(ttnn::Shape({rows, cols}), ttnn::DataType::BFLOAT16,
                   ttnn::Layout::TILE, device);
  }
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s != nullptr && t.data == s->host) {
    s->device = dev;
    s->dev_rows = rows;
    s->dev_cols = cols;
    s->device_current = true;
    s->host_current = true;
  }
  return dev;
}

// (StageWeightBf16ForTest is defined at vt::tenstorrent scope below — it
// needs external linkage for the focused test and calls EnsureDevice2D,
// which stays TU-internal here.)

// DEBUG (BACKEND-TENSTORRENT-QWEN35 W2c): ensure the tensor is staged on
// device exactly as a consuming kernel would, then read the DEVICE copy back.
// Comparing this against the host master exposes staging corruption that a
// host-side dump cannot see.
// True when `t` already has a device-resident shadow matching [rows, cols]
// (exact shape). Used to pick device vs host kernels without forcing upload.
bool DeviceShadowExact(const Tensor& t, uint32_t rows, uint32_t cols) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  return s != nullptr && s->device_current && s->device.has_value() &&
         s->dev_rows == rows && s->dev_cols == cols;
}

// to host (the residency win). Host is marked stale until EnsureHost.
// Device tensor is stored as logical [rows, cols] TILE (may differ from out's
// rank-3 view as long as numel matches) so a later Reshape+EnsureDevice2D hits.
void CommitDeviceLogical2D(Tensor& out, ttnn::Tensor dev, uint32_t rows, uint32_t cols) {
  VT_CHECK(out.IsContiguous(), "tenstorrent: CommitDeviceLogical2D expects contiguous out");
  VT_CHECK(out.Numel() == static_cast<int64_t>(rows) * static_cast<int64_t>(cols),
           "tenstorrent: CommitDeviceLogical2D numel mismatch");
  {
    int64_t vol = 1;
    const auto ds = dev.logical_shape();
    for (uint32_t i = 0; i < ds.rank(); ++i) vol *= ds[i];
    VT_CHECK(vol == static_cast<int64_t>(rows) * static_cast<int64_t>(cols),
             std::string("tenstorrent: CommitDeviceLogical2D device volume ") +
                 std::to_string(vol) + " != rows*cols " +
                 std::to_string(rows * cols));
  }
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(out.data);
  if (std::getenv("VT_TT_SLOT_TRACE") != nullptr) {
    const auto ds = dev.logical_shape();
    int64_t vol = 1;
    for (uint32_t i = 0; i < ds.rank(); ++i) vol *= ds[i];
    std::fprintf(stderr,
                 "[TT-SLOT] commit out=%p rows=%u cols=%u tracked=%d vol=%" PRId64
                 " dt=%d\n",
                 static_cast<const void*>(out.data), rows, cols,
                 s != nullptr ? 1 : 0, vol, static_cast<int>(dev.dtype()));
  }
  if (s == nullptr) {
    // Untracked buffer (e.g. stack/test scratch): fall back to host write.
    DownloadToHost(dev, out, "CommitDeviceLogical2D(untracked)");
    return;
  }
  s->device = std::move(dev);
  s->dev_rows = rows;
  s->dev_cols = cols;
  s->device_current = true;
  s->host_current = false;
  s->conv_transposed = false;  // logical [rows, cols] — oracle layout
  s->device_reserved = false;  // real bytes committed — the reservation is spent
}



// Llama-3 frequency rescale (cpu_ops Llama3ScaleFreq); no-op when scaling_factor
// is unset. Kept so Qwen3 / Llama rope paths share one host implementation.
double Llama3ScaleFreq(double freq, const RopeArgs& a) {
  const double sf = static_cast<double>(a.llama3_scaling_factor);
  if (!(sf > 0.0)) return freq;
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const double lo = static_cast<double>(a.llama3_low_freq_factor);
  const double hi = static_cast<double>(a.llama3_high_freq_factor);
  const double omax = static_cast<double>(a.llama3_orig_max_position);
  const double low_freq_wavelen = omax / lo;
  const double high_freq_wavelen = omax / hi;
  const double wave_len = kTwoPi / freq;
  double smooth = 0.0;
  if (lo != hi) smooth = (omax / wave_len - lo) / (hi - lo);
  if (wave_len < high_freq_wavelen) return freq;
  if (wave_len > low_freq_wavelen) return freq / sf;
  return (1.0 - smooth) * freq / sf + smooth * freq;
}

// Expand per-token cos|sin [T, half] to per-(token,head) [T*H, half] for a
// device NeoX apply over the flat [T*H, D] view of qs/ks.
void ExpandCosSinPerHead(const float* cos_t, const float* sin_t, int64_t tokens,
                         int64_t heads, int64_t half, std::vector<float>& cos_exp,
                         std::vector<float>& sin_exp) {
  cos_exp.resize(static_cast<size_t>(tokens * heads * half));
  sin_exp.resize(static_cast<size_t>(tokens * heads * half));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      const size_t dst = static_cast<size_t>((t * heads + h) * half);
      const size_t src = static_cast<size_t>(t * half);
      std::memcpy(cos_exp.data() + dst, cos_t + src, static_cast<size_t>(half) * sizeof(float));
      std::memcpy(sin_exp.data() + dst, sin_t + src, static_cast<size_t>(half) * sizeof(float));
    }
  }
}

// Build per-token cos|sin from RopeNeox frequencies (double angles, f32 store).
void BuildCosSinFromPositions(const Tensor& pos, int64_t tokens, int rot, double base,
                              const RopeArgs& args, std::vector<float>& cos_t,
                              std::vector<float>& sin_t) {
  EnsureHost(pos);
  const int64_t half = rot / 2;
  cos_t.resize(static_cast<size_t>(tokens * half));
  sin_t.resize(static_cast<size_t>(tokens * half));
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t p =
        pos.dtype == DType::kI32 ? pos.Ptr<int32_t>()[t] : pos.Ptr<int64_t>()[t];
    for (int64_t i = 0; i < half; ++i) {
      double freq = std::pow(base, -2.0 * static_cast<double>(i) / static_cast<double>(rot));
      freq = Llama3ScaleFreq(freq, args);
      const double angle = static_cast<double>(p) * freq;
      cos_t[static_cast<size_t>(t * half + i)] = static_cast<float>(std::cos(angle));
      sin_t[static_cast<size_t>(t * half + i)] = static_cast<float>(std::sin(angle));
    }
  }
}


std::atomic<int64_t>& LastTraceBytes() {
  static std::atomic<int64_t> v{0};
  return v;
}
int64_t LastTraceBytesForTest() {
  return LastTraceBytes().load(std::memory_order_relaxed);
}

// W4a wave-3b-1 residency-policy probes — the contract lives in
// tenstorrent_device.h. Each reads its map under its own mutex; a nullptr or
// absent host reads false. Defined OUTSIDE the anonymous namespace (the
// wave-3a hook pattern): a -Werror=unused-function TU would reject an
// anonymous-namespace helper no other file-local code calls.
bool EmbedTableShadowPresentForTest(const void* host) {
  if (host == nullptr) return false;
  std::lock_guard<std::mutex> g(EmbedTableMutex());
  return EmbedTableShadows().find(reinterpret_cast<uintptr_t>(host)) !=
         EmbedTableShadows().end();
}

// ---- Called from TenstorrentBackend::Alloc/Free/Copy (no ttnn in that TU). ----

void RegisterHostBuffer(void* host, size_t bytes) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(SlotMutex());
  if (std::getenv("VT_TT_SLOT_TRACE") != nullptr)
    std::fprintf(stderr, "[TT-SLOT] register %p bytes=%zu\n", host, bytes);
  BufferSlot s;
  s.host = host;
  s.bytes = bytes;
  s.host_current = true;
  s.device_current = false;
  Slots()[reinterpret_cast<uintptr_t>(host)] = std::move(s);
}

void UnregisterHostBuffer(void* host) {
  if (host == nullptr) return;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    if (std::getenv("VT_TT_SLOT_TRACE") != nullptr)
      std::fprintf(stderr, "[TT-SLOT] unregister %p\n", host);
    Slots().erase(reinterpret_cast<uintptr_t>(host));
  }
  DropPagedKvShadow(host);
  DropEmbedTableShadow(host);
  DropDecodedWeightShadow(host);
  DropKeepQuantWordShadow(host);
  DropGroupedActShadow(host);
}

void MarkHostWritten(void* host) {
  if (host == nullptr) return;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s != nullptr) {
      s->host_current = true;
      s->device_current = false;
      s->device = std::nullopt;
      s->conv_transposed = false;
      s->device_reserved = false;  // real bytes now — the upload contract applies
    }
  }
  // Weight tables may be rewritten in place during load — drop embed cache.
  DropEmbedTableShadow(host);
  // ... and the PACKED word shadow: a weight re-staged in place must not keep
  // serving the pre-rewrite words (same in-place-load hazard as the embed twin).
  DropKeepQuantWordShadow(host);
}

void MarkScratchAcquired(void* host) {
  if (host == nullptr) return;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s != nullptr) {
      // The state MarkHostWritten registers, plus the W7 reservation: the
      // bytes on BOTH sides belong to the previous tenant. host_current stays
      // true so host reads behave exactly as before (stale bytes served);
      // only the first device stage is gated.
      s->host_current = true;
      s->device_current = false;
      s->device = std::nullopt;
      s->conv_transposed = false;
      s->device_reserved = true;
    }
  }
  // Same drop MarkHostWritten performed for every caller — a pool block must
  // never serve a stale embed-table staging either.
  DropEmbedTableShadow(host);
}

// TRUSTED dump (BACKEND-TENSTORRENT-QWEN35 W2c measurement reset): whole
// tensors only, typed header, DUAL-READ verified. Two independent
// Synchronize+Copy passes must agree byte-for-byte or the file records
// verified=0 (a mismatch means residency state changed under us and the
// payload must not be trusted). File layout: magic 'TDMP', dtype u32,
// rank u32, dims[4] u32, numel u64, verified u8, payload.
void TrustDump(Queue& q, const char* dir, const char* name, const Tensor& t) {
  static std::atomic<uint64_t> seq{0};
  // Defense-in-depth refresh (review finding F2, #1715): Backend::Copy below
  // already runs EnsureHostBytes on the source before its memcpy, so this
  // entry call is redundant today; it is kept explicit because TrustDump's
  // contract ("the payload reflects current truth") must not depend on a
  // copy-path implementation detail elsewhere.
  EnsureHostBytes(t.data);
  const size_t bytes = static_cast<size_t>(t.Numel()) * vt::SizeOf(t.dtype);
  std::vector<uint8_t> a(bytes), b(bytes);
  Backend& tb = GetBackend(DeviceType::kTENSTORRENT);
  tb.Synchronize(q);
  tb.Copy(q, a.data(), t.data, bytes);
  tb.Synchronize(q);
  tb.Copy(q, b.data(), t.data, bytes);
  tb.Synchronize(q);
  const bool verified = (a == b);
  uint32_t dims[4] = {0, 0, 0, 0};
  for (int i = 0; i < t.rank && i < 4; ++i)
    dims[i] = static_cast<uint32_t>(t.shape[i]);
  char hdr[48];
  const uint32_t magic = 0x504D4454;  // 'TDMP'
  const uint32_t dt = static_cast<uint32_t>(t.dtype);
  const uint32_t rk = static_cast<uint32_t>(t.rank);
  const uint64_t numel = static_cast<uint64_t>(t.Numel());
  const uint8_t ver = verified ? 1 : 0;
  std::memcpy(hdr, &magic, 4);
  std::memcpy(hdr + 4, &dt, 4);
  std::memcpy(hdr + 8, &rk, 4);
  std::memcpy(hdr + 12, dims, 16);
  std::memcpy(hdr + 28, &numel, 8);
  std::memcpy(hdr + 36, &ver, 1);
  const uint64_t sid = seq.fetch_add(1, std::memory_order_relaxed);
  std::FILE* f = std::fopen(
      (std::string(dir) + "/" + std::to_string(sid) + "_" + name + ".tdmp").c_str(),
      "wb");
  if (f != nullptr) {
    std::fwrite(hdr, 1, 40, f);
    if (verified) std::fwrite(a.data(), 1, bytes, f);
    std::fclose(f);
  }
}


std::vector<float> DebugDeviceReadbackF32(Queue& q, const Tensor& t) {
  (void)q;
  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev = EnsureDevice2D(t, device);
  std::vector<float> v = dev.to_vector<float>();
  return v;
}

void EnsureHostBytes(void* host) {
  if (host == nullptr) return;
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] EnsureHostBytes DURING CAPTURE\n");
  ttnn::Tensor dev;
  size_t bytes = 0;
  void* base = nullptr;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s == nullptr || s->host_current) return;
    VT_CHECK(s->device_current && s->device.has_value(),
             "tenstorrent: EnsureHostBytes with no current device data");
    dev = *s->device;
    bytes = s->bytes;
    base = s->host;
  }
  // TT-27B-STEP-DECOMPOSE (VT_TT_STEP_PHASES): to_vector is THE blocking
  // sync point on this backend (no queue-synchronize primitive exists), so
  // the phase clock brackets it — a read that lands after a replay launch
  // reports the step's completion wait. Zero cost when the knob is unset.
  vt::tenstorrent::StepPhaseReadBegin();
  std::vector<float> result = dev.to_vector<float>();
  vt::tenstorrent::StepPhaseReadEnd(static_cast<int64_t>(bytes));
  const size_t n = result.size();
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s == nullptr || s->host_current) return;
    if (s->conv_transposed) {
      // Time-major shadow -> oracle [slots, C, sl] byte order.
      const uint32_t slots = s->conv_slots, Cc = s->conv_c, sl = s->conv_sl;
      const uint32_t R = slots * Cc;
      const int64_t numel = static_cast<int64_t>(slots) * Cc * sl;
      if (bytes >= static_cast<size_t>(numel) * sizeof(float)) {
        auto* dst = static_cast<float*>(base);
        for (int64_t i = 0; i < numel; ++i) {
          const int64_t j = i % sl, rc = i / sl;
          dst[i] = result[static_cast<size_t>(j * R + rc)];
        }
      } else if (bytes >= static_cast<size_t>(numel) * sizeof(uint16_t)) {
        auto* dst = static_cast<uint16_t*>(base);
        for (int64_t i = 0; i < numel; ++i) {
          const int64_t j = i % sl, rc = i / sl;
          dst[i] = F32ToBF16(result[static_cast<size_t>(j * R + rc)]);
        }
      } else {
        VT_CHECK(false, "tenstorrent: EnsureHostBytes host buffer too small");
      }
      s->host_current = true;
      return;
    }
    // Device results are f32 via to_vector. Host Alloc is typically
    // numel*sizeof(float) (tests/f32 path) or numel*2 (bf16 activations).
    if (bytes >= n * sizeof(float)) {
      std::memcpy(base, result.data(), n * sizeof(float));
    } else if (bytes >= n * sizeof(uint16_t)) {
      auto* dst = static_cast<uint16_t*>(base);
      for (size_t i = 0; i < n; ++i) dst[i] = F32ToBF16(result[i]);
    } else {
      VT_CHECK(false, "tenstorrent: EnsureHostBytes host buffer too small");
    }
    s->host_current = true;
  }
}

// ITEM 5: persistent zero tensors, created OUTSIDE capture (ttnn::zeros
// host-fills + to_device()s = an enqueue_write, illegal during trace capture).
// EnsureDevice2D primes the cache during the eager warmup so the captured
// res.Zero finds its entry and replays a warm device->device ttnn::copy.

// HOST-FREE-FORWARD R2: device->device copy when capturing, so Backend::Copy
// does not to_vector inside the captured region. Both dst and src must carry a
// current device shadow of equal byte size; dst's shadow becomes a copy of src.
// The byte volume a shadow's LOGICAL shape holds, or 0 for a dtype this
// contract does not cover (conservative: the D2D copy then declines and the
// host path runs).
static size_t ShadowLogicalBytes(const ttnn::Tensor& t) {
  const auto dt = t.dtype();
  size_t esz = 0;
  if (dt == ttnn::DataType::FLOAT32)
    esz = 4;
  else if (dt == ttnn::DataType::BFLOAT16)
    esz = 2;
  if (esz == 0) return 0;
  uint64_t vol = 1;
  const auto ds = t.logical_shape();
  for (uint32_t i = 0; i < ds.rank(); ++i) vol *= ds[i];
  return static_cast<size_t>(vol) * esz;
}

 bool CopyDeviceDeviceIfCapture(void* dst, const void* src, size_t bytes) {
  // Run the device->device copy when EITHER capturing OR in host-free-decode
  // mode (the env opt-in). The latter is essential so the EAGER warmup step
  // (which the decode-graph framework runs BEFORE capture) also exercises
  // ttnn::empty+ttnn::copy, compiling those programs into the cache so the
  // subsequent capture doesn't hit "Cannot load new binaries during trace
  // capture." Read LIVE, not cached in a static: the inertness-guard case in
  // test_tenstorrent_backend unsets the env mid-process and must observe the
  // decline, and a suite run under an ambient flag must not pin the armed
  // behavior for cases that unset it.
  const bool host_free = HostFreeDecodeEnabled();
  if (!tt_capture_active() && !host_free) return false;
  static bool once = [&] {
    // Enable program cache once on the first host-free path use — ttnn trace
    // requires every captured op to be program-cache-warm.
    MeshDevice& device = SharedMeshDevice();
    device.enable_program_cache();
    return true;
  }();
  (void)once;
  ttnn::Tensor src_dev;
  size_t copy_bytes = 0;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(const_cast<void*>(src));
    BufferSlot* d = FindSlot(dst);
    if (s == nullptr || !s->device_current || !s->device.has_value()) return false;
    if (d == nullptr) return false;
    // The Copy contract is `bytes`, NOT slot capacity. The registered
    // s->bytes / d->bytes are the HOST blocks' pool capacities, and the
    // best-fit lend (#1922) hands a DBuf a block from a LARGER size class:
    // Qwen3-4B #1625 capture staged s.hidden ([1,2560] bf16 = 5120 B) in a
    // 10240-B block lent from the prefill K/V class while the working copy
    // landed on an exactly-5120-B block. Capacity equality declined, the
    // host-path fallthrough issued a D2H read mid-capture, and tt-metal
    // TT_FATALs ("Reads are not supported during trace capture"). The content
    // contract is the SRC SHADOW's logical volume — the tensor the clone
    // reproduces — which must hold exactly the bytes this Copy names.
    if (ShadowLogicalBytes(*s->device) != bytes) return false;
    src_dev = *s->device;
    copy_bytes = bytes;
  }
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr) {
    void* bt[8];
    const int nbt = ::backtrace(bt, 8);
    char** sym = ::backtrace_symbols(bt, nbt);
    std::fprintf(stderr,
                 "[TT-TRACE] device->device copy (capture-safe) dst=%p src=%p "
                 "bytes=%zu frames=%d\n",
                 dst, src, copy_bytes, nbt);
    for (int i = 1; sym != nullptr && i < nbt; ++i)
      std::fprintf(stderr, "[TT-TRACE]   bt[%d]=%p %s\n", i, bt[i], sym[i]);
    std::fflush(stderr);
    std::free(sym);
  }
  MeshDevice& device = SharedMeshDevice();
  // Allocate a destination device tensor matching src's shape/dtype/layout,
  // then copy. No host readback.
  ttnn::Tensor cloned = ttnn::empty(src_dev.logical_shape(), src_dev.dtype(),
                                    src_dev.layout(), &device,
                                    src_dev.memory_config());
  cloned = ttnn::copy(src_dev, cloned);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* d = FindSlot(dst);
    if (d == nullptr) return false;
    // The shadow just installed is SRC-shaped, and equal byte size does not
    // mean equal geometry: the record must name the served geometry, or a
    // later stage at dst's recorded exact geometry would hit the fast path
    // and be handed a wrongly-shaped tensor. The capture lane mirrors the
    // eager device-resident arm (#2294).
    const auto ds = src_dev.logical_shape();
    d->device = std::move(cloned);
    d->dev_rows = static_cast<uint32_t>(ds[0]);
    d->dev_cols = static_cast<uint32_t>(ds[1]);
    d->device_current = true;
    d->host_current = false;
    d->device_reserved = false;  // real bytes installed — reservation spent
  }
  StagingAvoidedDeviceCopy().fetch_add(1, std::memory_order_relaxed);
  return true;
}

// HOST-FREE-FORWARD R3: on-device fill (for DBuf::Zero -> Backend::Memset)
// when host-free decode is active, so no host write happens inside capture.
// Reinterprets the buffer as a 2D [rows, cols] f32 tensor matching the
// existing device shadow's numel (zeros is the only value the forward uses).
bool MemsetDeviceIfCapture(void* p, int value, size_t bytes) {
  // Live read for the same reason as CopyDeviceDeviceIfCapture above.
  const bool host_free = HostFreeDecodeEnabled();
  if (!tt_capture_active() && !host_free) return false;
  if (value != 0) return false;  // only zero-fill is handled on-device
  // Need an existing shadow to know shape/dtype; or allocate from the slot.
  std::optional<ttnn::Tensor> dev;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(p);
    if (s != nullptr && s->device_current && s->device.has_value()) {
      dev = *s->device;
    }
  }
  std::optional<ttnn::Tensor> fresh;
  if (!dev.has_value()) {
    // HOST-FREE-FORWARD R4 (#1105): a brand-new buffer (registered at Alloc,
    // no device tensor yet) still takes the device lane. res.Zero at the top
    // of the captured layer region must leave the slot device-resident, or
    // the first EnsureDevice2D(*residual) restages from the recycled slot's
    // persistent buffer — an enqueue_write, which trace capture fatals on
    // (fd_mesh_command_queue.cpp:760).
    // bf16-only, the same polarity as the W7 reservation arm: the geometry
    // is derived from the registered byte size, which is dtype-unambiguous
    // only for 2-byte elements.
    // CAPTURE-ONLY: an eager fresh-slot zero keeps the host fallback. The
    // byte size does not name a dtype (the f32 KV masters share these pool
    // blocks), so serving one eagerly would install a wrongly-typed shadow;
    // inside the capture the write is banned and the buffer is scratch whose
    // every consumer reads on device, which is what makes the guess safe.
    // The capture-time zero still finds its tensor: the cold step's
    // EnsureDevice2D restage primed the zero at this exact spec
    // (ZeroCachePrime) and the copy program is warm from the eager copy
    // lane — ZeroCacheGet refuses a capture-time miss by design.
    if (!tt_capture_active()) {
      // Prime the zero-cache AND warm the copy program for this spec during
      // eager warmup: the capture-time lane runs ttnn::copy(zero_src, *fresh)
      // whose CopyDeviceOperation hash is shape-specific, so a copy never
      // executed during warmup is not in the program cache and trace capture
      // fatals on the missing binary. Also prime ZeroCacheGet for the
      // [1, cols] bf16 TILE spec — a fresh-slot memset whose geometry never
      // staged (the 27B bench: a 20480-B res.Zero → [1,10240] bf16 TILE)
      // would miss mid-capture.
      if (bytes > 0 && (bytes % 2) == 0) {
        uint32_t cols = static_cast<uint32_t>(bytes / 2);
        MeshDevice& md = SharedMeshDevice();
        md.enable_program_cache();
        auto shape = ttnn::Shape({1u, cols});
        ZeroCachePrime(shape, ttnn::DataType::BFLOAT16,
                       ttnn::Layout::TILE, md);
        ttnn::Tensor zero_src = ZeroCacheGet(
            shape, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, md);
        ttnn::Tensor tmp = ttnn::empty(shape, ttnn::DataType::BFLOAT16,
                                       ttnn::Layout::TILE, &md,
                                       ttnn::MemoryConfig{});
        ttnn::copy(zero_src, tmp);
      }
      return false;
    }
    MeshDevice& device_fresh = SharedMeshDevice();
    device_fresh.enable_program_cache();
    uint32_t cols = 0;
    {
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(p);
      if (s == nullptr) return false;  // untracked buffer: host memset
      // The MEMSET's byte count names the geometry, not the slot's
      // registered capacity: the #1922 best-fit lend hands a DBuf a block
      // from a larger class (Qwen3-4B #1625: a fresh [1,2560] bf16 res got
      // a 10240-B block), and cols derived from capacity staged a
      // [1,5120] zero whose key no warm step primed — a zero-cache miss
      // VT_CHECK-fails mid-capture. Requested bytes are the contract, the
      // same shadow-volume rule the D2D copy lane above now applies.
      if (bytes == 0 || (bytes % 2) != 0) return false;
      cols = static_cast<uint32_t>(bytes / 2);
      if (s->persistent.has_value() && s->persist_rows == 1 &&
          s->persist_cols == cols) {
        // Recycled staging: zero the persistent buffer IN PLACE — the device
        // address stays stable across steps, which is what lets the captured
        // zero-copy replay against the same buffer.
        fresh = *s->persistent;
      }
    }
    if (!fresh.has_value()) {
      fresh = ttnn::empty(ttnn::Shape({1u, cols}), ttnn::DataType::BFLOAT16,
                          ttnn::Layout::TILE, &device_fresh,
                          ttnn::MemoryConfig{});
      std::lock_guard<std::mutex> g(SlotMutex());
      if (BufferSlot* s = FindSlot(p)) {
        // W5 semantics: the allocation becomes the slot's persistent buffer,
        // so the next zero reuses the same device address.
        s->persistent = fresh;
        s->persist_rows = 1;
        s->persist_cols = cols;
      }
    }
    ttnn::Tensor zero_src = ZeroCacheGet(*fresh, device_fresh);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] device zero-fill (fresh slot %p cols=%u)\n",
                   p, cols);
    ttnn::Tensor z = ttnn::copy(zero_src, *fresh);
    (void)z;
    {
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(p);
      if (s == nullptr) return false;
      s->device = *fresh;
      s->dev_rows = 1;
      s->dev_cols = cols;
      s->device_current = true;
      s->host_current = false;
      s->conv_transposed = false;
      s->device_reserved = false;  // real zeros installed — reservation spent
    }
    return true;
  }
  MeshDevice& device = SharedMeshDevice();
  const ttnn::Tensor& shadow = *dev;
  // ITEM 5: ttnn::zeros/full is NOT capture-safe — full_impl host-fills and
  // to_device()s (creation.cpp:52-71), i.e. an enqueue_write that ttnn trace
  // fatals on. The plugin pattern instead: keep PERSISTENT zero tensors
  // (created outside capture, at warmup) and ttnn::copy one onto the target —
  // a device->device program that is captured/replayed like any other warm op.
  ttnn::Tensor zero_src = ZeroCacheGet(shadow, device);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] device zero-fill (capture-safe)\n");
  // Copy the persistent zero onto the shadow IN PLACE (keeps the shadow's
  // device address stable — the whole point of persistent buffers).
  ttnn::Tensor z = ttnn::copy(zero_src, shadow);
  (void)z;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(p);
    if (s == nullptr) return false;
    s->device_current = true;
    s->host_current = false;
    s->device_reserved = false;  // real zeros installed — reservation spent
  }
  return true;
}

// BACKEND-TENSTORRENT-QWEN35 W7 (#2282): the EAGER zero-fill. Outside capture
// (the capture lane is MemsetDeviceIfCapture above), a FULL-slot memset(0) of
// a device-resident slot fills the shadow on-device and KEEPS it: the caller
// still memsets the host bytes, so both sides hold zeros and the next device
// read serves the shadow instead of restaging the zeros. Zero is the only
// value handled — memset writes byte patterns, and only the all-zero pattern
// is a valid value in every dtype the shadows use.
bool MemsetDeviceFill(void* p, int value, size_t bytes) {
  if (value != 0) return false;
  if (tt_capture_active()) return false;  // capture-unsafe refusals keep semantics
  std::optional<ttnn::Tensor> dev;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(p);
    if (s == nullptr || !s->device_current || !s->device.has_value()) return false;
    if (s->bytes != bytes) return false;  // a partial memset keeps the host path
    dev = *s->device;
  }
  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor zero_src = ZeroCacheGet(*dev, device);
  // In place into the slot's owning shadow — shared TensorAttributes carry the
  // write, the shadow's device address stays stable.
  ttnn::Tensor z = ttnn::copy(zero_src, *dev);
  (void)z;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(p);
    if (s == nullptr) return false;
    s->device_current = true;  // shadow KEPT; the caller zeroes the host bytes
    s->host_current = true;
    s->device_reserved = false;  // real zeros installed — reservation spent
  }
  StagingAvoidedMemset().fetch_add(1, std::memory_order_relaxed);
  return true;
}

// BACKEND-TENSTORRENT-QWEN35 W7 (#2282): the EAGER device->device copy.
// Outside capture (the capture lane is CopyDeviceDeviceIfCapture above), a
// copy whose SOURCE shadow is current and whose destination already owns a
// persistent buffer goes device->device: the host path would download the
// source, memcpy, drop the destination's shadow, and re-upload the same bytes
// on the next device read. A destination WITHOUT a persistent buffer is a
// host reader's buffer (d2h readback, dump) and keeps the host path. Base
// pointers only: FindSlot resolves interior views to the base slot, and a
// sub-range copy is not served by a whole-shadow copy.
bool CopyDeviceDeviceIfResident(void* dst, const void* src, size_t bytes) {
  if (tt_capture_active()) return false;
  ttnn::Tensor src_dev;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(const_cast<void*>(src));
    BufferSlot* d = FindSlot(dst);
    if (s == nullptr || !s->device_current || !s->device.has_value()) return false;
    if (d == nullptr || s->host != src || d->host != dst) return false;
    // Same shadow-volume contract as the capture lane above: slot bytes are
    // HOST BLOCK CAPACITY (the #1922 best-fit lend pads it), not the content
    // the shadow holds. The clone must reproduce exactly `bytes`.
    if (d->bytes < bytes) return false;
    if (ShadowLogicalBytes(*s->device) != bytes) return false;
    if (!d->persistent.has_value()) return false;
    src_dev = *s->device;
  }
  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor cloned = ttnn::empty(src_dev.logical_shape(), src_dev.dtype(),
                                    src_dev.layout(), &device,
                                    src_dev.memory_config());
  cloned = ttnn::copy(src_dev, cloned);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* d = FindSlot(dst);
    if (d == nullptr) return false;
    // The shadow just installed is SRC-shaped, and equal byte size does not
    // mean equal geometry: the record must name the served geometry, or a
    // later stage at dst's recorded exact geometry would hit the fast path
    // and be handed a wrongly-shaped tensor.
    const auto ds = src_dev.logical_shape();
    d->device = std::move(cloned);
    d->dev_rows = static_cast<uint32_t>(ds[0]);
    d->dev_cols = static_cast<uint32_t>(ds[1]);
    d->device_current = true;
    d->host_current = false;
    d->device_reserved = false;  // real bytes installed — reservation spent
  }
  StagingAvoidedDeviceCopy().fetch_add(1, std::memory_order_relaxed);
  return true;
}


// ITEM 5 (rope): driver-side warm hook. The decode-graph driver calls this
// for the step's (padded) positions BEFORE BeginCapture — the exact
// SizeSlot::Refresh slot in qwen3.cpp — so the persistent cos/sin tensors
// are populated outside capture and the captured rope cache-HITs on content.
// hq/hk select the expanded layouts to warm; base/args must match RopeNeox.
void WarmRopeCosSin(const int32_t* positions, int64_t tokens, int64_t hq,
                    int64_t hk, int64_t rot, double base) {
  if (!HostFreeDecodeEnabled()) return;
  MeshDevice& device = SharedMeshDevice();
  std::vector<float> cos_t, sin_t;
  Tensor pos = Tensor::Contiguous(const_cast<int32_t*>(positions), DType::kI32,
                                  Device{DeviceType::kTENSTORRENT, 0}, {tokens});
  const RopeArgs no_scale{};  // plain rope only on the warm path
  BuildCosSinFromPositions(pos, tokens, rot, base, no_scale, cos_t, sin_t);
  // Byte-exact with what the captured rope reads: the per-step cos|sin CACHE
  // stores f32-built values into a BF16 tensor (RopeCosSinCacheKernel's
  // StoreElemF32 rounds), and the rope-side gather reads them back. Round the
  // warm content through the same bf16 round-trip so the content-HIT
  // comparison is exact.
  for (auto& v : cos_t) v = BF16ToF32(F32ToBF16(v));
  for (auto& v : sin_t) v = BF16ToF32(F32ToBF16(v));
  auto warm_one = [&](int64_t heads) {
    std::vector<float> ce, se;
    ExpandCosSinPerHead(cos_t.data(), sin_t.data(), tokens, heads, rot / 2, ce, se);
    const uint32_t thu = static_cast<uint32_t>(tokens * heads);
    const uint32_t halfu = static_cast<uint32_t>(rot / 2);
    std::lock_guard<std::mutex> g(RopeCSMutex());
    auto& c = RopeCSCache();
    const std::string k = RopeCSKey(thu, halfu);
    auto it = c.find(k);
    if (it == c.end()) {
      RopeCSEntry e;
      e.cos = UploadRows(ce.data(), thu, halfu, device);
      e.sin = UploadRows(se.data(), thu, halfu, device);
      e.cos_host = ce;
      c[k] = std::move(e);
    } else if (it->second.cos_host != ce) {
      // In-place CONTENT refresh of the SAME device tensors: a captured rope
      // op reads the address recorded at capture time, so replacing the
      // tensors here would leave every replay reading the capture-step
      // cos/sin (stale positions). The host tensors are built with the
      // identical bf16 TILE spec so copy_to_device writes byte-matching
      // data. Legal here: the driver calls this outside capture.
      // VT_TT_NO_ROPE_REFRESH: stall bisection only — skip the per-step H2D
      // copies AFTER the first capture (stale cos/sin on replays, numerically
      // wrong, mechanics test only).
      if (ReplayRegimeBisectSkip("VT_TT_NO_ROPE_REFRESH")) return;
      ttnn::Tensor cos_h = ttnn::Tensor::from_vector<float>(
          ce, TileSpecOf(thu, halfu), nullptr);
      ttnn::Tensor sin_h = ttnn::Tensor::from_vector<float>(
          se, TileSpecOf(thu, halfu), nullptr);
      ttnn::copy_to_device(cos_h, it->second.cos);
      ttnn::copy_to_device(sin_h, it->second.sin);
      it->second.cos_host = ce;
    }
  };
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] WarmRopeCosSin tokens=%lld hq=%lld hk=%lld"
                 " rot=%lld first_pos=%d cos_first=%f\n",
                 (long long)tokens, (long long)hq, (long long)hk,
                 (long long)rot, (int)positions[0],
                 cos_t.empty() ? -1.0f : cos_t.front());
  warm_one(hq);
  warm_one(hk);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr) {
    std::lock_guard<std::mutex> g(RopeCSMutex());
    for (auto& [k, e] : RopeCSCache())
      std::fprintf(stderr, "[TT-TRACE] warm stored key=%s first=%f n=%zu\n",
                   k.c_str(), e.cos_host.empty() ? -1.0f : e.cos_host.front(),
                   e.cos_host.size());
  }
}

// kAttnQkNormRopeGate's per-step cos|sin table: the dense decode-graph driver
// calls this EVERY step (capture + replay), outside capture, with the same
// positions vector the in-region RopeCosSinCacheKernel refill reads, so the
// captured fused preamble serves a table matching this step (WarmRopeCosSin's
// populate-outside/content-HIT-inside pattern; the SizeSlot::Refresh shape).
// Byte-exactness: RopeCosSinCacheKernel stores plain f32 (no bf16 round-trip
// needed — the DBuf is kF32), and BuildCosSinFromPositions carries the same
// double-pow + Llama3ScaleFreq math the kernel inlines.
void WarmAttnCosSin(const int32_t* positions, int64_t tokens, int64_t rot,
                      double base) {
  if (!HostFreeDecodeEnabled()) return;
  if (rot <= 0 || (rot % 2) != 0 || tokens < 1) return;
  Tensor pos = Tensor::Contiguous(const_cast<int32_t*>(positions), DType::kI32,
                                    Device{DeviceType::kTENSTORRENT, 0}, {tokens});
  std::vector<float> cos_t, sin_t;
  BuildCosSinFromPositions(pos, tokens, static_cast<int>(rot), base, RopeArgs{},
                             cos_t, sin_t);
  const int64_t half = rot / 2;
  std::vector<float> table(static_cast<size_t>(tokens) * rot);
  for (int64_t i = 0; i < tokens; ++i) {
    for (int64_t j = 0; j < half; ++j) {
      table[static_cast<size_t>(i) * rot + j] =
          cos_t[static_cast<size_t>(i) * half + j];
      table[static_cast<size_t>(i) * rot + half + j] =
          sin_t[static_cast<size_t>(i) * half + j];
    }
  }
  const uint32_t tu = static_cast<uint32_t>(tokens);
  const uint32_t rotu = static_cast<uint32_t>(rot);
  const std::string dbg_key = AttnCSKey(tu, rotu);
  std::lock_guard<std::mutex> g(AttnCSMutex());
  auto& c = AttnCSCache();
  const std::string k = dbg_key;
  auto it = c.find(k);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr,
                 "[TT-TRACE] WarmAttnCosSin tokens=%lld rot=%lld key=%s "
                 "found=%d first=%f\n",
                 (long long)tokens, (long long)rot, k.c_str(),
                 it != c.end() ? 1 : 0,
                 table.empty() ? -1.0f : table.front());
  if (it == c.end()) {
    AttnCSEntry e;
    e.cs_host = table;
    e.cs = UploadTensor(std::move(table), ttnn::Shape({tu, rotu}),
                          ttnn::DataType::FLOAT32, ttnn::Layout::TILE,
                          SharedMeshDevice());
    c[k] = std::move(e);
  } else if (it->second.cs_host != table) {
    // In-place content refresh of the SAME device tensor: a captured op reads
    // the address recorded at capture time, so replacing the tensor would
    // strand every replay on the capture-step positions. Legal here — the
    // driver calls this outside capture.
    // Same f32 TILE spec the entry was uploaded with (TileSpecOf is the BF16
    // rope-cache spec — a dtype mismatch here is a TT_FATAL in copy_to_device).
    ttnn::Tensor h = ttnn::Tensor::from_vector<float>(
        table,
        SpecOf(tt::tt_metal::Shape({tu, rotu}), ttnn::DataType::FLOAT32,
               ttnn::Layout::TILE),
        nullptr);
    ttnn::copy_to_device(h, it->second.cs);
    it->second.cs_host = std::move(table);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] WarmAttnCosSin REFRESHED key=%s\n",
                   k.c_str());
  }
}

// Serveability query for the transposed conv-state shadow: mirrors the
// EnsureConvStateTransposed fast path READ-ONLY (no download, no upload), so
// the decode driver can gate capture/replay on it outside capture. A
// prefill-bearing step's ssm/cache role transition (GdnStateGather) clears
// the shadow and replaces the slot's device tensor; decode must then run the
// step eagerly — which rebuilds the shadow — before any graph may capture or
// replay against the buffer again.
bool ConvShadowServeable(const void* conv_state_data, int64_t slots,
                         int64_t conv_dim, int64_t state_len) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(const_cast<void*>(conv_state_data));
  if (s == nullptr) return false;
  const uint32_t uslots = static_cast<uint32_t>(slots);
  const uint32_t uc = static_cast<uint32_t>(conv_dim);
  const uint32_t usl = static_cast<uint32_t>(state_len);
  return s->device_current && s->device.has_value() && s->conv_transposed &&
         s->conv_slots == uslots && s->conv_c == uc && s->conv_sl == usl &&
         s->device->dtype() == ttnn::DataType::FLOAT32 &&
         s->device->layout() == ttnn::Layout::TILE &&
         s->device->logical_shape()[0] == usl + 1 &&
         s->device->logical_shape()[1] == uslots * uc;
}

// ---- BACKEND-TENSTORRENT-QWEN35 W4 (#2107): staging counters ----
// Readers for the tenstorrent_device.h API; the increment sites live with the
// staging paths in the anonymous namespace above.
StagingStats GetStagingStats() {
  StagingStats s;
  s.uploads_bulk_bf16 = StagingBulkUploads().load(std::memory_order_relaxed);
  s.staged_bulk_bf16_bytes = StagingBulkBytes().load(std::memory_order_relaxed);
  s.staged_f32_elems = StagingF32Elems().load(std::memory_order_relaxed);
  s.uploads_persistent_bf16 =
      StagingPersistentWrites().load(std::memory_order_relaxed);
  s.uploads_persistent_allocs =
      StagingPersistentAllocs().load(std::memory_order_relaxed);
  s.staged_persistent_bf16_bytes =
      StagingPersistentBytes().load(std::memory_order_relaxed);
  s.stages_avoided_reservation =
      StagingAvoidedReservation().load(std::memory_order_relaxed);
  s.stages_avoided_device_memset =
      StagingAvoidedMemset().load(std::memory_order_relaxed);
  s.stages_avoided_device_copy =
      StagingAvoidedDeviceCopy().load(std::memory_order_relaxed);
  return s;
}

void ResetStagingStats() {
  StagingBulkUploads().store(0, std::memory_order_relaxed);
  StagingBulkBytes().store(0, std::memory_order_relaxed);
  StagingF32Elems().store(0, std::memory_order_relaxed);
  StagingPersistentWrites().store(0, std::memory_order_relaxed);
  StagingPersistentAllocs().store(0, std::memory_order_relaxed);
  StagingPersistentBytes().store(0, std::memory_order_relaxed);
  StagingAvoidedReservation().store(0, std::memory_order_relaxed);
  StagingAvoidedMemset().store(0, std::memory_order_relaxed);
  StagingAvoidedDeviceCopy().store(0, std::memory_order_relaxed);
}

}  // namespace vt::tenstorrent

// Tenstorrent mesh-trace capture + W4d alloc trace — split stage 5 of
// tenstorrent_ops.cpp (.agents/specs/tenstorrent-ops-split.md,
// ISSUE-LOCAL-01M2M2PZC0E998B88C9XKJT8JJ). Mechanical move from
// tenstorrent_ops.cpp: the ttnn mesh-trace capture state machine
// (TraceBeginCapture/TraceEndCapture/TraceReplay, the multi-graph
// TraceEndCaptureGraph/TraceReplayGraph/TraceDestroyGraph handles, the
// GraphCapturesDone bisection counter), the HOST-FREE-DECODE persistent
// decode ids + capture-safe embedding (WarmDecodeIds/EmbedDeviceIdsInto),
// and the W4d W0 (#3042) device-side allocation trace with its W3/W6
// attribution, DRAM-probe and warm-release/staging helpers. Registrations
// stay in tenstorrent_ops.cpp's Registrar.

#include "vt/tenstorrent/tenstorrent_internal.h"

#include <chrono>
#include <vector>

namespace vt::tenstorrent {

// ---- ttnn mesh-trace capture (Backend graph-capture mapping) ----------------
// Process-local single-slot capture + multi-graph handles (opaque MeshTraceId*).
// Mirrors the CUDA backend's single-exec_ vs EndCaptureGraph split.

namespace {
struct TraceState {
  bool capturing = false;
  bool has_replay = false;
  ttnn::MeshTraceId capturing_id{0};
  ttnn::MeshTraceId replay_id{0};
};
TraceState& TraceSlot() {
  static TraceState s;
  return s;
}
constexpr auto kTraceCq = ttnn::QueueId(0);

}  // namespace

// Stall-bisection helper: counts completed TraceEndCaptureGraph calls. The
// bisection skip flags (VT_TT_NO_*_WARM) must NOT fire on the capture step
// itself — the captured rope cache-HIT guard requires fresh warm content —
// so they skip only once a graph exists (steady replay regime).
std::atomic<int>& GraphCapturesCounter() {
  static std::atomic<int> n{0};
  return n;
}
int GraphCapturesDone() { return GraphCapturesCounter().load(); }
void NoteGraphCaptured() { GraphCapturesCounter().fetch_add(1); }
bool ReplayRegimeBisectSkip(const char* flag) {
  return std::getenv(flag) != nullptr && GraphCapturesDone() > 0;
}
namespace {


// A replayed trace rewrote the device memory of every tensor the captured
// region produced, but the slot registry cannot know which host buffers those
// shadows belong to. Mark the host cache of EVERY device-current slot stale so
// the next host read re-downloads. Without this, DBuf::Download ->
// Backend::Copy -> EnsureHostBytes short-circuits on host_current and serves
// the bytes captured at trace time on every later replay (frozen logits).
// Replay is non-blocking, so device writes may still be in flight here; the
// invalidation only marks device memory as newer than the host copy, and the
// re-download at the next host read is the blocking sync point.
// Input shadows (weights, embeddings) are only re-read, never re-uploaded:
// replay does not modify them, and the extra download is identical bytes.
void InvalidateHostCachesAfterTrace() {
  std::lock_guard<std::mutex> g(SlotMutex());
  for (auto& [addr, slot] : Slots())
    if (slot.device_current && slot.device.has_value()) slot.host_current = false;
}
}  // namespace

void TraceBeginCapture() {
  TraceState& s = TraceSlot();
  VT_CHECK(!s.capturing, "tenstorrent: nested TraceBeginCapture");
  MeshDevice& device = SharedMeshDevice();
  s.capturing_id = ttnn::operations::trace::begin_trace_capture(&device, kTraceCq);
  s.capturing = true;
  tt_capture_active() = true;
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] BeginCapture (flag set)\n");
}

void TraceEndCapture() {
  TraceState& s = TraceSlot();
  VT_CHECK(s.capturing, "tenstorrent: TraceEndCapture without Begin");
  MeshDevice& device = SharedMeshDevice();
  ttnn::operations::trace::end_trace_capture(&device, s.capturing_id, kTraceCq);
  // W4a wave-3a: expose the device-reported live trace demand so the
  // chunked E=1 arm's fit inside the 50 MiB trace region is a measurement,
  // not an assumption (the wave-1b falsification class).
  LastTraceBytes() = static_cast<int64_t>(device.get_trace_buffers_size());
  // Drop previous single-slot replay if any.
  if (s.has_replay) {
    try {
      ttnn::operations::trace::release_trace(&device, s.replay_id);
    } catch (...) {
    }
  }
  s.replay_id = s.capturing_id;
  s.has_replay = true;
  s.capturing = false;
  tt_capture_active() = false;
}

void TraceReplay() {
  TraceState& s = TraceSlot();
  VT_CHECK(!s.capturing, "tenstorrent: TraceReplay during capture");
  VT_CHECK(s.has_replay, "tenstorrent: TraceReplay with no captured trace");
  MeshDevice& device = SharedMeshDevice();
  // NON-BLOCKING, matching models/common/models/executor.py's long-decode
  // pattern (execute_trace(blocking=False) + a later blocking readback):
  // repeated blocking replays hang the mesh trace completion wait after a
  // few dozen executions on this tt-metal build. The caller's post-replay
  // device readback (logits Download) provides the synchronization; queue
  // order keeps any later input refresh behind the replay.
  ttnn::operations::trace::execute_trace(&device, s.replay_id, kTraceCq, /*blocking=*/false);
  InvalidateHostCachesAfterTrace();
}

void* TraceEndCaptureGraph() {
  TraceState& s = TraceSlot();
  VT_CHECK(s.capturing, "tenstorrent: TraceEndCaptureGraph without Begin");
  MeshDevice& device = SharedMeshDevice();
  ttnn::operations::trace::end_trace_capture(&device, s.capturing_id, kTraceCq);
  LastTraceBytes() = static_cast<int64_t>(device.get_trace_buffers_size());
  NoteGraphCaptured();
  s.capturing = false;
  tt_capture_active() = false;
  // Opaque handle: heap-allocated MeshTraceId for the multi-graph API.
  return new ttnn::MeshTraceId(s.capturing_id);
}

void TraceReplayGraph(void* graph) {
  VT_CHECK(graph != nullptr, "tenstorrent: TraceReplayGraph null");
  VT_CHECK(!TraceSlot().capturing, "tenstorrent: TraceReplayGraph during capture");
  MeshDevice& device = SharedMeshDevice();
  const auto id = *static_cast<ttnn::MeshTraceId*>(graph);
  // NON-BLOCKING — see TraceReplay: the qwen3 graph driver downloads the
  // logits right after this call, and that blocking readback is the sync
  // point (the upstream traced-decode executor's pattern).
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-STEP] execute_trace begin\n");
  ttnn::operations::trace::execute_trace(&device, id, kTraceCq, /*blocking=*/false);
  // TT-27B-STEP-DECOMPOSE: stamp the launch so the first blocking read after
  // it (the logits download) reports the step's completion wait.
  StepPhaseNoteLaunch("replay");
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-STEP] execute_trace enqueued\n");
  InvalidateHostCachesAfterTrace();
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-STEP] replay step complete\n");
}

void TraceDestroyGraph(void* graph) {
  if (graph == nullptr) return;
  auto* id = static_cast<ttnn::MeshTraceId*>(graph);
  try {
    MeshDevice& device = SharedMeshDevice();
    ttnn::operations::trace::release_trace(&device, *id);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-STEP] release_trace ok\n");
  } catch (const std::exception& ex) {
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-STEP] release_trace THREW: %s\n", ex.what());
  } catch (...) {
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-STEP] release_trace THREW (unknown)\n");
  }
  delete id;
}

// ---- TT-27B-STEP-DECOMPOSE (VT_TT_STEP_PHASES): the step phase clock ---------
// A READ-ONLY instrument for the captured decode step's cost decomposition
// (.agents/specs/tenstorrent-27b-step-decompose.md). The driver brackets its
// own phases (warmup refreshes, embed, capture begin/body/end, replay launch)
// and prints one line per step; the SEAM half here closes the loop on the one
// phase the driver cannot see — the completion wait. Replay enqueue is
// NON-BLOCKING (see TraceReplayGraph), so the step's device time surfaces at
// the next blocking host read (EnsureHostBytes' to_vector). StepPhaseNoteLaunch
// stamps that launch; the first read after it reports the wait and consumes
// the stamp. Zero cost when the env is unset.
namespace {
struct StepPhaseState {
  std::chrono::steady_clock::time_point launch{};
  std::chrono::steady_clock::time_point read_begin{};
  bool has_launch = false;
  char kind[16] = "none";
};
StepPhaseState& StepPhaseSt() {
  static StepPhaseState* s = new StepPhaseState();  // never destroyed (#1486)
  return *s;
}
double StepPhaseMs(std::chrono::steady_clock::time_point a,
                   std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
}  // namespace

bool StepPhasesEnabled() { return std::getenv("VT_TT_STEP_PHASES") != nullptr; }
const char* StepPhaseMode() {
  const char* e = std::getenv("VT_TT_STEP_PHASES");
  return e != nullptr ? e : "";
}
void StepPhaseNoteLaunch(const char* kind) {
  if (!StepPhasesEnabled()) return;
  StepPhaseState& s = StepPhaseSt();
  s.launch = std::chrono::steady_clock::now();
  s.has_launch = true;
  std::snprintf(s.kind, sizeof s.kind, "%s", kind);
}
void StepPhaseReadBegin() {
  if (!StepPhasesEnabled()) return;
  StepPhaseSt().read_begin = std::chrono::steady_clock::now();
}
void StepPhaseReadEnd(int64_t bytes) {
  if (!StepPhasesEnabled()) return;
  StepPhaseState& s = StepPhaseSt();
  const auto t_end = std::chrono::steady_clock::now();
  const double wait_ms = StepPhaseMs(s.read_begin, t_end);
  if (s.has_launch) {
    // The first blocking read after the step's last launch pays the device
    // tail: gap = host work between launch and the read, wait = the read
    // itself (queue drain + D2H).
    std::fprintf(stderr,
                  "[TT-STEP-PHASE] sync kind=%s gap_ms=%.3f wait_ms=%.3f "
                  "read=%lld\n",
                  s.kind, StepPhaseMs(s.launch, s.read_begin), wait_ms,
                  static_cast<long long>(bytes));
    s.has_launch = false;
  } else {
    // No pending launch (a read outside the driver's step bracket, e.g. the
    // prefill forward's own logits drain): its own duration is the device
    // tail it paid.
    std::fprintf(stderr,
                  "[TT-STEP-PHASE] sync kind=host-read wait_ms=%.3f read=%lld\n",
                  wait_ms, static_cast<long long>(bytes));
  }
}
void StepPhaseSyncProbe() {
  if (!StepPhasesEnabled()) return;
  // The per-layer sampling probe's queue drain (VT_TT_STEP_PHASES=sync): a
  // blocking read of a persistent 1-element device tensor. The CQ is
  // in-order, so the read cannot complete before every prior enqueue has;
  // per-layer probe time is therefore that layer's true DEVICE cost. The
  // tensor is created on first use OUTSIDE capture and never freed (#1486).
  static ttnn::Tensor* probe = nullptr;
  MeshDevice& device = SharedMeshDevice();
  if (probe == nullptr) {
    const auto spec = SpecOf(tt::tt_metal::Shape({1}), ttnn::DataType::UINT32,
                             ttnn::Layout::ROW_MAJOR);
    probe = new ttnn::Tensor(ttnn::Tensor::from_vector<uint32_t>(
        std::vector<uint32_t>{0}, spec, &device));
  }
  (void)probe->to_vector<uint32_t>();
}
bool TraceCaptureActive() { return tt_capture_active(); }


// ---- HOST-FREE-DECODE: persistent decode ids + capture-safe embedding -----
// The replay step must perform ZERO eager device allocations: per-step eager
// alloc/free churn (the from_vector + embedding output of the old EmbedInto
// refresh) eventually hands a live trace's fixed buffer addresses to new
// allocations — tt-metal warns allocations while a trace exists "may be
// corrupted once a trace is executed", observed as a device hang ~60 replays
// in. The embedding therefore moves INSIDE the captured region: ids are
// refreshed into one persistent device tensor (allocation-free
// copy_to_device), the captured ttnn::embedding runs over that stable
// address, and its output tensor is kept alive so the trace's write address
// is never returned to the allocator.
namespace {
struct DecodeIdsEntry {
  ttnn::Tensor ids;  // device ROW_MAJOR UINT32 [n], content refreshed in place
  ttnn::Tensor out;  // embedding output [n, hidden] TILE; held for the trace
  bool allocated = false;
};
std::map<int64_t, DecodeIdsEntry>& DecodeIdsCache() {
  static std::map<int64_t, DecodeIdsEntry>* m = new std::map<int64_t, DecodeIdsEntry>(); // never destroyed (#1486)
  return *m;
}
std::mutex& DecodeIdsMutex() {
  static std::mutex m;
  return m;
}
}  // namespace

void WarmDecodeIds(const int32_t* ids, int64_t n) {
  if (!HostFreeDecodeEnabled()) return;
  if (ids == nullptr || n < 1) return;
  MeshDevice& device = SharedMeshDevice();
  std::vector<uint32_t> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    VT_CHECK(ids[i] >= 0, "tenstorrent WarmDecodeIds: negative id");
    host[static_cast<size_t>(i)] = static_cast<uint32_t>(ids[i]);
  }
  const auto spec = SpecOf(
      tt::tt_metal::Shape({static_cast<uint32_t>(n)}),
      ttnn::DataType::UINT32, ttnn::Layout::ROW_MAJOR);
  std::lock_guard<std::mutex> g(DecodeIdsMutex());
  DecodeIdsEntry& e = DecodeIdsCache()[n];
  const bool dbg_ids = std::getenv("VT_TT_TRACE_DEBUG") != nullptr;
  if (dbg_ids) std::fprintf(stderr, "[TT-STEP] WarmDecodeIds begin n=%lld\n", (long long)n);
  // VT_TT_NO_IDS_WARM: stall bisection only — skip the per-step H2D copy
  // after the first capture (same token embedded every replay, numerically
  // wrong, mechanics test only).
  if (ReplayRegimeBisectSkip("VT_TT_NO_IDS_WARM")) {
    if (dbg_ids) std::fprintf(stderr, "[TT-STEP] WarmDecodeIds skipped\n");
    return;
  }
  if (!e.allocated) {
    e.ids = ttnn::Tensor::from_vector<uint32_t>(host, spec, &device);
    e.allocated = true;
  } else {
    // Allocation-free refresh: host staging tensor + H2D copy into the SAME
    // device buffer (the WarmRacIdx pattern).
    ttnn::Tensor h = ttnn::Tensor::from_vector<uint32_t>(host, spec, nullptr);
    ttnn::copy_to_device(h, e.ids);
  }
  if (dbg_ids) std::fprintf(stderr, "[TT-STEP] WarmDecodeIds done\n");
}

void EmbedDeviceIdsInto(void* out_host, int64_t rows, int64_t cols,
                        const void* table_host, int64_t vocab, int64_t hidden,
                        int64_t n) {
  ttnn::Tensor dev_ids;
  {
    std::lock_guard<std::mutex> g(DecodeIdsMutex());
    auto it = DecodeIdsCache().find(n);
    VT_CHECK(it != DecodeIdsCache().end() && it->second.allocated,
             "tenstorrent: EmbedDeviceIdsInto without WarmDecodeIds(n)");
    dev_ids = it->second.ids;
  }
  ttnn::Tensor dev_table;
  {
    std::lock_guard<std::mutex> g(EmbedTableMutex());
    auto it = EmbedTableShadows().find(reinterpret_cast<uintptr_t>(table_host));
    VT_CHECK(it != EmbedTableShadows().end() && it->second.device.has_value() &&
                 it->second.vocab == static_cast<uint32_t>(vocab) &&
                 it->second.h == static_cast<uint32_t>(hidden),
             "tenstorrent: EmbedDeviceIdsInto without a warmed embed table "
             "(run one eager embedding step before capture)");
    dev_table = *it->second.device;
  }
  ttnn::Tensor dev_out = ttnn::embedding(dev_ids, dev_table, /*pad_token=*/std::nullopt,
                                         /*layout=*/ttnn::Layout::TILE);
  if (dev_out.logical_shape().rank() != 2 ||
      dev_out.logical_shape()[0] != n || dev_out.logical_shape()[1] != hidden) {
    dev_out = ttnn::reshape(
        dev_out, ttnn::Shape({static_cast<uint32_t>(n),
                              static_cast<uint32_t>(hidden)}));
  }
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(out_host);
    VT_CHECK(s != nullptr && s->device_current && s->device.has_value() &&
                 s->dev_rows == static_cast<uint32_t>(rows) &&
                 s->dev_cols == static_cast<uint32_t>(cols),
             "tenstorrent: EmbedDeviceIdsInto hidden shadow not resident");
    ttnn::copy(dev_out, *s->device);
    s->host_current = false;
  }
  // Hold the embedding output for the trace's lifetime (its address is baked
  // into the captured command sequence; freeing it would return the buffer
  // to the allocator).
  {
    std::lock_guard<std::mutex> g(DecodeIdsMutex());
    DecodeIdsCache()[n].out = dev_out;
  }
}

// ---- W4d W0 (#3042): device-side allocation trace ---------------------------
// Interleaves get_memory_view snapshots between ops to attribute the 27B OOM.
// Gated by VT_TT_ALLOC_TRACE. Logs free/allocated bytes per DRAM bank to stderr
// with a running delta; the max negative delta (largest single allocation) is
// the red-first test observable.
namespace {
bool AllocTraceEnabled() {
  return std::getenv("VT_TT_ALLOC_TRACE") != nullptr;
}
struct AllocTraceState {
  std::mutex mtx;
  int64_t snapshot_count = 0;
  int64_t prev_free_total = -1;  // -1 = no previous snapshot
  int64_t max_alloc_delta = 0;   // largest single allocation (bytes)
};
AllocTraceState& AllocTraceSt() {
  static AllocTraceState* s = new AllocTraceState();  // never destroyed (#1486)
  return *s;
}
}  // namespace

void AllocTraceSnapshot(MeshDevice& device, const char* label) {
  if (!AllocTraceEnabled()) return;
  const auto view = tt::tt_metal::detail::GetMemoryView(
      &device, tt::tt_metal::BufferType::DRAM);
  const int64_t num_banks = static_cast<int64_t>(view.num_banks);
  const int64_t free_per_bank =
      static_cast<int64_t>(view.total_bytes_free_per_bank);
  const int64_t alloc_per_bank =
      static_cast<int64_t>(view.total_bytes_allocated_per_bank);
  const int64_t largest_free =
      static_cast<int64_t>(view.largest_contiguous_bytes_free_per_bank);
  const int64_t total_free = num_banks * free_per_bank;

  std::lock_guard<std::mutex> g(AllocTraceSt().mtx);
  int64_t delta = 0;
  if (AllocTraceSt().prev_free_total >= 0)
    delta = total_free - AllocTraceSt().prev_free_total;  // negative = allocated
  if (delta < 0 && -delta > AllocTraceSt().max_alloc_delta)
    AllocTraceSt().max_alloc_delta = -delta;
  AllocTraceSt().prev_free_total = total_free;
  AllocTraceSt().snapshot_count++;

  std::fprintf(stderr,
      "[TT-ALLOC] #%lld label=%s banks=%lld free_per_bank=%lld "
      "largest_free=%lld alloc_per_bank=%lld total_free=%lld delta=%lld\n",
      (long long)AllocTraceSt().snapshot_count, label,
      (long long)num_banks, (long long)free_per_bank,
      (long long)largest_free, (long long)alloc_per_bank,
      (long long)total_free, (long long)delta);
}

int64_t AllocTraceSnapshotCountForTest() {
  std::lock_guard<std::mutex> g(AllocTraceSt().mtx);
  return AllocTraceSt().snapshot_count;
}
int64_t AllocTraceMaxDeltaForTest() {
  std::lock_guard<std::mutex> g(AllocTraceSt().mtx);
  return AllocTraceSt().max_alloc_delta;
}
void StageWeightBf16ForTest(const Tensor& t, MeshDevice& device) {
  (void)EnsureDevice2D(t, device);
}

// W4d W3 (ISSUE-LOCAL-01M2AA4ZVD9EWJG5NNQD5DWZXS): after the ctor's cold
// pre-warm (which runs at the full batched shape) and the capture, every
// warm activation it committed via CommitDeviceLogical2D is garbage — the
// captured decode only reads step-shaped buffers — yet the slots hold it
// forever (~14 GiB at 27B: banks at 93 percent before the first decode, and
// the repair chain's contiguity ask fragments against it). Release slots
// whose consumer shadow has EXACTLY the warm forward's row count; word
// shadows, persistent weight stagings and the embed twin live in other
// holders and are untouched. Recipe-gated via VT_TT_RELEASE_WARM_ROWS.
void ReleaseWarmShapeSlots(uint32_t rows) {
  std::vector<ttnn::Tensor> dead;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    for (auto& kv : Slots()) {
      BufferSlot& s = kv.second;
      if (!s.device) continue;
      const auto ls = s.device->logical_shape();
      if (ls.rank() >= 1 && ls[0] == rows) {
        dead.push_back(*s.device);
        s.device.reset();
      }
    }
  }
  if (!dead.empty()) {
    std::fprintf(stderr,
                 "[TT-WARM-RELEASE] rows=%u: released %zu warm slot(s)\n",
                 rows, dead.size());
    TTReclaimPlanes(SharedMeshDevice(), dead);
  }
}

// W4d W6: stage the keep-quant word shadow for ONE block-quant weight at
// LOAD time (the residency pre-pass). A lazy first-matmul stage at 27B
// asked the head's 1 GiB words against a fragmented, 95-percent-full bank
// set and died; staged here — immediately after the weight's own resident
// upload, while the banks are unfragmented — the words land contiguously
// and the warm's EnsureKeepQuantWords hits the cache (same key: the
// ResidentWeight device view's pointer). No-op for non-block dtypes.
void StageKeepQuantWordsFor(const Tensor& packed) {
  if (!vt::IsBlockQuant(packed.dtype)) return;
  const int64_t nb =
      static_cast<int64_t>(packed.shape[1]) /
      static_cast<int64_t>(BlockElems(packed.dtype));
  // TT-STAGING-FOOTPRINT: kQ8_0/kQ6_K stage the ROW-PACKED form at load —
  // the tight page-per-row layout the dense int8-dot arm serves (the legacy
  // grid's x1.88/x1.22 inflation is what OOM'd UD-Q6_K_XL on one P150).
  // Q4_K/Q5_K and every other registered encoding keep the legacy grid.
  if (KeepQuantRowPacked(packed.dtype)) {
    (void)EnsureKeepQuantRowWords(packed, packed.dtype,
                                  static_cast<int64_t>(packed.shape[0]), nb,
                                  SharedMeshDevice());
    return;
  }
  (void)EnsureKeepQuantWords(packed, packed.dtype,
                             static_cast<int64_t>(packed.shape[0]), nb,
                             SharedMeshDevice());
}

int64_t FreeDeviceDramBytesForTest() {
  MeshDevice& device = SharedMeshDevice();
  const auto view = tt::tt_metal::detail::GetMemoryView(
      &device, tt::tt_metal::BufferType::DRAM);
  return static_cast<int64_t>(view.num_banks) *
         static_cast<int64_t>(view.total_bytes_free_per_bank);
}

// TT-27B-STEP-DECOMPOSE (VT_TT_STEP_PHASES): free DRAM across banks — the
// public-name twin of FreeDeviceDramBytesForTest above (which stays test-
// scoped), so the model-side phase clock can read the per-step retention
// state without a ForTest name riding in product code.
int64_t DeviceDramFreeBytes() { return FreeDeviceDramBytesForTest(); }

// Total DRAM across banks — the number the placement fit needs as the
// platform's probed total (ISSUE-LOCAL-01M2ACXRJYFW7R7BP2ABQS3VY2: without
// it the budget is UNKNOWN and the MoE fit refuses, landing the model on
// CPU). Same view the free-bytes helper reads; capacity, not free.
int64_t DeviceDramTotalBytes() {
  MeshDevice& device = SharedMeshDevice();
  const auto view = tt::tt_metal::detail::GetMemoryView(
      &device, tt::tt_metal::BufferType::DRAM);
  return static_cast<int64_t>(view.num_banks) *
         static_cast<int64_t>(view.total_bytes_per_bank);
}

// Attribution (W4d W3, ISSUE-LOCAL-01M2AA4ZVD9EWJG5NNQD5DWZXS): walk the
// slot table and dump resident device bytes by holder — the consumer shadow
// (`device`), the persistent staged buffer (`persistent`), the gemma affine
// form — plus the top holders by volume. The 27B warm pass leaves ~14 GiB
// of slot residency; this census names which slots hold it.
void DumpSlotCensus(const char* label) {
  std::lock_guard<std::mutex> g(SlotMutex());
  auto bytes_of = [](const ttnn::Tensor& t) -> size_t {
    size_t esz = 4;
    switch (t.dtype()) {
      case ttnn::DataType::BFLOAT16: esz = 2; break;
      case ttnn::DataType::UINT8: esz = 1; break;
      default: esz = 4; break;
    }
    size_t n = 1;
    for (uint32_t d : t.logical_shape()) n *= d;
    return n * esz;
  };
  size_t dev_bytes = 0, pers_bytes = 0, gemma_bytes = 0;
  size_t n_dev = 0, n_pers = 0;
  std::map<std::string, std::pair<size_t, size_t>> dev_hist;  // shape -> bytes, slots
  std::map<std::string, std::pair<size_t, size_t>> pers_hist;
  for (auto& kv : Slots()) {
    BufferSlot& s = kv.second;
    if (s.device) {
      const size_t b = bytes_of(*s.device);
      dev_bytes += b;
      ++n_dev;
      auto& e = dev_hist[std::to_string(s.dev_rows) + "x" +
                         std::to_string(s.dev_cols)];
      e.first += b;
      e.second += 1;
    }
    if (s.persistent) {
      const size_t b = bytes_of(*s.persistent);
      pers_bytes += b;
      ++n_pers;
      auto& e = pers_hist[std::to_string(s.persist_rows) + "x" +
                          std::to_string(s.persist_cols)];
      e.first += b;
      e.second += 1;
    }
    if (s.gemma_device) gemma_bytes += bytes_of(*s.gemma_device);
  }
  std::fprintf(stderr,
               "[TT-SLOT-CENSUS] %s: dev=%zu B in %zu slots, "
               "pers=%zu B in %zu slots, gemma=%zu B\n",
               label, dev_bytes, n_dev, pers_bytes, n_pers, gemma_bytes);
  for (auto& [shape, e] : dev_hist)
    std::fprintf(stderr, "[TT-SLOT-CENSUS]   dev %s: %zu B in %zu slots\n",
                 shape.c_str(), e.first, e.second);
  for (auto& [shape, e] : pers_hist)
    std::fprintf(stderr, "[TT-SLOT-CENSUS]   pers %s: %zu B in %zu slots\n",
                 shape.c_str(), e.first, e.second);
}
void ResetAllocTraceForTest() {
  std::lock_guard<std::mutex> g(AllocTraceSt().mtx);
  AllocTraceSt().snapshot_count = 0;
  AllocTraceSt().prev_free_total = -1;
  AllocTraceSt().max_alloc_delta = 0;
}

}  // namespace vt::tenstorrent

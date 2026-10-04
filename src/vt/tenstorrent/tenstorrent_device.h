// Shared Tenstorrent mesh-device lifecycle for the backend, op providers, and
// the platform registrar (BACKEND-TENSTORRENT W0, .agents/specs/
// tenstorrent-backend.md). vllm.cpp original; no upstream mirror.
#pragma once

#include <cstdint>
#include <vector>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>

// ttnn::MeshDevice (ttnn/api/ttnn/device.hpp) is a `using` alias for this real
// type, brought into ttnn:: scope via `using namespace device;` there — the
// alias itself is not forward-declarable, so this header names the concrete
// type it resolves to.
namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

// Forward declarations: this header stays light (the backend TU includes
// only vt/backend.h); the definition links against vt/ops.h types.
// MSVC C4099 (W-X): the real definitions are structs (vt/tensor.h:15,
// vt/device.h:107); forward-declaring them as class breaks /WX builds on
// windows the moment a TU includes vt/ops.h before this header.
namespace vt { struct Tensor; struct Queue; }

namespace vt::tenstorrent {

using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// True iff at least one Tenstorrent device is enumerable on this host. Cheap
// (tt::tt_metal::GetNumAvailableDevices(), no device open) — the same
// runtime-probe-before-registering shape vulkan_context.h's Available() uses,
// so a Tenstorrent-enabled build on a host with no Blackhole card registers
// nothing instead of throwing during static init.
bool DeviceAvailable();

// True iff host-free decode is enabled. Default ON; VT_TT_HOST_FREE_DECODE=0
// opts out (reproduces the pre-flip default path). No function-local static
// caching: tests toggle this env per case in one process (#1604).
inline bool HostFreeDecodeEnabled() {
  const char* e = std::getenv("VT_TT_HOST_FREE_DECODE");
  return e == nullptr || std::string_view(e) != "0";
}

// True iff captured decode is enabled. Default ON since the #1625 flip — same
// parse as HostFreeDecodeEnabled ("0" opts out and restores the eager
// host-free arm) — where before the flip capture additionally required the
// env to be present, because the captured arm hung on the first multi-request
// run. That hang was the short-chunk device KV push clobber (#2669), repaired
// on this row; the captured multi-request battery is 16/16 green and
// hang-free. The parity test's golden selection calls this same function, so
// the gate can never adjudicate a different arm than the engine runs. No
// function-local static caching: tests toggle this env per case in one
// process.
inline bool DecodeCaptureEnabled() {
  const char* e = std::getenv("VT_TT_DECODE_CAPTURE");
  return e == nullptr || std::string_view(e) != "0";
}

// The pre-flip polarity: capture only when EXPLICITLY requested. The decode
// graph drivers whose TT captured arm has no committed gate evidence conjunct
// this, so the #1625 default-on flips only the arm classes with evidence
// (Qwen3-dense, Mistral-7B and, since the GDN device-pure wave, the
// Qwen3.5-GDN dense driver: its captured arm resolves the #2812 blocker —
// the GDN ops are device-resident under capture — and gates against its own
// committed teacher-forced pair).
inline bool DecodeCaptureRequested() {
  const char* e = std::getenv("VT_TT_DECODE_CAPTURE");
  return e != nullptr && std::string_view(e) != "0";
}

// The #1625 flip's evidence families: Qwen3 dense causal-LM, Mistral
// checkpoints and the Qwen3.5 GDN dense driver. Each captured arm gates
// against its own committed teacher-forced pair (0.6B/4B, Mistral-7B,
// Qwen3.5-0.8B — #2812); Llama and InternLM2 construct the SAME graph class
// and keep the explicit opt-in until their captured arms are brought up
// (#2812).
inline bool DecodeCaptureDefaultArch(
    const std::vector<std::string>& architectures) {
  for (const auto& a : architectures) {
    if (a == "Qwen3ForCausalLM") return true;
    if (a == "MistralForCausalLM") return true;
    if (a == "Qwen3_5ForConditionalGeneration") return true;
  }
  return false;
}

// Opens (lazily, once) and returns the single process-wide mesh device this
// W0 skeleton targets — device index 0 only, no multi-device mesh. Throws if
// DeviceAvailable() was not already checked true. DELIBERATELY LEAKED (see
// the .cpp): a plain static shared_ptr's destructor reproducibly segfaults at
// process exit, inside MeshDevice's own teardown chain, a cross-DSO static
// destruction ordering hazard the real device-driver state survives fine
// either way.
MeshDevice& SharedMeshDevice();

// Host-allocation registry for device-resident shadows (BACKEND-TENSTORRENT
// residency). Implemented in tenstorrent_ops.cpp (the only TU that links
// ttnn); called from TenstorrentBackend::Alloc/Free/Copy so Free drops any
// cached ttnn::Tensor that still owns device pages for that host pointer.
// No-ops until the ops registrar has loaded (static init order: backend may
// Free before ops if a test tears down early — Unregister is tolerant).
// Defined in tenstorrent_ops.cpp (needs ttnn). Declared here so model TUs
// can ask for a device-side readback of a staged tensor without linking
// ttnn themselves. `Queue`/`Tensor` are vt types (vt/ops.h is in scope
// wherever this header is included after it).
std::vector<float> DebugDeviceReadbackF32(Queue& q, const Tensor& t);
// TRUSTED dump: whole tensor, typed header, dual-read verified (see ops.cpp).
void TrustDump(Queue& q, const char* dir, const char* name, const Tensor& t);
// Review finding F1 (#1715): these two are defined only in the TT-gated ops
// TU, but model TUs call them under a RUNTIME env check. Follow the
// WarmPagedKvShadow pattern: inline no-ops when the backend is not linked,
// so a default (non-tt-metal) configure of examples still links.
#ifdef VLLM_CPP_TENSTORRENT
#else
inline std::vector<float> DebugDeviceReadbackF32(Queue&, const Tensor&) {
  return {};
}
inline void TrustDump(Queue&, const char*, const char*, const Tensor&) {}
#endif
void RegisterHostBuffer(void* host, size_t bytes);
void UnregisterHostBuffer(void* host);
// Host bytes at `host` (or any interior pointer into that allocation) were
// written by a host-side path (Backend::Copy, Memset, host op). Invalidates
// any device shadow so the next EnsureDevice re-uploads.
void MarkHostWritten(void* host);
// DevicePool::Get hands a retained block to a NEW tenant: the slot state
// MarkHostWritten registers, plus the W7 (#2282) reservation — the resident
// device allocation and the host bytes both belong to the previous tenant, so
// the new tenant's first device use is served WITHOUT staging; the pending op
// establishes the content by writing the buffer it is served. Any host write
// (CommitHost / MarkHostWritten) clears the reservation and restores the
// upload-on-stale contract.
void MarkScratchAcquired(void* host);
// If `host` is inside a registered buffer whose device shadow is the source
// of truth, download to host. Used by Backend::Copy so D2H-style reads see
// device-resident results without every op writing host eagerly.
void EnsureHostBytes(void* host);
// HOST-FREE-FORWARD R2: when capture is active and BOTH dst and src carry a
// current device shadow, do a device->device copy (ttnn) instead of staging
// through host. Returns true if it performed a device copy, false if the
// caller should fall back to host memcpy.
bool CopyDeviceDeviceIfCapture(void* dst, const void* src, size_t bytes);
// HOST-FREE-FORWARD R3: when capture/host-free is active, fill the buffer's
// device shadow on-device (ttnn::zeros, matching the shadow's own
// shape/dtype) instead of host memset. Only value==0 is handled; every other
// value declines so Backend::Memset falls back to host memset. Requires the
// buffer to already carry a current device shadow.
bool MemsetDeviceIfCapture(void* p, int value, size_t bytes);
// BACKEND-TENSTORRENT-QWEN35 W7 (#2282): the EAGER arms of the same two
// primitives, for steps that run outside capture and outside host-free decode.
// MemsetDeviceFill zero-fills a device-resident slot's shadow on-device for a
// FULL-slot memset(0) and KEEPS the shadow (the caller still memsets the host
// bytes), so the next device read serves the shadow instead of restaging the
// zeros. CopyDeviceDeviceIfResident runs the device->device copy whenever the
// source shadow is current and the destination already owns a persistent
// buffer — a cold destination is a host reader's buffer (d2h readback, dump)
// and keeps the host path. Each returns false when a precondition does not
// hold and the caller falls back to the host write it has always done.
bool MemsetDeviceFill(void* p, int value, size_t bytes);
bool CopyDeviceDeviceIfResident(void* dst, const void* src, size_t bytes);

// BACKEND-TENSTORRENT-KEEPQUANT W3 capture-safety probe: how many staging
// writes (host repack + from_vector upload) the keep-quant decode path has
// performed while a trace capture was active. The staged arm performs ZERO —
// the i32 word shadow is resident before capture and the per-call decode
// recomputes from it — so any positive count inside a captured run is the
// #2812 class (a captured graph reading a buffer its replay cannot refresh).
// The reset hook exists for the op-level red-first test only. Defined only
// when the backend is built (tenstorrent_ops.cpp); inline no-ops otherwise,
// the WarmPagedKvShadow pattern, so device-agnostic TUs may call them.
#ifdef VLLM_CPP_TENSTORRENT
int64_t KeepQuantCaptureStagingWrites();
void ResetKeepQuantCaptureStagingWritesForTest();

// BACKEND-TENSTORRENT-KEEPQUANT W4a wave-3a test hooks (the red/green
// measurement pair for the chunked E=1 arm):
//  - LastTraceBytesForTest: the device-reported total live trace-buffer
//    bytes as of the last EndCapture — the observable for the wave-1b
//    falsification class (a whole-weight tile inside a captured graph
//    demanding 425,754,624 B against the 52,428,800 B region). The reading
//    accumulates across every live trace, so tests release a graph before
//    measuring the next one in isolation.
//  - KeepQuantChunkRowsOverrideForTest: forces the E=1 slice-decode chunk
//    row count (0 = the production policy default), so the sweep and the
//    bit-exact legs execute MANY chunks at shapes whose policy chunk would
//    cover the whole [N, K] slice in one pass.
int64_t LastTraceBytesForTest();
void KeepQuantChunkRowsOverrideForTest(int64_t rows);

// BACKEND-TENSTORRENT-KEEPQUANT W4a wave-3b-1 residency-policy probes, one
// per shadow map, so the twin policy is a measurement and not a claim:
//  - KeepQuantWordShadowPresentForTest: the PACKED i32 word shadow for a host
//    weight is staged. The dense keep-quant matmul's residency — the chunked
//    E=1 arm decodes from it every call.
//  - DecodedWeightShadowPresentForTest: a decoded bf16 TWIN for a host weight
//    exists. The dense keep-quant matmul must NOT build one — an entry here
//    for a matmul weight is the wave-2 twin residency surviving the switch.
//  - EmbedTableShadowPresentForTest: the embedding gather's bf16 table twin
//    exists — the gather-class survivor the twin policy deliberately keeps
//    (the vehicle's tied head shares the table: its GATHER keeps the twin,
//    its MATMUL stages only the packed words).
bool KeepQuantWordShadowPresentForTest(const void* host);
bool DecodedWeightShadowPresentForTest(const void* host);
bool EmbedTableShadowPresentForTest(const void* host);
// W4d W0 (#3042): device-side allocation trace — interleaves
// get_memory_view snapshots between ops to attribute the 27B OOM.
// Gated by VT_TT_ALLOC_TRACE. The snapshot count and max-allocation-delta
// probes are the red-first observables: zero before any device work,
// positive after the keep-quant path runs instrumented.
void AllocTraceSnapshot(MeshDevice& device, const char* label);
int64_t AllocTraceSnapshotCountForTest();
int64_t AllocTraceMaxDeltaForTest();
void ResetAllocTraceForTest();
// W4d W2 (#3042): total free DRAM across banks, the residency-reclaim
// observable — the W1 trace proved the decode planes never return to the
// allocator; this is the red-first probe the reclaim test asserts on.
int64_t FreeDeviceDramBytesForTest();
// Total DRAM across banks (capacity, not free) — the platform's probed
// total for the placement fit (ISSUE-LOCAL-01M2ACXRJYFW7R7BP2ABQS3VY2).
int64_t DeviceDramTotalBytes();
// W4d W3 focused-test hook: stage `t`'s bf16 TILE form exactly as
// EnsureDevice2D does (creating the slot's persistent buffer), reproducing
// the loader's weight staging before a keep-quant matmul.
void StageWeightBf16ForTest(const Tensor& t, MeshDevice& device);
// W4d W6: stage the keep-quant word shadow for a block-quant weight at
// LOAD time (the residency pre-pass; no-op for non-block dtypes).
void StageKeepQuantWordsFor(const Tensor& packed);
// Attribution (W4d W3): dump resident slot bytes by holder (device shadow /
// persistent staged buffer / gemma) + top holders. Gated by the caller.
void DumpSlotCensus(const char* label);
// W4d W3: release consumer shadows whose rows match the warm forward's
// shape (recipe-gated via VT_TT_RELEASE_WARM_ROWS; see the ops-side comment).
void ReleaseWarmShapeSlots(uint32_t rows);
// TT-27B-STEP-DECOMPOSE (VT_TT_STEP_PHASES): the captured decode step's
// host-side phase clock. READ-ONLY instrument — it brackets existing calls
// and reads existing counters, touches no numeric surface, and the anchor's
// token stream is byte-identical with the knob set (the row's gate 1). Zero
// cost when the env is unset: every entry point is a getenv compare.
//  - StepPhasesEnabled/StepPhaseMode: the knob and its raw value ("1" = the
//    per-step phase lines; "sync" additionally enables the per-layer probe).
//  - StepPhaseNoteLaunch: record the pending launch timestamp (the trace
//    replay enqueue inside TraceReplayGraph, or an eager forward's last
//    enqueue) so the FIRST blocking host read after it can report the step's
//    completion wait.
//  - StepPhaseReadBegin/End: bracket the blocking read (EnsureHostBytes'
//    to_vector); End prints the per-read sync line and consumes the pending
//    launch. A read with no pending launch reports its own duration (an
//    eager path's device tail).
//  - DeviceDramFreeBytes: free DRAM across banks — the per-step retention
//    read-out (the ~950 MB/request staircase, tt-metal#57970).
//  - StepPhaseSyncProbe: a blocking queue drain on a persistent 1-element
//    device tensor, so the per-layer sampling probe (VT_TT_STEP_PHASES=sync)
//    can report true per-layer DEVICE time on the eager pass. The probe
//    tensor is created on first use OUTSIDE capture and never freed (#1486).
//  - TraceCaptureActive: is a trace capture open right now (the layer probe
//    is inert during capture — a readback inside the region is prohibited).
bool StepPhasesEnabled();
const char* StepPhaseMode();
void StepPhaseNoteLaunch(const char* kind);
void StepPhaseReadBegin();
void StepPhaseReadEnd(int64_t bytes);
int64_t DeviceDramFreeBytes();
void StepPhaseSyncProbe();
bool TraceCaptureActive();
#else
inline int64_t KeepQuantCaptureStagingWrites() { return 0; }
inline void ResetKeepQuantCaptureStagingWritesForTest() {}
inline int64_t LastTraceBytesForTest() { return 0; }
inline void KeepQuantChunkRowsOverrideForTest(int64_t) {}
inline bool KeepQuantWordShadowPresentForTest(const void*) { return false; }
inline bool DecodedWeightShadowPresentForTest(const void*) { return false; }
inline bool EmbedTableShadowPresentForTest(const void*) { return false; }
inline void AllocTraceSnapshot(MeshDevice&, const char*) {}
inline int64_t AllocTraceSnapshotCountForTest() { return 0; }
inline int64_t AllocTraceMaxDeltaForTest() { return 0; }
inline void ResetAllocTraceForTest() {}
inline int64_t FreeDeviceDramBytesForTest() { return 0; }
inline int64_t DeviceDramTotalBytes() { return 0; }
inline void StageWeightBf16ForTest(const Tensor&, MeshDevice&) {}
inline void StageKeepQuantWordsFor(const Tensor&) {}
inline void DumpSlotCensus(const char*) {}
inline void ReleaseWarmShapeSlots(uint32_t) {}
inline bool StepPhasesEnabled() { return false; }
inline const char* StepPhaseMode() { return ""; }
inline void StepPhaseNoteLaunch(const char*) {}
inline void StepPhaseReadBegin() {}
inline void StepPhaseReadEnd(int64_t) {}
inline int64_t DeviceDramFreeBytes() { return 0; }
inline void StepPhaseSyncProbe() {}
inline bool TraceCaptureActive() { return false; }
#endif

// ITEM 5 (rope): driver-side warm hook — populate the persistent device
// cos/sin tensors for the step's positions BEFORE BeginCapture (the
// SizeSlot::Refresh slot), so the captured rope cache-HITs. No-op unless
// VT_TT_HOST_FREE_DECODE is set. vt::RopeArgs is declared in vt/ops.h
// (included by every TU that needs the args); this header stays ttnn-free.
// (Plain-field args keep this header free of vt/ops.h; llama3 rope scaling
// is NOT supported on the warm path — TT host-free decode is Qwen3/Mistral
// plain-rope only, matching the current allowlist.)
// ITEM 5 (RAC): eagerly create the paged-KV device shadow (ttnn tensor) for
// the given k_cache / v_cache host buffers during warmup, so the captured
// RAC + PA find the shadow without an in-region upload. No-op unless
// VT_TT_HOST_FREE_DECODE. Takes raw host ptrs + geometry (no ttnn types).
#ifdef VLLM_CPP_TENSTORRENT
void WarmPagedKvShadow(void* k_cache_data, void* v_cache_data,
                      int64_t num_blocks, int64_t block_size,
                      int64_t num_kv_heads, int64_t head_size,
                      int64_t used_blocks);
// TEST-ONLY (the DeviceShadowExact pattern): read the paged-KV DEVICE shadow
// for this cache buffer back to host (row-major [nb, bs, nkv, d] floats), so
// a focused case can verify what the captured RAC replay wrote without going
// through the stale host master.
bool ReadPagedKvShadowForTest(const void* k_cache_data, float* dst, int64_t n);
#else
inline void WarmPagedKvShadow(void*, void*, int64_t, int64_t, int64_t, int64_t,
                              int64_t) {}
inline bool ReadPagedKvShadowForTest(const void*, float*, int64_t) {
  return false;
}
#endif

// GDN conv-state shadow serveability (decode side): true when the transposed
// [sl+1, slots*C] f32 TILE shadow that the captured CausalConv1dUpdate
// fast-path serves is current for this buffer. A prefill-bearing step's
// ssm/cache role transition (GdnStateGather) clears it; the decode driver
// must then run the step eagerly — which rebuilds the shadow — BEFORE any
// graph may capture or replay against the buffer again.
#ifdef VLLM_CPP_TENSTORRENT
bool ConvShadowServeable(const void* conv_state_data, int64_t slots,
                         int64_t conv_dim, int64_t state_len);
#else
inline bool ConvShadowServeable(const void*, int64_t, int64_t, int64_t) {
  return true;
}
#endif

// (W2 of the capture-warmup redesign removed the GDN-shadow
// snapshot/restore API: the warmup's scatter commit IS the geometry the
// capture's first read serves, so restoring pre-warmup shadows only
// discarded the warmup's state update — the frozen GDN recurrence.
// ISSUE-LOCAL-01M3918KQ580Z3NHVRNXVF15FZ.)
// for THIS slot mapping, outside capture (driver Refresh slot). No-op unless
// VT_TT_HOST_FREE_DECODE. slot_mapping_owner is the host buffer the captured
// ReshapeAndCache will see as its slot_mapping (keyed identity). page_table is
// the block table (virtual→physical block mapping, int32, [num_reqs, cols]).
// positions are the sequence positions for this step (int32). Both are used to
// build the update_idxs (positions) and page_table tensors paged_update_cache
// reads at replay time.
#ifdef VLLM_CPP_TENSTORRENT
void WarmRacIdx(const void* slot_mapping_owner, const int64_t* slots,
                int64_t num_slots, int64_t block_size,
                const int32_t* page_table, int64_t page_table_cols,
                const int32_t* positions);
#else
inline void WarmRacIdx(const void*, const int64_t*, int64_t, int64_t,
                       const int32_t*, int64_t, const int32_t*) {}
#endif

// ITEM 5 (PA): warm persistent page_table + cur_pos device tensors.
// When `advance_on_device` is true, cur_pos is seeded once and then advanced
// in-trace via CaptureDecodePosAdvance (ttnn::plus_one); WarmPaMeta's cur_pos
// copy_to_device is skipped in steady-state replay (R2 on-device state advance).
#ifdef VLLM_CPP_TENSTORRENT
void WarmPaMeta(const int32_t* block_table, int64_t num_reqs, int64_t max_blocks,
                int64_t bt_row_stride, int64_t bt_col_stride,
                const int32_t* seq_lens);
#else
inline void WarmPaMeta(const int32_t*, int64_t, int64_t, int64_t, int64_t,
                       const int32_t*) {}
#endif

// R2 (on-device state advance): seed the persistent cur_pos device tensor
// (= seq_lens - 1) for this step. Called on every cold/warm/capture step
// (re-seed), never on a replay step — the captured plus_one advances it
// on-device. Also warms the plus_one program (program cache) so
// CaptureDecodePosAdvance can run inside the trace without a "load new
// binaries during capture" fatal.
#ifdef VLLM_CPP_TENSTORRENT
// replay_regime=true: this step replays a captured trace whose plus_one owns
// the cur_pos advance — do not touch the tensor. replay_regime=false (cold,
// warm, or capture step): re-seed cur_pos = seq_lens-1 for THIS step. The
// re-seed is what makes a RE-capture correct (#1476): Reset() releases the
// trace but the cold eager step that follows runs no plus_one, so without a
// re-seed the newly captured trace reads cur_pos one position behind.
void WarmDecodePos(const int32_t* seq_lens, int64_t num_reqs, bool replay_regime);
#else
inline void WarmDecodePos(const int32_t*, int64_t, bool) {}
#endif

// R2: capture ttnn::plus_one(cur_pos) at the END of the trace body (after all
// reads of cur_pos in sdpa_decode / paged_update_cache), so the NEXT replay
// sees cur_pos+1. Must be called INSIDE BeginCapture/EndCapture.
#ifdef VLLM_CPP_TENSTORRENT
void CaptureDecodePosAdvance(int64_t num_reqs);
#else
inline void CaptureDecodePosAdvance(int64_t) {}
#endif

// HOST-FREE-DECODE: stage the persistent device decode-ids tensor (UINT32 [n],
// ROW_MAJOR) for THIS step's token ids, outside capture. The captured
// embedding reads this stable address; each replay step only refreshes its
// content (copy_to_device — allocation-free). No-op unless
// VT_TT_HOST_FREE_DECODE.
#ifdef VLLM_CPP_TENSTORRENT
void WarmDecodeIds(const int32_t* ids, int64_t n);
#else
inline void WarmDecodeIds(const int32_t*, int64_t) {}
#endif

// HOST-FREE-DECODE: capture-safe embedding over the persistent decode-ids
// tensor (see WarmDecodeIds) into the hidden buffer whose host base is
// `out_host` — the same device->device in-place refresh EmbeddingKernel
// performs on its warm path, but with no host ids upload, so it can run
// INSIDE the captured region. `table_host` is the embedding table's host
// base (device shadow cached by the kEmbedding path). Requires WarmDecodeIds
// to have staged ids for this n first.
#ifdef VLLM_CPP_TENSTORRENT
void EmbedDeviceIdsInto(void* out_host, int64_t rows, int64_t cols,
                        const void* table_host, int64_t vocab, int64_t hidden,
                        int64_t n);
#else
inline void EmbedDeviceIdsInto(void*, int64_t, int64_t, const void*, int64_t,
                               int64_t, int64_t) {}
#endif



#ifdef VLLM_CPP_TENSTORRENT
void WarmRopeCosSin(const int32_t* positions, int64_t tokens, int64_t hq,
                    int64_t hk, int64_t rot, double base);
#else
inline void WarmRopeCosSin(const int32_t*, int64_t, int64_t, int64_t, int64_t,
                           double) {}
#endif

// Fused-preamble cos|sin table (kAttnQkNormRopeGate): the dense decode-graph
// driver refreshes the persistent device [T, rot] table OUTSIDE capture every
// step so the captured preamble serves the step's content (WarmRopeCosSin
// pattern). No-op off the TT host-free-decode lane.
#ifdef VLLM_CPP_TENSTORRENT
void WarmAttnCosSin(const int32_t* positions, int64_t tokens, int64_t rot,
                    double base);
#else
inline void WarmAttnCosSin(const int32_t*, int64_t, int64_t, double) {}
#endif

// TT-GDN-REGION-REPLAY attribution instrument: checksum the CURRENT device
// shadow a GDN state cache (ssm or conv) holds, between steps and outside
// capture (the driver calls this from the decode graph; a replay runs no
// host code and a download is the read a trace capture refuses). Advances
// print changing FNVs; a slot frozen at the capture-time values prints one
// identical checksum at every replay. Read-only; inert while a trace capture
// is open.
#ifdef VLLM_CPP_TENSTORRENT
void GdnShadowProbe(const void* host_ptr, const char* what);
#else
inline void GdnShadowProbe(const void*, const char*) {}
#endif

// ---- ttnn mesh-trace capture (Backend graph-capture mapping) --------------
// Maps vt::Backend::{BeginCapture,EndCapture,Replay} onto
// ttnn::operations::trace::{begin,end,execute}_trace_capture. Implemented in
// tenstorrent_ops.cpp so the backend TU stays free of ttnn headers.
//
// Contract (same class as CUDA graphs): every op between Begin and End must
// stay async on the mesh CQ with fixed device buffers; host Ensure/Download
// and fresh program compiles during capture are illegal and will throw.
// Warm the program cache with an identical shape before BeginCapture.
void TraceBeginCapture();
void TraceEndCapture();           // stores the default single-slot replay id
void TraceReplay();               // replay the single-slot id
void* TraceEndCaptureGraph();     // returns an opaque MeshTraceId* (caller owns)
void TraceReplayGraph(void* graph);
void TraceDestroyGraph(void* graph);

// ---- BACKEND-TENSTORRENT-GDN W2: GDN state/conv-shadow PCIe traffic counters
// The spec's Evidence gate requires the "no per-token state round-trip" claim
// PROVEN by a counter, not assumed. The counters cover ONLY the state /
// conv-state caches the GDN ops own (decode state, conv state, the gather /
// scatter cache): bytes the ops themselves upload (h2d) or download (d2h), and
// one step per kGdnDecode / kCausalConv1dUpdate invocation. A decode replay of
// T tokens must show steps == T, h2d == exactly ONE upload of the cache, and
// d2h == 0. Implemented in tenstorrent_ops.cpp; ttnn-free on the no-op arm.
struct GdnShadowTraffic {
  uint64_t state_h2d_bytes = 0;  // state/conv cache host→device uploads
  uint64_t state_d2h_bytes = 0;  // state/conv cache device→host downloads
  uint64_t decode_steps = 0;     // kGdnDecode + kCausalConv1dUpdate calls
};
#ifdef VLLM_CPP_TENSTORRENT
GdnShadowTraffic GetGdnShadowTraffic();
void ResetGdnShadowTraffic();
#else
inline GdnShadowTraffic GetGdnShadowTraffic() { return {}; }
inline void ResetGdnShadowTraffic() {}
#endif

// ---- BACKEND-TENSTORRENT-QWEN35 W4 (#2107): bulk host→device staging -------
// counters. The W4 gate pins the ROUTE, not only the bytes: EnsureDevice2D
// must stage a contiguous bf16 master through the bulk path (raw bf16 bytes
// handed to ttnn from_span — no f32 intermediate, no per-element dtype
// dispatch), and the counter is the observable that proves the route ran for
// a given staging. staged_f32_elems counts elements that still go through the
// per-element f32 reference path (f16/f32 masters, genuine conversions — the
// f32 logits GEMM output keeps its declared dtype). Implemented in
// tenstorrent_ops.cpp; ttnn-free on the no-op arm.
struct StagingStats {
  uint64_t uploads_bulk_bf16 = 0;   // bulk bf16 uploads from EnsureDevice2D
  uint64_t staged_bulk_bf16_bytes = 0;
  uint64_t staged_f32_elems = 0;    // elements staged via the f32 path
  // ---- BACKEND-TENSTORRENT-QWEN35 W5 (#2244): allocation-free staging ----
  // Subdivision of uploads_bulk_bf16 (which keeps counting every bulk bf16
  // staging, W4 route pin unchanged). uploads_persistent_bf16 counts the
  // uploads written IN PLACE into the per-slot persistent device buffer via
  // the mesh command queue; uploads_persistent_allocs counts the (re)
  // allocations of that buffer — cold slot or staging-geometry change. A
  // steady-state decode step must show allocs == 0 and every bulk upload on
  // the persistent route; churn in allocs is visible debt for the profile.
  uint64_t uploads_persistent_bf16 = 0;
  uint64_t uploads_persistent_allocs = 0;
  uint64_t staged_persistent_bf16_bytes = 0;
  // ---- BACKEND-TENSTORRENT-QWEN35 W7 (#2282): staging-write elimination ----
  // The W6 probe read the per-step staging-write count off
  // uploads_persistent_bf16 (~7-8/step) and traced every one to the residency
  // state dropping a shadow a consumer could have served. These count the
  // staging writes W7 eliminates, per class: reservation (a pool-acquired
  // scratch block served from its resident allocation instead of restaging
  // garbage), device_memset (an eager full-slot zero-fill that keeps the shadow
  // alive), device_copy (a device-resident D2D copy that skips the
  // download+restage pair). A post-W7 step's write count is
  // uploads_persistent_bf16; the avoided counters attribute which precise-
  // residency arm removed each write the pre-W7 baseline paid.
  uint64_t stages_avoided_reservation = 0;
  uint64_t stages_avoided_device_memset = 0;
  uint64_t stages_avoided_device_copy = 0;
};
#ifdef VLLM_CPP_TENSTORRENT
StagingStats GetStagingStats();
void ResetStagingStats();
#else
inline StagingStats GetStagingStats() { return {}; }
inline void ResetStagingStats() {}
#endif

}  // namespace vt::tenstorrent

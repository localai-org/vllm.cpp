# Gap analysis: kolibri1 TT captured decode (dispatch batching)

Row: `BACKEND-TENSTORRENT`. Issue:
`ISSUE-LOCAL-01M4N8Y7XPDC74SED0WEDN0YFR` (in
`.agents/issues/BACKEND-TENSTORRENT/`). Date: 2026-10-11. Branch:
`row/tt-kolibri-capture`. Read-only study plus one contained eager
prototype; no capture wiring landed.

## Symptom

kolibri1 P150 decode: ~960 ms/step over 50 layers (~19 ms/layer, ~10
device ops/layer). The profile is host-serialized, not
silicon-bound. The captured-decode mechanism is default ON since #1625
and serves other models on this same card, but kolibri1 never captures.

## How captured decode works (the qwen3.5 reference)

Capture on TT is a ttnn mesh trace: `BeginCapture` /
`EndCaptureGraph` / `ReplayGraph` map onto `ttnn::begin_trace_capture`
/ `end_trace_capture` / `replay_trace` (`src/vt/tenstorrent/tenstorrent_ops.cpp`,
the `Trace*` family; `SupportsGraphCapture() == true`). TT also
overrides `support_static_graph_mode() == true`
(`src/vllm/platforms/tenstorrent.cpp:98`), so the platform gate is
open.

The model-side driver is `Qwen3_5DenseDecodeGraph::Step`
(`src/vllm/model_executor/models/qwen3_5.cpp`, ~11900-12900): a
per-padded-size slot state machine `cold (eager) -> warm -> capture ->
replay`:

- **Driver scaffold**: padded capture sizes (`PadToCaptureSize` /
  `BuildPaddedDecode`), one `SizeSlot` per shape holding PERSISTENT
  hidden and `[S,vocab]` f32 logits buffers, in-place `Refresh()` of
  persistent host inputs, `vt::BreakableGraph` +
  `vt::GraphCaptureScope` (kFull) around `ForwardLayers`, replay per
  step.
- **Warm\* driver hooks** (all OUTSIDE capture, every step; they
  refresh persistent device inputs the captured trace re-reads):
  `WarmAttnCosSin`, `WarmPagedKvShadow` (per KV layer), `WarmDecodePos`
  (with in-trace `CaptureDecodePosAdvance` plus_one for the R2
  continuation), `WarmRacIdx` (persistent RAC idx tensors + sharded
  input; required by the device `TryReshapeAndCacheDeviceDecode` arm),
  `WarmPaMeta`, `WarmDecodeIds` / `EmbedDeviceIdsInto`
  (`src/vt/tenstorrent/tenstorrent_capture.cpp`, `tenstorrent_device.h:309-448`).
- **Pool discipline**: `MarkStepBoundary` / `StepDemandProfile` (cold
  step) / `PreGrowForCapture` / `PinForGraph` (#1380/#2274) so the
  captured region does zero allocations.
- **Host-free captured region**: no `to_vector`/`Download`, no host
  sync, no host-side data-dependent control flow between ops
  (ttnn aborts capture on any device-to-host read inside the region;
  the original NO-GO is recorded in
  `.agents/specs/tenstorrent-trace-runner.md` ## Outcome).

Measured reference point: a served captured replay on this P150 class
runs ~12 ms wall / 0.3 ms replay (`.agents/specs/tenstorrent-capture-economics.md`,
#3327), i.e. three orders of magnitude under kolibri1's 960 ms step.

## What kolibri1 lacks (the hook list, concrete)

`ForwardKolibri1TTResidentForward`
(`src/vllm/model_executor/models/kolibri1_tt_forward.cpp:903`) is a
hand-rolled eager loop. Missing items, each mapped to its qwen3.5
counterpart:

1. **No decode-graph driver**: no size slots, no cold/warm/capture
   state machine, no `vt::BreakableGraph` / `GraphCaptureScope`, no
   padding to capture sizes, no persistent hidden/logits buffers (every
   step allocates fresh `DBuf`s through the whole loop), no
   `Replay()`.
2. **No Warm\* hooks at all**: `grep -n 'Warm' kolibri1_tt_forward.cpp`
   is empty. The step metadata (`Kolibri1BuildStepInputs`, :973) is
   uploaded per step through the generic staging path; nothing primes
   persistent device inputs a trace could bake.
3. **Per-layer device-to-host readback in the MoE router** (:478
   `dlog.Download`) plus (before this branch's prototype) a per-layer
   bias D2H copy with a blocking `Synchronize` (:483-484). Two blocking
   host round-trips per layer, 100 per step. This alone forbids trace
   capture: the router readback sits mid-region.
4. **Host-side data-dependent routing + streaming expert fetch**: the
   sigmoid-logit-add top-k runs on host over the readback
   (`SigmoidLogitAddRouting`, :522); the routed experts then stream from
   host through `BuildKolibri1TTSlotFetchList` with host-built jobs,
   per-expert H2D staging, a full-slot D2H readback pivot with
   byte-exact verification (:693-701), and a CPU dequant per missed
   slot. This is host control flow on data the region computes —
   uncapturable by construction.
5. **Device RAC / PA decode arms decline on kolibri geometry**
   (`src/vt/tenstorrent/tenstorrent_paged.cpp:666` and `:352`):
   `if ((d % 32u) != 0u || (bs % 32u) != 0u) return false;` — head_dim
   and block_size must be TILE multiples for
   `ttnn::paged_scaled_dot_product_attention_decode`. The checkpoint's
   cache block_size (16) fails `bs % 32`, and no `WarmRacIdx` entry
   exists for its shapes (the RAC arm's precondition check at
   `tenstorrent_paged.cpp:766/772` refuses by name).

### Is the geometry decline hard?

The `%32` guard is a ttnn `sdpa_decode` kernel validation, hard for a
fixed block_size layout. But block_size is OUR cache allocation, not a
property of the checkpoint: allocating kolibri1's TT KV cache at
block_size 32 satisfies the guard without touching weights or numerics.
head_dim (128 on the real checkpoint) already passes. So: **not a hard
constraint — a padded-layout/configuration fix** (cache block_size 32
on the TT arm), though the device RAC arm additionally needs a warmed
`WarmRacIdx` entry and TILE-shadow k/v from the device rope path before
it serves.

## Why full captured decode is NOT feasible here (the honest verdict)

Items 1, 2, 5 are the ordinary porting work (the driver scaffold +
hooks + block_size fix; the qwen3.5 template exists and is proven on
this card). Items 3 and 4 are architectural to the B2b-ii streaming-MoE
design and cannot be captured around:

- The routed expert set is data-dependent per token per layer and
  arrives through host-built fetch jobs with host-verified readbacks.
  A captured trace bakes fixed device pointers and fixed op sequences;
  expert streaming changes both every step.
- Making it capturable means the qwen3.5 fused-MoE shape: ALL experts
  resident on device, device-side routing (`vt::MoeRouterTopK` — which
  has NO TT implementation today; `src/vt/` shows cpu/cuda/rocm/vulkan
  only), and grouped expert GEMM on device. kolibri1 has 384 experts x
  50 layers; resident fp8 experts plus dequant working set against the
  P150 DRAM budget that the B2b-ii record already describes as
  allocated to within ~0.6 MB/bank is the open sizing question, and the
  `MoeRouterTopK` TT port plus a grouped-GEMM TT arm are each
  multi-day rows on their own.

Estimate: driver scaffold + Warm hooks + block_size fix alone ≈ 3-5
device-days, and would deliver ~zero win while items 3/4 still serialize
every layer on the host. Capture only pays after the MoE arm is
device-resident — a campaign, not a task. This satisfies the stop
condition and is the deliverable verdict.

## What DOES pay first (eager round-trip elimination)

The step is host-serialized: every blocking D2H in the loop drains the
whole queue. Removing synchronous round-trips needs no capture and
compounds with any later capture work:

- **Router bias host cache** (PROTOTYPED, this branch): the
  `e_score_correction_bias` is a static weight, yet the loop copied it
  D2H with a blocking `Synchronize` every layer, every step — 50
  round-trips per step for constant bytes. Now cached once per device
  buffer (`kolibri1_tt_forward.cpp`, `BiasCache()`). Measured result
  below.
- **Router logits readback** (:478): required by host routing today;
  replaced only by the `vt::MoeRouterTopK` TT port (one small D2H of
  ids/weights instead, if the streaming fetch stays host-driven).
- **Per-expert readback pivot + CPU dequant** (:693-701): a full
  packed-slot D2H plus byte-exact verify per missed slot. Memoized per
  slot, so it decays with resident-slot hit rate, but every miss costs
  megabytes of D2H. A verify-knob (trust the H2D staging for
  non-first-touch) or device-side dequant removes most of it without
  touching numerics-critical paths.

## Prototype result

`tests/vllm/models/test_kolibri1_tt_decode_bench.cpp` (cherry-picked
clean from 4dab09abd on `row/tt-kolibri-decode-bench`) measured on the
P150, same recipe (tt-smi reset, cache/shm clean,
`TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium`):

| arm | prefill walk (128 tok) | decode (64 steps) | ms/step | tok/s |
|---|---|---|---|---|
| baseline run 1 (origin/main) | 193.486 s | 68.918 s | **1077** | 0.929 |
| baseline run 2 (origin/main, control) | 149.693 s | 80.225 s | **1254** | 0.798 |
| bias host cache (prototype) | 165.979 s | 73.701 s | **1152** | 0.868 |

**Verdict: no measurable win.** The prototype sits inside the
run-to-run noise band of the baseline itself (1077-1254 ms/step across
two identical-baseline runs, ~+-10%), and the greedy chains are NOT
reproducible run-to-run in any arm (all three chains differ from the
first token — consistent with the B2b-ii record's state-dependent
7-flip residue; the bench prints, it does not adjudicate).

**Finding (more valuable than the prototype): the per-layer bias copy
is NOT a decode lever.** Removing 50 blocking `Synchronize` points per
step moved nothing, so those syncs were cheap — each waits only on the
tiny router GEMM tail, not the queue. The step's serialization lives
in the round-trips that wait on the FULL layer tail: the router-logits
`Download` (kolibri1_tt_forward.cpp:478, one full-tail drain per
layer) and the streaming expert path (per-miss full-slot D2H readback +
byte-exact verify + CPU dequant, :693-701). The re-ranked eager levers
are therefore: (1) the expert fetch path's readback/verify cost per
miss, (2) device-side routing (`vt::MoeRouterTopK` TT port) to remove
the per-layer full-tail download, (3) cache block_size 32 to un-decline
the device RAC/PA arms. The prototype edit is kept (it removes real
work and is inert to numerics — cached bytes are identical) but is not
a perf result.


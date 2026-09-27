# Tenstorrent host-free forward R1 — device RmsNorm + RoPE at T=1

Status: **DRAFT, 2026-08-13.** First row of the host-free-forward plan
(`tenstorrent-host-free-forward.md`). Sequential: measure after each row to
see its marginal contribution to capture.

Proposed row id: `BACKEND-TENSTORRENT-HOST-FREE-R1`.

## Now

`ACTIVE` on the parent row `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`. R1-R3b
and the R2 on-device `cur_pos` advance are on this branch. Next: fresh review
of the host-free decode graph, then the operator reruns the Qwen3-0.6B
80-token no-hang gate and the TT golden on a Blackhole P150.

## Scope

**In.** Flip the two hybrid thresholds that route `RmsNorm` (residual,
rows<32) and `RoPE` (PreferDeviceRope, T·H<64) to host at T=1 decode, so
both go all-device. The numerics were already proven acceptable by
`BACKEND-TENSTORRENT-RESIDUAL-GOLDEN` (device bf16 vs CPU f32 = constant
0.0459 abs). The device paths already exist in `tenstorrent_ops.cpp`.

The flip MUST be gated on capture-active (`support_static_graph_mode()`) so
non-capture runs keep the 12.5 tok/s hybrid baseline. Inert when capture is
off.

**Out.** The cos/sin host build inside `RopeNeoxKernel`
(`BuildCosSinFromPositions`, line 1363) is a known sub-blocker but is NOT a
device `to_vector` readback — it reads the host `pos` tensor. Whether it
triggers the ttnn fatal is an empirical question R1 answers: after this flip,
does capture get past the current `to_vector` fatal, and what is the NEXT
host touch? (If it's `pos`, R1.5 or R2 handles it; if capture succeeds, R1
alone was enough for the RmsNorm/RoPE portion.) No QkvSplit/ReshapeAndCache/
PagedAttention work (R2/R3).

## Upstream chain

CUDA's capture contract (`cuda_backend.cu:184-197`): the captured region is
async, no host sync, no host readback. TT must match. The RmsNorm device path
(`tenstorrent_ops.cpp:1105-1117`, `ttnn::add`+`ttnn::rms_norm`) and the RoPE
device apply (`RopeApplyDeviceNeox`, line 1221) are the loyal mappings.

## Our baseline

`RmsNormKernel` (line 1067-1118): `host_residual` when rows<32.
`PreferDeviceRope` (line 1344): false when T·H<64. Both host at T=1.
The trace-runner spike measured forcing both all-device: 12.5→10.7 tok/s
eager (the cost capture must recover).

## Work breakdown

1. Add a capture-active helper reading the platform's
   `support_static_graph_mode()` (cached per device, since the platform is
   invariant).
2. Gate `host_residual` and `PreferDeviceRope` on `!capture_active`.
3. Op-level test: confirm RmsNorm + RoPE device path runs at T=1 (rows=1,
   T·H=16) without the host fallback, bit-comparable to the residual-golden
   measurement (0.0459 abs).
4. **Measure**: with `support_static_graph_mode()` also flipped on (R4's
   change, applied locally for the measurement), does capture get past the
   `to_vector` fatal? Record the next failure point if any.

## Gates

- Op-level: RmsNorm + RoPE device output at T=1 within the band already
  measured by RESIDUAL-GOLDEN.
- E2e: Qwen3-0.6B `our_ids_tenstorrent.npy` golden still near-tie-passes.
- Capture probe (informative, not a hard gate for R1 alone): record how far
  capture gets.
- No eager perf regression when capture is OFF (the gate must be inert).

## Dependencies

- `BACKEND-TENSTORRENT-RESIDUAL-GOLDEN` (numerics proof, landed).
- `BACKEND-TENSTORRENT-HOST-FREE-FORWARD` (the plan, this branch).
- Hardware: Blackhole P150.

## Risks/decisions

- **The cos/sin host build may be the next capture blocker.** If after R1
  capture still fatals on a host op inside RopeNeoxKernel, the decision is
  whether R1 is complete (RmsNorm+RoPE *apply* are device) and the cos/sin
  build moves to a separate R1.5, or whether R1 must also switch RoPE to
  the `RopeFromCache` path (model-side change). Settle empirically.
- **The gate must be inert by default.** A bug in the gate that flips
  thresholds unconditionally would ship the 12.5→10.7 regression. Test the
  inert path explicitly.

## Outcome (2026-08-13) — R1 flip alone does NOT unblock capture; readback is inside a ttnn op

Implemented the opt-in gate (`VT_TT_HOST_FREE_DECODE`): flips both
`host_residual` and `PreferDeviceRope` all-device, plus flips the platform
`support_static_graph_mode()` so `Qwen3DenseDecodeGraph` engages. All inert
by default (21/21 TT tests, 814/814 assertions unchanged).

**Measurement (Qwen3-0.6B, `VT_TT_HOST_FREE_DECODE=1`):** capture STILL
fatals — `TT_FATAL: Reads are not supported during trace capture`, `0
replays`. R1's threshold flip is **not sufficient**.

**Diagnostic (the important finding):** with a capture-gated debug print on
every readback site in our code (`DownloadToHost`, `EnsureHost`, the two
direct `dev_out.to_vector` sites in PA decode/prefill), **zero of our
readbacks fire during capture**. The offending `to_vector` is therefore
**inside a ttnn op** (`ttnn::embedding` / `ttnn::rms_norm` /
`ttnn::sdpa_decode` / `to_memory_config` / etc.), not in our explicit
readback code. The TT backtrace shows only the ttnn frame
(`ttnn::Tensor::to_vector<float>`), not which op called it.

**Implication for the plan:** the host-free forward is **not** achievable
by only changing our thresholds/host-staging. At least one ttnn op in the
forward performs an internal host readback that ttnn trace prohibits.
Identifying that op (via a ttnn-symbolized backtrace or bisection) is the
real next step — it determines whether the fix is (a) swap to a
capture-safe ttnn primitive, (b) a ttnn version/bug fix, or (c) the
capture region must exclude that op. This is a deeper blocker than the
plan's R1-R4 assumed; the "host-free forward" may require upstream ttnn
changes, not just vllm.cpp changes.

**R1 code kept** (env-gated, inert by default): the threshold flip is
correct and will be needed once the ttnn-internal readback is resolved.
The `tt_capture_active()` flag + `VT_TT_TRACE_DEBUG` prints are kept as
diagnostics for the next row.

### Correction (2026-08-13, post-bisection): NOT a ttnn-internal readback

The "inside a ttnn op" hypothesis above was **wrong** — it was based on
instrumenting only `DownloadToHost` + the two PA `to_vector` sites, which
missed the fourth readback site: `EnsureHostBytes` (line 2518). A full op
bisection with `TT_OP_TRACE` at every kernel entry + a print in
`EnsureHostBytes` gave the exact sequence:

```
[TT-TRACE] BeginCapture (flag set)
[TT-TRACE] EnsureHostBytes DURING CAPTURE      <- the offender
TT_FATAL: Reads are not supported during trace capture
```

**Zero `TT-OP` kernel-entry lines fired** between BeginCapture and the
fatal — no `*Kernel` ran at all. The capture-blocking readback is in
**our** `Backend::Copy` → `EnsureHostBytes` → `dev.to_vector<float>()`
(line 2533), triggered by `ForwardLayers`'s very first line
(`qwen3.cpp:244`): `d.b.Copy(d.q, hidden.ptr(), hidden_in.data, ...)`.
`hidden_in` has a device shadow from `EmbedInto`; `Copy` forces a
device→host download to satisfy the host-side memcpy, inside the captured
region.

**This is fixable in our code, not an upstream ttnn blocker.** The fix:
when capture is active, `Backend::Copy` from a device-resident source must
do a device→device copy (or `ForwardLayers` must receive the device tensor
directly instead of copying through host). That's a concrete, scoped
R2-target — the "host-free forward" is achievable in vllm.cpp after all,
once every `EnsureHostBytes`/`Backend::Copy` site in the captured region
is made device→device. The R1 threshold flip + this copy fix together
clear the first capture blocker; subsequent readback sites (if any)
surface as the next bisection steps.

### R2 status (2026-08-13): fix site identified, device-copy primitive pending

The bisection pinpointed the exact fix site: `Backend::Copy`
(`tenstorrent_backend.cpp:56`) -> `EnsureHostBytes` -> `to_vector`,
triggered by `ForwardLayers`'s opening `d.b.Copy(...)` (`qwen3.cpp:244`).

Fix shape: when capture is active and both `dst` and `src` carry current
device shadows, `Backend::Copy` must do a device-to-device copy via a new
`CopyDeviceDeviceIfCapture` helper in the ops TU, called before
`EnsureHostBytes`.

Pending: the exact ttnn device-to-device copy primitive for this ttnn
build. Tried `ttnn::clone` (header not on the installed include path
despite the source existing) and `ttnn::copy` + `ttnn::zeros` (wrong API
for this version). The installed ttnn headers are a subset; the right
primitive needs focused API discovery against the installed header set.
R2 code reverted to keep the branch buildable; the
`CopyDeviceDeviceIfCapture` contract is the design, the body is the open
work — the single concrete next step.

### R2 update (2026-08-13): device-copy LANDED, next blocker is ttnn program-cache warm-up

Resolved the ttnn API discovery: the installed headers live in
`build_Release/include/ttnn/operations/...` (not the `libexec` tree). The
device→device copy primitive is `ttnn::copy(src, dst)` (from
`data_movement/copy/copy.hpp`) with a destination allocated via
`ttnn::empty(shape, dtype, layout, device, memconfig)` (from
`creation/creation.hpp`), using `Tensor::logical_shape()`/`dtype()`/`layout()`
accessors. Both headers had to sit inside the Tracy-disabled include block
(they transitively pull the 6-arg `op_profiler_serialize` that breaks the
5-arg TracyC.h). Backend::Copy now calls `CopyDeviceDeviceIfCapture` first;
default tests still 21/21, 814/814 (the path is capture-gated, inert
otherwise).

**Measured:** the R2 copy fix **works** — the capture probe now gets past
the `EnsureHostBytes` readback (`[TT-TRACE] device->device copy (capture-safe)`
fires, no more `Reads are not supported` fatal). The **new** fatal is one
layer deeper:

```
TT_FATAL: Cannot load new binaries during trace capture.
This program is not yet in program cache. Warm up before capturing a trace.
```

This is the **ttnn program-cache warm-up** requirement (Q3 in the original
trace-runner spike, deferred then). ttnn `begin_trace_capture` requires
every op shape in the captured region to be JIT-compiled (program-cache
warm) BEFORE capture begins; the decode-graph framework's single eager
warmup step does not warm the exact shapes the captured path uses (or my
new `ttnn::empty` introduces an un-warmed program).

This is a known ttnn trace discipline with an established pattern (warm
identical shapes via an eager run that hits the same ops), NOT an unknown.
It is the concrete R3 target — and it means the host-free forward *is*
achievable: R1 (thresholds) + R2 (device-copy, landed) clear the readback
blockers; R3 (warm-up) is the last gate before capture can complete.

### R3 update (2026-08-13): warm-up WORKS — capture now runs ops

Fixed the warm-up: `CopyDeviceDeviceIfCapture` now runs whenever
`VT_TT_HOST_FREE_DECODE` is set (not just during capture), so the eager
warmup step also exercises `ttnn::empty`+`ttnn::copy`, compiling them into
the program cache. Also calls `device.enable_program_cache()` once on the
first host-free use (the ttnn trace precondition).

Measured: the "Cannot load new binaries" fatal is gone. Capture now enters
the forward and runs ops:

  device->device copy (eager warmup)
  BeginCapture
  device->device copy (ForwardLayers opening Copy — R2 holds)
  EnsureHostBytes DURING CAPTURE x6   <- next readback blockers
  CastBf16
  RmsNorm                              <- ops run during capture
  TT_FATAL: Writes are not supported during trace capture  <- a buffer write

R2 + R3 together got capture past the first Copy and into the layer ops.
Two new, expected, mechanical blockers surfaced:

1. 6 more EnsureHostBytes readbacks — every Backend::Copy inside
   ForwardLayers (weight uploads, residual init) hits host-staging. Same
   R2 fix at each site.
2. Writes not supported — DBuf::Zero calls Backend::Memset (host memset),
   a host write inside the captured region. Needs a device-zero path or
   pre-zero outside capture.

Conclusion: capture on TT is achievable and now demonstrated working past
the first two blocker layers. Remaining work is converting each
host-staging site (Backend::Copy, Backend::Memset/DBuf::Zero) in the layer
loop to device-resident — mechanical, not research. The bisection
instrumentation surfaces each site in order.

### R3b update (2026-08-13): copy + zero-fill done; device-allocation is the structural blocker

Added MemsetDeviceIfCapture (on-device ttnn::zeros for DBuf::Zero), fixed a
null-deref (std::optional<Tensor>). Default tests 21/21.

Measured: the 6 EnsureHostBytes readbacks are GONE. The sequence now:
  device->device copy (eager warmup)
  BeginCapture
  device->device copy (ForwardLayers opening)
  device zero-fill (DBuf::Zero)
  Writes are not supported during trace capture   <- structural blocker

The Writes fatal is ttnn forbidding device allocations during capture
(same as CUDA's no-cudaMalloc-during-capture). The TT ops do per-call
from_vector host->device uploads (weights/inputs) and ttnn::empty scratch
inside kernels; those are fresh device writes, forbidden during capture.
CUDA solves this with a pre-warmed DevicePool + fixed-address persistent
weight buffers; TT has no equivalent, and its weights are not in stable
device buffers persisting across warmup->capture.

This is the structural hard part: a TT scratch-pool analogue + stable
weight residency so no allocation/upload happens during capture. Real
engineering, the natural scope of a dedicated row.

COMPLETE BLOCKER MAP (the experiment's deliverable):
1. RmsNorm/RoPE host thresholds -> R1 (flip, done)
2. Backend::Copy host readback -> R2 (device->device copy, done)
3. ttnn program-cache warm-up -> R3 (enable + eager-warm, done)
4. Backend::Memset/DBuf::Zero host write -> R3b (device zero-fill, done)
5. per-op device allocation/upload (from_vector, ttnn::empty) -> REMAINING;
   needs a TT scratch pool + stable weight residency

Items 1-4 landed, measured, inert-by-default. Item 5 is the open
engineering gate before decode capture can complete and replay tok/s can
be measured.

### Upstream investigation (2026-08-13): item 5 may be a non-issue on newer ttnn

Searched tt-metal/tt-nn issues. The "Writes are not supported during trace
capture" fatal is a **known limitation with an upstream fix**:

- **tt-metal issue [#13690](https://github.com/tenstorrent/tt-metal/issues/13690)**
  "Enable allocation of new buffers with a warning to allow running decode
  with trace and prefill without trace" — filed by Tenstorrent **for vLLM**
  (referenced by tenstorrent/vllm#14). The exact use case: interleaving a
  traced decode with untraced prefill needs buffers allocated while a trace
  is live.
- **Fixed in PR [#13696](https://github.com/tenstorrent/tt-metal/pull/13696)**
  (commit `f0b2483`): instead of `TT_FATAL`, it now prints a warning and
  allows the allocation, safe as long as untraced intermediates are consumed
  before a trace runs.
- **This build does NOT have the fix** — `fd_mesh_command_queue.cpp:760`
  still uses `TT_FATAL(!trace_id_.has_value(), "Writes are not supported...")`.

**Implication:** bumping the tt-metal build to one including #13696 may
eliminate item 5 entirely (the upload-during-capture becomes a warning,
not a crash). Worth testing before building a scratch-pool subsystem.

Additionally:
- `ttnn::create_device_tensor(spec, device)` (from
  `graph/graph_query_op_constraints.hpp`) allocates an empty device tensor
  **without** a host→device write — the capture-safe allocation pattern.
  The canonical capture sequence (graph_query_op_runtime.hpp:76-90) uses it
  to create input tensors pre-capture, warm, then capture. Our ops use
  `from_vector` (which writes); converting uploads to
  `create_device_tensor` + a pre-capture warm would also avoid the fatal.
- `TraceBufferPool` (PR #18523) — ttnn already has trace buffer management
  infrastructure.

**Two concrete paths to clear item 5, in order of effort:**
1. **Bump tt-metal** to a build with #13696 and re-run the capture probe.
   If the warning-only path works, capture completes and we get replay
   tok/s immediately — no vllm.cpp changes.
2. If the bump is not possible or insufficient: convert the TT ops' weight
   uploads to pre-capture `create_device_tensor` (stable, pinned addresses
   — the "pin addresses for a stable pool" approach) so no write happens
   during capture. Bounded work, no new subsystem.

### Correction (2026-08-13): bump will NOT help — our fatal is a write guard, not the allocator guard

Verified `f0b2483` IS an ancestor of the installed tt-metal build (the #13690
fix is present). But #13690 only relaxed the **allocator** (`allocator.cpp`
+ `device.cpp`) — it allows **buffer allocation** during a live trace.
Our fatal is at `fd_mesh_command_queue.cpp:760`, the **`enqueue_write`**
(host→device write) guard, which is a *separate* assertion that #13690 did
NOT touch (all three `Writes are not supported` fatals in
`fd_mesh_command_queue.cpp` are still hard `TT_FATAL`s).

So bumping tt-metal will not clear item 5. The real fix is path 2: avoid
the `enqueue_write` during capture by pre-allocating device tensors with
`create_device_tensor` (which does not write) and uploading their contents
*before* capture, so the captured ops reference stable device buffers with
no host→device write. This is the "pin addresses for a stable pool" approach
— confirmed feasible by `ttnn::create_device_tensor` existing and being the
canonical capture-safe allocation path (graph_query_op_constraints.hpp).

### Architecture answer (2026-08-13): mirror the tt-metal vLLM plugin's design

Read the official Tenstorrent vLLM plugin
(tt/vllm/plugins/vllm-tt-plugin/.../model_runner.py). It solves this
exactly, and the answer is a vllm.cpp-side architecture change, not a
tt-metal patch:

1. Two-phase warmup (model_runner.py:3216-3262): Phase 1 compiles all ops
   into the program cache with enable_trace=False; Phase 2 captures with
   every op compiled. Our Qwen3DenseDecodeGraph already does the
   single-step version.
2. Persistent device tensors at warmup shape (model_runner.py:480-487):
   block tables, positions, inputs allocated as persistent ttnn device
   tensors at the max padded shape during warmup so capture replays against
   stable device addresses.
3. Per-step inputs pushed BEFORE the captured region, not inside it: the
   plugin uses ttnn.copy_host_to_device_tensor (= C++ copy_to_device ->
   enqueue_write_tensor) to populate stable buffers. Crucially,
   copy_to_device hits the SAME enqueue_write path that fatals during
   capture (fd_mesh_command_queue.cpp:760), so the plugin calls it BEFORE
   capture (warmup populate) and BEFORE each replay (per-step refresh),
   NEVER inside the captured region.

Implication: our Backend::Copy/EnsureHostBytes fatal during capture is
fundamental -- copy_to_device itself would fatal there too. The fix is
architectural: the captured ForwardLayers region must reference only
pre-allocated, pre-populated device tensors. Per-step inputs (token id,
position, slot mapping, block table) must be written to stable device
buffers BEFORE ReplayGraph, the same way CUDA's decode graph does (its
SizeSlot::Refresh writes host buffers that a captured async-copy re-reads,
qwen3.cpp:528).

So path-2 is: make the TT decode-graph slot hold persistent device tensors
for inputs, populate them before capture/replay via copy_to_device, and
ensure the captured ops read those device tensors without any internal
from_vector/to_vector. That is the real host-free forward -- a bounded
architecture port of the plugin's design, not a new subsystem and not an
upstream tt-metal fix.

### Steady-state perf baselines (2026-08-14, real Blackhole P150)

Qwen3-0.6B, `vllm-cli --prompt "Hello" --max-tokens 64 --repeat 3`:

| config | warm tok/s (runs 2/3) | ms/tok |
|--------|----------------------|--------|
| default hybrid | **7.30 / 7.31** | ~137 |
| all-device eager (`VT_TT_HOST_FREE_DECODE=1` + `VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH=0`) | **6.87 / 6.92** | ~145 |

Two corrections to the earlier smoke numbers, both measured:

1. **The 4-token smoke (12.5 tok/s) is NOT steady state.** At 64 tokens the
   same config sustains 7.3 tok/s — per-token cost grows with context (PA
   decode over a growing KV), so the handoff's ~12.3 and our 12.5 both
   over-report. The honest reference for capture work is 7.3.
2. **The all-device eager cost at steady state is ~0.4 tok/s (~6%), not the
   ~1.8 tok/s (~14%) the 4-token smoke suggested.** This materially improves
   the capture break-even: capture only needs to recover ~6% of eager time
   to beat the hybrid baseline at 64-token scale — a much lower bar than
   the spike's 14% framing assumed.

(The `DECODE_GRAPH=0` opt-out is required for the all-device run: with the
flag on, `support_static_graph_mode()` flips true and the decode-graph
framework would otherwise attempt capture and abort on item 5.)

Mistral-7B-v0.3 reference on the same box: 4.26 tok/s warm at 32 tokens
(recorded in tenstorrent-mistral.md).

**What is still NOT measurable until item 5 lands: capture/replay tok/s.**
Capture cannot complete (the enqueue_write fatal fires mid-forward), so the
replay number — the actual payoff — remains open. The numbers above bound
it: replay must exceed 7.3 (the hybrid eager baseline) to be a win, and
starts from a 6.9 eager floor on the all-device path.

### Item 5 progress (2026-08-14): two sites fixed; frontier now mid-layer-0, at rope cos/sin

Instrumented all 16 `from_vector` sites (capture-gated `[TT-UP]` prints,
incl. ptr+shape on UploadRows) and iterated the bisection. Two real item-5
fixes landed:

1. **ttnn::zeros is NOT capture-safe** (creation.cpp `full_impl` host-fills
   + `to_device()` = an enqueue_write) — my own R3b helper was an offender.
   Fixed the plugin way: a persistent ZERO TENSOR CACHE (keyed by
   shape/dtype/layout) created OUTSIDE capture, primed during the eager
   warmup by `EnsureDevice2D`, and applied in-region by
   `ttnn::copy(zero, shadow)` — a device->device program that is captured
   and replayed. Cache-miss during capture is a hard VT_CHECK (must warm
   first), which is exactly what forced the priming fix.
2. **QkvSplit's device path was already correct** (ttnn::slice +
   CommitDevice2D) — the earlier suspicion was wrong; with MatmulBT's
   shadow it fires and hands q/k/v shadows downstream.

**Measured frontier after both fixes** — capture now runs DEEP into
layer 0 and dies at a precisely-identified site:

```
BeginCapture -> device-copy -> zero-fill -> [6 EnsureHostBytes readbacks
= the weight DBuf copies, handled by R2] -> CastBf16 -> RmsNorm ->
MatmulBT -> QkvSplit -> RmsNorm(q-norm) -> RmsNorm(k-norm)
-> [TT-UP] UploadRows ptr=... rows=16 cols=64   <- THE blocker
-> TT_FATAL: Writes are not supported during trace capture
```

`[16, 64]` is the **RoPE cos/sin table** (Hq=16, rot/2=64): host-computed
by `BuildCosSinFromPositions` inside `RopeNeoxKernel` and uploaded
in-region. This is the cos/sin sub-blocker the R1 spec predicted, and it
is the plugin's "per-step input" case: the fix is a PERSISTENT device
cos/sin buffer populated before capture/replay (positions change per step,
so the decode-graph driver must copy_to_device the step's rows BEFORE
ReplayGraph — the same pattern as CUDA's SizeSlot::Refresh async-copy).

**Remaining sites after rope (not yet reached by the bisection, expected
from the readback map):** ReshapeAndCache's KV writes (host-staged),
PagedAttention's metadata uploads, the lm_head/logits path. Each is the
same pattern; the rope fix establishes the template.

Status: item 5 is now a SCOPED multi-site port (rope cos/sin + RAC + PA
metadata + logits), with two sites landed and the third precisely
characterized. Not complete; the replay-tok/s payoff measurement remains
blocked behind the remaining sites.

### Item 5: rope cos/sin SOLVED (2026-08-14, measured on card)

The rope blocker took three fixes working together:

1. **Persistent cos/sin cache** keyed by (tokens*heads, rot/2), entries
   created/refreshed OUTSIDE capture, replayed in-region via the captured
   program (no per-call upload). Content-identity checked against the exact
   bytes the kernel will use — a stale table is a hard VT_CHECK during
   capture, never silent corruption.
2. **Driver warm hook** `WarmRopeCosSin(positions, ...)` called from the
   decode-graph driver's Refresh slot (qwen3.cpp, right after
   SizeSlot::Refresh) — THE per-step populate point, the exact plugin
   SizeSlot::Refresh analogue. Crucially it warms the UNPADDED T-row
   positions (what si.positions/rope sees), not the padded ppos — the
   first attempt used ppos and always missed.
3. **Byte-exact content**: the captured rope path (RopeFromCache, the
   default VT_QWEN3_ROPE_CACHE route) reads cos/sin from the per-step bf16
   CACHE table (RopeCosSinCacheKernel's StoreElemF32 rounds f32->bf16), so
   the warm content must round-trip through bf16 (BF16ToF32(F32ToBF16(v)))
   — f32 warm content never matches (cos(1)=0.540302 f32 vs 0.539062 bf16).

Measured: rope cache **HIT for both q (16x64) and k (8x64)** during
capture (`content_eq=1`), capture proceeds PAST rope. Also discovered en
route: the dense decode path routes rope through RopeFromCacheKernel (not
RopeNeoxKernel) by default — the first debug print in the wrong kernel
never fired, which is what exposed it.

**New frontier: ReshapeAndCache** — the next fatal is a to_vector readback
inside RAC (the KV-write path), right after rope in layer 0. This is the
"queued" RAC item from the original plan, now live. After RAC: PA metadata,
then the logits path. RAC is the most delicate remaining site: KV writes
inside a captured+replayed region also raise a REPLAY-SEMANTICS question
(every replay re-appends the same KV row) that must be answered alongside
the mechanical fix — the CUDA graph solves this by capturing the append
against fixed slot addresses refreshed per step.

Default-path safety re-verified after all rope changes: 23/23 cases,
830/830 assertions.

### Post-rebase benchmark confirmation (2026-08-14, rebased tree)

Rebased onto main (47 commits; picked up the MISTRAL row landing via the
bot and the windows C4456 fixes). One MAIN-RED found while rebuilding:
MUSIC3 W6 (aa3643b6) placed a C++ helper returning a C++ reference inside
`extern "C"` in src/capi/vllm_c.cpp — clang rejects it (every clang build
is broken on that commit; MSVC/Windows was lax). Applied the minimal local
fix (hoist the helper out of the extern-C block) to unblock; reported
upstream.

Benchmark re-run on the rebased tree (64 tokens, batch 1, 3 reps):

| config | warm tok/s |
|--------|-----------|
| default hybrid | **7.13 / 7.23** (clean EXIT=0) |
| all-device eager (rope-cache additions included) | **6.68 / 6.80** |

Consistent with the pre-rebase 7.30/7.31 and 6.87/6.92 (within run noise);
the rope-cos/sin-cache additions cost ~0.1-0.2 tok/s eager, the price of
capture-safety on that path. Default-path suite on the rebased tree:
23/23 cases, 831/831 assertions (main's merged tests grew the count).

### Item 5 frontier: ReshapeAndCache analysis complete (2026-08-14)

The readback: `ReshapeAndCacheKernel`'s first act is `EnsureHost(k)` — the
rope output carries a device shadow, so the download (a to_vector) fires
inside capture. Even on a shadow hit, the device push re-uploads: every
existing device path (`TryDevicePagedFill/UpdateBatch/FusedUpdateBatch`)
builds its input AND page table via `from_vector` (enqueue_write, also
fatal). The host NHD cache is the RAC/LMCache source of truth; the ttnn
shadow is a mirror patched from host floats.

**The capture-safe RAC design (next implementation step):**

1. Device-resident k/v input: the rope output shadow [T*H, D] bf16 TILE
   must feed `paged_update_cache` directly. Layout gap ([T*H,D] flat vs
   the sharded [C, nkv*d] input MakeHeightShardedUpdateInput builds today)
   resolved ON DEVICE via capturable reshape/permute ops.
2. Persistent PAGE-TABLE device tensor, per-step refreshed outside capture
   (the driver Refresh slot — same pattern as WarmRopeCosSin; the padded
   block table already lives in the SizeSlot host buffer, so the refresh
   source exists).
3. Persistent UPDATE-IDX device tensor likewise (paged_update_cache takes
   update_idxs_tensor — a device tensor — natively).
4. The host NHD mirror patch moves OUT of the captured region: done at the
   per-step refresh point from the same k/v tokens, before capture, so the
   LMCache contract (host NHD = source of truth) is preserved.

Replay semantics note: each replay re-writes the same KV slots the capture
baked in. That is only correct if the slot indices come from a
per-step-refreshed device tensor — the same reason CUDA's graph refreshes
slot_mapping per step. The design above has that property (2/3).

`ttnn::experimental::paged_update_cache`'s signature confirms feasibility:
it accepts a device `input_tensor`, a device `update_idxs_tensor`, and a
device `page_table` — all three can be persistent/refreshed, no host
floats needed in-region.

This is the largest single remaining piece (bigger than rope: on-device
layout conversion + two new refreshed buffers + moving the mirror patch).
After RAC: PA metadata (same refresh pattern), then logits.

### Item 5: RAC progress (2026-08-14)

Implemented `TryReshapeAndCacheDeviceDecode` + `WarmRacIdx` driver warm hook
+ shape-keyed idx cache (same content-refresh pattern as rope). The warm
hook fires correctly (slot0=32 warmup, slot0=33 capture step), the content
check matches, but the device branch bails because the **paged-KV device
shadow is empty** (`k=0 v=0`) — it was never created.

Root cause: the paged-KV shadow is created by `EnsurePagedKvTtnn` (inside
`TryPagedAttentionDeviceDecode`), but PA's device path doesn't run during
the eager warmup (its preconditions aren't met on the non-capture path).
So by capture time the shadow was never populated.

Fix needed: eagerly create the paged-KV shadow during the warm hook (call
`EnsurePagedKvTtnn` from `WarmRacIdx`, or prime it from the KV cache
metadata the driver has via `attn_kv`). This is the same "prime outside
capture" pattern as the zero cache and rope cos/sin.

After the shadow exists, the remaining RAC path (device→device
`paged_update_cache` with persistent idx tensors) should work — the idx
content already matches (verified), the k/v shadows exist (post-rope
`CommitDevice2D`), and the paged_update_cache signature accepts all
device tensors.

### Item 5: RAC device branch EXECUTES; paged_update_cache warm hangs

The shadow priming (WarmPagedKvShadow) works — both k and v shadows exist
(`k=1 v=1`). The RAC device branch (`TryReshapeAndCacheDeviceDecode`) fires
during capture: `[TT-TRACE] RAC device->device update (capture-safe)`.
But `paged_update_cache` is not program-cache-warm (the eager forward's RAC
bailed to host because the shadow didn't exist yet), and the capture call
hits `Writes are not supported` (new binary load during capture).

Attempted to warm `paged_update_cache` from `WarmRacIdx` with a dummy
input of the correct geometry (`[1,1,nkv_pad,d]` = `[1,1,32,128]`).
The warm call HANGS — `paged_update_cache` appears to deadlock when called
from the warm-hook context (outside the regular forward flow). This may be
a ttnn device-state issue (the mesh device's CQ is in a state that doesn't
support the op outside a forward step) or a geometry mismatch in the
warm-call's page_table/idx tensors vs what paged_update_cache expects.

NEXT: investigate why the warm `paged_update_cache` hangs. Options:
(a) call it from within the eager forward (not the warm hook) by making the
    eager RAC step also take the device branch (prime the shadow BEFORE the
    eager forward, not after it — move WarmPagedKvShadow before the eager
    step in the driver flow);
(b) use a simpler ttnn op (e.g. just `ttnn::copy`) as a warm substitute
    that compiles the same program path;
(c) move the shadow priming into the eager forward itself (call
    EnsurePagedKvTtnn at the top of the eager PA, not just the capture PA).

Option (a) is the most promising: the eager forward already runs the full
op chain; if the shadow exists at eager time, the eager RAC takes the
device branch, which warms `paged_update_cache` naturally (same context,
same CQ state). The issue is the ordering: the framework runs the eager
step BEFORE the Refresh slot (where the warm hooks fire). Moving the shadow
priming to BEFORE the eager step (at slot creation, not Refresh) would fix
the ordering.

### Item 5: RAC — `paged_update_cache` is NOT capture-safe (internal allocation)

After fixing:
- shadow priming for all layers (not just layer 0)
- used=block+1 (off-by-one in block coverage)
- idx tensor dtype INT32 (not UINT32 — ttnn requirement)
- input sharding (paged_update_cache requires height-sharded input)

The RAC device branch now EXECUTES on both cold and capture steps
(`RAC device->device update (capture-safe)` fires). But `paged_update_cache`
itself triggers `Writes are not supported during trace capture` — the op
does an internal allocation (result tensor) that is an `enqueue_write`.

This is NOT a program-cache issue (the cold step compiled the program).
`ttnn::experimental::paged_update_cache` allocates a new output tensor
even when the program is cached — that allocation is a device write,
forbidden during capture.

This is a ttnn API limitation: the op is not capture-safe by design.
The plugin's approach (persistent device tensors + before-replay populate)
works for ops that take pre-allocated outputs, but `paged_update_cache`
returns a new tensor. The fix would be either:
(a) an upstream ttnn change to support in-place update (pass output tensor)
(b) pre-allocate the result and use a different capture-safe scatter op
(c) capture only the ops AFTER RAC (skip RAC from the captured region,
    do it before replay) — but RAC mutates the KV cache, which PA reads
    inside the captured region, so it can't be moved out.

Option (a) is the cleanest (an upstream issue/PR to ttnn). This is the
genuine gate — not a vllm.cpp code issue but a ttnn API limitation.

### Item 5: RAC — paged_update_cache IS in-place; writes from build_padded

Key discovery: `paged_update_cache::create_output_tensors` returns
`tensor_args.cache_tensor` — it's an **in-place** operation (no output
allocation). The `Writes are not supported` error was NOT from
`paged_update_cache` itself but from `build_padded`'s helper ops:
`ttnn::to_memory_config` (sharding allocates a new buffer) and possibly
`ttnn::concat`/`ttnn::zeros`.

Attempted: pre-build the sharded zero input in WarmRacIdx and use
`ttnn::copy` (capture-safe) in `build_padded`. Crashed (segfault 139)
during the cold step — likely a shape/lifetime mismatch between the
pre-built sharded tensor and what `build_padded` produces. The
`sharded_zero` may be default-constructed (empty) if the warm loop didn't
find a shadow, or the shapes don't align.

NEXT: debug the sharded_zero lifetime/shape, or take the simpler approach
(b) — replace `paged_update_cache` with a manual `ttnn::copy` into a
pre-sliced cache region (simpler op, no sharding requirement, proven
capture-safe by the R2 copy fix).

### Item 5: sharded_zero crash FIXED; Writes still from build_padded

Fixed the segfault: the warm order was wrong — `WarmRacIdx` ran BEFORE
`WarmPagedKvShadow`, so the shadow loop found 0 entries and the
sharded_zero was default-constructed (empty). Swapped the order in the
driver: shadows first, then RAC idx. No more segfault.

But the `Writes are not supported` fatal persists. `paged_update_cache`
is in-place (confirmed: create_output_tensors returns cache_tensor).
The writes come from `build_padded`'s helper ops — specifically
`ttnn::copy(reshaped, sharded_zero)` where reshaped is TILE and
sharded_zero is height-sharded. The copy between different memory configs
triggers an implicit layout conversion (a write/allocation).

NEXT: approach (b) — replace `paged_update_cache` + the sharded input
with a manual `ttnn::copy` into a pre-sliced cache region. The cache
shadow is a persistent ttnn tensor; slicing it and copying the k/v rows
into the slice is all-capture-safe (proven by R2's device->device copy).
No sharding requirement, no paged_update_cache, no layout conversion.

### Item 5: approach (b) — RAC SKIPPED during capture; next blocker = PA

Implemented approach (b): `TryReshapeAndCacheDeviceDecode` returns true
immediately during capture (skipping the KV write). This is INCORRECT for
real decode (stale KV) but proves the approach works — capture proceeds
PAST RAC to the next op.

Measured on card: `[TT-TRACE] RAC skip during capture (approach b probe)`
fires, capture continues to `TryPagedAttentionDeviceDecode` which then hits
`Reads are not supported during trace capture` (fd_mesh_command_queue.cpp:807
= the READ guard, not the write guard at :760). So PA is doing a
`to_vector` readback — likely `EnsurePagedKvTtnn` re-uploading the stale
shadow (marked stale by the skip), or PA reading query/metadata via
`EnsureHost`.

The remaining sites after RAC are:
1. PA metadata (block_table, seq_lens, query_start_loc via EnsureHost)
2. PA's EnsurePagedKvTtnn (re-upload the stale KV shadow)
3. PA output (to_vector to read the attention result)
4. Logits (lm_head output)

The real fix for RAC: move it OUT of the captured ForwardLayers region
entirely — do the KV write at the driver Refresh slot (before BeginCapture),
same as the plugin's per-step copy_host_to_device_tensor pattern. This means
splitting the captured region: RAC runs before capture, PA+forward runs
inside capture. That's a driver-level change (the captured region starts
after EmbedInto + RAC, not at ForwardLayers).

### Item 5: PA — KV shadow skip works; next = page_table + cur_pos uploads

Fixed PA's KV shadow re-upload: during capture, skip EnsurePagedKvTtnn and
use the cached shadow directly (it was primed by WarmPagedKvShadow at the
Refresh slot). `PA using cached KV shadows (k_nb=2 v_nb=2)` prints,
`PA KV shadows OK, building page_table` prints.

Next fatal: BOTH Writes (:760) and Reads (:807) — the from_vector uploads
for dev_pt (page_table) and dev_pos (cur_pos) at PA lines 2104/2114.
Same pattern as rope: per-step data (block_table, seq_lens) that needs
persistent device tensors warmed at the Refresh slot.

Remaining sites after PA metadata:
1. PA page_table upload (from_vector, line 2104) — persistent device tensor
2. PA cur_pos upload (from_vector, line 2114) — persistent device tensor
3. PA sdpa_decode output (to_vector at line 2164+) — device→device commit
4. Logits (lm_head output)

Each is the same persistent-buffer + driver-warm pattern. The path is
proven (RAC skip + PA shadow skip both work); it's mechanical repetition.

### Item 5: PA — metadata warm works; sdpa_decode not compiled (cold bail)

PA metadata warm (WarmPaMeta) works: `PA using cached meta (pt+cp)` fires
during capture. But `sdpa_decode` hits `Cannot load new binaries during
trace capture` — it was never compiled during the cold step because the
cold step's PA device path bails before `sdpa_decode`.

The cold step's PA enters `TryPagedAttentionDeviceDecode` (28 times,
verified via `PA reached EnsurePagedKvTtnn cap=0 used_nb=2`), but
`PA q_from_device OK cap=0` NEVER prints — meaning the cold step's
`EnsureDevice2D(query)` either throws (caught by the try/catch) or the
`identity_q` check fails. No `PA q_from_device FAILED` print either.

Root cause TBD: either the query's device shadow doesn't exist during
the cold step's PA (rope didn't commit it, or the pointer differs), or
`EnsureDevice2D` throws an exception that the outer try/catch swallows.
NEXT: add a print at the `identity_q` check and at the `EnsureDevice2D`
call to find the exact bail point.

### CAPTURE COMPLETE — replay tok/s measured (2026-08-15)

The cold step's PA device path was bailing because `KvSlice` returns a
non-contiguous strided view that `EnsurePagedKvTtnn`'s VT_CHECK rejects.
Fixed by using the cached shadow (from WarmPagedKvShadow) on BOTH cold
and capture steps, bypassing EnsurePagedKvTtnn's contiguous check +
from_vector upload. The cold step now runs sdpa_decode on all 28 layers
(program compiled), and the capture step's sdpa_decode hits the program
cache.

**MEASURED on real Blackhole P150** (Qwen3-0.6B, `vllm-cli --prompt Hello
--max-tokens 4 --repeat 3`):

| run | secs | tok/s | note |
|-----|------|-------|------|
| 1 (cold JIT + capture) | 18.5 | 0.22 | first compile + capture |
| 2 (warm replay) | 0.046 | **86.5** | replay only |
| 3 (warm replay) | 0.051 | **77.9** | replay only |

**~12× speedup over the eager baseline (7.3 tok/s).** EXIT=0 (clean).

Default-path safety: 23/23 cases, 831/831 assertions (inert without the
env flag).

NOTE: the 4-token smoke over-reports (per-token cost grows with context,
as established). A 64-token measurement will give the honest steady-state
number. But even at 4 tokens, 86 tok/s vs 12.5 tok/s (the old 4-token
smoke) is a 7× speedup. The replay collapses ALL host-API overhead into
a single ReplayGraph call, exactly as the CUDA decode graph does.

Remaining caveats:
- RAC is SKIPPED during capture (stale KV — correctness is wrong for real
  decode; the real fix is moving RAC out of the captured region).
- The PA output path (CommitDeviceLogical2D) works because identity_order
  is true (pure decode).
- The logits/lm_head path hasn't been checked — capture may hit a
  readback there. The 4-token run EXIT=0 suggests it completed, but the
  output correctness hasn't been verified.

### RAC in-capture: paged_update_cache IS capture-safe; k/v copy needs warm

RAC now runs INSIDE the captured region via `paged_update_cache` (in-place,
capture-safe). The sharded input uses the pre-built `sharded_zero` from
WarmRacIdx. Capture completes: EXIT=0, 83/77 tok/s replay.

The k/v copy into the sharded buffer (`build_input`) currently fails during
capture because `ttnn::zeros` + `ttnn::concat` are writes (caught by the
try/catch, falls back to zeros = stale KV). The fix: pre-build the padded
k/v tensor at warm time (same persistent-cache pattern as rope cos/sin).
The warm hook already has the k/v device shadows from the cold step; it
can build the padded sharded input and store it in the RacIdxEntry, then
the captured `build_input` uses `ttnn::copy` (capture-safe) to refresh it.

NOTE: the k/v data changes per step (it's the rope output for the current
token), so the warm must happen at the Refresh slot — but the k/v aren't
available at Refresh (they're computed inside ForwardLayers). This is the
fundamental circular dependency: RAC needs the k/v from the current step's
forward, which is inside the captured region.

The plugin solves this by capturing the k/v write as part of the graph
(the k/v are device-resident from the rope, and the paged_update_cache
reads them directly). Our issue is only the sharded input construction
(zeros + concat). If we can pre-allocate the padded tensor and use only
`ttnn::copy` (from the device k/v shadow into the padded sharded buffer),
it should work. The `ttnn::copy` between TILE and height-sharded may
still allocate (layout conversion) — that's the remaining question.

### RAC k/v copy: slice+copy produces wrong output (layout mismatch)

The `ttnn::copy` from a TILE `[1,1,nkv,d]` into a height-sharded
`[1,1,nkv_pad,d]` slice silently fails or produces wrong data (the
try/catch swallows the error, leaving zeros). The output is
`] (zheimerzheimerzheimer` instead of ` Answer! I'm`.

The capture mechanism is COMPLETE — EXIT=0, 83/76 tok/s replay. The
issue is purely the k/v data copy into the sharded buffer: ttnn::copy
between different memory configs (TILE vs height-sharded) doesn't work
as a simple memcpy.

The fix: either
(a) find a ttnn op that copies TILE→sharded without allocation, or
(b) pre-build the k/v as a height-sharded tensor at warm time (the
    warm hook has the device k/v shadow from the cold step), or
(c) use a different approach entirely — skip paged_update_cache and
    do a manual scatter via ttnn::slice + ttnn::copy on the cache
    shadow itself (which is TILE, not sharded).

Option (c) is promising: the cache shadow `[nb,nkv,bs,d]` is TILE. We
can slice it at `[block, :, offset, :]` → `[1,nkv,1,d]` and copy the
k/v `[nkv,1,d]` (reshaped from the rope output) into it. All TILE→TILE
copies, no sharding. But the TILE constraint means offset must be
tile-aligned (multiple of 32) — which it isn't for arbitrary decode
positions.

## Session checkpoint (2026-08-15)

### Complete state

**CAPTURE WORKS** — measured 83 tok/s replay on real Blackhole P150
(12x over the 7.3 tok/s eager baseline). EXIT=0, default path inert
(23/23, 831/831).

### What works (all env-gated, inert by default)
- R1: RmsNorm/RoPE threshold flip
- R2: Backend::Copy device→device (CopyDeviceDeviceIfCapture)
- R3: Program-cache warm (enable_program_cache + eager warm)
- R3b: Backend::Memset/Zero device fill (persistent zero cache)
- 5a: ttnn::zeros → persistent zero cache (capture-safe fill)
- 5b: Rope cos/sin → persistent cache + bf16 round-trip + driver warm
- 5c: RAC → paged_update_cache IS capture-safe (in-place); k/v copy
     needs fix (TILE→sharded layout mismatch → wrong output)
- 5d: PA KV shadow → cached shadow skip (bypass EnsurePagedKvTtnn's
     contiguous check on KvSlice's strided view)
- 5e: PA metadata (page_table + cur_pos) → persistent device tensors
     + driver warm (WarmPaMeta)
- 5f: sdpa_decode → compiled on cold step (all 28 layers), program-cache
     hit on capture step

### What's left
1. RAC k/v copy correctness — tnn::copy TILE→sharded produces wrong data.
   Fix: "RAC before replay" — do the KV write at the driver Refresh slot
   (before BeginCapture/ReplayGraph) using the previous step's rope output
   shadow. The captured graph skips RAC; the driver does it outside capture.
2. 64-token steady-state tok/s measurement (the 83 tok/s is a 4-token smoke).
3. Correctness gate vs the Qwen3-0.6B TT golden (verify output matches).
4. Fresh review of the complete change.

### Performance summary
| config | warm tok/s | note |
|--------|-----------|------|
| Qwen3-0.6B default hybrid (64 tok) | 7.13 / 7.23 | eager baseline |
| Qwen3-0.6B all-device eager (64 tok) | 6.68 / 6.80 | capture prerequisite cost ~6% |
| Qwen3-0.6B capture replay (4 tok) | 83 / 76 | ~12x speedup (smoke; 64-tok pending) |
| Mistral-7B-v0.3 (32 tok) | 4.26 | first Mistral number |

### PRs
| PR | Row | Status |
|----|-----|--------|
| #431 | MISTRAL | MERGED |
| #694 | HOST-FREE | draft, all work on this branch |
| #393 | RESIDUAL-GOLDEN | merged |
| #541 | TRACE-RUNNER | closed (superseded by #694) |
| #805 (issue) | MAIN-RED | filed (MUSIC3 extern C bug) |

### RAC flush approach: one-step lag (PA reads stale KV)

The FlushPendingRac approach works mechanically (per-layer flush fires,
86 tok/s) but produces wrong output because of a ONE-STEP LAG: the KV
write for token N happens at the Refresh of step N+1, but PA at step
N+1 needs token N+1's KV (written by RAC during the captured forward of
step N+1, which is skipped). PA always reads one token behind.

The CUDA decode graph handles this correctly: the captured graph
INCLUDES RAC (the KV write happens BEFORE PA within the same captured
forward). Our TT capture skips RAC, so PA never sees the current token.

The correct fix: RAC must be inside the captured region. The blocker
was the sharded input construction (zeros+concat+to_memory_config are
writes during capture). The solution: build the sharded input at the
WARM hook from the k/v device shadow, and during capture only do
ttnn::copy (capture-safe) from the rope output into the pre-built
sharded buffer + paged_update_cache (in-place, capture-safe).

The k/v shadow from the rope is available INSIDE the captured region
(rope runs before RAC). The copy into the sharded buffer is the TILE→
sharded issue — needs testing whether ttnn::copy with an explicit
output memory config works without allocating.

### In-region RAC: every input-construction path is a write

Tried during capture:
1. `to_memory_config(padded, sharded)` → Writes fatal (sharding allocates)
2. `ttnn::copy(padded_tile, sharded_zero)` → Writes fatal (copy between
   different memory configs allocates or enqueues a write)

The fundamental constraint: ANY tensor shape/layout construction during
capture is a write (enqueue_write). The sharded input that
paged_update_cache requires cannot be built during capture from a TILE
source.

The only remaining approaches:
(a) Pre-build the sharded input at the warm hook with the CORRECT rope
    output data — but the data isn't known at warm time (it's computed
    during the forward).
(b) Have the rope output go DIRECTLY into the sharded layout (rope's
    output tensor IS the sharded buffer). This requires modifying the
    rope's output allocation.
(c) Accept the one-step lag (FlushPendingRac) — output is wrong but the
    capture+replay mechanism works (86 tok/s).
(d) Skip RAC during capture + do the KV write at Refresh with the
    correct k/v — same one-step lag.
(e) Patch ttnn to allow writes during capture (the tt-metal issue #13690
    fix only relaxed the allocator, not the write guard).

Option (b) is the most promising but requires restructuring the rope
output. Options (c)/(d) give wrong output. Option (e) is upstream.

STATUS: capture+replay WORKS (86 tok/s, 12x speedup) with stale KV.
Correct KV inside the captured region requires one of the above.

### Post-recovery verification (2026-08-15, after tt-umd update + device reset)

The tt-metal patch experiments corrupted device state (needed a tt-umd
update + PCI reset to recover). All tt-metal patches REVERTED — the
build is clean upstream tt-metal. The vllm.cpp side retains the working
skip+flush RAC (86 tok/s).

Verified on the recovered device:
- Default (no flag): " Answer! I'm" correct, 12.4 tok/s, EXIT=0
- Capture (VT_TT_HOST_FREE_DECODE=1): 87.3/82.2 tok/s replay, EXIT=0,
  output wrong (the known one-step-lag KV issue)

### tt-metal patch experiment: conclusion

The 4 patches (write guard → warning, program-cache miss → warning,
binary-commit → warning, DRAM overlap → warning) DID let the capture
run through all 28 layers including in-region RAC — no fatals. But:
1. The DRAM-overlap relaxation corrupted allocator state
2. The corruption survived tt-smi -r, required a PCI-level reset
3. The corruption may have permanently damaged the device state
   (needed the tt-umd update to fully recover)

The patches are valuable as PROOF that in-region RAC works (all 28
layers' sdpa_decode ran during capture, only warnings), but they're too
dangerous for production. The upstream proposal should be a narrowly
scoped "capture-safe writes" API rather than blanket guard relaxations.

### trace_region_size spike: CONCLUSIVE (2026-08-15)

**The plugin's key device parameter, found and applied:**
`ttnn::open_mesh_device(device_id, l1_small_size, trace_region_size=50MB)`
(worker.py:710 — the plugin sets 50000000 when trace_mode is on). Our device
creation now passes it. Effect: the trace buffer gets a dedicated DRAM
region, so the "Trace buffer overlaps with DRAM activity" FATAL is gone.

**Then tested the full in-region RAC (slice+copy only, no zeros/concat/
to_memory_config during capture):**
1. Write fatal GONE (view ops + ttnn::copy don't host-write)
2. Program-cache-miss fatal appears (slice+copy program not warmed)
3. Unified eager+capture paths (same slice+copy ops) → program warms
4. Result: allocator WARNING "Allocating device buffers is unsafe due to
   the existence of an active trace" — then DEVICE HANG (240s timeout,
   tt-smi -r also hangs)

**CONCLUSION: even with trace_region_size, allocating during an active
trace CORRUPTS the trace buffer and hangs the device.** The ttnn::copy
between TILE and height-sharded memory configs allocates a conversion
temp — that allocation lands in the trace region and corrupts it. The
warning is ttnn telling us exactly this.

The plugin avoids it by NEVER allocating during capture: everything is
pre-allocated at warmup; per-step data goes through
copy_host_to_device_tensor BEFORE capture/replay only.

**The spike answer: in-region RAC needs a capture-safe device->device
copy between memory configs, which ttnn does not provide today. The
upstream ask is precisely that API (or a copy_into_sharded variant of
ttnn::copy).** Until then, the working configuration is skip+flush RAC
(one-step lag, 86 tok/s) + trace_region_size=50MB.

## Session checkpoint (2026-08-16) — host-free decode graph fidelity + stall characterization

### Complete state

**HOST-FREE DECODE GRAPH LANDS.** Captured replay is token-exact vs
host-free eager (22/22 argmax-identical, per-step `VT_TT_DUMP_KV` diff,
monitor-57 vs monitor-58). Marginal decode step: captured ~30 ms (~33
tok/s) vs default eager ~175 ms (~5.7 tok/s) ⇒ **5.8× speedup** at the
step level (short-prompt startup ~21.4 s still dominates wall time).

### What works (all env-gated, inert by default)

- Frozen-replay fix: `InvalidateHostCachesAfterTrace()` after every
  `execute_trace` (in `TraceReplay`/`TraceReplayGraph`, both
  `blocking=false` — the qwen3 driver's blocking logits readback is the
  sync point, mirroring upstream executor.py's traced-decode pattern).
- KV shadow pool-sized allocation: `EnsurePagedKvTtnn` rewritten — used
  prefix `from_vector` + `ttnn::zeros` tail + `ttnn::concat`; the shadow
  buffer is allocated ONCE at the full pool size and never moves (the old
  per-block-boundary realloc freed a buffer a live trace still
  referenced → all-zero logits at the first block-table growth).
- Embedding moved INSIDE the captured trace: `WarmDecodeIds` (persistent
  device UINT32 ids tensor, content refreshed in place, zero eager
  allocs per replay) + `EmbedDeviceIdsInto` (capture-safe embedding over
  the persistent ids tensor writing the persistent hidden shadow). A
  replay step now performs ZERO eager device allocations.
- Re-capture workaround `VT_TT_RECAPTURE_EVERY=N`: destroy + re-capture
  the graph every N replays (see stall, below).

### The two walls (characterized + cause ISOLATED 2026-08-16)

1. **~38-replay completion hang.** A single trace id deterministically
   stops completing (futex wait in the post-replay readback) after ~38–50
   consecutive replays. `execute_trace` enqueues fine (0.08 ms); the
   device-side completion never arrives.
2. **~2 healthy capture generations.** `VT_TT_RECAPTURE_EVERY=N`:
   N=32 → gen1=32 ✓, gen2=32 ✓, gen3 died @11 (~75 replays total).

**Upstream machinery is NOT the cause** (monitors 61-66). Ported upstream
`test_mesh_trace.cpp` Sanity + 4 scratch variants (120 replays of one trace,
Finish-per-replay, 10 recapture gens, 50-program big trace) — ALL PASS on
our card. The raw mesh-trace path is sound.

**Cause ISOLATED by per-class bisection** (monitors 68-72):

| skip flag(s) ON | copies active | result | replays |
|---|---|---|---|
| (none, recapture=32) | rope+idx+PAmeta+ids | HANG | 75 |
| `VT_TT_NO_ROPE_REFRESH` | idx+PAmeta+ids | HANG | 39 |
| `VT_TT_NO_IDX_WARM` | rope+ids | **PASS** | 79 |
| `VT_TT_NO_IDS_WARM` | rope+idx+PAmeta | HANG | 50 |
| all three | (none) | **PASS** | 79 |

The toxic copy class is `WarmRacIdx` + `WarmPaMeta` (the
`VT_TT_NO_IDX_WARM` flag), both calling `ttnn::copy_to_device` into INT32
`page_table`/`update_idxs`/`cur_pos` buffers the captured trace READS.
Rope (bf16) and decode-ids (UINT32) are innocent.

`ttnn::copy_to_device` (ttnn/core/tensor/tensor_ops.cpp:182) →
`enqueue_write_tensor`: a DIRECT host→device write into the existing
device buffer on the trace's CQ, NO allocation. So the leak is NOT an
allocation (the `mark_allocations_unsafe` warning correctly never fired) —
it's a write to a trace-input buffer on the trace's CQ between replays.

**Root mechanism (code-read 2026-08-16):** `copy_to_device` → mesh CQ
`write_to_core` (fd_mesh_command_queue.cpp:579) →
`device_dispatch::write_to_core`, which threads
`expected_num_workers_completed` (impl/buffers/dispatch.cpp:486+,
737/985/1304/1350) — the SAME counter the trace replay path
(`update_worker_state_post_trace_execution`, tt_metal/impl/trace/dispatch.cpp)
assigns and `mark_completely_full()`s per replay. After ~38–50 interleaved
writes, the accounting desyncs and the next replay's device-side completion
never arrives → the post-replay readback futex-hangs.

The toxic class is specifically idx/PAmeta: with idx/PAmeta OFF (rope+ids
ON) the run PASSES 79 replays; with idx/PAmeta ON it hangs regardless of
the others. The wall count (39/50/75) is run-to-run variance, not
evidence that rope/ids contribute. Why idx/PAmeta specifically (not rope
bf16, not ids UINT32) is not yet distinguished — candidates: they're read
by `sdpa_decode`/`paged_update_cache` dispatch (prefetch dependency), or
the INT32 ROW_MAJOR 2D page_table write is the specific dispatch shape
that leaks.

Memory notes: `tt-trace-replay-wall-not-upstream-machinery.md`,
`tt-replay-wall-toxic-copy-class-idx-pameta.md`,
`tt-replay-wall-root-mechanism-write-to-core.md`.

### Performance (Qwen3-0.6B, "Hello", bf16, 1 req)

| config | step | marginal tok/s |
|--------|------|----------------|
| default eager | ~175 ms | ~5.7 |
| captured replay | ~30 ms | ~33 |
| speedup | — | 5.8× |

Fixed startup ≈21.4 s dominates short runs. Re-capture cycle <1 s.

### Mode difference (NOT a capture bug)

Host-free mode (eager or captured) vs default mode diverge at generation
step 18 (host-free emits " first." where default emits "."), re-sync
after. Both coherent; near-tie flip caused by host-free ops changes
(pool-sized KV shadow, padded-view paths). The golden-gate near-tie
arrays are the arbiter — not yet run.

### Files modified this session (all uncommitted)

- `src/vt/tenstorrent/tenstorrent_ops.cpp`:
  `InvalidateHostCachesAfterTrace()` + calls; `EnsurePagedKvTtnn`
  rewrite; `DecodeIdsCache`/`WarmDecodeIds`/`EmbedDeviceIdsInto`;
  `GraphCapturesCounter`/`Done`/`NoteGraphCaptured`/`ReplayRegimeBisectSkip`;
  stall-bisection skip flags (`VT_TT_NO_ROPE_REFRESH`/`VT_TT_NO_IDX_WARM`/
  `VT_TT_NO_IDS_WARM`, replay-regime only); `TraceDestroyGraph`
  `release_trace` diagnostics; `[TT-STEP]` brackets in `TraceReplayGraph`.
- `src/vt/tenstorrent/tenstorrent_device.h`: declarations for
  `WarmDecodeIds`, `EmbedDeviceIdsInto`.
- `src/vllm/model_executor/models/qwen3.cpp`: `VT_TT_RECAPTURE_EVERY`
  driver logic; replay-branch timing (`replay_ns`/`replay_steps` on Impl);
  `~Impl()` stats extension.

### What's left

1. **Fix the toxic copy path** (idx/PAmeta `copy_to_device` into trace-input
   buffers). Fix directions (ranked): (a) on-device state advance inside the
   trace (upstream executor.py pattern, biggest change); (b) capture-safe
   write primitive (tiny captured program); (c) raw `enqueue_write_buffer`
   bypass of ttnn's tensor write path; (d) separate CQ. Correctness gate
   (needs >75 replays) is blocked until a fix or longer-lived workaround.
2. Fresh review of the complete host-free change; commit everything
   (nothing committed yet — all changes uncommitted).
3. Upstream issue to tt-metal: the ~38-replay wall under our ttnn-op
   interleaving + the ~2-generation re-capture wall + the earlier
   padded-shape view trap + the TILE 4D view mapping rule.

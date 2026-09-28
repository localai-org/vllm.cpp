ID: ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ
Title: 27B serve: TT_FATAL writes during trace capture from CaptureSafeReshape tiled reshape
Row: BACKEND-TENSTORRENT-QWEN35
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-28
Updated: 2026-09-28
Closed: -

## Problem

At main 8b5435bb0, the Qwen3.8-27B-Q4_K_M served arm (2x128/32 c2, both VT_TT_KEEPQUANT_INT8DOT=0 and =1) crashes in Qwen3_5DenseDecodeGraph::Step during trace capture: TT_FATAL 'Writes are not supported during trace capture' (tt-metal fd_mesh_command_queue.cpp:826). Chain: EnsureDevice2D -> CaptureSafeReshape -> ttnn::reshape (tiled) -> ReshapeViewTiledProgramFactory::create_program_artifacts -> ttnn::to_device host write mid-capture. Logs: /tmp/int8dot-leg0.log, /tmp/int8dot-leg1.log (2026-09-28). Suspects: the capture-warmup redesign #3321 or the GDN state-binding fix #3327 changed the capture shape; or the pinned tt-metal (9161e8fdb27+4) tiled-reshape path now materializes at artifact creation. The 27B serve arm has not run since those landed; the 9B did. The 92/92 suite does not cover this shape.

## Resolution

- 2026-09-28 (worktree row/tt-27b-capture-write, base 40990825d): root cause is
  `EnsureDevice2D`'s exact-rows/cols arm in
  `src/vt/tenstorrent/tenstorrent_residency.cpp` running a DIFFERENT chain in
  the capture pass than the eager pass. A decode op commits a rank-3 device
  result under a flat 2D slot record (`CommitDeviceLogical2D`), so the next
  2D `EnsureDevice2D` lands on the arm whose logical shape mismatches. The
  eager pass warms `to_layout(ROW_MAJOR) -> reshape (free view) ->
  to_layout(TILE)`; the `tt_capture_active()` branch instead ran a bare
  `ttnn::reshape` on the TILED rank-3 shadow — a spec the eager pass never
  warmed — so the first decode capture created
  `ReshapeViewTiledProgramFactory::create_program_artifacts`
  (tt-metal reshape_tiled_program_factory.cpp:275), whose unconditional
  `to_device` of the page-mapping tensor is the mid-capture write that
  fatals at fd_mesh_command_queue.cpp:826. Fix: run ONE chain in both
  passes (the W4 doctrine), deleting the capture-active branch. Red:
  new doctest `EnsureDevice2D rank-3 reshape is capture-safe` reproduces the
  exact TT_FATAL on the old branch (/tmp/red-focused.log) and passes with the
  fix (/tmp/green-focused.log). Suite: 93/93 (/tmp/green-suite.log). 27B
  serve leg: /tmp/leg-27b-green.log.
- 2026-09-28 UPDATE: the 27B serve leg still fatals after the arm fix — a
  SECOND, distinct divergence. With VT_TT_TRACE_DEBUG=1 (/tmp/leg-27b-diag.log)
  the capture pass hits EnsureDevice2D's same-numel arm with spec
  {1,10240} -> {2,5120}, a reshape the eager pass never ran (its arm726 specs
  were {96,128}->{2,6144} and {3072,128}->{64,6144}); the reshape_tiled program
  is created mid-capture and the same TT_FATAL fires. This is the "slot state
  differs between passes" class: the producer commits {1,10240} in the capture
  step where the warmup step committed a different shape. Issue stays OPEN for
  that second site; the arm-710 unification and its red/green doctest stand as
  committed.
- 2026-09-28 (worktree row/tt-27b-capture-write, ab7cdb359 + follow-up): site 2
  ROOT CAUSE: `MemsetDeviceIfCapture`'s fresh-slot lane in
  `src/vt/tenstorrent/tenstorrent_residency.cpp` was CAPTURE-ONLY for the
  shadow install. Under capture, a fresh-slot `DBuf::Zero` (the 20480-B
  residual) installed a `{1,10240}` bf16 TILE shadow; the eager pass only
  primed the zero and returned false (host memset + `MarkHostWritten`), so
  eager ended with the consumer-shaped shadow (`{2,5120}` from kRmsNorm's
  staging) and capture ended with the memset-shaped one. The capture consumer
  then hit EnsureDevice2D's same-numel arm with the never-warmed
  `{1,10240} -> {2,5120}` reshape (bench trace `arm726 rows=2 cols=5120
  dev=1x10240 cap=1`) and the ReshapeViewTiled program's `to_device` wrote
  mid-trace. FIX (W4 doctrine, one install in both passes): the fresh-slot
  lane now installs the `{1, cols}` persistent shadow in BOTH passes, so the
  eager consumer runs — and warms — the same-numel reshape and capture
  replays it as a program-cache hit. Eager installs are bounded to
  scratch-scale memsets (bytes <= 64 KiB): an unbounded eager install
  retained the 3-8 MB weights-load slots and OOMed DRAM (a 268 MB
  `ttnn::where` then missed by ~7 MB); larger slots keep the pre-fix
  host-fallback priming. RED: doctest `fresh-slot Memset installs the same
  shadow in both passes` reproduces the exact bench fatal on the old code
  (`arm726 rows=2 cols=5120 dev=1x10240 cap=1` -> TT_FATAL at
  fd_mesh_command_queue.cpp:826, /tmp/red-site2.log); GREEN with the fix
  (arm726 cap=0 warms, cap=1 cache-hit, /tmp/green-site2.log). SUITE 94/94
  (/tmp/suite-final2.log). The case also exposed a suite-hygiene bug, fixed
  here: the `kRopeNeox (small)` case leaked `VT_TT_HOST_FREE_DECODE=0` into
  every later case; it now restores the ambient value.
- 2026-09-28 DEVICE GATE (27B leg, /tmp/leg-27b-final3.log, c1:
  /tmp/leg-27b-c1.log): BENCH_EXIT=1 — the site-2 fatal is gone (capture
  passes the `{1,10240}->{2,5120}` reshape), but two FURTHER blockers, both
  previously masked because the leg died at site 2 first, now surface in
  order: (1) at --concurrency 2, `TryReshapeAndCacheDeviceDecode`
  (tenstorrent_paged.cpp:643) declines `num_slots > 1` ("decode T=1 only for
  now"), RAC falls to the host path and `EnsureHost(k)` readbacks mid-capture
  (fd_mesh_command_queue.cpp:873, "Reads are not supported"); (2) at
  --concurrency 1 the whole decode capture replays cleanly but
  `end_trace_capture` OOMs: the trace buffer needs 3,153,969,152 B against
  ~298 MB free (MeshTrace::populate_mesh_buffer). These are new owed sites
  (multi-slot RAC device path; 27B decode-trace DRAM fit), not regressions
  of this fix. Issue stays OPEN for them.
-

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

At main 8b5435bb0, the Qwen3.8-27B-Q4_K_M served arm (2x128/32 c2, both VT_TT_KEEPQUANT_INT8DOT=0 and =1) crashes in Qwen3_5DenseDecodeGraph::Step during trace capture: TT_FATAL 'Writes are not supported during trace capture' (tt-metal fd_mesh_command_queue.cpp:826). Chain: EnsureDevice2D -> CaptureSafeReshape -> ttnn::reshape (tiled) -> ReshapeViewTiledProgramFactory::create_program_artifacts -> ttnn::to_device host write mid-capture. Logs: <logs-dir>/int8dot-leg0.log, <logs-dir>/int8dot-leg1.log (2026-09-28). Suspects: the capture-warmup redesign #3321 or the GDN state-binding fix #3327 changed the capture shape; or the pinned tt-metal (9161e8fdb27+4) tiled-reshape path now materializes at artifact creation. The 27B serve arm has not run since those landed; the 9B did. The 92/92 suite does not cover this shape.

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
  exact TT_FATAL on the old branch (<logs-dir>/red-focused.log) and passes with the
  fix (<logs-dir>/green-focused.log). Suite: 93/93 (<logs-dir>/green-suite.log). 27B
  serve leg: <logs-dir>/leg-27b-green.log.
- 2026-09-28 UPDATE: the 27B serve leg still fatals after the arm fix — a
  SECOND, distinct divergence. With VT_TT_TRACE_DEBUG=1 (<logs-dir>/leg-27b-diag.log)
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
  fd_mesh_command_queue.cpp:826, <logs-dir>/red-site2.log); GREEN with the fix
  (arm726 cap=0 warms, cap=1 cache-hit, <logs-dir>/green-site2.log). SUITE 94/94
  (<logs-dir>/suite-final2.log). The case also exposed a suite-hygiene bug, fixed
  here: the `kRopeNeox (small)` case leaked `VT_TT_HOST_FREE_DECODE=0` into
  every later case; it now restores the ambient value.
- 2026-09-28 DEVICE GATE (27B leg, <logs-dir>/leg-27b-final3.log, c1:
  <logs-dir>/leg-27b-c1.log): BENCH_EXIT=1 — the site-2 fatal is gone (capture
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
- 2026-09-28 (worktree row/tt-27b-capture-write) BLOCKER A FIXED — multi-slot
  decode RAC is capture-safe. ROOT CAUSE: the decline `if (num_slots > 1)
  return false` at tenstorrent_paged.cpp:642 is NOT new — it is byte-identical
  since 79ff8f310 (2026-08-18, the #1105 whole-graph capture) and was already
  present at the W3-era c2 bench commit ead93289b. What changed since W3 is
  the STATE OF k/v AT THE FALLBACK: the capture-warmup redesign's device-
  residency waves (28c154d78 "grouped-quant activations serve device-resident"
  and successors; the suite-repair 12d5d75ee) left the rope K/V outputs as
  device_current TILE shadows at RAC time under capture. In W3 the host
  fallback's `EnsureHost(k)` (tenstorrent_paged.cpp:890) found host bytes and
  did a pure host write; at HEAD the same fallback triggered the device
  readback mid-capture — TT_FATAL "Reads are not supported during trace
  capture" (fd_mesh_command_queue.cpp:873). The decline that was a benign
  perf shortcut in August became a capture-fatal in September.
  FIX (W4 doctrine, warm in eager what capture replays; no capture-active
  branch): TryReshapeAndCacheDeviceDecode admits the batched decode
  (num_slots == T, one token per user) and runs the PROVEN single-user
  sequence once per user — slice this user's [nkv, d] rows from the rope
  shadow, 1.0-multiply into a fresh native [1,1,nkv,d], ttnn::copy into that
  user's own single-shard persistent input (K on worker core u, V on C+u),
  then one paged_fused_update_cache per user against that user's own [1]
  update_idx and [1, cols] page-table row; the fused op's
  override_runtime_arguments re-patches the per-user addresses on the shared
  cached program, the same mechanism the C=1 replay uses. WarmRacIdx
  allocates and per-step-refreshes the per-user idx tensors outside capture
  (the per-step update_idx copy is the C=1 lane's on-device plus_one not yet
  extended here — recorded as owed perf debt). An earlier design that packed
  all C users into ONE C-shard sharded input was tried and REJECTED on two
  tt-metal facts measured in this change: ttnn::copy demands LOGICAL shape
  equality while a C-shard dest needs padded height C*np (unrepresentable
  from logical C*nkv, tt_metal::Shape carries no padding), and concat does
  not preserve per-input padding into its output storage.
  RED: new doctest `kTENSTORRENT batched decode RAC is capture-safe
  (num_slots=2)` reproduces the exact leg fatal with the decline restored
  (<logs-dir>/red-multislot.log, TT_FATAL fd_mesh_command_queue.cpp:873); GREEN
  with the fix: capture + replay complete and BOTH users' KV verified
  token-exact (128/128 K, 128/128 V) in the paged-KV device shadow via the
  new ReadPagedKvShadowForTest hook (<logs-dir>/green-multislot.log). The leg also
  needed no residency change — the fix is confined to tenstorrent_paged.cpp.
- 2026-09-28 BLOCKER B ANALYSIS — 27B whole-graph decode trace does not fit;
  recommendation is REGION-SCOPED capture. The numbers: (1) DEMAND: the
  captured decode graph records 1,037 tt-metal op entries (the step-decompose
  legE in-capture census, identical across all four captures —
  docs/bench-evidence/tt-27b-step-decompose-20260926.md) and end_trace_capture
  asks for one 3,153,969,152 B DRAM buffer (<logs-dir>/leg-27b-c1.log) — 3.04 MB
  PER RECORDED COMMAND. The MeshTrace buffer is the replay staging DRAM for
  the whole recorded command stream; at 1,037 commands the per-command region
  (launch descriptors, CB/semaphore state, runtime-arg and buffer-descriptor
  records) dominates, NOT tensor staging (every in-region tensor is
  persistent/preallocated by the warmup discipline; the 27B DRAM ledger shows
  the slot census flat) and NOT program binaries (cached outside the trace).
  (2) SUPPLY: 298,568,896 B free, largest block 280,928,384 B — and ~948
  MiB of that pressure is the tt-metal#57970-class GDN scatter retention
  (docs/bench-evidence/tt-ttm-retention-rootcause-20260925.md), i.e. at most
  ~1.9 GiB recoverable for a 2-request run, taking free to ~2.2 GiB. Even a
  FULL upstream retention fix leaves 3.15 GB > 2.2 GB: NO whole-graph scope
  fits the 27B decode step at this command density, and the gap is
  structural (3,037 more commands than a ~90-command budget of 280 MB).
  (3) PRECEDENT: region-scoped capture FITS and SERVES — the GDN-region row
  captured and replayed a multi-layer region inside this same step
  (.agents/specs/tenstorrent-gdn-region-replay.md,
  docs/bench-evidence/tt-gdn-region-replay-20260927.md), and the chunked E=1
  arm already targets a 50 MiB trace region (tenstorrent_capture.cpp's
  LastTraceBytes discipline). A region of ≤ ~90 commands fits today's free;
  ≤ ~16 commands fits a 50 MiB region.
  RECOMMENDATION: region-scoped capture for the 27B decode — capture the
  recurring per-layer compute region(s) (the kernel-dense GDN/attention/GEMM
  chain), replay per layer from a host loop, and keep the heterogeneous
  preamble/RAC/sampling ops eager. Whole-graph stays the right shape for the
  small models it already serves; forcing 27B into it is bounded by tt-metal's
  per-command trace cost, which no vllm.cpp-side discipline can shrink. Do
  not attempt this inside this issue — it is a fresh row (trace-budget,
  region segmentation, per-region state binding) with its own spec.
- 2026-09-28 DEVICE GATE (c2 leg, <logs-dir>/leg-27b-c2b.log = monitor
  bench-c2b): BENCH_EXIT=1 — BLOCKER A's fatal is GONE (the leg no longer
  dies at RAC; it served thousands of decode steps across ~16 minutes) but
  fails FURTHER DOWN at the NEXT site of the same class:
  PagedAttentionKernel's host fallback read mid-capture (EnsureHost ->
  DownloadToHost -> to_vector, TT_FATAL "Reads are not supported during
  trace capture" at fd_mesh_command_queue.cpp:873) during a late re-capture
  — TryPADecodeDevice's multi-slot condition declines in some late state
  (width change / boundary re-capture) and the leg falls to the PA host
  path. This is a NEW owed site (multi-slot PA device path under re-capture),
  the third of the class after the two capture-write sites and the RAC one.
  Also fixed en route: the batched user split handles the 27B's real rope
  shadow geometry — rank-2 token-row [T, nkv*d] (per-head d-ALIGNED column
  slices, the proven recipe) as well as rank-3 [C, nkv, d].
  RESIDUAL (recorded, not hidden): the new doctest passes standalone and in
  12-case subsets, but in the FULL 95-case suite it measures 96/128 K elems
  exact — user 1's SECOND head (and only that head) reads uninitialized bytes
  from the paged-KV shadow after the eager pass, deterministically, with the
  same 32-elem count across eager/capture/replay; the per-user
  mesh_command_queue().finish() sync did not clear it. The fused-update
  input dumps verify the device inputs are correct (out_headmax exact for
  all four per-user copies), so the divergence is inside tt-metal's
  copy/program-cache interaction for the LAST per-user input in a deep
  program-cache history. Owed: bisect which preceding case poisons the
  variant, then either the tt-metal report or a per-user program isolation.
  SUITE: 94/95 passed (the residual is the only failure; every pre-existing
  case stays green).
- 2026-09-28 Blocks A and B status: A = the decline is REMOVED and the
  batched RAC device path serves (warm in eager, replay in capture —
  <logs-dir>/red-multislot.log reproduces the old fatal, <logs-dir>/green-multislot.log
  the green case); the serve leg now reaches the PA site above. B = the
  trace-budget analysis above stands (region-scoped recommended). Issue
  stays OPEN for the PA multi-slot site, the batched-lane residual, and the
  27B decode-trace DRAM fit.
- 2026-09-28 (worktree row/tt-27b-capture-write) PA MULTI-SLOT SITE FIXED —
  the same doctrine as Blocker A, third site of the class. ROOT CAUSE
  (diagnosed live, <logs-dir>/leg-27b-diag.log = monitor bench-c2-diag, the leg
  re-run with VT_TT_TRACE_DEBUG=1): the final traces before the fatal are
  "PA q_from_device FAILED: tenstorrent PA: batched (B>1) Q 4D materializa-
  tion is not capture-safe; the host Q path must serve this step" ->
  "PA device decode FAILED: vt: tenstorrent: PA Q host path is not capture-
  safe (from_vector readback)" (tenstorrent_paged.cpp:1280) -> the host PA
  oracle's EnsureHost readback mid-trace (TT_FATAL fd_mesh_command_queue.cpp:873).
  The decline was the explicit `if (tt_capture_active() && Bu > 1) throw` in
  TryPagedAttentionDeviceDecode's identity-Q path — a stale W3-era guard.
  Its premise ("that program calls to_device — forbidden during trace
  capture") predates the W4 doctrine: the B>1 arm runs the IDENTICAL
  multiply(reshape(...)) chain in both passes, so the eager step warms the
  reshape program for the exact input/output spec and the captured call is a
  program-cache HIT; a spec the warmup did not warm still fatals loudly at
  the miss (the W4 divergence detector, not a defect to guard against).
  FIX: the guard is deleted; both passes run the same chain (one hunk in
  tenstorrent_paged.cpp, no capture-active branch).
  RED: new doctest `kTENSTORRENT batched decode PagedAttention is capture-
  safe (num_reqs=2)` (mirror of the RAC case) fails on HEAD for the right
  reason (<logs-dir>/red-pa.log): the capture pass takes the decline (trace in
  <logs-dir>/red-pa2.log shows the exact q_from_device FAILED chain) and the
  captured+replayed output mismatches the device-path reference 2048/2048 —
  the test stages generation-B K/V through RAC into the DEVICE paged-KV
  shadow before the capture, so only a device-served PA can reproduce it.
  GREEN: <logs-dir>/green-pa.log — capture pass serves (q_from_device OK cap=1),
  replay-vs-eagerB 0/2048 mismatched elems, both users nonzero.
  EN ROUTE BUG (own issue ISSUE-LOCAL-01M3KM4R2KQN5WXTM57W8BD849): the new
  case exposed that the batched RacIdxCache lane lacks the C=1 lane's
  page-table width-change guard — fixed in the same change (evidence in that
  issue). The RAC residual flake is NOT this: it still reproduces in the
  full suite and stays owed.
  SUITE: 95/96 (<logs-dir>/suite-pa3.log) — every pre-existing case green, the PA
  case green in-suite (0/2048), the only failure the recorded RAC residual
  (126/128 K/V, user-1 second head).
  DEVICE GATE: see the next dated entry (bench-c2c).
- 2026-09-28 DEVICE GATE (c2 leg, <logs-dir>/leg-27b-c2c.log = monitor bench-c2c,
  post-fix HEAD): BENCH_EXIT=1, but the PA multi-slot site is GONE — the leg
  served ~12+ minutes of batched decode through SIX successful boundary
  re-captures (11:24:06, 11:26:40, 11:29:13, 11:31:46, 11:34:15, 11:36:40)
  that previously died at the PA host readback. The fatal moved PAST the PA
  site to the ALREADY-RECORDED Blocker B structural limit: the last
  end_trace_capture asks for 3,128,655,872 B of DRAM trace staging (2.9 GiB,
  ~3 MB/recorded command) against 278,858,624 B free / 266,655,872 B largest
  block — bank_manager.cpp:495 OOM, same whole-graph-does-not-fit conclusion
  as the Blocker B analysis above (3.15 GB demand, ~2.2 GiB recoverable best
  case). No TPOT table: the leg died at a re-capture before completing the
  32-token horizon. Blocker B stays with its fresh trace-budget row; this
  issue's PA site is closed by the red/green + full-suite evidence above.
- 2026-09-28 (worktree row/tt-27b-region-capture-spec, dc99071ad + f376b512c):
  the Blocker-B region-scoped recommendation was implemented as the row's wave
  1 and the fit wall was MEASURED. The dense decode driver captures ONE REGION
  PER LAYER under VLLM_CPP_REGION_CAPTURE=1 through the bare GraphBreak seam;
  the red-first device gate proves a TWO-region capture replays byte-identical
  to eager across the boundary (region 0 = 2,048 B, region 1 = 3,088,384 B).
  The 27B c1 leg then died at the 8th segment close: tt-metal mesh_trace.cpp:125,
  trace buffer 4,226,469,888 B vs allocation high-water 4,229,506,816 B — the
  64 live regions SUM to the whole graph's ~3.15 GB staging (each region owns
  its trace staging until release, and all 64 replay every step), so
  segmentation does not shrink the fit demand. The whole-graph OOM and this
  collision are one structural limit. Evidence:
  docs/bench-evidence/tt-region-capture-20260928.md. The 27B decode-trace
  DRAM-fit site stays OPEN, now with the segmentation result recorded: it
  closes only on a tt-metal-side change (shared/reusable trace staging, a
  trace_region_size policy, or a #57970 retention recovery that actually
  covers 3.15 GB) — escalated with these numbers beside tt-metal#57970. The
  in-flow RAC C=1 segfault the legs exposed is ISSUE-LOCAL-01M3M0K390EM40W5R9BR5A2KZ7.

- 2026-09-28 UPLOAD-GUARD WAVE (worktree row/tt-27b-region-capture-spec,
  commits 6473ae731/286947603/39e2ca8ef, evidence
  docs/bench-evidence/tt-capture-upload-guard-20260928.md): the
  trace-record-audit inline-upload lever landed and the c1 leg FALSIFIED the
  attribution. The guard (UploadRows/UploadRowsBf16 refuse capture-scope
  uploads by name; the AddKernel broadcast operand warms into a hash-keyed
  cache) is red-first proven (pre-fix: raw TT_FATAL
  fd_mesh_command_queue.cpp:826, no refusal; post-fix: named refusal, warmed
  capture 1,024 B). SUITE: 98/526,778 assertions, only the pre-recorded RAC
  residual flake fails. MONEY LEG: 27B whole-graph c1 BENCH_EXIT=1, no TPOT —
  but zero capture-scope uploads fired (no [TT-UP] line in the leg) and the
  end_trace_capture demand is byte-identical 3,153,969,152 B; the warmed
  MatmulBT region still closes at 3,088,384 B. Model C (inline H2D payload) is
  falsified: the ~3 MB/command is the quant-matmul program class's own
  recorded launch stream on this tt-metal pin. This issue's fit site stays
  OPEN with a sharper next step: dump the tt::LogDispatch command stream for
  one captured MatmulBTQuantGrouped launch and attribute the ~3 MB of
  bypass_data (candidates: per-launch relay of kernel-binary pages for
  programs missing the 1,024 KB prefetch ringbuffer fit at
  fd_mesh_command_queue.cpp:453, or per-launch config/RTA page writes scaling
  with the quant program's footprint). The escalation beside tt-metal#57970
  now carries a reproducing op-scale case: a single warmed MatmulBT region
  closes at 3,088,384 B on this pin.
- 2026-09-28 ATTRIBUTION (worktree row/tt-27b-region-capture-spec, evidence
  docs/bench-evidence/tt-launch-record-attribution-20260928.md): the
  binary-relay candidate is FALSIFIED and the ~3 MB/command is attributed to
  tt-metal's per-core launch-record path. Source (pin d20b8e27f29): the
  1,024 KB prefetch-ringbuffer fit (fd_mesh_command_queue.cpp:453,
  dispatch_settings.cpp:72) only chooses relay_paged vs relay_ringbuffer —
  BOTH record the kernel binary BY REFERENCE to the resident DRAM kernels
  buffer (dispatch.cpp:1942-1971, 2079-2116); binary bytes never enter
  bypass_data; and load_binaries refuses first-time loads mid-capture by name
  (mesh_workload.cpp:201-205). Our keep-quant kernel measures .text 51,648 B +
  .data 3,372 B — 18x under the threshold. Device (region-handoff doctest under
  gdb breakpins, capture window gated on tt_capture_active()): the 3,088,384 B
  region record is exactly 284 issue_queue_reserve chunks (~10.9 KB each) of
  the recorded command stream for the ONE full-grid MatmulBTQuantGrouped
  program, with ZERO in-capture buffer-data writes (write_to_device_buffer: 0
  real hits) — so no inline H2D payload, ours or tt-metal's. The record scales
  with the keepquant program's per-core config/RTA dispatch footprint
  (RmsNorm-class program: 2,048 B total), i.e. tt-metal-side. Our-side grid
  shrink only scales the record linearly (halving the grid halves 3.15 GB —
  still OOM) and is recorded as a bound, not a fix. OPEN NEXT (one step): a
  logging-enabled pin build (TT_METAL_ENABLE_LOGGING=ON + TT_METAL_LOGGER_
  LEVEL=TRACE, names verified at tt-logger.hpp:98,182 — current release builds
  compile LogDispatch out, which is why the cheap logger leg was silent) to
  name the dominant per-chunk class; the attribution does not depend on it.
- 2026-09-29 (pin advanced to upstream-live 98134127a7b, pin head 6449cf13f7b,
  logging build dir build_logging): the logger discriminator CLOSED the open
  next step, and it flips the locus to OURS. On the new pin the focused leg
  reads region 0 = 2,048 B; region 1 = 2,965,504 B (<logs-dir>/tregion-newpin.log,
  1/1 case, 1,032/1,032 assertions) — the record barely moved, so upstream's
  576+ commits did not touch the per-core record path (create_trace_node /
  issue_queue_reserve unchanged in dispatch.cpp). The capture window is
  exactly 254 one-shot command-sequence fetches summing 2,961,024 B and
  contains 11,040 per-core Unique RTA (UNICAST) writes (40-48 B payload each,
  page-granular when recorded) plus 110 full-grid CB/DFB config pages. Those
  per-core RTAs are OUR keepquant program's SetRuntimeArgs stream
  (tenstorrent_keepquant.cpp:2108-2130): 12 words per core, 10 of them
  shape-global constants and only r0 = c*tcols / rc = clamp(...) varying —
  both derivable in-kernel from the core coordinate. CONCRETE FIX (ours):
  compute r0/rc in the kernel, launch with SetCommonRuntimeArgs only, delete
  the per-core SetRuntimeArgs stream. Expected record: ~2.97 MB -> the
  RmsNorm-class floor (~2-16 KB per captured launch, ~200x), which closes the
  27B whole-graph 3.15 GB trace demand. The earlier "tt-metal-side" locus is
  thereby refined: tt-metal faithfully records what our program asks it to
  dispatch per core; the shrink lever is ours.
- 2026-09-29 (worktree row/tt-27b-region-capture-spec, fix commits a1661114b +
  the RTA-removal commit): the CONCRETE FIX above LANDED and was measured.
  The keepquant program now launches SetCommonRuntimeArgs-only (14 words) and
  derives r0/rc in-kernel from the core coordinate
  (tenstorrent_keepquant.cpp, CARG_* table; host per-core SetRuntimeArgs loop
  deleted). Correctness held: E=1 grouped keep-quant capture-x2 byte-identity
  PASS with a partial last core, host derivation-parity doctest added. 27B c1
  arbiter (fresh build2, pin 6449cf13f7b): trace demand
  3,153,969,152 -> 2,925,109,248 B (-228,859,904 B = 1,037 launches x ~920
  cores x one 256 B recorded RTA page) — the lever is real and SPENT, but
  BENCH_EXIT=1: the whole-graph trace STILL does not fit; c1 does not serve.
  Two small-shape A/Bs (region-handoff 2,965,504 B both binaries; grouped
  capture 48,316,416 B both binaries, dispatch-log Unique-RTA counts
  identical) show the ~2.9 MB per captured command at those shapes was never
  the RTA stream: the dominant remaining class is per-launch full-grid
  program command-sequence payload (CB/DFB config pages per sequence). The
  KB-floor gate for this vehicle was removed as unreachable by this lever
  (records: docs/bench-evidence/tt-keepquant-rta-fix-20260929.md §3-§5). The
  fit wall stands at 2,925,109,248 B; next lever is the per-launch
  command-sequence payload class (issue stays OPEN, evidence above updated).
- 2026-09-29 (worktree row/tt-27b-region-capture-spec, this leg): the
  per-launch "config-page payload" hypothesis is REFUTED, and the record's
  locus is now program COUNT in the grouped decode chain. The §4 synthetic
  bisect (docs/bench-evidence/tt-trace-config-page-repro-20260929.md §5,
  repro_bisect_program_shape.cpp) built the keepquant program's exact shape
  raw (full-grid CoreRange DM kernel, common RTAs) and swept CB count
  (4/8), CB page size (4/16 KiB) and kernel binary size (32-256 KiB .rodata
  tables): every variant records 1,024 B/launch — the packed relay collapses
  all of them, so non-identical per-core config pages are not the 2.82 MB.
  The region-handoff doctest on HEAD (now gated at 64 KiB, RED measured
  2,965,504 B, <logs-dir>/region-red.log) showed the default dispatch there is the
  W4a GROUPED arm: the region is ceil(N/8)=8 chunks x (~85 eltwise decode
  programs from DecodeKeepQuantWordsF32 Q6_K + ~8 matmul-chain programs) ≈
  680 programs x the 4-17 KB per-program floor = 2.97 MB. NEXT LEVER (one
  step): collapse the decode to ONE custom-kernel program per launch — the
  bit-exact int8-dot kernel is the existence proof at the floor — and the
  64 KiB doctest gate is the arbiter. The 27B fit wall keeps its
  2,925,109,248 B bound; c1/c2 re-measure legs stay blocked behind the
  fusion. Issue stays OPEN.
- 2026-09-29 (same leg, suite): the 64 KiB region gate landed RED by design
  (commit 7a0f1ca3f). Full TT suite (build2, pin 6449cf13f7b): 99 cases,
  97 passed, 2 failed — the owed 2261 flake (126 == 128) and the new
  intentional red gate (11527, 2,965,504 > 65,536); 527,817/527,819
  assertions; all keepquant bit-exact capture-x2 cases green
  (<logs-dir>/suite-final.log, teardown segfault after the run is pre-existing).
  27B money legs NOT run: no fix landed this leg, so c1 would reproduce the
  recorded 2,925,109,248 B / BENCH_EXIT=1 outcome; the legs stay blocked
  behind the decode-fusion lever.
- 2026-09-30 (worktree row/tt-decode-fusion, commit 1d84f00a6, TT-DECODE-FUSION):
  the NEXT LEVER above LANDED and was measured. The ~85-program Q6_K
  eltwise decode chain is now ONE full-grid custom kernel per chunk launch
  (kKeepQuantDecodeFusedKernelSrc + DecodeKeepQuantWordsFusedQ6K,
  tenstorrent_keepquant.cpp; int8dot house style: one SetCommonRuntimeArgs
  vector, uniform self-cycled CBs, in-kernel row0/rowc, warm-first cache;
  the chain stays as the named VT_TT_KEEPQUANT_FUSED=0 fallback). The 64 KiB
  arbiter is GREEN: region 1 reads 32,768 B (was 2,965,504 B RED),
  <logs-dir>/kq-focused.log. Bit-exactness: fused vs chain byte-identity through
  the grouped P=1 decode op on idle-core and partial-last-core tails with
  planted zero-d/zero-scale blocks, 7/7 assertions (<logs-dir>/kqf-bitexact.log).
  Full TT suite: 99/100 cases, 527,825/527,826 assertions, the only failure
  the owed 2261 flake (<logs-dir>/kqf-suite.log). Evidence:
  docs/bench-evidence/tt-decode-fusion-20260930.md. 27B c1 INT8DOT=0
  (<logs-dir>/kqf-c1-int8dot0.log): BENCH_EXIT=1 — the whole-graph trace still
  does not fit (populate_mesh_buffer overlap fatal, buffer address
  4,015,745,024, allocation high-water 4,228,372,992); no TPOT. The decode
  region record fell ~90x at the vehicle shape, but the whole-graph demand
  (matmul chain ~8 programs/chunk + the rest of the graph) still exceeds
  free DRAM. NEXT LEVER (one step): the matmul-chain fuse, then re-measure
  the whole-graph bound; the fit wall keeps standing until that lands.
  INT8DOT=1 c1 and c2 queued behind the device lock. Issue stays OPEN.

- 2026-09-30 (row/tt-matmul-record-audit @ 913392513): the matmul-class
  attribution landed. Standalone repro `docs/bench-evidence/
  tt-matmul-record-attribution-20260930.md` on pin 6449cf13f7b: stock
  ttnn::matmul at 27B shapes ([64,5120]x[5120,5120] bf16, full 110-core
  grid) records 8,484 B/launch (small shape 3,510; down-proj 4,242;
  empty-trace floor 1,024). Control: the old inline-H2D attribution is
  FALSIFIED on this pin — a from_vector issued inside a capture FATALS at
  fd_mesh_command_queue.cpp:830 ("Writes are not supported during trace
  capture") and aborts, <logs-dir>/mmrepro-run3.log. Verdict OURS: the ~8
  MB/launch the ~357 non-eltwise launches need cannot come from tt-metal's
  matmul record path; fix locus is our MatmulBT/attention program
  construction in src/vt/tenstorrent/ — next leg discriminates per-core
  config-page packing failure vs prefetch-ring overflow vs captured
  device-side movement on one post-fusion-pin MatmulBT region. No upstream
  defect to file. Issue stays OPEN; whole-graph fit wall keeps standing.

- 2026-10-01 (row/tt-matmul-class-split, worktree <worktree>):
  the class split LANDED and the last unexplained class is attributed. New
  focused doctest `VT_TT_MMCLASS` (tests/vt/test_tenstorrent_backend.cpp):
  one warmed MatmulBT capture at [64,5120]x[5120,5120] closes at **147,456 B**
  (stock ttnn::matmul same shape: 10,240 B); launches=8 is linear at
  ~143,945 B/launch; N=17408 gate/up records 540,672 B/launch. Logging-build
  dispatch census (<logs-dir>/mmsplit/logging1.full.log): the launch enqueues **20
  tt-metal programs**, whose Command Sequence Summary TOTALs sum to 147,264 B
  — the region close minus the 192 B header floor, i.e. FULLY attributed.
  Candidates (a) non-identical pages (packed MCAST relay works),
  (b) prefetch-ring overflow (15,872 B max one-shot fetch vs 1,024 KB fit),
  (c) captured D2D (guards held) all REFUTED. Dominant class: (d) many
  programs per launch, each at tt-metal's KB floor. Wave-2 fix: fuse the
  chain to 1-2 full-grid kernels (kKeepQuantDecodeFusedKernelSrc style),
  locus tenstorrent_keepquant.cpp:1279 + MatmulBT dispatch in
  tenstorrent_ops.cpp; expected 147,456 -> ~8-16 KB/launch (~10-18x).
  Evidence: docs/bench-evidence/tt-matmul-class-split-20261001.md. Issue
  stays OPEN; c1/c2 re-measure legs stay blocked behind the wave-2 fuse.

- 2026-10-02 (row/tt-matmul-fusion, worktree <worktree>):
  the wave-2 whole-decode fuse LANDED on the safe route the class-split
  verdict names. The E=1 dense arm now serves Q6_K shapes whose whole
  decoded f32 plane fits the chunk budget through ONE fused-kernel decode
  launch + ONE stock matmul + a TILE-domain partial tail (~6 tt-metal
  programs per launch, was ~20); the exact-f32 decode arm, over-budget
  planes, and every unserved encoding fall to the proven chunk chain by
  name, and VT_TT_KEEPQUANT_MM_CHAIN=1 forces the chain (the named kill
  switch). MEASURED: the captured [64,5120]x[5120,5120] launch records
  **23,552 B** where the chain recorded 147,456 B (red-first: the new
  gate read 147,456 B on HEAD — byte-exact the class-split number — and
  failed the <= 32,768 B gate; green after). Bit-exactness: the
  fused-vs-chain memcmp golden is byte-identical on four shapes (prefill
  partial/idle tails, single-chunk prefill, exact-f32 P=1 both forms,
  zero d/scale blocks mixed in). Evidence:
  docs/bench-evidence/tt-matmul-fusion-wave2-20261002.md. Issue stays
  OPEN pending the 27B c1/c2 re-measure legs this wave unblocks.

- 2026-10-02, second entry (row/tt-matmul-fusion): the 27B re-measure
  legs RAN on the wave-2 tree. c1 INT8DOT=1 **BENCH_EXIT=0 with a full
  TPOT table — the first 27B serve on the live capture arm** (TPOT mean
  6,758.97 ms, TTFT 821,024 ms); c2 INT8DOT=1 BENCH_EXIT=0 (TPOT
  25,684.37 ms, TTFT 1,218,612 ms — the two streams serialize on the one
  replay queue). Both INT8DOT=0 legs still die on the whole-graph fit
  wall (mesh_trace.cpp:126) — attributed to the checkpoint's Q4_K ffn/attn
  weights, which the wave-2 Q6_K fuse does not serve: the P=1 dense arm
  runs the per-chunk chain there. OWED NEXT: the Q4_K arm of the fused
  whole-decode dispatch (each encoding through its golden before its
  default flips, per the standing rule). Evidence:
  docs/bench-evidence/tt-matmul-fusion-wave2-20261002.md. Issue stays
  OPEN; the fit wall keeps standing on the Q4_K arm only.

- 2026-10-02 (row/tt-q4k-fusion, worktree <worktree>,
  HEAD 17a42d831): the wave-3 Q4_K arm's money legs RAN — the fit wall
  is GONE. All four legs BENCH_EXIT=0 with full TPOT tables: c1
  INT8DOT=0 **31,345.59 ms** TPOT (THE GOAL — the first INT8DOT=0 27B
  serve; wave 2 died here on `mesh_trace.cpp:126`, DRAM high-water
  4,211,219,712 B), c2 INT8DOT=0 31,573.54 ms, c1 INT8DOT=1 6,680.05 ms
  (wave-2 6,758.97, −1.2%, no regression), c2 INT8DOT=1 25,434.18 ms
  (wave-2 25,684.37, −1.0%, no regression). The checkpoint's
  quantization serving coverage is COMPLETE: both arms, both
  concavities. The INT8DOT=0 f32-out arm runs ~4.7x the INT8DOT=1 TPOT
  (twice the decode-plane bytes) — honest first numbers, TPOT levers
  stay individually traceable. Evidence:
  docs/bench-evidence/tt-q4k-fusion-20261001.md. Logs
  <logs-dir>/leg-c{1,2}-i{0,1}.log.

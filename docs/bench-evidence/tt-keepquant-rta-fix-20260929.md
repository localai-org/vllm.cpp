# the per-core RTA fix: measured, and what it did NOT close (2026-09-29)

Worktree `row/tt-27b-region-capture-spec` at the fix commit. Follows
[tt-launch-record-attribution-20260928.md](tt-launch-record-attribution-20260928.md),
whose verdict named our keepquant program's per-core `SetRuntimeArgs`
(`tenstorrent_keepquant.cpp:2108-2130`) as the ~2.9 MB per captured launch.

## 1. The fix (landed)

`src/vt/tenstorrent/tenstorrent_keepquant.cpp`:

- The kernel (`kernel_main`) reads ALL runtime words from
  `SetCommonRuntimeArgs` (14 words: the 3 bank bases + M/K/N/nb/wpb/act_f32/
  mtile/qb_pad/enc/tcols + grid_x) and derives the per-core slice in-kernel:
  `c = get_relative_logical_y() * grid_x + get_relative_logical_x()`,
  `row0 = c * tcols`, `rowc = row0 >= N ? 0 : min(tcols, N - row0)` — the
  exact guard the deleted host loop applied, including the fully-idle tail.
  Grid is part of the workload key, so `grid_x` is shape-global per program.
- The host per-core `SetRuntimeArgs` loop (12 words × grid_cores per call) is
  DELETED; every word moves to the common-args vector, set once on a
  workload miss and updated in place on a hit. This call site is the only
  caller of the program; nothing else needs per-core args on it.

Correctness: the keepquant capture-x2 byte-identity suite stays green on the
fix (E=1 grouped keep-quant capture, full test suite below); the shape
carries a partial last core, so the in-kernel clamp is device-proven. A new
host doctest pins the in-kernel derivation to the deleted loop's values for
every core across partial/idle tail shapes.

## 2. Measured on the 27B whole-graph capture (the arbiter)

c1 leg (`Qwen3.8-27B-Q4_K_M`, 2x128/32, `VT_TT_TRACE_DEBUG=1`), fresh
build2 against the new pin `6449cf13f7b`:

| | trace demand at end_mesh_capture |
|---|---|
| pre-fix (attribution doc) | 3,153,969,152 B |
| post-fix (this leg, /tmp/money-c1.log) | 2,925,109,248 B |

The fix removed **228,859,904 B** ≈ 1,037 captured launches × ~920 grid
cores × one recorded 256 B RTA page each — exactly the per-core
`SetRuntimeArgs` stream the fix deleted. `BENCH_EXIT=1`: the capture still
fatals `Out of Memory: Not enough space to allocate 2925109248 B` (bank
manager, 8 banks). **The whole-graph trace still does not fit DRAM; c1 does
not serve.** The remaining ~2.93 GB is NOT per-core RTAs.

## 3. The attribution doc's per-launch magnitude was wrong; its direction was right

Two controlled A/Bs on small keepquant captures (same command, red vs green
binary, new-pin libs):

- region-handoff doctest: region 1 = **2,965,504 B on BOTH binaries** —
  byte-identical. Dispatch-log counts (TT_METAL_LOGGER_LEVEL=TRACE,
  build_logging): 22,079 Unique-RTA lines on both.
- E=1 grouped keep-quant capture: device trace demand
  **48,316,416 B on BOTH binaries**, capture-x2 byte-identity PASS on both.

So at these shapes the per-core RTA stream contributed ~0 to the recorded
region — the 2.9 MB per captured command is dominated by the ~250 remaining
"one-shot program command sequence" fetches per launch (full-grid CB/DFB
config pages and per-sequence chunks), which are per-launch, not per-core.
The 27B A/B is the honest measurement: −228.9 MB real, wall standing.

## 4. OPEN NEXT (updated)

The per-core RTA lever is SPENT (landed, correct, ~7.3% of the demand). The
dominant remaining class is per-launch program command-sequence payload:
~1,037 launches × ~2.7 MB, i.e. tt-metal records each launch's full-grid
CB/DFB configuration per sequence. Candidate levers, in traceable order:

1. Count the remaining capture-window classes with the existing
   logging-enabled discriminator on a 27B capture attempt (the 254-fetch
   census of the attribution doc, rerun post-fix) — name the per-sequence
   payload composition before touching anything.
2. tt-metal-side: whether `create_trace_node` can dedupe/re-reference
   unchanged full-grid config pages across replays of the same program
   (upstream question; pin-local experiment first).
3. Our-side: fewer full-grid CB/config-bearing programs per launch (merge
   programs), or capture at coarser launch granularity — our-side grid
   shrink only scales linearly and stays OOM (recorded as a bound, not a
   fix).

## 5. Test reconciliation

The region-handoff KB-bound gate added red-first for this fix measured
2,965,504 B before AND after (§3), so the KB floor is not reachable by
removing per-core RTAs and the gate was removed rather than left red. The
keepquant capture-x2 byte-identity gates and the derivation-parity doctest
stand. The 27B trace-fit assertion remains the bench leg (the only vehicle
at that scale), still failing at 2,925,109,248 B — the row's fit wall.

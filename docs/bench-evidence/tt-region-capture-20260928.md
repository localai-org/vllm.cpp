# tt-27b-region-capture — wave-1 evidence (2026-09-28)

Worktree `row/tt-27b-region-capture-spec`, HEAD ec4e8a824 + the commits below.
Host thalia (P150), all device work under `/home/lu_zero/gpu.lock`, device reset
before every leg, `~/Sources/tt/env-tt-common.sh`, tt-metal pin
`~/Sources/tt/tt-metal-pin` (read-only). One fresh process per leg.

## The region handoff (spec tests 1+2, op scale)

- RED: the pre-row tree has no per-region census —
  `tests/vt/test_tenstorrent_backend.cpp`'s
  `kTENSTORRENT region replay: state handoff across a region boundary` does not
  compile there (no `vt::BreakableGraph::region_bytes()`, no
  `vt::GraphRegionBytesProbe`), and with `VLLM_CPP_REGION_CAPTURE=1` set the
  whole-graph arm dies at the trace OOM (every leg below, EXIT=139/1).
- GREEN (focused, `/tmp/tregion.log`): TWO regions on the real tt-metal trace
  backend — region 0 `RmsNorm` writes the PERSISTENT norm buffer in place,
  `vt::GraphBreak()` (bare form, no eager call) closes segment 1 and opens
  segment 2, region 1 `MatmulBT` reads that same buffer. Per-region census:
  **region 0 = 2,048 B, region 1 = 3,088,384 B**, both ≤ 50 MiB. Each replay is
  BYTE-IDENTICAL to the eager reference (memcmp, 4,096 B f32). The reviewer's
  mutation target is the #3327-class defect: give region 0 a FRESH output
  buffer and region 1's baked address reads freed storage — the replay stops
  being the eager bytes.
- Host-side census + predicate (`/tmp` run of `test_breakable_graph -tc="tt-27b*"`):
  12/12 assertions — `WholeGraphTraceFits` declines only on a measured
  over-budget estimate (1,037 × 3,187,104 B > 298,568,896 B free = FALSE;
  zeroed fields = TRUE), and every segment records its staging delta
  (4 segments from 3 breaks on the recording backend).

## The in-flow bug: RAC C=1 lane (ISSUE-LOCAL-01M3M0K390EM40W5R9BR5A2KZ7)

- RED (`/tmp/leg-control-c1.log`, `/tmp/leg-region-c1-diag.log`): the c1 leg
  segfaults in `ttnn::copy` inside `ReshapeAndCacheKernel` at the FIRST COLD
  eager decode step (capturing=0) — identical WITH and WITHOUT
  `VLLM_CPP_REGION_CAPTURE`, so pre-existing on the base, not the region arm.
- Cause: e39f2cf3f's batched-lane rewrite routed C=1 through the batched
  arrays WarmRacIdx never allocates for one user.
- GREEN (`/tmp/leg-region-c1-fix.log`): C=1 restored verbatim
  (shared `sharded_in`/`update_idxs`/`page_table`); the cold step and the
  capture pass now run — the leg reaches deep into per-segment capture (8
  `BeginCapture`s) before hitting the fit wall below.

## The 27B c1 leg — the fit wall, measured

`/tmp/leg-region-c1-fix.log` (2026-09-28 14:52, fresh reset):

- Per-segment capture proceeds segment by segment; the 8th segment close dies
  in tt-metal `populate_mesh_buffer` (`mesh_trace.cpp:125`):
  `Trace buffer at address 4,226,469,888 overlaps with DRAM activity during
  trace capture. Allocation high water mark: 4,229,506,816`.
- **The verdict: segmentation does not shrink the total staging demand.** Each
  live region owns its trace staging until released, and all 64 regions must
  stay live to replay every step, so the 64 per-layer regions sum to the same
  ~3.15 GB (1,037 commands × ~3.04 MB/command) the whole graph asked for —
  against the same free DRAM. The whole-graph OOM and this region-arm
  collision are ONE structural limit wearing two fatals.
- `BENCH_EXIT`: 1 (engine fatal). **No TPOT table: the leg cannot complete a
  step horizon.** The honest baseline stands as the spec records it: the
  whole-graph arm serves NOTHING (BENCH_EXIT=1 at its own trace OOM) and the
  region arm serves nothing YET — the 27B TT arm stays masked.
- Per the spec's stop conditions: the census is recorded, the per-command
  trace cost is not shrinkable from vllm.cpp, and the numbers go beside
  tt-metal#57970. The arm remains env-gated (`VLLM_CPP_REGION_CAPTURE`) and
  default-off; the over-cap decline (named, sticky, per size) is in the tree.

## What did NOT get measured

- The TPOT table (no completed leg), the boundary-re-capture rate, the c2 leg,
  the INT8DOT correctness leg — all blocked by the fit wall above.
- The in-tree model-scale served-replay doctest for the region arm — owed to
  the wave that lands a serving arm.

## Suite

See the dated entry in the row spec's `## Now` (filled after the run).

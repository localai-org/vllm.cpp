# QWEN38-DFLASH2-STAGE-A — the trace contract and the both-ways single-token measurement (2026-11-07)

Row: `QWEN38-DFLASH2-STAGE-A` (record `BACKEND-TENSTORRENT-QWEN38-DFLASH2-STAGE-A`),
issue `ISSUE-LOCAL-01M6VZZE9W9Q1JB8VX4GH4ADET`, spec
`.agents/specs/qwen38-dflash2-stage-a.md`. Stage 0's study:
`docs/design/qwen38-dflash2-integration-20261102.md` (on `mine/row/QWEN38-DFLASH2`).

Head at instrumentation: `5400fd5b9` (see `head-sha.txt`). Host: thalia
(aarch64, 128 cores), local Blackhole P150, `flock $HOME/gpu.lock` mutex,
`luwen` reset + 15 s + `/dev/shm` lock cleanup per leg, idle gate logged,
governor `performance` logged per leg (`governor-log.txt`). Substrate
unchanged: `LD_LIBRARY_PATH=/tmp/umdtrial-install/lib64:/tmp/umdtrial-install/libexec/tt-metalium`,
`TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium`,
`TT_METAL_HOME=~/Sources/tt/tt-metal-pin`. Build `/tmp/build-stagea`, Release,
`VLLM_CPP_TENSTORRENT=ON`, `taskset -c 64-127 -j 64`. Bench legs `taskset -c 32-63`.
`VT_TT_KEEPQUANT_INT8DOT=1` MANDATORY on every leg.

## The measurement

Production `vllm-bench` (AsyncLLM), batch 1, input 32, output 200, seed 0,
`--num-blocks 64 --max-num-batched-tokens 64`. 3 arms x 3 reps, fresh reset
per leg, medians. Instrumentation env on every leg (default-off knobs, the
`VT_TT_STEP_PHASES` pattern): `VT_TT_STEP_PHASES=1 VT_FLINT_PHASE_TIMING=1
VT_DECODE_GRAPH_STATS=1 VT_REGION_CENSUS=1`.

Arms:

- **defaults** — the rebench recipe's env (no capture knobs set). The
  whole-graph capture path.
- **region** — `VLLM_CPP_REGION_CAPTURE=1`, the region-scoped arm, one
  region per layer.
- **eager** — `VT_TT_DECODE_CAPTURE=0`, every step dispatches op-by-op.

## Result table

| Arm | kind= per leg | Median TTFT (ms) | Median TPOT (ms) | ms/step (device read) | step wall (host) | op dispatches/step | replay enqueues/step |
|---|---|---:|---:|---:|---:|---:|---:|
| defaults r1 | cold, capture, replay x197 | 213386.14 | 6637.35 | 6623 | 3.7 | 0 | 1 |
| defaults r2 | cold, capture, replay x197 | 213320.38 | 6638.19 | — | 3.7 | 0 | 1 |
| defaults r3 | cold, capture, replay x197 | 213521.52 | 6636.68 | — | 3.7 | 0 | 1 |
| **defaults median** | | **213386** | **6637.35** | | **3.7** | **0** | **1** |
| region r1 | cold, capture, replay x197 | 213308.62 | 6646.16 | — | 3.9 | 0 | 65 |
| region r2 | cold, capture, replay x197 | 213664.84 | 6645.72 | — | 3.9 | 0 | 65 |
| region r3 | cold, capture, replay x197 | 213745.42 | 6645.69 | — | 3.9 | 0 | 65 |
| **region median** | | **213664.84** | **6645.72** | | **3.9** | **0** | **65** |
| eager r1 | no graph driver; eager forward x200 | 214168.17 | 6682.12 | 359 (queue pre-drained) | 6336 | ~963 | 0 |
| eager r2 | no graph driver; eager forward x200 | 214418.26 | 6680.56 | — | — | ~963 | 0 |
| eager r3 | no graph driver; eager forward x200 | 213002.43 | 6680.35 | — | — | ~963 | 0 |
| **eager median** | | **214168.17** | **6680.56** | | — | **~963** | **0** |

Run spread inside a warm cell is ≤0.2% (defaults 6636.68-6638.19;
region 6645.69-6646.16; eager 6680.35-6682.12).

The `~963` eager dispatch count is measured on the capture step's recorded
body (`dispatches=963` in defaults-r1's `kind=capture` line — the same
forward, dispatched op-by-op once); the eager arm's decode bypasses the
dense decode-graph driver entirely (`VT_TT_DECODE_CAPTURE=0` makes the
driver's `enabled` false and the model forward never enters `Step`), so
its per-step line comes from the runner bracket (`[FLINT-STEP]`) and the
per-layer probes instead: 64 layers x 200 steps, steady-state host
dispatch ~98 ms/layer = **6.29 s/step of host op dispatch**
(`eager-r1.stderr.log`, steps 50/100/150/199 sums 6282/6289/6298/6307 ms).

## The verdict: COMPUTE-BOUND

The per-step numbers decide the study's §3 bound question:

1. **The host is not the bottleneck on the captured arm.** A replayed step
   costs wall_ms=3.7 on the host, resolves ZERO op-table dispatches
   (`dispatches=0` — the trace replays through one `execute_trace` enqueue),
   and stages 20 warm/H2D refreshes (warms_ms=0.1). The device forward tail
   — what the sampler's blocking logits read waits on — is 6623 ms of the
   6637 ms TPOT (99.9%). The step is device-work-bound, end of question.
2. **Region splitting costs nothing measurable**: 65 replay enqueues/step
   instead of 1 moves TPOT 6637→6646 ms (+0.14%). The dispatch-vs-compute
   question for the CAPTURED arm is closed in favor of compute at BOTH
   region granularities.
3. **The eager arm quantifies the dispatch counterfactual**. The eager
   forward spends **6.29-6.34 s/step of HOST op dispatch** (~98 ms/layer
   over 64 layers), against the captured arm's 3.7 ms — a ~1700x host
   reduction. End-to-end TPOT moves only 6637→6681 ms (+0.65%), because
   at THIS shape the host dispatch cost nearly equals the device work and
   the two overlap; the eager arm is dispatch-bound at the layer level
   with the device tail hidden underneath it. The dispatch bill is real
   and the capture removes it; at a faster kernel set (or more rows per
   step) the ~6.3 s/step dispatch bill would become the dominant term —
   which is exactly the trap the on-card DFlash2 loop (one host touch
   per iteration) avoids and the eager drafter-host loop would hit.
4. **Row scaling**: at batch 1 / input 32 the TPOT is 6637 ms — versus the
   rebench's 25230 ms at batch 2 / input 128 (S=2 padded, longer context).
   The step cost tracks the served rows and context, not a fixed host
   overhead. A K+1=9-row verify block therefore pays ~9 rows of the same
   device work unless the stage-B measurement shows otherwise; the DFlash2
   steady state (verify 9 rows / ~6.5 accepted) must clear that arithmetic
   at the stage-D acceptance gate. This row's GO/NO-GO rule
   (compute-bound → stages proceed) holds; the break-even burden travels
   with it.

## The arm the rebench served

**The CAPTURED whole-graph arm.** The rebench recipe sets no capture env;
defaults-r1..r3 prove that env captures at S=1: `kind=cold` on the boundary
step, `kind=capture` on the next, then 197 `kind=replay` steps, and
`[DenseDecodeGraph] captured ... for padded size S=1` + `198 total replays
across 1 captured size(s)` in the stats. The sticky-decline eager arm never
engaged. The rebench README's silence is resolved: its 25.23 s TPOT was a
replayed trace, one replay enqueue per step, at S=2.

## The sticky-decline cliff: it cannot fire

`SizeSlot::region_declined` (`src/vllm/model_executor/models/qwen3_5.cpp:12254`,
field + sticky comment) has NO writer anywhere in the tree: the over-budget
"decline BY NAME" the tt-27b-region-capture comment describes is declared,
not implemented, and the `vt::WholeGraphTraceFits` predicate
(`include/vt/breakable_graph.h:221`) has no caller. Measured: the whole
graph at S=1 stages **125,558,784 B (119.7 MiB)** in ONE segment against
~2.2 GiB free DRAM and captures fine; the region arm's 65 segments each
stage 0.24-2.20 MiB (max 2,195,456 B) against the 50 MiB per-region budget
— 23x headroom. The cliff the study feared does not exist at our shape
today: capture succeeds in both scopes. The real residual hazard is the
one the code retains: a capture that FAILS propagates (`capture_failed()`
rethrow), it does not decline to eager — that repair, and wiring the fit
predicate, are stage-B/owed work, not stage-A behavior changes.

## The trace contract for stages B-D (written from measurement)

For the `(S, 9, spec=true)` verify slot:

1. **Slot key**: `DecodeGraphSlotKey{S, Q, spec}` (`qwen3_5.cpp:11104`) — a
   9-row verify is a NEW slot key on the EXISTING state machine, measured
   here cold→capture→replay stable for 197+ steps. The exact-shape capture
   (S==B, no padding) the spec path takes is the same lane the W8 aux hook
   already sizes (`s.aux [S, H*taps]`). Distinct query lengths per driver
   are bounded by `VT_SPEC_GRAPH_MAX_QLENS` (default 2); DFlash2's
   steady-state `1+k` plus the clamped tail fits that bound, but a third
   shape goes EAGER by `qlen_capped` — the stage-B risk to log per step.
2. **Fixed-address persistent inputs**: measured working — token/position
   staging lands in the slot's persistent buffers, the trace re-reads them
   every replay (197 replays, byte-stable kind=replay line, dram_free flat
   at 2234-2235 MiB). A verify slot needs the same for 9 rows plus the
   persistent aux DBuf; the aux seam on TT has the recorded defect (the
   D2D Copy arm re-shadows the aux slot with per-tap clones instead of
   writing the [S, H*taps] buffer, `qwen3_5.cpp` VT_TT_GDN_REPLAY_TAP
   comment) — stage B must fix or route around it before taps feed a
   drafter.
3. **Capture scope**: both work and both are cheap on the host. Region
   scope's 50 MiB/region budget has 23x headroom at 9 rows' expected
   per-region growth (row-dependent ops scale the command stream, not the
   per-command staging). Whole-graph at 9 rows stages ~9x the row-dependent
   bytes of the S=1 119.7 MiB measurement — stage B should census the
   actual 9-row fit before choosing the scope; the fit predicate does not
   decide it today (see the cliff section).
4. **GDN state advance/rollback** — the largest stage-B mechanism, and the
   capture machinery makes the naive form WRONG: on TT the captured graph
   does not read state-slot indices at replay; `GdnDecodeKernel` /
   `CausalConv1dUpdate` resolve them on the HOST and the trace BAKES the
   content-keyed gathers/scatters at capture time (the TT-GDN-SLOT-CHURN
   discipline, `qwen3_5.cpp`). A rejected verify tail therefore cannot be
   undone by re-seeding indices into a replayed trace: the state advance
   of the rejected rows is IN the trace. Stage B needs the two-buffer A/B
   commit-or-restore of the touched states (or recompute-from-checkpoint),
   plus the existing changed-state reset lane (`gdn_idx_changed` →
   graph Reset → eager rebuild → re-capture) as the fallback. Every
   rollback claim must be mutation-tested against the greedy-exactness
   gate on multi-block trajectories.

## The eager arm

Served through the model's non-graph forward (see the table note). Steady
state per decode step: `[FLINT-STEP] fwd_ms=6335.9 step_wall_ms=6695.4`,
`sample_ms=359` (the blocking read is short because the slow host
dispatch drains the queue ahead of it); per-layer host dispatch
~98 ms/layer (gdn) / ~128 ms (fa), 64 layers, 200/200 steps. Median TPOT
6680.56 ms, +0.65% over the captured arm at this shape.

## Governor and idle

`performance` found and left on every leg (`governor-log.txt`, before and
after each leg); idle-gate 1-min load recorded per leg in the leg header
(max seen 14.21, below the 16 gate); per-leg `ps` top-CPU snapshots in
`ps-top-*.txt`.

## Raw logs

`defaults-r{1,2,3}.log` + `.stderr.log`, `region-r{1,2,3}.log` +
`.stderr.log`, `eager-r{1,2,3}.log` + `.stderr.log`, `governor-log.txt`,
`ps-top-*.txt`, `head-sha.txt`.

## Deviations

- TPOT/TTFT at this shape (batch 1 / input 32) are NOT comparable to the
  rebench's 25.23 s / 1203.6 s cells (batch 2 / input 128 / concurrency 2);
  the two measurements serve different questions and both are recorded.
- The smoke leg (`smoke-region.*`, 6 output tokens) preceded the series to
  validate the instrumentation; its TPOT 8050 ms is a 5-step median and is
  not part of any cell.

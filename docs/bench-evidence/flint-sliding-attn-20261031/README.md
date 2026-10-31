# FLINT-SLIDING-ATTN — the sliding-phase seq growth attributed, then the serial (token,head) split fixed

Row: `FLINT-SLIDING-ATTN`
Spec: `.agents/specs/flint-sliding-attn.md`
Issue: `.agents/issues/FLINT-SLIDING-ATTN/ISSUE-LOCAL-01M68S34FDEF6M6CARKGQFEP0V.md`
Date: 2026-10-31. Host: thalia (aarch64 128 cores). Governor **performance** AS
FOUND (unchanged). Idle gate (1-min load < 16) before every leg; the 32t
baseline legs were repeated after a foreign session (`test_flint_tt_w`,
TT-ANCHORING) released cpu 34 inside the bench zone — the first 32t triple ran
under that contamination and is discarded (`base-32-rep{1,2,3}.log` kept for
the record, numbers NOT used). Artifact:
`/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`, sha256 `20c89707…5927e` (full value
in `host-state.txt`).

Recipe (per leg) — identical workload both sides, `VT_FLINT_PHASE_TIMING=1`:

```sh
VT_FLINT_KEEP_QUANT=1 VT_FLINT_PHASE_TIMING=1 VLLM_CPP_CPU_THREADS=$TH \
taskset -c $ZONE ./examples/vllm-bench \
  --model /mnt/models/flint-GGUF/flint-Q4_K_M.gguf \
  --num-prompts 1 --input-len 8 --output-len 200 --concurrency 1 \
  --seed 0 --temperature 0 --ignore-eos
```

One `[FLINT-PHASE] batch=1` line per decode step (199 steady steps per rep;
the first line is the prefill). Step s attends keys `p+1 = s+9` (input 8 +
greedy tail; positions 0-based). Windows below use steps 10–20 (early,
p ≈ 19–29) and 185–198 (late, p ≈ 194–207), medians.

## Attribution verdict

**The growth is the per-key term `keys = p+1` of an UNCLIPPED window (p ≤ 207
< 513, so `jmin = 0` every measured step — the window bound never binds in the
measured range) executed ENTIRELY ON ONE THREAD. H3 confirmed (the amplifier
and the fixable share). H1 refuted. H2 refuted in the measured range. The
unclipped-window growth itself (named fourth) is correct kernel behavior and
plateaus past seq 513; it is not a defect.**

### The 16t baseline curve (3 reps, pre-fix binary)

`attn_sliding` is LINEAR in the key count, three reps within ±0.15%:

| rep | fit (ms, steps 12–198) | early med | late med | ratio |
|---|---|---:|---:|---:|
| 1 | 39.34 + 0.2634·step | 43.19 | 89.27 | 2.067 |
| 2 | 40.44 + 0.2633·step | 43.72 | 90.15 | 2.062 |
| 3 | 40.97 + 0.2630·step | 44.47 | 91.20 | 2.051 |

`moe_routed` over the same windows: 0.974 / 1.006 / 1.005 — flat, replicating
the spec's 0.994.

### H1 (KV-append / table maintenance inside the timer) — REFUTED

- Structurally: the scope (`flint.cpp:1063-1221`) wraps qkv/o_proj GEMVs, head
  norms, RoPE, `WriteKvCache`, the slot-mapping derivation, `PagedAttention`,
  and the residual add. The append (`WriteKvCache`) and the slot map are
  O(query tokens) = O(1) at serving batch 1; nothing in the scope outside the
  paged kernel reads the sequence length.
- Numerically: the seq-independent content IS the intercept, and it is
  constant (39.34 / 40.44 / 40.97 ms across reps, and `attn_full`'s intercept
  likewise 13.23 / 13.59 / 13.56). A growing maintenance term inside the
  timer would bend the intercept, not the slope. No timer-scope change is
  needed.

### H2 (arena-scatter cache misses at constant key count) — REFUTED (measured range)

- The premise does not hold: the key count is NOT constant — the window
  (left = 512) never clips below p = 513, so keys grow 17 → 208 across the
  measured run.
- The slope is LINEAR in the key count to ±0.15% across reps while the pages
  a gather touches grow from 2 to 14 blocks: a scatter term growing with the
  arena would curve the slope upward. It does not. (H2 can only become live
  past seq 513, where the window binds and the key count flattens; nothing in
  this row's evidence reaches that regime.)

### H3 (batch-1 serial attention) — CONFIRMED, with the fourth named

- The whole per-key term runs on one thread: `ParallelForRows` splits by
  TOKEN rows (`cpu_paged_attn.cpp`); at concurrency 1 there is exactly one
  row, so 32 heads × p keys × 128 dims execute serially while the pool idles.
- Cross-check that isolates the term from any per-layer fixed cost: the FULL
  attention layers (14, causal, also keys = p+1) grow with slope
  0.1051 / 0.1061 / 0.1061 ms/step. The sliding slope / full slope =
  2.506 / 2.483 / 2.478 ≈ 35/14 = 2.50 — per-layer per-key cost is the same,
  and the entire growth is the key term. (Sliding layers additionally pay
  RoPE, reflected in the intercept ratio 39.3/13.2 ≈ 2.96 > 2.5.)
- Per-key serial cost: slope 0.2633 ms/step over 35 layers = 7.5 µs per key
  per layer = 32 heads × 128 dims × 2 passes = 8192 MAC → ~1.1 GFLOP/s —
  single-core scalar-scale throughput. The 1t control leg (`ctl-1t-rep1.log`,
  pre-fix binary, 1 pool thread, cpu 40): fit 158.63 + 0.2574·step — the
  SLOPE is within 2% of 16t's 0.2630 (the per-key kernel term is identical at
  1 and 16 threads: serial), while the intercept is 4x larger (158.6 vs
  ~40 ms): the seq-independent GEMV content DOES parallelize, the kernel does
  not. 16 threads buy nothing on the key term at batch 1.

### The named fourth: unclipped-window key growth (correct behavior)

The window left bound is 512 and the measured positions stop at 207, so
`jmin = 0` throughout: per-token work is O(p+1) by design until p exceeds
513, where it plateaus at 513 keys. The 1.94x–2.06x measured growth is this
legitimate term amplified 16x by H3's serial execution. No kernel-math change
(spec non-goal); H3's split removes the amplification.

## The fix (H3)

`src/vt/cpu/cpu_paged_attn.cpp`: the parallel axis moves from TOKEN rows to
`(token, head)` units — `ParallelForRows(CurrentThreadpool(), total_q * hq,
…)` with unit u = (u/hq, u%hq). Per-head scratch (`probs`, `acc`, one head's
widened query row) per unit; every unit walks jmin..jmax in the SAME order
with the same per-element math, so each output row is produced by one thread
exactly as the serial code produced it — **bit-identical by construction**
(the parallel axis crosses heads, never a head's key loop; the MiMoV2
sink-seeding path is untouched — its gates below).

Gates (all on the fixed head, build /tmp/build-sliding, CPUs 64-127):

| gate | result |
|---|---|
| `test_flint_paged` | 208/208 assertions PASS |
| `test_ops_paged_attn_dtype` (bit-identity) | 172 assertions, 3 cases PASS |
| `test_flint_gguf` keep-quant serving gate | PASS (184 assertions, 4 cases; 0 hard flips) |
| MiMoV2 sink gates (`test_mimov2_forward` 178, `test_mimov2_w1` 51, `test_mimov2_w2` 67) | ALL PASS |

## Before / after

(16t = taskset -c 32-47, 32t = -c 32-63, 3-rep medians. After-legs gated per
rep on 1-min load < 6 and no >80%-cpu foreign process inside the zone; one
single-cpu foreign workload (`test_flint_tt_w`, the TT-ANCHORING claim's test
loop) lives in the zone and migrates across cpus — documented, ~1 of 16-32
cpus. Earlier post triples whose early-step median betrayed a burst are
quarantined under the `contaminated-` filename prefix (a nested subdir would fail check-pr-size's per-run evidence pattern). The early-step median is the contention
canary: the fix leaves the intercept untouched, so a clean rep reproduces it.)

| metric | before (base-16-rep{1,2,3}) | after (post-16-rep{1,2,3}) |
|---|---:|---:|
| attn_sliding early med, ms | 43.19 / 43.72 / 44.47 | 38.99 / 39.00 / 38.11 |
| attn_sliding late med, ms | 89.27 / 90.15 / 91.20 | 43.29 / 43.90 / 42.70 |
| late/early growth ratio | 2.067 / 2.062 / 2.051 | 1.110 / 1.126 / 1.120 |
| per-key slope, ms/step | 0.2634 / 0.2633 / 0.2630 | 0.0269 / 0.0284 / 0.0279 |
| decode tok/s @16t | 5.50 / 5.48 / 5.29 (med 5.48) | 7.01 / 7.01 / 6.87 (med 7.01) |
| attn_sliding early med (32t), ms | 41.05 / 42.25 / 43.46 | 35.16 / 36.02 / 35.97 |
| attn_sliding late med (32t), ms | 86.20 / 87.47 / 89.88 | 39.59 / 39.52 / 40.37 |
| late/early growth ratio (32t) | 2.100 / 2.070 / 2.068 | 1.126 / 1.097 / 1.122 |
| decode tok/s @32t | 4.91 / 5.61 / 4.90 (med 4.91; prior clean record 5.60 — see note) | 6.98 / 6.83 / 6.42 (med 6.83) |

32t note: the baseline 32t triple ran on the pre-fix binary
(`/tmp/build-ttnr/examples/vllm-bench`, which carries the landed timers but
not this row's fix) under today's bursty sharing; its decode rates are
depressed by post-leg load bursts (18-21 one-minute load at leg end) while its
SLIDING medians are internally consistent (early 41-43, late 86-90, ratio
2.07-2.10 — the same curve as 16t). The prior row's clean same-recipe 32t
record (5.60, `docs/bench-evidence/tt-nonrouted-attribution-20261031/`) is the
calmer before-reference; after (6.83 median) clears both. The phase numbers,
not the wall rates, are this row's primary evidence: they are within-run
quantities and robust to the bursts.

32t post baseline check: the sliding intercept DROPS 41-43 -> 35-36 at 32
threads (the seq-independent GEMV content uses the wider pool), consistent
with the 1t control's intercept behavior; the key term's late/early ratio
collapses 2.07 -> 1.10-1.13 both widths.

The intercept is unchanged (39-41 before, 38-39 after): the fix moved only the
key term. The key term drops ~9.5x (32 heads over 16 threads; the ideal split
is 16x — the residual is pool-kick overhead per 35 layer calls and the 2
heads-per-thread quantization). The remaining growth (late/early 1.12) is the
same unclipped-window key term at its parallel floor; it plateaus at seq 513
by design.

## Raw logs

- `base-16-rep{1,2,3}.log` — baseline 16t (pre-fix binary, clean).
- `base-32-rep{1,2,3}.log` — baseline 32t, CONTAMINATED (foreign session on
  cpu 34); kept for the record, not used.
- `base-32-clean-rep{1,2,3}.log` — baseline 32t on the pre-fix binary, clean
  (canary-gated; burst tries quarantined under the `contaminated-` filename prefix (a nested subdir would fail check-pr-size's per-run evidence pattern)).
- `ctl-1t-rep1.log` — 1-thread control leg (pre-fix binary
  `/tmp/build-ttnr/examples/vllm-bench`, which carries the landed timers but
  not this row's fix).
- `post-*-rep*.log` — after legs (this row's fix).
- `host-state.txt`.

## Preflight / record

`agent-preflight.sh` on this host: green except the KNOWN host-specific
failure `check-windows-portability.py` (dotnet is not installed on this Linux
box; the injected fake-tool contract cannot run — the named-host exception).
`check-agent-record` OK. Commit style + trailer gates green on
`origin/main..HEAD`.

## Governor

performance (as found, unchanged). Idle gate per rep: 1-min load < 6 (16t/32t
after-legs; < 16 for the first baseline triple) plus a no->80%-cpu-foreign-in
-zone check; early-step-median canary on every kept leg.

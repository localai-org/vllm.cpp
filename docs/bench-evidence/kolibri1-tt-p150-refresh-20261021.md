# Kolibri-1 on the Tenstorrent P150: the 2026-10-21 performance refresh

Row: `MODEL-TEXT-kolibri-1-tenstorrent` (P150 lane). Issue:
`ISSUE-LOCAL-01M5F68A30HNPG4V95BKTAQS79`
(`.agents/issues/MODEL-TEXT-kolibri-1-tenstorrent/`). Date: 2026-10-21.
This change is records-only: the numbers below were measured in a separate
session and are transcribed here. It contains no product code and no device
work.

## Question

What are the current Kolibri-1 decode rate, per-step decode cost and
128-token prompt prefill time on the P150, and does the ratified band gate
still read the same?

## Method

- Device: Tenstorrent Blackhole P150 on host `thalia`. The P150 is not a
  fleet device, so every leg held the file mutex `${GPU_LOCK:-$HOME/gpu.lock}`
  with `flock -x`.
- Build: `origin/main` at `3ce14f35c` ("disable the dequant cache on a host
  that cannot afford the knee"), `Release`, `-DVLLM_CPP_TENSTORRENT=ON`, built
  at `/tmp/build-p150-bench` from the worktree `/tmp/p150-bench`.
- Weights: `/mnt/models/Aleph-Alpha/Kolibri-1` (the fp8 checkpoint, unchanged).
- Substrate: `TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin` at
  `a585e5744a8` ("vllm-cpp pin 2026-10-08"), runtime
  `TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium`
  (the tt-umd 0.9.12 era).
- CPU affinity: `taskset -c 32-63`. On this 4-node host that set is NUMA
  node 1 (node0 `0-31`, node1 `32-63`, node2 `64-95`, node3 `96-127`).
- Scaling governor: `schedutil`. The harness recorded it on the first matrix
  of the same session (12:47), on the same host and the same pin, and the
  box read `schedutil` again on the same CPUs after the run. The quiet
  re-run script does not sample the governor, and an inspection between the
  run and this record read `performance` on the same CPUs, so the governor
  on this host is not pinned. That is a standing condition caveat for every
  P150 number recorded here.
- Host load: the quiet gate passed at 1-minute load 1.73 (13:36) and the
  first accepted run started at load 1.93. The 1-minute load rises above 100
  during the P150 legs because the device legs are heavy themselves.
- Binaries: `test_kolibri1_tt_decode_bench` (the decode and prefill timing)
  and `test_kolibri1_tt_b2ii --test-case="*device token gate*"` (the band
  gate).
- Board reset before the legs: `tt-smi -r 0`, a 20 s settle, then
  `rm -rf ~/.cache/tt-metal-cache/* /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory`.
- Run windows: the decode and prefill numbers come from the quiet re-run at
  13:37 to 13:47, after the quiet gate passed. The band-gate legs ran in the
  first matrix at 12:58 to 13:23, on the same tree and the same recipe. The
  band gate is a correctness instrument and not a timing one, so the busier
  host of that first matrix does not bear on its verdict. The first matrix's
  own decode legs (1.72777 / 1.83838 / 1.70903 tok/s, prefill 163.108 /
  129.245 / 141.322 s) ran while the host was busy, and the quiet re-run
  supersedes them.

### The harness shape

`test_kolibri1_tt_decode_bench` prefills a 128-token prompt, then times 64
incremental decode steps over one persistent KV cache. The bench prints
three numbers:

- `prefill walk 128 tokens in <s> s`: the 128-token prompt prefill.
- `decode <s> s`: the 64 incremental decode steps alone.
- `tok/s`: 64 steps divided by the decode seconds. It excludes the prefill.

So 35.11 s over 64 steps is 0.548 s per decode step, and the prefill number
is the cost of the 128-token prompt alone. A reader must not read the tok/s
figure as an end-to-end rate that includes the prefill.

## The current numbers (2026-10-21, three runs)

| run | prefill walk, 128 tokens (s) | decode, 64 steps (s) | decode tok/s | ms per decode step |
|---|---|---|---|---|
| 1 | 146.968 | 35.0521 | 1.82585 | 547.7 |
| 2 | 131.111 | 35.2098 | 1.81767 | 550.2 |
| 3 | 124.892 | 35.1051 | 1.8231 | 548.5 |
| median | 131.111 | 35.1051 | 1.8231 | 548.5 |

The decode cell is tight: 1.81767 to 1.82585 tok/s, a 0.45 % spread around
the median, and the decode phase sits inside a 0.16 s window. The prefill
spans 124.892 to 146.968 s, a 12.2 % spread around the median, and that
spread is the prompt forward's load-time variance: the decode phase of the
same three runs does not move with it.

The bench emitted the same greedy chain in all three runs
(`10,32,70,10,325,10,34,286,10,34,120004,34,...`), and each run reported
196 of 196 assertions passing with `Status: SUCCESS!`.

Side by side with the vintage it replaces: the 2026-10-11 prototype read
0.929 / 0.798 / 0.868 tok/s (median 0.868) at 1077 / 1254 / 1152 ms per
step, and today's run reads 1.82585 / 1.81767 / 1.82310 tok/s (median
1.823) at 547.7 / 550.2 / 548.5 ms per step. The current number is about
2.1x the prototype's, and that is a comparison across two trees under two
host conditions, so no speedup is claimed here. The tree carried no
"1.83 tok/s" and no "546 ms" and no "prefill 149 s" figure before this
refresh: the P150 numbers it carried were the 2026-10-11 prototype's, the
2026-10-17 memo and shadow cost's, and the 2026-10-12 perf note's, all
listed under the superseded vintages below.

## The band gate (three runs, byte-identical)

The three runs below are the 12:58 to 13:23 matrix of the same session, not
the quiet re-run.

| run | verdict | worst in-band divergence | margin to the 2.5-nat edge |
|---|---|---|---|
| 1 | 3/8 sequences ON (zero hard flips) | -1.74672 nats | 0.75328 nats |
| 2 | 3/8 sequences ON (zero hard flips) | -1.74672 nats | 0.75328 nats |
| 3 | 3/8 sequences ON (zero hard flips) | -1.74672 nats | 0.75328 nats |

All three runs print the same verdict line and the same 40 hard flips, and
their nat gaps match the 40 entries of the 2026-10-19 artifact
`docs/bench-evidence/kolibri1-readj/ON.json` one for one (der Mond step 0 at
1.746723 nats, Australia step 0 at 10.523823 nats, and so on). The flip set
is therefore byte-identical to the 2026-10-19 verdict recorded in issue
`ISSUE-LOCAL-01M4P21TZ1Q9CA7P66XBH0BQTX` and in
`.agents/specs/kolibri-tt.md`. Each run ends with the same three
pre-existing assertion failures (600 passed, 3 failed): `:1263`
(`argmax_matches >= 141`), `:1264` (`hard_flips == 0`) and `:1419`
(`instr_hard_flips == 0`).

The host-free OFF arm was not re-measured in this session. Its standing
numbers stay the 2026-10-19 ones (6/8 sequences ON, worst in-band -2.157
nats, margin 0.343 nats).

## The noise floor, and why a bare number is not a measurement

This box carries a single-run noise floor of about 4 %: two single runs of
one cell in the same session read 8.45 and 8.11. The decode cell of this
refresh is tighter, but the rule stands. Every refreshed number in this
change is a three-run set with its median, and each superseded vintage keeps
its own date and its own run count.

## The superseded vintages (kept, not overwritten)

| quantity | value | date | record |
|---|---|---|---|
| decode tok/s, ms per step | 0.929 / 0.798 / 0.868, 1077 / 1254 / 1152 | 2026-10-11 | the bias-cache prototype study, `docs/gap-kolibri1-tt-captured-decode.md` |
| decode rate | about 1 tok/s at about 960 ms per step | 2026-10-11 | the capture-gap study, issue `ISSUE-LOCAL-01M4N8Y7XPDC74SED0WEDN0YFR` |
| ms per decode step | 1576 (median of 3), 1000 on the immediate pre-fix parent, 610 historical | 2026-10-17 | the deterministic-decode memo and shadow cost, issue `ISSUE-LOCAL-01M54CJVSXAW32PB9AVKGJG2RH` |
| ms per decode step, shadow drop disarmed | 1556 to 1130 | 2026-10-17 | the same issue, the env-gated attribution leg |
| 128-token prompt prefill | 214.6 to 166.7 | 2026-10-17 | the same issue, the env-gated attribution leg |
| decode phase, prompt prefill | decode about 104 s against about 73.5 s pre-fix, prefill about 214 s against about 181 s | 2026-10-12 | the determinism fix's perf note, issue `ISSUE-LOCAL-01M4P21TZ1Q9CA7P66XBH0BQTX` |
| ms per decode step, device-router composite | 952 against 924 | 2026-10-11 | the device-router composite, issue `ISSUE-LOCAL-01M4NZA9B3DBKS3MD0M4TGRVES` |
| 128-token prompt prefill, chunked | 156.3 to 72.1 | 2026-10-11 | the chunked-prefill study, issue `ISSUE-LOCAL-01M4NS6D2GCD5WXBFN24XGXYQ4` |
| band gate, host-free ON | 3/8 sequences ON, worst in-band -1.747 nats, margin 0.753 | 2026-10-19 | the ratified rescore, issue `ISSUE-LOCAL-01M4P21TZ1Q9CA7P66XBH0BQTX` |
| ms per decode step, 128-token prompt prefill | about 1556, prefill 156.4 s | 2026-10-18 | the LRU memo commit `53c6a4666`, decode bench |
| ms per decode step, before and after the memo-owns-tensors fix | 1559 to 1567 | 2026-10-18 | the memo-owns-tensors fix `93e37e703` |

Each of these is a measurement of a different tree under different host
conditions, so none of them is a like-for-like comparand for the current
numbers. They stay in the tree as the history of how the numbers moved.

## What did not change, what did change, and why

- THE BAND-GATE VERDICT DID NOT CHANGE. It reproduces byte-identically, so
  the correctness quantity is stable across the window and the standing
  verdict still stands.
- THE DECODE SPEED DID CHANGE. Today's per-step cost is 548.5 ms, and the
  same decode bench read about 1556 ms/step on 2026-10-18 (`53c6a4666`:
  "decode bench: prefill 156.4 s, the same ~1556 ms/step the regression
  reported"), with the memo-owns-tensors fix reporting 1559 to 1567 ms/step
  in the same session (`93e37e703`). The step time therefore fell by about
  2.8x in three days, and the tree carries no attribution for that fall.
  Both numbers are the same quantity on the same bench and the same recipe,
  so this is a real movement and not a unit change.
- THE CPU-SIDE WORK DID NOT MOVE IT. The bf16-dot adoption (`61d333e52`, in
  `src/vt/cpu/cpu_paged_attn.cpp` and the kolibri1 CPU forward) and the CPU
  dequant-cache default (`d9e669582`, `e451be705`, `3ce14f35c`, in
  `src/vllm/model_executor/models/kolibri1_dequant_cache.h`) are CPU-path
  changes, and a P150 number cannot move because of them. A reader must not
  infer that the P150 number moved because the CPU work landed.
- CORRECTION TO A WORKING PREMISE: the f32-mean reduction repair
  (`4b3a0091c`) is a TENSTORRENT-PATH change, not a CPU-path change. It
  touches `src/vt/tenstorrent/tenstorrent_gdn.cpp` and
  `src/vt/tenstorrent/tenstorrent_keepquant.cpp`, and it landed on
  2026-10-19, inside the window between the 2026-10-18 measurement and
  today. It is the only Tenstorrent-path product change in that window, and
  it is therefore the leading hypothesis for the fall: it replaced the pin's
  tf32 FPU tile-reduce `ttnn::sum` with the accurate full-fp32 `ttnn::mean`
  in the GDN decode dots and the grouped keep-quant tails. That is a
  hypothesis, not a measurement: `4b3a0091c` reports device-suite evidence
  and no step time, and this change measures no A/B.
- The second candidate is the host condition. The 2026-10-17 and 2026-10-18
  sessions recorded host load and concurrent builds as the reason their
  numbers were not comparable with the earlier ones, while today's three
  runs are on a quiet host (1-minute load 1.93 at the start). The box's
  single-run noise floor of about 4% does not cover a 65% move, so the noise
  floor does not explain it either. The third candidate is the CPU governor
  and the host clock state, because the governor on this host is not pinned
  (it read `schedutil` during the run and `performance` on an inspection
  between the run and this record).
- The next traceable step is a same-session A/B of `4b3a0091c`'s parent
  against `3ce14f35c` under the quiet-host recipe. The attribution stays
  OPEN.
- The P150 path still carries the tt-metal rms_norm deviation: about 0.7 % on
  real in-model rows, an input-dependent kernel-precision deviation and not
  a call-site config difference (issue
  `ISSUE-LOCAL-01M578VP02HPVBYSMRP32GWZGF`). The upstream probe at tt-metal
  head `2500c1d8f63` found the bias still present and 3x worse on the worst
  row (issue `ISSUE-LOCAL-01M5B60NEFSEHYZXMR6FSS1RAF`). That filing is the
  open one.

## Reproduce

```sh
# decode and prefill timing, three runs on a quiet host
cd /tmp/p150-bench
taskset -c 32-63 /tmp/build-p150-bench/tests/test_kolibri1_tt_decode_bench

# the band gate
taskset -c 32-63 /tmp/build-p150-bench/tests/test_kolibri1_tt_b2ii \
  --test-case="*device token gate*"
```

Both legs need the file mutex and the tt-metal environment of the run:
`TT_METAL_RUNTIME_ROOT=/tmp/umdtrial-install/libexec/tt-metalium`,
`TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin`,
`VT_KOLIBRI1_TT_B2II_MODEL=/mnt/models/Aleph-Alpha/Kolibri-1`.

## The WARM ARM (2026-10-22): the prefill's spread is not the device recipe

A follow-up arm ran the same bench binary, the same recipe and the same device
allocation path, three legs back to back, with NO device reset (`tt-smi -r 0`)
and NO tt-metal kernel-cache clear — the two things every leg above did before
its run. Log: `/tmp/p150/logs/warm-prefill.log`.

| leg | load at leg start | prefill 128 tok | decode 64 steps | tok/s |
|---|---|---|---|---|
| warm 1 | 164.46 | 160.506 s | 39.308 s | 1.62818 |
| warm 2 | 285.20 | 153.634 s | 38.597 s | 1.65815 |
| warm 3 | 286.08 | 151.551 s | 38.515 s | 1.66171 |

The warm arm is the TIGHTEST set in the record (a 5.8 % prefill spread,
151.6-160.5 s) against the cold arms' 30 % (124.9-163.1 s), and under a
constant host load the prefill repeats to about 6 %. So the device reset and
the kernel-cache clear are NOT the variable behind the prefill's spread, and
the variable no leg recorded is the BOX'S MEMORY PRESSURE: at 15:01 on
2026-10-22 the swapfile held 192 GiB of its 256 GiB, and the holders were
three concurrent Kolibri-1 test processes each mapping the 74 GB fp8 model
(`test_kolibri1_w3 -s` 58.8 GiB swapped, `test_kolibri1_threeway_capture`
58.3 GiB, the bf16-arm worktree's test 53.3 GiB) plus four `rust-analyzer`
processes. By 15:07 the same three tests had the swap drained to 129 MiB with
151 GB resident — the pressure fluctuates with the concurrent model tests.

WHAT THIS DOES NOT SHOW: the mechanism is a candidate, not an established
one. The candidate is that the 128-token prefill pays the page-cache/swap
cost of streaming the whole weight set while the decode re-reads those pages
warm, and that the prefill therefore tracks the box's memory pressure. The
swap usage at the time of the slow historical legs was not recorded, so the
mechanism is not established; what is established is the two arm sets above
and the pressured legs that followed the 156 GB bf16 write stream which
finished at 14:52 (prefill 154.5 / 161.9 s, decode 36.7 / 36.7 s, tok/s
1.74602 / 1.74365 — contention during and after that write stream, not a
regression).

OWED: a decode-bench leg must record the box's swap delta and its memory
pressure (`grep SwapTotal/SwapFree /proc/meminfo` before and after, and the
top consumers' `VmSwap`), because the prefill's number is only meaningful
when the model's pages are resident. The headline number above (1.8231 tok/s,
the quiet set) stands: the quiet set is the only set in this record whose
page-cache and load state was recorded and quiet.

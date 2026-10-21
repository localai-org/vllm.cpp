ID: ISSUE-LOCAL-01M54CJVSXAW32PB9AVKGJG2RH
Title: kolibri1 P150: the deterministic-decode memo/shadow fixes cost ~1.6x step time — the shadow drop re-stages evicted expert weights
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-17
Updated: 2026-10-21
Closed: -

## Problem

ad8d9b45b (memo eviction order + weight-view shadow drops at UnregisterHostBuffer/MarkScratchAcquired) fixed the run-to-run nondeterminism but read 1576 ms/step (median of 3, taskset -c 32-63, tt-smi reset recipe) vs 1000 ms/step on the immediate pre-fix parent under the identical recipe; the historical 610 ms/step eager-lever median was NOT reproducible today (host load, concurrent builds). Direct attribution (env-gated shadow-drop no-op measurement leg): prefill 214.6->166.7 s, decode 1556->1130 ms/step with the drop disarmed — the drop forces full re-staging of the 7.86 MB expert dequant views after every memo eviction; the residual gap to pre-fix is the eviction-order change plus host noise. PRICED FIX: make the memo entry own its staged device tensors (stage the three views at dequant time into the SlotDequant, serve from there) so eviction drops them exactly once and the pointer-keyed shadow map never caches transient-buffer views; UnregisterHostBuffer then needs no range scan on the pool re-hand path.

## Resolution

2026-10-21: STILL TRUE, with the attribution changed. The PRICED FIX this issue
names landed as `93e37e703` (PR #3468): the memo entry now owns its staged
device tensors (`SlotDequant::staged`,
`src/vllm/model_executor/models/kolibri1_tt_forward.h:172-180`, staged at
`kolibri1_tt_forward.cpp:882-892`), the shadow sweeps are off the decode hot
path (172,800 owned serves, 142,740 owned stages, 0 sweep events), and the
pointer-keyed shadow map no longer caches views of the transient buffer. The
mechanism this issue names -- the shadow drop re-staging evicted expert weights
-- no longer exists.

The cost, however, persists and is re-attributed: step time is unchanged (1559
-> 1567 ms/step) because the memo thrashes. The cap of 512 entries against about
300 distinct experts per forward gives a hit rate near zero, so every touch
re-uploads 7.86 MB H2D. The pre-#3452 1000 ms/step was fast because the stale
pointer-keyed cache skipped that upload while serving the previous tenant's
weights, which is the bug #3452 closed.

What settles it: one of the three levers the landing commit names -- a
content-keyed expert staging cache, a device-side fp8->bf16 dequant, or raising
the cap -- followed by a step-time measurement. The residual has no open owner:
the memo-bound record `ISSUE-LOCAL-01M4MJG2CKSM1FK53PZ44QFN1J` is CLOSED.

- CURRENT PER-STEP COST (2026-10-21, `row/P150-NUMBERS-REFRESH`, records-only):
  the per-step decode cost on the P150 is 0.548 s per token, which is
  547.7 / 550.2 / 548.5 ms per decode step over three runs (median 548.5 ms),
  measured on `origin/main` `3ce14f35c` (Release,
  `-DVLLM_CPP_TENSTORRENT=ON`, built at `/tmp/build-p150-bench`; host
  `thalia`, P150 Blackhole; `taskset -c 32-63`; governor `schedutil`;
  tt-metal substrate pin `a585e5744a8`; file mutex `$HOME/gpu.lock`; model
  `/mnt/models/Aleph-Alpha/Kolibri-1`). The same three runs read decode
  1.82585 / 1.81767 / 1.82310 tok/s (median 1.823), a decode phase of
  35.05 / 35.21 / 35.11 s (median 35.11 s), and a 128-token prompt prefill of
  146.97 / 131.11 / 124.89 s (median 131.1 s).

  WARM ARM (2026-10-22, `row/TT-DEVICE-VS-CPU-LAYER0` follow-up,
  `/tmp/p150/logs/warm-prefill.log`): the same bench binary, the same recipe
  and the same device allocation path, three legs back to back, with NO device
  reset and NO tt-metal kernel-cache clear. The three legs read prefill
  160.506 / 153.634 / 151.551 s, decode 39.308 / 38.597 / 38.515 s, and
  1.62818 / 1.65815 / 1.66171 tok/s, all at load 164-286.

  WHAT THE WARM ARM SETTLES: the device reset and the kernel-cache clear are
  NOT the variable behind the prefill's spread. The warm arm had neither and
  was the TIGHTEST set in the table (a 5.8 % prefill spread, 151.6-160.5 s)
  against the cold arms' 30 % (124.9-163.1 s), and under a constant host load
  the prefill repeats to about 6 %. Both phases track the host, and the
  prefill tracks it harder (the decode's spread across every arm is
  34.8-39.3 s, 12 %). The variable no leg recorded is the BOX'S MEMORY
  PRESSURE: measured on 2026-10-22 15:01 the swapfile held 192 GiB of its
  256 GiB, and the holders were three concurrent Kolibri-1 test processes
  (`test_kolibri1_w3 -s` at 58.8 GiB swapped, `test_kolibri1_threeway_capture`
  at 58.3 GiB, the bf16-arm worktree's test at 53.3 GiB) each mapping the
  74 GB fp8 model, plus four `rust-analyzer` processes; by 15:07 the same
  three tests had the swap drained to 129 MiB with 151 GB resident. The
  128-token prefill is the phase that first streams the whole weight set; the
  decode then re-reads those pages warm. The candidate mechanism is therefore
  stated as a CANDIDATE: the prefill pays the page-cache/swap cost of the
  weight stream, and its spread tracks the box's memory pressure. The swap
  usage at the time of the slow historical legs was NOT recorded, so the
  mechanism is not established; what is established is the two arm sets above
  and the two pressured legs that followed the 156 GB bf16 write stream which
  finished at 14:52 (prefill 154.5 / 161.9 s, decode 36.7 / 36.7 s, tok/s
  1.74602 / 1.74365 — contention during and after that write stream, not a
  regression). OWED: a decode-bench leg must record the box's swap delta and
  its memory pressure (`grep SwapTotal/SwapFree /proc/meminfo` before and
  after, and the top consumers' `VmSwap`), because the prefill's number is
  only meaningful when the model's pages are resident.

  HARNESS CORRECTION: the decode bench times 64 incremental decode steps
  AFTER a 128-token prompt prefill, and its tok/s figure covers the decode
  phase alone. So 0.548 s per token is the per-step decode cost, and the
  prefill number is the cost of the 128-token prompt. A bare tok/s is not a
  measurement on this box, which carries a single-run noise floor of about
  4% (measured: two single runs of one cell read 8.45 and 8.11), so every
  number above is a three-run set with its median.

  THE SUPERSEDED VINTAGES STAY, each with its date: the 1576 ms/step median
  of 3 and the 1000 ms/step pre-fix parent and the 610 ms/step historical
  eager-lever median (2026-10-17, this issue); the 1556 -> 1130 ms/step
  attribution leg with the shadow drop disarmed and the prefill
  214.6 -> 166.7 s leg (2026-10-17, this issue); the about 1556 ms/step and
  the 156.4 s prefill the LRU commit measured (2026-10-18, `53c6a4666`);
  the 1559 -> 1567 ms/step the memo-owns-tensors fix measured (2026-10-18,
  `93e37e703`); the 0.929 / 0.798 / 0.868 tok/s at 1077 / 1254 / 1152 ms
  per step prototype runs (2026-10-11,
  `docs/gap-kolibri1-tt-captured-decode.md`); and the decode about 104 s
  against about 73.5 s pre-fix with prefill about 214 s against about 181 s
  (2026-10-12, issue ISSUE-LOCAL-01M4P21TZ1Q9CA7P66XBH0BQTX). Every one of
  them is a different tree under different host conditions, so none is a
  like-for-like comparand for the current numbers.

  THE PRICED FIX LANDED, AND ITS PERF EXPECTATION WAS FALSIFIED (recorded
  here because this issue's Resolution carried no resolution before
  2026-10-21): `93e37e703` (PR #3468, 2026-10-18) shipped the memo entry
  owning its staged device tensors, and the step time did not move (1559 to
  1567 ms/step). The falsification carries the reason: the memo thrashes,
  because the cap is 512 against about 300 distinct experts per forward,
  so the hit rate is near zero and every touch re-uploads 7.86 MB H2D. The
  LRU commit of the same day (`53c6a4666`) then measured a 100% memo hit
  rate in steady state and still the same about 1556 ms/step, so the
  eviction order is not the cause. The issue stays OPEN, and the open cause
  is the two hypotheses that commit names: the #3452
  `DropWeightViewShadowsInRange` drain and the per-fill readback pivot. The
  title's "~1.6x step time" is that 2026-10-17 and 2026-10-18 pair (about
  1556 ms/step against the 1000 ms/step pre-fix parent), and it is not the
  current state.

  WHAT TODAY'S NUMBER SAYS ABOUT THAT: today's per-step cost is 548.5 ms
  (median of three runs) on `origin/main` `3ce14f35c`, a tree that carries
  both the memo-owns-tensors fix and the LRU eviction, against about
  1556 ms/step on 2026-10-18 on the same bench and the same recipe. The
  step time therefore fell by about 2.8x after those two commits landed,
  and the tree carries no attribution for the fall. The only
  Tenstorrent-path product change after 2026-10-18 08:06 is the f32-mean
  reduction repair (`4b3a0091c`, 2026-10-19), which replaced the pin's
  tf32 FPU tile-reduce `ttnn::sum` with the accurate full-fp32
  `ttnn::mean` in the GDN decode dots and the grouped keep-quant tails;
  that is the leading hypothesis, and it is not measured here. The second
  candidate is the host condition, because the 2026-10-17 and 2026-10-18
  sessions recorded host load and concurrent builds while today's three
  runs are on a quiet host. The third candidate is the CPU governor and the
  host clock state, because the governor on this host is not pinned (it read
  `schedutil` during the run and `performance` on an inspection between the
  run and this record). The attribution stays OPEN, and the next traceable
  step is a same-session A/B of `4b3a0091c`'s parent against `3ce14f35c`
  under the quiet-host recipe.

  THE CPU-SIDE WORK DID NOT MOVE THE P150 NUMBER: the bf16-dot adoption
  (`61d333e52`) and the CPU dequant-cache default (`d9e669582`,
  `e451be705`, `3ce14f35c`) are CPU-path changes, and a P150 number cannot
  move because of them. Evidence:
  `docs/bench-evidence/kolibri1-tt-p150-refresh-20261021.md`; issue
  ISSUE-LOCAL-01M5F68A30HNPG4V95BKTAQS79.


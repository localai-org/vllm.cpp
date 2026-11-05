# QWEN38-P150-REBENCH — the qwen3.8-27B Q4_K_M P150 re-bench at current head (2026-11-05)

Row: `QWEN38-P150-REBENCH`. Head at measurement: `702b19167` (see
`head-sha.txt`). Host: thalia (aarch64, 128 cores). Substrate unchanged:
tt-metal runtime `/tmp/umdtrial-install`, `TT_METAL_HOME` = `~/Sources/tt/tt-metal-pin`.
Artifact: `/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf`
(16314.3 MB), the same file the 2026-09-14 campaign served.
Build: `/tmp/build-q38rebench`, Release, `VLLM_CPP_TENSTORRENT=ON`,
`-DCMAKE_PREFIX_PATH=/tmp/umdtrial-install/lib64/cmake;/tmp/umdtrial-install/share/cmake`,
built `taskset -c 64-127 -j 64`. `VT_TT_KEEPQUANT_INT8DOT=1` MANDATORY on
every leg (the default lm_head arm OOMs: 636 MB dequant chunk).

## Recipe (identical to the 09-14 measurement, for comparability)

```sh
flock -x $HOME/gpu.lock; luwen reset + 15 s + /dev/shm lock cleanup;
taskset -c 32-63 vllm-bench \
  --model /mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --num-prompts 2 --input-len 128 --output-len 32 --concurrency 2 \
  --num-blocks 64 --max-num-batched-tokens 64 --seed 0
```

via the production AsyncLLM path (the W3 invocation recorded in
`docs/benchmarks/tt-keepquant-27b-decode.md`). Governor method per leg
(the kolibri1 TT-ANCHORING discipline): governor logged, idle gate
(1-min load < 16), fresh device reset, 3 reps per cell, median. Cells:
`performance` r1-r3, `schedutil` r1-r3, `performance` r4 (return-pin
confirm). Governor pin and log: `governor-log.txt`, `idle-gate.txt`,
per-leg top-CPU snapshots `ps-top-*.txt`.

## Comparison table

(09-14 baseline: TTFT 1389 s, TPOT 29.7 s, 0.03 tok/s output, 0.14 tok/s
total, duration 2326 s, JIT 912/916 = 99.6%. Single un-swept leg under
an unrecorded governor.)

| Cell | Governor | Reps (TTFT s / TPOT s) | Median TTFT (s) | Median TPOT (s) | Out tok/s | Total tok/s | Median duration (s) | JIT hits | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-09-14 (single leg, governor unrecorded) | ? | 1389 / 29.7 | 1389 | 29.7 | 0.03 | 0.14 | 2326 | 912/916 (99.6%) | baseline |
| head `702b19167` r1 (COLD tt-metal kernel cache) | performance | 1283.3 / 26.85 | — | — | 0.03 | 0.15 | 2131 | 166/819 (20.3%) | cold-cache outlier, excluded from the median rationale below, kept for the record |
| head r2 | performance | 1203.6 / 25.22 | | | 0.03 | 0.16 | 1996 | 819/819 (100%) | |
| head r3 | performance | 1203.4 / 25.23 | | | 0.03 | 0.16 | 1996 | 819/819 (100%) | |
| head r4 (return-pin confirm) | performance | 1204.3 / 25.24 | | | 0.03 | 0.16 | 1997 | 819/819 (100%) | |
| **head, performance median (r1-r3)** | performance | | **1203.6** | **25.23** | **0.03** | **0.16** | **1996** | 100% warm | **−13.4% TTFT, −15.0% TPOT, −14.2% duration vs 09-14** |
| head sched r1 | schedutil | 1215.9 / 25.45 | | | 0.03 | 0.16 | 2016 | 819/819 (100%) | |
| head sched r2 | schedutil | 1215.5 / 25.46 | | | 0.03 | 0.16 | 2015 | 819/819 (100%) | |
| head sched r3 | schedutil | 1214.5 / 25.45 | **1215.5** | **25.45** | **0.03** | **0.16** | **2015** | 100% warm | +1.0% vs performance — the governor is NOT the story |

Notes:

- The 09-14 leg was a single un-swept measurement; the governor question
  the spec posed is answered the kolibri1 way: schedutil costs ~1.0% on
  this path, so the 13-15% delta is real code delta, not governor state.
- Rep r1 ran on a COLD `~/.cache/tt-metal-cache` (the first leg after
  the build): its 20.3% JIT hit rate and inflated TTFT/duration are the
  cold-cache cost, not a regression. The 09-14 99.6% figure was a
  warm-cache run; the comparable current number is 819/819 = 100%.
  Fresh `luwen` resets per leg do NOT clear the on-disk kernel cache.
- ms/step: decode = 32 output tokens per request here (the W3 recipe's
  `--output-len 32`), median decode TPOT 25.23 s/token under
  performance.

## Verdicts

1. **No regression; the intervening fixes delivered.** TTFT 1389 →
   1203.6 s (−13.4%), TPOT 29.7 → 25.2 s (−15.0%), duration 2326 →
   1996 s. The run spread inside a warm cell is ≤0.1% (r2-r4), so the
   delta is far beyond run variance.
2. **The governor sweep closes the open question**: +1.0% for
   schedutil, same class as the kolibri1 +2.8% finding.
3. **Attribution of the delta per lever** (expected vs measured — the
   bench measures the whole path, so per-lever effect is bounded by the
   total, not isolated):

## Intervening fixes and expected effect on the TT path

| Fix | Since 09-14 | Expected effect on this path | Measured |
|---|---|---|---|
| pool-match / threadpool sizing (`FLINT-KQ-CPU-PERF`, 4.6x on the CPU keep-quant arm) | host-side threadpool under taskset oversubscribes; the TT host legs run on the same pool | applies to every host-side leg | consistent with part of the −13/−15%; not isolated (whole-path bench) |
| NEON q4_K/q6_K dots (`FLINT-NEON-Q4K-DOT`) | CPU-side compute legs | applies only where dequant/dot work stays on host (keep-quant gathers, host lm_head arm — NOT the INT8DOT device lm_head this bench uses) | sub-total effect; the INT8DOT arm bypasses the host lm_head dot |
| sliding-attention decomposition (`FLINT-SLIDING-ATTN` / `FLINT-SLIDING-PAGESCAN`) | paged kernel (token,head) parallelization + page-scan hoist | applies if qwen3.8's sliding phase runs the same paged kernel | plausible contributor to the TTFT/duration cut; not isolated |
| router total order | determinism only | no throughput change expected | none observed (token counts identical: 64 generated, RC=0 every leg) |
| slot-serve cure (`FLINT-TT-DENSE-MLP-STAGING` `f4725780b`, allocation-order insensitivity) | the P150 band-gate victim class | applies at capture time; removes allocation-order sensitivity | every leg RC=0 with identical totals across 7 fresh resets — the allocation-order sensitivity is gone |

## Raw logs

`performance-r{1,2,3,4}.log`, `schedutil-r{1,2,3}.log`, `governor-log.txt`,
`idle-gate.txt`, `ps-top-*.txt`, `head-sha.txt`.

## Deviations

- One transient `luwen reset` failure (`Failed to reset chips [0]`, the
  schedutil r2 reset, 2026-11-05 18:57 in the session log); the leg ran
  anyway and landed inside its cell's 0.1% spread, so the median stands.
- Governor pin needs `sudo tee` on this host (sysfs not writable by the
  user); the found state at session start was `performance` on all 128
  CPUs, restored to `performance` at session end.

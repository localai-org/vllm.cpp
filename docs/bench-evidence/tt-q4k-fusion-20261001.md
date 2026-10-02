# the wave-3 Q4_K arm: the INT8DOT=0 27B legs serve, the fit wall is gone (2026-10-02)

Wave-3 money legs for the Q4_K arm of the fused whole-decode dispatch
(commit `640709002`, the fused kernel with the scale-index repair —
`scales[g+8]`/`scales[g+4]` for the `is >= 4` super-block groups — plus
its golden pin). Branch `row/tt-q4k-fusion`, worktree
`/tmp/vllm-region-capture-spec`, build `build2/build2-rescue2`, device
`thor:gpu0`-class Blackhole under `flock $HOME/gpu.lock`, own device
reset + 15 s before each leg, pin-first env
(`TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal-pin`,
`TT_METAL_RUNTIME_ROOT` the same,
`build_release_script/lib64` + `libexec/tt-metalium`), no other
`TT_METAL*`. One device job at a time.

Workload: Qwen3.8-27B Q4_K_M
(`/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf`),
`--num-prompts 2 --input-len 128 --output-len 32 --num-blocks 64
--max-num-batched-tokens 64 --seed 0`, greedy, `VT_TT_KEEPQUANT_INT8DOT`
selects the arm, `timeout -k 10 5400` per leg.

## 1. The legs (wave-2 baselines from
[tt-matmul-fusion-wave2-20261002.md](tt-matmul-fusion-wave2-20261002.md))

| leg | wave 2 BENCH_EXIT | wave 3 BENCH_EXIT | TPOT mean/median (ms) | TTFT mean (ms) | note |
|---|---|---|---|---|---|
| c1 INT8DOT=0 | 1 (fit wall) | **0** | **31,345.59 / 31,345.59** (P99 31,431.64) | 106,087.29 | **THE GOAL: the first BENCH_EXIT=0 INT8DOT=0 27B serve — the Q4_K fused arm clears the whole-graph fit wall that killed this leg in wave 2** (`mesh_trace.cpp:126`, DRAM high-water 4,211,219,712 B) |
| c2 INT8DOT=0 | 1 (fit wall) | **0** | 31,573.54 / 31,573.54 (P99 32,059.65) | 177,318.77 | serves; same queue-serialization shape as the INT8DOT=1 c2 leg |
| c1 INT8DOT=1 | 0 — 6,758.97 | **0** | **6,680.05 / 6,680.05** (P99 6,693.04) | 798,983.58 | regression leg: −78.92 ms vs wave 2 (−1.2%), inside run-to-run noise — **no regression** |
| c2 INT8DOT=1 | 0 — 25,684.37 | **0** | **25,434.18 / 25,434.18** (P99 38,032.32) | 1,215,318.10 | completed the 2x2 matrix; −1.0% vs wave 2, no regression |

Leg-4 detail (c2 INT8DOT=1): TPOT mean/median 25,434.18 ms (P99
38,032.32), TTFT mean 1,215,318.10 ms — −1.0% vs wave 2, the same
two-streams-on-one-replay-queue shape, no regression.

Logs `/tmp/leg-c1-i0.log`, `/tmp/leg-c2-i0.log`, `/tmp/leg-c1-i1.log`,
`/tmp/leg-c2-i1.log` (each ends with its own `BENCH_EXIT=` line, read
from that log).

## 2. Verdict

- The campaign's quantization coverage for this checkpoint is COMPLETE
  for serving: both arms, both concavities serve with full TPOT tables
  and `BENCH_EXIT=0`.
- The INT8DOT=0 f32-out arm runs ~4.7x the INT8DOT=1 TPOT (31.3 s vs
  6.7 s per token at c1) — expected, the f32-out decode plane moves
  twice the bytes through the same launch budget. Honest first numbers,
  not targets; the TPOT levers stay individually traceable.
- The INT8DOT=1 legs hold their wave-2 numbers within noise (−1.2% c1,
  −1.0% c2), so the Q4_K fused arm costs the INT8DOT=1 path nothing.

## 3. Records

Same issue (`ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ`), same row
(TT-DECODE-FUSION wave 3).

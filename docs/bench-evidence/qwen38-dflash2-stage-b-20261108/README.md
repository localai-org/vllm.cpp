# QWEN38-DFLASH2-STAGE-B — the captured verify trace serves DFlash2: the device gate (2026-11-08)

Row: `QWEN38-DFLASH2-STAGE-B`, issue
`ISSUE-LOCAL-01M6X99TTQDM83FCM2M2PFF9VB`, spec
`.agents/specs/qwen38-dflash2-stage-b.md`. Head at measurement:
`8a9ee9776` (13 commits, `9996ce94d..8a9ee9776`).

## The gate

Spec Gates 3: **greedy decode token identity, traced (defaults) vs eager,
identical workload**, batch 1, input 32, output 200, seed 0,
`--num-blocks 64 --max-num-batched-tokens 64`, `VT_TT_KEEPQUANT_INT8DOT=1`
on every leg, governor `performance` logged per leg (`*-env.txt`), fresh
`luwen` reset + 15 s + `/dev/shm` lock cleanup per leg, file mutex
`${HOME}/gpu.lock`, idle gate (1-min load < 16) before each leg, bench
zone `taskset -c 32-63`, per-leg top-CPU snapshots (`ps-top-*.txt`).
Greedy token IDs captured per rep (`*-token-ids.json`).

## Result: PASS — byte-identical token streams

All six reps produced the SAME greedy token stream:

```
d49c53d23ac5bc4180096b30ebe3fb6990ba6f6eff7d6098760685f9035019ff
  defaults-1 defaults-2 defaults-3 eager-1 eager-2 eager-3
```

Divergence would mean a wrong implementation (the accept rule is the
target's argmax) and would have stopped the row. It did not fire.

## Per-leg timing (production vllm-bench, warm cell r2-r3)

| Leg | TTFT (ms) | TPOT (ms) |
|---|---:|---:|
| defaults r1 (cold: warmup + capture) | 314521.93 | 6947.95 |
| defaults r2 | 214433.23 | 6605.36 |
| defaults r3 | 216027.19 | 6606.84 |
| **defaults median (r2-r3)** | **216027.19** | **6606.84** |
| eager r1 | 215591.87 | 6648.40 |
| eager r2 | 214216.66 | 6653.26 |
| eager r3 | 214256.92 | 6648.14 |
| **eager median** | **214256.92** | **6648.14** |

The captured arm's warm TPOT is within 0.6% of eager (6606.84 vs
6648.14 ms) — the D2D recorded-geometry fix and the GDN shadow staging
cost nothing measurable on the replay path. r1 on both arms carries the
cold weight-load; the defaults r1 additionally carries graph capture.

## Capture engagement proof

The six gate legs ran without instrumentation knobs, so they carry no
capture markers. `capture-proof.log` / `capture-proof.stderr.log` is one
short defaults leg (output-len 24, same input-len/blocks/seed, same env
plus `VT_TT_STEP_PHASES=1 VT_FLINT_PHASE_TIMING=1
VT_DECODE_GRAPH_STATS=1`) showing the dense decode driver's
`kind=host → kind=cold → "captured Qwen3.5 dense decode graph for padded
size S=1" → kind=capture → kind=replay...` sequence: the defaults arm
served through the captured trace, the eager arm
(`VT_TT_DECODE_CAPTURE=0`) through op-by-op dispatch.

## Run recipe

`run-gate.sh` in this directory (self-contained: env, reset, idle gate,
per-leg logging). Build:
`cmake -DCMAKE_PREFIX_PATH="$HOME/Sources/tt/tt-metal-pin-durable/lib64/cmake;$HOME/Sources/tt/tt-metal-pin-durable/share/cmake"
-DCMAKE_CXX_FLAGS="-isystem $HOME/Sources/tt/tt-metal-pin-durable/libexec/tt-metalium/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device"`
Release, `VLLM_CPP_TENSTORRENT=ON`. Runtime:
`LD_LIBRARY_PATH=$HOME/Sources/tt/tt-metal-pin-durable/lib64:<substrate>`,
`TT_METAL_RUNTIME_ROOT=$HOME/Sources/tt/tt-metal-pin`,
`TT_METAL_HOME=$HOME/Sources/tt/tt-metal-pin`.

## Companion host gates (same head)

- preflight `--staged` to completion: green except the two known host
  gaps (`check-windows-portability.py` missing dotnet,
  `test_check_test_registration` fetch-site-count) — both verified
  failing without this series' changes (see the -host evidence dir).
- Focused cases 5/5 green (97 assertions): the equal-numel
  recorded-geometry pair, the dtype-decline case, the interior-pointer
  decline case, the GDN spec-state shadow rollback unit, the #2294
  sibling. Mutation reds recorded in the -host dir.
- MiMoV2 sink gates: forward 178, w1 51, w2 67 — ALL PASS.

## Known open items (recorded, not waived)

- The kRmsNormGated committed-gate-shadow defect is real but PRE-EXISTING
  (fails byte-identically at the pre-branch baseline in isolation);
  filed as `ISSUE-LOCAL-01M6Y0TC5D24YGF74H0TQ2VAN5` on row
  `BACKEND-TENSTORRENT-GDN`. See the -host dir's reconciled record.
- PagedAttention pure-decode order/state dependence and the full-suite
  teardown segfault remain owned follow-ups on this row's issue.
- `VT_TT_SPEC_STATE_SHADOW` documented in docs/ENVIRONMENT.md;
  `VT_SPEC_GRAPH_MAX_QLENS` default (2) NOT moved — the spec shape
  consumes 1 of 2 slots.

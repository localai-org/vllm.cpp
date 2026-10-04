# TT-KEEPQUANT-INT8DOT-DEFAULT: the stopped spec's gates re-run on the LIVE 27B arm (2026-10-03/04, P150)

Row `BACKEND-TENSTORRENT-KEEPQUANT`, spec
[tenstorrent-keepquant-int8dot-default.md](../../../.agents/specs/tenstorrent-keepquant-int8dot-default.md),
issue `ISSUE-LOCAL-01M37W0HP6JTNJP55S74159T02`, branch `row/int8dot-default-flip`
(base `origin/main` `ade662308`). The 2026-09-23/24 gate run
([tt-int8dot-flip-gates-20260923.md](tt-int8dot-flip-gates-20260923.md))
fired on the dead engine; wave 3 landed the live capture arm
([tt-q4k-fusion-20261001.md](tt-q4k-fusion-20261001.md)), so the gates
re-ran there. Every leg: `BENCH_EXIT=0`.

## Verdict up front

**DONT-FLIP (this unit).** At c1 the arms are token-identical (2/2
prompts, 64/64 tokens). At c2 the arm-pair divergence turned out to be
WITHIN-ARM run-to-run nondeterminism, not a lever effect: a c2 INT8DOT=0
re-run reproduced the INT8DOT=1 stream byte-for-byte, so the f32-out arm's
own identical-flag runs straddle both streams of the pair. What still
blocks the flip is the spec's gate 1: the 500-mnat teacher-forced band
needs the pinned llama.cpp `b10451` batch decode, and the pin REFUSES this
artifact (`missing tensor 'blk.64.ssm_conv1d.weight'`; the
`qwen35.nextn_predict_layers` KV override is inert at the pin). No
denominator exists for the unsloth Q4_K_M on this pin, so the band cannot
hold. The sibling gate and the `=0` opt-out identity also remain unrun on
the live arm. The lever stays opt-in.

## Setup

- Tree: `row/int8dot-default-flip` at `ade662308`, fresh build
  `/tmp/vllm-int8dot-flip/build` (Release, Ninja, `-DVLLM_CPP_TENSTORRENT=ON`,
  `-DVLLM_CPP_BUILD_EXAMPLES=ON`); `nm vllm-bench | grep -c tenstorrent` = 427.
- Device: personal Blackhole P150a under `flock $HOME/gpu.lock`, one device
  job at a time, `~/Sources/tt/luwen/target/release/reset && sleep 15` before
  each leg (3 retries on failure; none needed).
- Env: `TT_METAL_HOME=TT_METAL_RUNTIME_ROOT=/home/lu_zero/Sources/tt/tt-metal-pin`,
  `LD_LIBRARY_PATH=$TT_METAL_HOME/build_release_script/lib64:$TT_METAL_HOME/build_release_script/libexec/tt-metalium`;
  `env-tt-common.sh` NOT sourced (stale tree). Run from the repo root.
- Workload: Qwen3.8-27B Q4_K_M
  (`/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 B),
  `--num-prompts 2 --input-len 128 --output-len 32 --num-blocks 64
  --max-num-batched-tokens 64 --seed 0`, greedy, c1 = `--concurrency 1`,
  c2 = `--concurrency 2`, `--output-token-ids` dumped per leg.

## Gate (a) — anchor/token gate: PASS at c1; UNDECIDABLE at c2 (within-arm nondeterminism)

Token streams (`--output-token-ids`, `/tmp/int8dot-flip/<leg>-tokens.json`):

| leg | BENCH_EXIT | TPOT mean (ms) | TTFT mean (ms) | request 0 stream (first ids) | request 1 stream |
|---|---|---:|---:|---|---|
| c1 INT8DOT=0 | 0 | 31,277.25 | 104,655.08 | `220,17,220,17,…` (220/17 alternating, all 32) | same, identical to req 0 |
| c1 INT8DOT=1 | 0 | 6,956.66 | 827,180.78 | **IDENTICAL to c1 INT8DOT=0** (byte-equal 64 ids) | identical |
| c2 INT8DOT=0 | 0 | 31,546.01 | 177,402.28 | `220,220,220,97787,96402,12305,…` (rich) | `220,17,…` (as c1) |
| c2 INT8DOT=1 | 0 | 25,389.87 | 1,215,344.73 | `220,220,220,0,198,248044,…` **diverges at token 3, cascades** | `220,17,…` identical |

- c1: the arms produce the IDENTICAL 2×32 token stream. PASS.
- c2 request 1: identical across arms. c2 request 0: the arms share the
  first 3 tokens (`220,220,220`) and diverge at generated index 3
  (97787 vs 0), with a full cascade after.

## The near-tie disposition is UNAVAILABLE for this artifact (the decisive blocker)

The spec's disposition prices a first-divergence by teacher-forcing OUR
stream through the pinned llama.cpp `b10451` batch decode and reading the
500-mnat band. For the unsloth Qwen3.8-27B-Q4_K_M artifact that oracle
cannot run at the pin:

```
llama_model_load: error loading model: missing tensor 'blk.64.ssm_conv1d.weight'
```

The artifact declares `qwen35.nextn_predict_layers` nowhere (the KV is
absent from the GGUF metadata), stores its MTP block at `blk.64` as an
attention+FFN block with NO SSM tensors, and the pin's
`src/models/qwen35.cpp` classifies layer 64 as a recurrence layer when
`n_layer_nextn == 0`, so the trunk loader demands `blk.64.ssm_conv1d.weight`.
`mp.kv_overrides` (`LLAMA_KV_OVERRIDE_TYPE_INT` `qwen35.nextn_predict_layers=1`,
verified through a `llama_log_set` callback that no
`Using metadata override` line ever printed) is INERT on this pin for this
key. `oracle-dump` therefore cannot teacher-force the 27B artifact either —
the same load path. The 2026-09-23 anchor adjudication used the
APEX-I-Nano artifact for exactly this reason; this record makes the gap
explicit for the unsloth Q4_K_M.

Without the band, the c2 divergence cannot be shown to be a near-tie
coin-flip, and the spec admits no flip on an unpriced divergence.

## A/B TPOT for the record (identical flags, one fresh device hold per leg)

| leg | INT8DOT=0 TPOT (ms) | INT8DOT=1 TPOT (ms) | ratio |
|---|---:|---:|---:|
| c1 | 31,277.25 | 6,956.66 | 4.50x |
| c2 | 31,546.01 | 25,389.87 | 1.24x |

The c1 4.50x confirms the operator's 4.7x default-path lever on the live
arm (wave-3 recorded 31,345.59 vs 6,680.05 = 4.69x; this session sits
within run-to-run noise of both). The c2 ratio is the queue-serialization
shape already recorded in wave 2/3, not a kernel difference.

## Sibling gate (b) and opt-out identity (c)

NOT RUN this window — the device budget went to the anchor gate and the
determinism re-run below. The 2026-09-23 sibling verdicts (0.8B gate stale
against the tt-metal rebase, both arms; 27B gate unreachable) predate wave
3 and are invalidated with the dead engine; they are re-owed on the live
arm.

## Determinism control — the c2 divergence is NOT lever-attributable

A c2 INT8DOT=0 re-run (`c2-i0-rep`, identical flags, fresh device hold,
`BENCH_EXIT=0`, TPOT 31,543.29 ms) reproduced **the INT8DOT=1 request-0
stream byte-for-byte** (`220,220,220,0,198,248044,2,…`, all 32 ids) — not
its own earlier stream. The INT8DOT=0 arm therefore is NOT
self-deterministic across device holds at c2: its two identical-flag runs
straddle BOTH streams of the arm pair. Consequences:

- The c2 arm-pair difference (97787-stream vs 0-stream at token 3) is
  run-to-run batched-decode nondeterminism (a near-tie coin at generated
  index 3 that flips between holds), NOT an INT8DOT kernel difference.
- The INT8DOT=1 stream is inside the INT8DOT=0 arm's own run-to-run set.
- c1 remains the clean comparison: single-stream decode, arms identical.

The token-identity evidence therefore reads: **c1 PASS (arms identical,
2/2 prompts, 64/64 ids); c2 UNDECIDABLE-BY-COMPARISON** — both arms land in
a shared stream set whose member selection varies per hold, and the
500-mnat near-tie band that would price the shared coin is unavailable
(the pin refuses the artifact, above).

## Artifacts

- `/tmp/int8dot-flip/` — per-leg `*-tokens.json` / `*-metrics.json`,
  `c2-i0-rep.log`, `gen-seq.cpp` (the prompt-replication + teacher-force
  seq writer), `load.err` (the pin's refusal).
- Monitor log (all four legs, one lock hold):
  `/home/lu_zero/.local/logs/maki/CeRgUcXiFK5bYWGSPS4sy/monitor-1791063940-f4c1/stdout.log`
  (each leg ends with its own `BENCH_EXIT=0`; `ALL_LEGS_EXIT=0`).

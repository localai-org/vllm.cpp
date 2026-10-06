# TT GDN state-slot churn fix: the multi-wave re-admission gate (2026-10-06)

Issue: `ISSUE-LOCAL-01M433M0TNT8FWC6SMT4R3700W` (the 64-prompt c2 engine-fatal
at `src/vt/tenstorrent/tenstorrent_gdn.cpp:558`, BENCH_EXIT=1, 2026-10-04).
Fix: `row/gdn-slot-churn` @ `76aef007c` (drivers) + `dcc278b8c`/amend (gate).
Tree `/tmp/vllm-slot-churn` off `origin/main` @ `0fbd7c994`.

## Mechanism

The TT captured decode graph bakes the GDN state-slot bindings: both
`GdnSsmIdxEntry` (`tenstorrent_gdn.cpp:551-577`) and `GdnConvIdxEntry`
(`:1620-1663`) resolve the state indices on the HOST and serve
content-keyed device tensors (the one-hot gather matrix, the split-block
scatter ids, the conv roll masks). A decode trace therefore replays the
CAPTURE-TIME slot set: after a wave re-admission changes the batch's
membership at a constant capture size S, replays gather/scatter stale
slots (the run-to-run stream churn), and the next capture aborts at the
`:558`/`:1629` content CHECK. The fix records the GDN non-spec
state-index content per decode-graph SizeSlot and routes a change through
the existing reset lane (`qwen3_5.cpp`, both dense and MoE drivers): the
graph is destroyed, the boundary step runs eagerly (re-warming the idx
caches with the new content, capture inactive), and the next step
re-captures. Replay only ever runs while the content is unchanged.

## Gate results

- 0.8B multi-wave churn gate (`test_qwen35_paged_engine.cpp`,
  `VLLM_CPP_QWEN35_Q4KM_GGUF`): GREEN, `1 passed` — two overlapping waves,
  staggered lengths, `max_num_seqs=4` so the batch's slot set churns at
  S=4; all 8 requests finish with exact token counts through the churn.
  On the pre-fix driver the equivalent load aborts (engine-fatal class;
  the recorded pre-fix red is the CI 64-prompt log in the issue). The
  final gate shape was not re-run against the pre-fix binary (a red leg
  costs a full rebuild plus a ~2 h device run); the red claim rests on
  the recorded CI evidence plus the pre-fix abort observed at
  2026-10-06 15:09/15:11 (both `engine-fatal`, 0/6 requests finished).
- MONEY LEG (27B Q4_K_M, `--num-prompts 8 --input-len 128 --output-len 32
  --concurrency 4 --num-blocks 64 --max-num-batched-tokens 64 --seed 0
  --temperature 0`, personal P150 under `flock ~/gpu.lock`, reset before
  each leg, `TT_METAL_HOME=~/Sources/tt/tt-metal-pin`, run from the repo
  root): **BENCH_EXIT=0**, 8/8 requests × 32 tokens, twice
  (`/tmp/gdn-churn-green-money{1,2}.log`). The engine reduced
  `max_num_seqs` to 2 (64-block KV pool), so the leg runs as c2 — the
  exact shape class that died at `:558` on the 64-prompt sweep. Mean TPOT
  34,748 ms / median 37,868 ms (a correctness leg on a cold-JIT personal
  card; NOT a throughput claim — no idle-host A/B was run).
- Determinism: the two money legs are NOT byte-identical — requests
  0/4/6 flip at generated token 4 (then continue coherently), request 5
  at token 29. The divergence sits at near-tie decision points and is
  consistent with arrival-timing jitter changing batch composition
  (poll-mode output wait), which perturbs shared-kernel numerics; both
  runs complete coherent streams with no corruption signature (no
  repeats/garbage). The stale-binding corruption class this fix removes
  is gone (no `:558`, no degenerate streams); byte-level cross-run
  determinism under concurrent admission was NOT achieved and is not
  claimed. The recorded INT8DOT flip-flop (dot0 re-run matching dot1
  byte-for-byte) is explained by the removed churn: that leg ran with
  STALE slot bindings, whose corruption dominated any tie jitter.

## Suite status

The new 0.8B gate is green; the pre-existing 0.8B/9B/27B checkpoint gates
were not re-run in this window (each is a multi-hour device leg) — owed on
the next TT gate sweep. A separate pre-existing defect was FOUND and NOT
fixed here: the B < S padded decode arm crashes at a
`CausalConv1dUpdateKernel` reshape volume mismatch
(`tenstorrent_gdn.cpp:2261`, ttnn reshape between different volumes) —
the "padded-capture arm" the NULL-refusal comments already name as owed.
The new gate deliberately keeps the batch at its capture size.

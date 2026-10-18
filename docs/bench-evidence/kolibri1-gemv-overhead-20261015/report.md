# Kolibri1 CPU decode GEMV overhead attribution (t32, NEON, 32 GiB dequant cache)

Row: `MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm`
Issue: `ISSUE-LOCAL-01M4YWC05Y4CQCVH70FSDPEKZ7` (kind gap)
Branch: `row/kolibri-gemv-overhead` (base `origin/main` @ `50d425bec`)
Date: 2026-10-15

## Question

Production kolibri decode (t32 threads + 32 GiB dequant cache + NEON paged
attention) runs ~175–266 ms/step. The linear GEMV stage was suspected of being
dispatch/sync-bound: ~3.1 Gelem/s aggregate across 32 threads while one thread
with a vbfdotq kernel reaches 10.5 Gelem/s (`/tmp/kolibri-micro/bfdot_micro.cpp`).
Hypothesis: pool kicks, per-call setup and sync waits dominate. Price the math
vs the dispatch/sync split, and price the restructures.

## Method

Scratch instrumentation, env-gated by `VT_GEMV_PROF` (default off, one static
getenv when off):

- `Threadpool::Kickoff` (`src/vt/cpu/cpu_threadpool.cpp`): counts every pool
  kick and the caller wall time inside the kick (`kicks`, `pool_ns`).
- `MatmulChunked` (`src/vt/cpu/cpu_ops.cpp`): sums per-worker time inside
  `MatmulOneChunk` only (`busy_ns`) — worker math, excluding steal/spin waits.
- `kolibri1_forward.cpp`: a RAII `gemvprof::Call` wraps every `MatmulBT` and
  attributes calls/elements/wall/pool/busy/kicks per site via snapshot deltas.
  Periodic report + reset every 16 decode steps (prefill dropped at the first
  `t > 1` forward).

Build: existing build tree `/tmp/build-gemv-overhead` (targets all green).

Run recipe (flock on `$HOME/gpu.lock` held):

```sh
env -u VT_KOLIBRI1_NUMA_INTERLEAVE -u VT_CPU_KOLIBRI_W8A8 \
    -u VT_CPU_PAGED_ATTN_BFDOT -u VT_CPU_KOLIBRI_BFDOT_GEMV \
    VLLM_CPP_CPU_THREADS=32 VT_KOLIBRI1_DEQUANT_CACHE_MB=32768 \
    VT_CPU_PAGED_ATTN_NEON=1 VT_GEMV_PROF=1 VT_KOLIBRI1_PROFILE=1 \
    /tmp/build-gemv-overhead/tests/test_kolibri1_decode_bench
```

## Attribution — per decode step (16-step steady-state window, 2 runs agreed within 3%)

| site | calls/step | Melem/step | wall ms | busy ms (per-worker equiv.) | dispatch+idle ms |
|---|---|---|---|---|---|
| attn_q | 50 | 786.4 | 18.36 | 16.56 | 1.80 |
| attn_o | 50 | 786.4 | 19.32 | 17.44 | 1.88 |
| attn_k | 50 | 65.5 | 2.15 | 1.32 | 0.83 |
| attn_v | 50 | 65.5 | 2.00 | 1.18 | 0.81 |
| router (f32) | 50 | 49.2 | 6.39 | 5.23 | 1.16 |
| routed_gate | 300 | 393.2 | 14.61 | 9.20 | 5.41 |
| routed_up | 300 | 393.2 | 14.62 | 8.70 | 5.93 |
| routed_down | 300 | 393.2 | 14.31 | 8.64 | 5.68 |
| shared_gate/up/down | 150 | 196.6 | 2.45 | 1.46 | 0.99 |
| lm_head | 1 | 327.7 | 7.18 | 7.09 | 0.09 |
| **TOTAL** | **1301** | **3457** | **105.8** | **80.2** | **25.6** |

Stage context per step (`VT_KOLIBRI1_PROFILE`, forwards 180→190 deltas):
`total_forward` 180 ms = attn_core 44 + linear_gemm 106 + moe_glue 52 +
attn_rope 10 + lm_head 8 + norms 4 (linear_gemm overlaps moe_glue; scopes nest).
Dequant-cache lookups (cold decode) only 2 ms/step; hits ≈ 1270/step,
misses ≈ 3/step, so the cache is doing its job.

## Findings

1. **The hypothesis is half right.** The GEMV stage costs 105.8 ms wall per
   step (59% of the 180 ms forward). Of that, 80.2 ms is worker math and
   **25.6 ms is dispatch/sync/idle** — ~20 µs per kick across 1301 kicks/step
   (the "900 kicks" premise was the right order). Parallel efficiency is 76%.
2. **Routed experts carry the worst dispatch overhead**: 300 kicks/step,
   17.0 ms/step dispatch+idle — 38% of their wall — because six small GEMVs
   (gate/up/down × per-expert) each pay a full pool kick for ~41 k elems/worker.
3. **The bigger gap is in the kernel, not the dispatcher.** Per-worker busy
   rate is 1.26 Gelem/s vs the 10.5 Gelem/s vbfdotq micro ideal (8.3×). The
   cause is structural: `MatmulOneChunk` (`src/vt/cpu/cpu_ops.cpp:156`, ported
   from `ggml_compute_forward_mul_mat_one_chunk`, ggml-cpu.c:1155-1243) is a
   deliberately scalar, sequential-over-K f32 accumulate — bit-exact by
   construction against the reference. No bf16 dot instruction path exists on
   this production route. If the kernel reached even half the micro rate, GEMV
   busy drops 80→~15 ms/step. That is a numerics-policy decision (the bit-exact
   constraint is documented as the whole argument at cpu_ops.cpp:74-90), not a
   restructure, and is out of scope here.
4. `dispatch+idle` for lm_head is ~0.09 ms: one big call amortizes the kick.
   This is the shape the expert batching should imitate.

## Restructure pricing

| restructure | mechanism | measured overhead removed | projected step |
|---|---|---|---|
| (a) batch 6-expert GEMVs into one pool call per layer (or one grouped call for gate/up/down across experts) | 300 routed kicks/step → ~30–90; each remaining kick is large enough to fill 32 workers (lm_head shape) | ~12–15 ms/step of the 17.0 ms routed dispatch+idle | 180 → ~166 ms |
| (b) batch attention q/k/v into one call | 2 small kicks/layer/step (k, v) merged into q's kick | ~1.2–1.6 ms/step | ~1 ms further |
| (c) eliminate per-call sync (persistent parallel region / `vt::FusedChain` across the layer's linears) | removes per-kick handoff+barrier for all 1301 kicks; overlaps (a)+(b), do not double count | total dispatch+idle 25.6 → ~5 ms/step | 180 → ~160 ms with (a) |

Combined (a)+(b)+(c): **~20 ms/step (11%)**, bringing the step to ~160 ms.
The remaining ~70 ms/step of headroom sits in finding 3 (kernel rate) and is
gated on the bit-exactness policy, not on dispatch.

## Recommended first restructure

**(a) Batch the routed-expert GEMVs into one pool call per layer** — largest
measured dispatch overhead (17 ms/step), no numerics change, and it routes
naturally through the shared seams (`layers::MlpGateUpMethodBase` /
`vt::MergedGemmGroup` for gate+up, a grouped GEMM for the per-expert set).

## Notes

- `dispatch+idle` per site is `wall − busy_summed/32`; it mixes true handoff
  cost with load imbalance and cache effects, and is an upper bound on pure
  dispatch.
- The bench's near-tie instrument reports `CHECK_EQ(hard_flips, 0)` FAIL on
  this host (HARD FLIP at position 5, neon chain `33382...`). The
  instrumentation adds counters only and cannot change arithmetic; this
  failure predates this branch and needs its own adjudication against the
  pinned oracle.
- Reported per-step times vary 175–325 ms across bench phases on this host
  (free-run neon 266 ms/step, profiled forward 180 ms/step); ratios, not
  absolute times, are the transferable result.

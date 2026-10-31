# TT-NONROUTED-ATTRIBUTION — the serving decode step, fully attributed

Row: `TT-NONROUTED-ATTRIBUTION` (measurement)
Spec: `.agents/specs/tt-nonrouted-attribution.md`
Issue: `.agents/issues/TT-NONROUTED-ATTRIBUTION/`
Date: 2026-10-31. Host: thalia (aarch64 128 cores), bench zone `taskset -c
32-47` (16t legs) / `-c 32-63` (32t legs). Governor **performance** AS FOUND
(left unchanged). Idle-gated before every leg (1-min load < 16). 3-rep medians
(tg200, greedy, single stream, `--ignore-eos`). No timeout guards. Box claim:
`~/.config/box-claims.txt` `TT-NONROUTED-ATTRIBUTION`. Artifact:
`/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`, sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`.

Recipe (per leg):

```sh
VT_FLINT_KEEP_QUANT=1 VT_FLINT_PHASE_TIMING=1 VLLM_CPP_CPU_THREADS=16 \
taskset -c 32-47 ./examples/vllm-bench \
  --model /mnt/models/flint-GGUF/flint-Q4_K_M.gguf \
  --num-prompts 1 --input-len 8 --output-len 200 --concurrency 1 \
  --seed 0 --temperature 0 --ignore-eos
```

(32t legs: `VLLM_CPP_CPU_THREADS=32`, `taskset -c 32-63`.) The instrument is
`VT_FLINT_PHASE_TIMING` (docs/ENVIRONMENT.md): scoped timers around the
production call sites of `FlintForwardPaged` / `FlintMoeBlock`
(`src/vllm/model_executor/models/flint.cpp`) plus the step-side forward /
sampler / inter-step wall clocks in
`src/vllm/v1/worker/gpu/runner.cpp`. One `[FLINT-PHASE]` + two `[FLINT-STEP]`
lines per step on stderr; the medians below pool every `batch=1` step after
the first 8 (prefill + warmup), 191 steady steps x 3 reps per leg. Read-only:
`test_flint_paged` 208/208 green at the instrumented head, and an unset env
pays one static `getenv`.

## The attribution table (medians of 3 reps, one serving decode step, batch 1)

| phase | 16t ms | 16t % | 32t ms | 32t % |
|---|---:|---:|---:|---:|
| attention — sliding layers (35) | 67.537 | 37.0% | 61.895 | 36.6% |
| attention — full layers (14, no RoPE) | 24.204 | 13.3% | 22.314 | 13.2% |
| dense MLP (layers 0-1) + residual | 1.844 | 1.0% | 1.260 | 0.7% |
| norms (input / post-attn / final RmsNorm) | 0.787 | 0.4% | 0.787 | 0.5% |
| MoE router + host top-k | 4.740 | 2.6% | 8.264 | 4.9% |
| MoE routed-expert loop | 60.313 | 33.1% | 58.063 | 34.3% |
| MoE shared expert + combine | 13.561 | 7.4% | 12.375 | 7.3% |
| lm_head (gather + untied head matmul) | 5.733 | 3.1% | 4.911 | 2.9% |
| other in forward (sink casts, slot mapping, post-MoE residual) | 2.316 | 1.3% | 2.389 | 1.4% |
| embedding | 0.009 | 0.0% | 0.009 | 0.0% |
| **SUM of phases** | **181.044** | **99.3%** | **172.267** | **101.9%** |
| step wall (median, cross-check) | 182.389 | 100% | 169.125 | 100% |
| sampler (runner-side) | 0.101 | 0.1% | 0.101 | 0.1% |
| orchestration-and-gaps (step − sample − fwd wall) | −0.544 | −0.3% | −0.335 | −0.2% |

Sum-vs-wall verification: at 16t the phases close to 99.3% of the step wall
and the residual orchestration bucket is NEGATIVE (−0.3%): there is no
unexplained gap bucket to name — the engine orchestration around the forward
is unmeasurable at this resolution (<0.5 ms/step). At 32t the phases sum to
101.9% of the step wall (+1.9% over-attribution): the per-phase medians are
taken over the pooled step population and the phase clock's own scope
overhead (~10 timers/step) plus run-to-run drift between the phase-median and
step-wall populations exceeds the residual. Both legs close the table to
within ±2% with NO large unexplained bucket.

Decode throughput medians (cross-check against the row floor): 5.46 tok/s
@16t (step wall 182.4 ms), 5.60 tok/s @32t (169.1 ms) — the same cluster as
the FLINT-KQ-CPU-PERF record (5.51 / 5.75), so the instrumented binary ran
the same serving regime.

## The finding: the "88% unattributed" premise was wrong, and so is its 12% complement

FLINT-KQ-CPU-PERF's stage-1 micro measured the routed-expert phase at
0.458 ms/layer (batch 1, matched pool) and the derived serving share was
46 layers x 0.478 ≈ 22 ms ≈ **12% of the 181 ms step**. In situ the routed
loop is **60.3 ms = 33.1%** — 2.7x the micro projection. The cache-hot
single-layer micro number does not transfer to a forward that streams 46
layers' worth of expert weights per step; this is the same
micro-wins-do-not-transfer-in-situ pattern the `maki-routed-8x8` claim
already recorded on this box. The unattributed remainder was therefore
~56% of the step, not 88% — and it now has names:

1. **Attention is the largest half of the step: 50.3% @16t (91.7 ms).** The
   35 sliding layers alone (67.5 ms) outweigh the entire routed-expert phase.
   Surprise inside the surprise: at the 12-step smoke (seq len ~20) sliding
   attention was ~42 ms/step; at seq len ~200 it is ~67 ms — the sliding
   window is supposed to CAP the attended context, so the per-step cost still
   growing with sequence length says the paged kernel's cost tracks pages
   scanned rather than window-resident keys (or the window is applied after
   the scan). Naming that precisely is the next traceable hypothesis and the
   single biggest lever on this arm.
2. **The routed-expert loop at 33% is still the second lever**, and it runs
   2.7x above its matched-pool micro floor — the NEON `vec_dot_q4_K_q8_K`
   port FLINT-KQ-CPU-PERF named stays the right lever for it.
3. **The shared expert + combine (7.4%)** is a bf16 GEMV pair at batch 1;
   it benefits from the same GEMV work as lm_head (3.1%), which measures
   5.7 ms — the maki-parity-profiling table prices the same lm_head shape at
   ~5.0 ms, so the head is already near its measured floor.
4. Router + host top-k (2.6% @16t) doubles at 32t (4.9%): the host
   `partial_sort` over 256 sigmoid scores per token is serial. Small today;
   it is the phase that scales worst with batch.
5. Norms (0.4%), dense MLP (1.0%), embedding (0.0%), sampler (0.1%): none of
   these is a lever. The dense layers and norms are noise-level; no row
   should open on them.

## Unseparable

None. Every phase boundary the deliverable names was separable with scoped
timers at existing call sites; the only fused pair is the shared expert with
`MoeCombine` inside `moe_shared` (the combine is a [t,h] bf16 add-combine,
microseconds at batch 1 — separating it needs a timer inside
`FlintMoeBlock`'s tail and prices nothing).

## Deviations

- The harness named by the brief (`tools/bench/flint_kq_moe_class_ab.cpp`)
  is not on `main` — it lives on the unmerged `row/FLINT-KQ-CPU-PERF`. This
  row instruments the production forward + runner directly instead
  (`VT_FLINT_PHASE_TIMING`), which the spec's method line explicitly allows
  ("instrument or reuse existing timers around the production forward").
  The class-ab micro's own numbers are quoted, not re-measured.
- The `moe_shared` bucket also carries the post-MoE residual add and
  `MoeCombine`; the post-dense-MLP residual rides `dense_mlp`. Both are
  single `vt::Add` calls at batch-1 scale.
- Phase percentages are quoted against the step-wall median; the sampler is
  reported separately (runner-side), so "lm_head + sampling" from the
  deliverable appears as the `lm_head` row plus the 0.1 ms sampler line.

## Raw logs

`kq-16-rep{1,2,3}.log`, `kq-32-rep{1,2,3}.log` (complete vllm-bench stderr +
stdout incl. every `[FLINT-PHASE]` / `[FLINT-STEP]` line), `host-state.txt`.

# FLINT-KQ-CPU-PERF — stage 1: the routed-expert compute-class A/B and the pool-mismatch attribution

Row: `FLINT-KQ-CPU-PERF`
Spec: `.agents/specs/flint-kq-cpu-perf.md`
Issue: `ISSUE-LOCAL-01M67BZR06KFQM7PAZ7A0X6H8X`
Date: 2026-10-30/31. Host: the shared aarch64 128-core dev box, bench zone
`taskset -c 32-47` (16t legs) / `-c 32-63` (32t legs). Governor **performance**
on all 128 CPUs AS FOUND (the prior flint-kq evidence legs were measured under
schedutil; the floor's 26.29 was also schedutil — recorded as a deviation, and
it can only favor our side). Idle-gated (1-min load < 16), 3-rep medians. Box
claim: `~/.config/box-claims.txt` `FLINT-KQ-CPU-PERF | RUNNING`.
Artifact: `/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`, sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`.
Harness: `tools/bench/flint_kq_moe_class_ab.cpp` (target
`vllm_flint_kq_moe_class_ab`; CI compiles it, never runs it). Weights come from
the PRODUCTION loader (`ModelRegistry::Load` with `VT_FLINT_KEEP_QUANT=1`), one
real MoE layer (layer 2), fixed deterministic top-6-of-256 routing shared by
every leg.

## The finding in one line

The 1.20 tok/s baseline was NOT the compute class. It was the CPU threadpool
defaulting to `hardware_concurrency()` (128) while the bench legs ran under a
16-CPU taskset — the oversubscription failure `cpu_threadpool.cpp:83-94`
already documents. With the pool sized to the cpus (`VLLM_CPP_CPU_THREADS`),
the SAME CIQ class, byte-identical tokens, serves at **5.51 tok/s @16t (5.75
@32t)** — 4.6x / 2.3x — with zero code change.

## Stage-1 table — routed-expert phase, one MoE layer, batch 1 / batch 8

(medians of 3, 16 cpus `taskset -c 32-47`; full logs `pool-sweep-16cpu-*.log`)

| class | batch 1 (ms) | batch 8 (ms) | notes |
|---|---:|---:|---|
| (a) CIQ per-expert dispatch, pool=128 (the serving default under the 16t taskset) | 8.584 | 72.146 | the baseline configuration |
| (a) CIQ per-expert dispatch, pool=16 | **0.458** | **4.394** | same kernel, same bytes |
| (b) dequant-at-load bf16 per-expert, pool=16 | 1.535 | 11.768 | +5.5 s dequant and +1.5 GiB RSS PER LAYER |
| (c) Q8_K block-dot gathers as a separate kernel | — | — | NOT REUSABLE AS DISTINCT: the existing CIQ kernel already IS this class |

Pool sweep at batch 1, 16 cpus (logs `pool-sweep-16cpu-t{4,8,16,32}.log`):
4→1.362 ms, 8→0.773 ms, 16→0.458 ms, 32→1.789 ms. The knee is pool == cpus.

### Why (c) is recorded not-reusable

`cpu_quant_traits.cpp` maps Q4_K to the Q8_K activation encoding and
`cpu_quant_gemm.cpp` quantizes the activation once per call and runs one
integer `vec_dot` per weight row — that IS mistral.rs's "quantize the
activation once, block-dot per expert row" class. There is no second kernel to
reuse, and the row brief forbids writing a new quant format. What the tree
lacks is a NEON `vec_dot_q4_K_q8_K`: the decode path (m=1) takes the SCALAR
generic port (`VecDotQ4_KQ8_K`, `cpu_quant_dot.cpp:285`) because the i8mm mmla
tier requires even m and the sdot tier is Q8_0-only. The scalar dot measures
0.231 ms per 512-row gate tower single-threaded — ~49% of the matched-pool
routed phase — and llama.cpp runs a NEON kernel at exactly this site. Porting
`ggml_vec_dot_q4_K_q8_K_neon` (mirroring the `llama-cpp` oracle) is the named
next lever, not this row's change.

### Class (b)'s memory trade, measured not estimated

One layer's experts (256 × {gate,up,down} [512/2048,·]): dequant 5.5 s,
packed 432 MB → bf16 1.6 GiB (3.7x for the expert towers). The whole tree
loads at 46.4 GiB RSS; dequantizing every MoE layer adds ~71 GiB → ~117 GiB
resident, for a class that is 3.3x SLOWER at batch 1 on this box. The trade is
refused on both axes (speed and memory); nothing to adjudicate.

## Profile split — where the routed-phase millisecond goes (pool=16, batch 1)

| component | ms per layer | share |
|---|---:|---:|
| scalar `vec_dot` inner loop (serial-rate extrapolation, 3 towers × 6 experts) | ~0.26 | ~55% |
| activation quantize (x6 rows, x3 projections) | 0.018 | 4% |
| weight-byte streaming floor (memcpy, 18 towers) | ~0.27 | — (overlapped) |
| everything else: 18 dispatch/threadpool fan-outs, gather, SiLU, scatter | ~0.19 | ~41% |

At serving level the split is different: matched-pool routed phase is
47 × 0.458 ≈ 21.5 ms/token, while serving runs 181 ms/token (5.51 tok/s) — so
~88% of serving decode time is OUTSIDE the routed-expert phase (attention
tail, shared expert, dense layers, lm_head, engine). That split is the next
attribution, unmeasured here.

## Serving (stage 3) — same recipe as the baseline plus the pool size

`vllm-bench`, tg200 shape, greedy, single stream, 3-rep medians
(`kq-t{16,32}-pool{16,32}-rep{1,2,3}.log`):

| threads | pool | tok/s (median) | reps | baseline | floor | ratio to floor |
|---:|---:|---:|---|---:|---:|---:|
| 16 | 16 | **5.51** | 5.50 / 5.58 / 5.51 | 1.20 | 26.29 | 0.21 |
| 32 | 32 | **5.75** | 5.76 / 5.75 / 5.68 | 2.46 | 26.38 | 0.22 |

Correctness: the keep-quant paged serving gate re-run at this configuration is
**byte-identical to the committed pin** — 53/60 greedy positions exact, 7
near-ties, 0 hard flips, worst gap 2.486 nats — as expected: no kernel, dtype,
or call-site changed, and the quant-dot suites assert thread-count
independence. No re-adjudication was needed because no compute-class change
landed.

## Disposition

- The row's premise ("the CIQ class is the wrong batch-1 shape") is FALSIFIED
  by measurement: CIQ beats dequant-at-load bf16 at batch 1 AND batch 8 on
  this box, and the Q8_K class is already the dispatch.
- The gap closed 1.20→5.51 with an environment/contract fix: serve with
  `VLLM_CPP_CPU_THREADS` equal to the cpu budget. The pool default remains
  `hardware_concurrency()` per its documented env contract; making the default
  affinity-aware (`sched_getaffinity`) is a product decision that owes its own
  spec and red-first gate.
- Residual gap to the floor: 4.8x. Open. Next levers named: (1) the NEON
  `vec_dot_q4_K_q8_K` port (~55% of the routed phase is scalar dots), (2) the
  88% non-routed serving split (attention tail / shared expert / engine), (3)
  the affinity-aware pool default. No ceiling declared.

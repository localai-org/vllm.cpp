# Flint on CPU, the keep-quant paged arm

The RecursiveSI Flint Q4_K_M k-quant container served PACKED on CPU through
the paged engine with `VT_FLINT_KEEP_QUANT=1`: the linear layers and
`lm_head` run the CPU compute-in-quant GEMM through the per-dtype dispatch
seam, no bf16 copy of the weights is materialized. Measured 2026-10-29/30
(baseline) and re-measured 2026-10-31 with the threadpool sized to the cpu
budget (`VLLM_CPP_CPU_THREADS`), the shared aarch64 dev box, idle-gated, 3-rep
medians, commit `row/FLINT-KQ-CPU-PERF` HEAD; container sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`.

Correctness: 53/60 greedy positions token-exact against the pinned
llama-cpp-flint oracle goldens, 0 hard flips, all divergences inside the
house 2.5-nat near-tie band
([evidence](../bench-evidence/flint-kq-cpu-20261030/report.md)); re-run
identically at the PERF configuration
([evidence](../bench-evidence/flint-kq-cpu-perf-20261030/README.md)).

Compute class: the packed CIQ dispatch is the MEASURED WINNER at decode batch
sizes — the stage-1 A/B
([flint-kq-cpu-perf-20261030](../bench-evidence/flint-kq-cpu-perf-20261030/README.md))
times it 0.478 ms/layer against 1.568 ms for dequant-at-load bf16 at batch 1
(4.405 vs 11.754 at batch 8), and the class (c) "Q8_K block-dot gathers" is
already what the CIQ kernel does. The dequant-at-load trade was refused on
both axes: 3.3x slower here and +~71 GiB RSS.

## Results

| threads | pool | keep-quant paged (tok/s, tg200 median) | floor: llama.cpp tg200 (tok/s) | ratio |
|---:|---:|---:|---:|---:|
| 16 | 16 (matched) | **5.51** | 26.29 | 0.21 |
| 32 | 32 (matched) | **5.75** | 26.38 | 0.22 |
| 16 | 128 (the 2026-10-30 default) | 1.20 | 26.29 | 0.046 |
| 32 | 128 (the 2026-10-30 default) | 2.46 | 26.38 | 0.093 |

The pool-size rows are the finding: the CPU threadpool defaults to
`hardware_concurrency()`, so a leg pinned to a 16-cpu budget ran a 128-thread
barrier — the oversubscription collapse `src/vt/cpu/cpu_threadpool.cpp:83-94`
documents. Serve with `VLLM_CPP_CPU_THREADS` equal to the cpu budget.

Recipe: `VT_FLINT_KEEP_QUANT=1 VLLM_CPP_CPU_THREADS=<n> build/examples/vllm-bench
--model /mnt/models/flint-GGUF/flint-Q4_K_M.gguf --num-prompts 1 --input-len 8
--output-len 200 --concurrency 1 --seed 0 --temperature 0 --ignore-eos`,
`taskset -c 32-47` (16t) / `-c 32-63` (32t) with the pool sized to match. The
2026-10-30 baseline legs ran under the schedutil governor; the 2026-10-31
re-measure ran under `performance` as found on the box.

## Disposition

The arm serves and is in-band. The 1.20 baseline was threadpool
oversubscription, not the compute class; with the pool matched the gap to the
floor is 4.8x and OPEN — the named next levers are the NEON
`vec_dot_q4_K_q8_K` port (the scalar generic dot is ~55% of the matched-pool
routed phase), the attribution of the ~88% of serving decode time outside the
routed-expert phase, and the affinity-aware pool default. No ceiling declared.

Update 2026-10-31: the NEON `vec_dot_q4_K_q8_K` port LANDED
(`FLINT-NEON-Q4K-DOT`, bit-identical to the scalar kernel; keep-quant token
gate unmoved at 53/60) — same-binary A/B on the loaded shared box measured
16t 5.57 -> 5.93 tok/s median (+6.5%), 32t noise-bound
([evidence](../bench-evidence/flint-neon-q4k-dot-20261031/README.md)). The
table above keeps the pre-port record; a quiet-box re-pin is owed.

Update 2026-11-04 (`FLINT-CPU-DESCENT-2`): the lm_head lever landed. The
untied `output.weight` in this container is [V, H] Q6_K, which the NEON q4_K
dot does not cover; a NEON `vec_dot_q6_K_q8_K` tier (`VT_CPU_Q6K_DOT=auto`,
bit-identical to the scalar kernel, 6370 asserts; keep-quant token gate
unmoved at 53/60, 0 hard, worst 2.48648) cut the in-situ lm_head phase 5.74 ->
4.55 ms at 16t. Same-binary interleaved A/B on a CONTENDED shared box (a peer
session benched the same zone; absolutes are not comparable to the records):
16t 5.12 -> 5.44 tok/s median (+6.3%), 32t 5.73 -> 5.83 (+1.7%), direction
consistent in 6/6 pairs. The row's re-attribution also split the old
"shared expert + combine" bucket: combine is 0.34 ms (0.2%); the shared
expert's three bf16 GEMVs are the bucket, and a bf16-dot tier for them is a
numerics change owed to a separate row. Router top-k (2.6-4.6%) cannot move
without a tie-order decision: `std::partial_sort`'s order among exact float
ties is unspecified, so a replacement selection cannot be proven
bit-identical (equivalence suite records the finding). Routed-expert loop
32-37% keeps its grouped-GEMM row. No ceiling declared.
[evidence](../bench-evidence/flint-cpu-descent-2-20261101/README.md).

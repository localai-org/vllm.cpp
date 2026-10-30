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
times it 0.458 ms/layer against 1.535 ms for dequant-at-load bf16 at batch 1
(4.394 vs 11.768 at batch 8), and the class (c) "Q8_K block-dot gathers" is
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

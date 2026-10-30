# Flint on CPU, the keep-quant paged arm

The RecursiveSI Flint Q4_K_M k-quant container served PACKED on CPU through
the paged engine with `VT_FLINT_KEEP_QUANT=1`: the linear layers and
`lm_head` run the CPU compute-in-quant GEMM through the per-dtype dispatch
seam, no bf16 copy of the weights is materialized. Measured 2026-10-29/30,
the shared aarch64 dev box, governor schedutil (the floor's governor state),
idle-gated, 3-rep medians, commit `row/FLINT-KQ-CPU-DISPATCH` HEAD; container
sha256 `20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`.

Correctness: 53/60 greedy positions token-exact against the pinned
llama-cpp-flint oracle goldens, 0 hard flips, all divergences inside the
house 2.5-nat near-tie band
([evidence](../bench-evidence/flint-kq-cpu-20261030/report.md)).

## Results

| threads | keep-quant paged (tok/s, tg200 median) | floor: llama.cpp tg200 (tok/s) | ratio |
|---:|---:|---:|---:|
| 16 | 1.20 | 26.29 | 0.046 |
| 32 | 2.46 | 26.38 | 0.093 |

Recipe: `VT_FLINT_KEEP_QUANT=1 build/examples/vllm-bench --model
/mnt/models/flint-GGUF/flint-Q4_K_M.gguf --num-prompts 1 --input-len 8
--output-len 200 --concurrency 1 --seed 0 --temperature 0 --ignore-eos`,
`taskset -c 32-47` (16t) / `-c 32-63` (32t).

## Disposition

The arm serves and is in-band; the throughput gap is OPEN — the gap is
~12-22x, wider than the bf16-dequant arm's 3.25 tok/s @16t, because the CIQ
path runs one tiny kernel call per expert projection (activation re-quantized
per call) in the MoE expert loop. Measured compute-class decision, recorded
in the evidence report; the named levers are the grouped expert GEMM the
kernel file already carries (`kMatmulBTQuantGrouped`), llama.cpp-style Q8_K
block-dot gathers, and the MoE gather/scatter fusion. No ceiling declared.

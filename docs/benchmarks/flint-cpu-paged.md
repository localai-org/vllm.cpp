# Flint on CPU, paged engine

The first throughput numbers for RecursiveSI's Flint on CPU, on the published
`flint-Q4_K_M.gguf` container served through the shared paged engine
(`vllm-bench`, full `LoadedEngine` stack). Measured 2026-10-27, the shared
aarch64 dev box, 32 threads (`taskset -c 0-31`, `nice -n 5`), idle, at commit
`92a217307`; container sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`.

This is a **host-specific baseline, not a parity claim**: vLLM registers no
flint architecture, so there is no vLLM denominator, and the llama-cpp-flint
oracle ([llama-cpp-flint](../../.agents/oracles/llama-cpp-flint.md)) serves
single-sequence only. The correctness record is the row's token gates
([evidence](../bench-evidence/flint-paged-20261027/report.md)).

| Axis | Value | Result |
|---|---:|---|
| Prefill (1×512 in) | **7.44 tok/s** | Baseline (first number) |
| Decode (1×8 in, 32 out, c=1) | **1.26 tok/s** (TPOT 797 ms) | Baseline (first number) |
| Serving (8×128 in, 32 out, c=4) | 0.97 tok/s aggregate, 0.49 per stream | Baseline (first number) |
| TTFT (8 in) | 5149 ms | Baseline (first number) |
| vLLM comparison | no vLLM run exists | **PENDING** a named external authority |

Open gap, next traceable lever: the MoE expert path runs the generic
per-expert gather/SiLU/scatter route; a grouped-GEMM expert path and
`MlpGateUpMethodBase` fusion is the first thing to trace against these
numbers. Batched decode scaling worse than batch 1 points at
memory-bandwidth contention in that gather/scatter, not at scheduling.

Recipe, raw session reports, and the full disposition:
[the evidence report](../bench-evidence/flint-paged-20261027/report.md).

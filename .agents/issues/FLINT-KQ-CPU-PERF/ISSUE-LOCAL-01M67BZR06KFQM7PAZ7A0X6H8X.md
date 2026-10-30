ID: ISSUE-LOCAL-01M67BZR06KFQM7PAZ7A0X6H8X
Title: Flint CPU keep-quant serves at 1.20 tok/s vs the 26.29 floor -- the batch-1 quantized compute class is the wrong one
Row: FLINT-KQ-CPU-PERF
State: CLOSED
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-31
Closed: 2026-10-31

## Problem

The keep-quant paged arm (landed bbd769bf6-lineage) serves flint-Q4_K_M.gguf at 1.20 tok/s @16t -- 2.7x SLOWER than the dequant-at-load bf16 arm (3.25) and ~22x under the gated llama.cpp floor (26.29, llama-cpp-flint oracle). The CIQ class (one tiny kernel per expert projection, activation re-quantized per call) is the wrong batch-1 shape. Three named levers: grouped expert GEMM (kMatmulBTQuantGrouped/MoeGateUpSwiGLUGrouped), llama.cpp-style Q8_K block-dot gathers, MoE gather/scatter fusion. Spec: .agents/specs/flint-kq-cpu-perf.md. Measurement-driven descent: stage 1 micro-times the three classes (dequant-at-load + grouped bf16 GEMM is a candidate winner with a memory trade to adjudicate), stage 2 adopts the winner through the shared seam with the serving gate re-adjudicated, stage 3 records. No ceiling claims.

## Resolution

- 2026-10-31 (FLINT-KQ-CPU-PERF): the premise is FALSIFIED by measurement —
  the CIQ class is the FASTEST available at decode batch sizes (stage-1 A/B:
  0.458 ms/layer vs 1.535 dequant-at-load bf16 at batch 1; the "Q8_K block-dot
  gathers" class is already what the CIQ kernel is). The 1.20 tok/s baseline
  was CPU-threadpool oversubscription: the pool defaults to
  `hardware_concurrency()` (128) while the serving legs ran under a 16-cpu
  budget. With `VLLM_CPP_CPU_THREADS` matched to the cpus the same class, with
  byte-identical tokens (gate re-run: 53/60, 0 hard, worst 2.486 — identical
  to the committed pin), serves at 5.51 tok/s @16t / 5.75 @32t. Evidence:
  `docs/bench-evidence/flint-kq-cpu-perf-20261030/README.md`; public record:
  `docs/benchmarks/flint-kq-cpu.md`. Residual 4.8x gap to the floor stays
  open; next levers named in the evidence. CLOSE with this evidence.

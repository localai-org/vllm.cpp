ID: ISSUE-LOCAL-01M67BZR06KFQM7PAZ7A0X6H8X
Title: Flint CPU keep-quant serves at 1.20 tok/s vs the 26.29 floor -- the batch-1 quantized compute class is the wrong one
Row: FLINT-KQ-CPU-PERF
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-30
Closed: -

## Problem

The keep-quant paged arm (landed bbd769bf6-lineage) serves flint-Q4_K_M.gguf at 1.20 tok/s @16t -- 2.7x SLOWER than the dequant-at-load bf16 arm (3.25) and ~22x under the gated llama.cpp floor (26.29, llama-cpp-flint oracle). The CIQ class (one tiny kernel per expert projection, activation re-quantized per call) is the wrong batch-1 shape. Three named levers: grouped expert GEMM (kMatmulBTQuantGrouped/MoeGateUpSwiGLUGrouped), llama.cpp-style Q8_K block-dot gathers, MoE gather/scatter fusion. Spec: .agents/specs/flint-kq-cpu-perf.md. Measurement-driven descent: stage 1 micro-times the three classes (dequant-at-load + grouped bf16 GEMM is a candidate winner with a memory trade to adjudicate), stage 2 adopts the winner through the shared seam with the serving gate re-adjudicated, stage 3 records. No ceiling claims.

## Resolution

-

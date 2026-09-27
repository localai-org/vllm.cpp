ID: ISSUE-GH-1928
Title: ROCm has no kMoeGroupedGemmBf16 provider (bf16-weight grouped MoE GEMM)
Row: -
State: OPEN
Kind: UNKNOWN
GitHub: 1928
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-25
Updated: 2026-08-25
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `OpId::kMoeGroupedGemmBf16` and its fused `kMoeGroupedGemmBf16GateUpSilu`
> sibling are registered CUDA-only
> ([`cuda_matmul_nvfp4.cu:2722`](../../../src/vt/cuda/cuda_matmul_nvfp4.cu#L2722)). ROCm's
> grouped-GEMM kernel
> ([`rocm_grouped_gemm.hip`](../../../src/vt/rocm/rocm_grouped_gemm.hip)) covers only
> **quantized-weight** grouped GEMM (Q8_0/Q4_K/Q5_K/Q6_K activations against
> keep-quant blocks); there is no bf16-weight grouped GEMM to extend on this
> backend.
>
> ## Where this bites
>
> A bf16-expanded MoE tower — reached whenever a GGUF's expert weights land in
> `GgufResidency::kExpandBf16` rather than a keep residency, which today is
> every MoE GGUF load on ROCm with `VT_GGUF_KEEP_QUANT=0` and every MoE GGUF
> whose quant encoding ROCm's keep-quant list does not cover — calls into
> `vt::OpRegistered(OpId::kMoeGroupedGemmBf16, kROCM)` at
> [`qwen3_5.cpp:7124`](../../../src/vllm/model_executor/models/qwen3_5.cpp#L7124) and
> [`deepseek_v2.cpp:325`](../../../src/vllm/model_executor/models/deepseek_v2.cpp#L325).
> Whether the fallback those two call sites take today already produces a
> message that names the missing op, or falls through to something less clear,
> is exactly what this issue's fix needs to verify and, if it does not already
> hold, repair — before any kernel work starts.
>
> ## Scope
>
> Not proposed as a design, only to bound the size of the eventual work: the
> CUDA implementation is a single TU spanning
> [`cuda_matmul_nvfp4.cu:1478`](../../../src/vt/cuda/cuda_matmul_nvfp4.cu#L1478)-`2732`,
> with a WMMA tensor-core prefill path, a split-K decode path, persistent
> CUDA-graph-safe scratch buffers (the `g_moe_fused_partials` /
> `g_moe_fused_gateup` retire-don't-free pattern), and a fused
> reduce+SwiGLU epilogue that folds `{gate; up; silu-mul}` into one launch. A
> ROCm port needs its own spec (upstream anchor: none — this is a vllm.cpp
> original, same as the CUDA source's own header states), its own red-first
> tests, and its own hardware-gated evidence on a `gfx1200`-class card (locally
> gateable — see `.agents/environment.md` and the AMD-only hardware note under
> this box's developer preferences).
>
> ## History
>
> Split out of [#1870](https://github.com/mudler/vllm.cpp/issues/1870)'s
> "Related gap, same area" section, filed separately per `AGENTS.md`'s owed-row
> rule rather than expanding that issue's scope: #1870's own reproduced bug (the
> device-fit refusal not accounting for an all-expand load policy) is a
> load-time-arithmetic fix with no upstream mirror and no kernel work, and does
> not block on this row landing first.
>
> Owned by `BACKEND-ROCM`.

## Resolution

The native provider implementation is tracked by
[#3094](https://github.com/mudler/vllm.cpp/issues/3094) and
[its committed spec](../../specs/rocm-bf16-moe.md). Both legacy provider IDs
are registered on ROCm, with separate typed native BF16 modes for the pinned
oracle's arithmetic. This issue remains open until the reviewed implementation
lands. The complete production token and performance gates are not yet satisfied.

ID: ISSUE-GH-2419
Title: IQ4_NL / Q5_0 / Q4_0 have no CUDA keep-quant GEMM, so the shipped qwen4_exp experts drain to the host or throw
Row: QUANT-CUDA-GATES
State: CLOSED
Kind: UNKNOWN
GitHub: 2419
Mirror: SYNCED
Availability: FULL
Created: 2026-08-31
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `QUANT-CUDA-KEEPQUANT-32B`
>
> Spec: [`.agents/specs/cuda-keepquant-32block.md`](../../specs/cuda-keepquant-32block.md).
>
> Discharges the keep-quant half of the debt #2380 records under `## Owed` for
> `MODEL-MM-QWEN4-EXP`.
>
> ## The gap
>
> `src/vt/cuda/cuda_quant_dot.cu::IsCudaKeepQuantSupported` admits twelve dtypes,
> all of them 256-element super-block encodings that dot a `BlockQ8_K`
> activation. **IQ4_NL, Q5_0 and Q4_0 are absent.** They are 32-element
> encodings that dot a `BlockQ8_0` activation, which that templated GEMM cannot
> take at all — the mismatch is structural, not a missing `case`.
>
> An absent dtype does not refuse. The three consumers behave in two ways:
>
> | Seam | Today, for these three dtypes |
> |---|---|
> | `MatmulBTQuantKernelCuda` | `cudaStreamSynchronize` then the CPU kernel over the same tensors. Correct numbers at host speed, and **not capturable** — the sync invalidates a decode graph. |
> | `MatmulBTQuantGroupedKernelCuda` | the same drain-and-fall-back. |
> | `MoeGateUpSwiGLUGroupedCuda` | **throws** `gate/up must be the SAME CUDA keep-quant dtype`. No fallback behind that seam. |
>
> Q8_0 is NOT in this gap despite also being absent from the predicate: it has
> its own dedicated on-device path (`MatmulQ8_0Cuda` / `MatmulQ8_0GroupedCuda`)
> dispatched *before* the predicate is consulted. #2380's sentence naming Q8_0
> alongside the other three is imprecise on that point.
>
> ## Why the released artifact needs exactly these
>
> `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S stores its routed-expert towers in
> IQ4_NL and Q5_0. It is not a free choice: `expert_feed_forward_length` is
> **640**, so every routed expert row is indivisible by 256 and cannot be encoded
> as any K-quant. `src/vllm/model_executor/models/qwen4_exp_gguf_weights.cpp:117-119`
> records that as the reason "the shipped file reaches for IQ4_NL".
>
> So the K=640 `ffn_down_exps` tower would be refused by the 256-super-block
> check even if its dtype *were* a K-quant. The 32-element lane is the only lane
> this checkpoint can use.
>
> ## Why no existing gate sees it
>
> A token gate cannot: before the change the tokens are right and the path drains
> to the host, after it they are right and it does not. The value comparisons in
> `tests/vt/test_cuda_quant_dot.cpp` read the host fallback as a pass for the same
> reason. The discriminator already in that file is the **stream-capture** case —
> `cudaStreamSynchronize` on a capturing stream fails — and it asserts a counted
> property (`captured == std::size(kCases)`) rather than a bare success.
>

## Resolution

FALSIFIED BY THE TREE, 2026-09-12. `593b888b5` ("feat(QUANT-CUDA-KEEPQUANT-32B): give IQ4_NL, Q5_0 and Q4_0 a device GEMM instead of a host drain") added `IsCuda32BlockKeepQuantSupported` (`src/vt/cuda/cuda_quant_dot.cu`), which admits exactly the three encodings this issue names, and both the dense `MatmulBTQuantKernelCuda` and the grouped `MatmulBTQuantGroupedKernelCuda` take that lane before reaching the `IsCudaKeepQuantSupported` test whose `default:` arm drains to the host. The comment on the 32-block predicate names this issue's own artifact: "the two encodings the released unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S stores its routed-expert towers in".

So the shipped qwen4_exp experts no longer drain to the host and no longer throw, and the title's condition no longer holds. Found while reading the same dispatch for the qwen4_exp decode profile; closed here with that evidence rather than re-specced, per AGENTS.md "An issue the tree falsifies closes with that evidence".

What this does NOT close: the grouped GEMM runs on device, but the MoE block AROUND it still round-trips the intermediates through host memory on the k-quant arm, which is a different defect and is filed separately as `.agents/issues/QUANT-CUDA-GATES/ISSUE-LOCAL-01M2AAACGC0SXYXFN2JQ3GFEAZ.md`. The CPU fallback at `cuda_quant_dot.cu:2551-2556` also still exists for MXFP4, which no shipped artifact on this row uses.

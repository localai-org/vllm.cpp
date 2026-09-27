ID: ISSUE-GH-1843
Title: The fp8 cuBLASLt lane still queries `cublasLtMatmulAlgoGetHeuristic` inside CUDA-graph capture because `VT_FP8_PLAN_CACHE` ships OFF (`fp8_plan_cache.h:49-59` @ `364f2a898`), so a captured decode on an fp8-tower model dies on CUDA 13.3 even with PR #1741 -- measured on `dgx:gpu0` (GB10, staged CUDA 13.3.73): #1741 alone fails on the fp8 lane, `VT_FP8_PLAN_CACHE=1` alone fails on the bf16-TN lane, both together pass the graphed 35B gate token-exact on all three arms. The fix is the default flip #1741's spec owed, with the same not-a-performance-knob polarity argument its `gemm_plan_cache.h` records. Claimed by row `FIX-FP8-PLAN-CAPTURE-1843` ([spec](../../specs/fix-fp8-plan-capture.md))
Row: FIX-FP8-PLAN-CAPTURE-1843
State: UNKNOWN
Kind: bug
GitHub: 1843
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:692`

### Frozen archive evidence

> | [#1843](https://github.com/mudler/vllm.cpp/issues/1843) | `FIX-FP8-PLAN-CAPTURE-1843` | The fp8 cuBLASLt lane still queries `cublasLtMatmulAlgoGetHeuristic` inside CUDA-graph capture because `VT_FP8_PLAN_CACHE` ships OFF (`fp8_plan_cache.h:49-59` @ `364f2a898`), so a captured decode on an fp8-tower model dies on CUDA 13.3 even with PR #1741 -- measured on `dgx:gpu0` (GB10, staged CUDA 13.3.73): #1741 alone fails on the fp8 lane, `VT_FP8_PLAN_CACHE=1` alone fails on the bf16-TN lane, both together pass the graphed 35B gate token-exact on all three arms. The fix is the default flip #1741's spec owed, with the same not-a-performance-knob polarity argument its `gemm_plan_cache.h` records. Claimed by row `FIX-FP8-PLAN-CAPTURE-1843` ([spec](../../specs/fix-fp8-plan-capture.md)) | bug |

## Resolution

-

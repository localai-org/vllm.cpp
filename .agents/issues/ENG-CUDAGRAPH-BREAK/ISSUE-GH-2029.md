ID: ISSUE-GH-2029
Title: With DFlash2 speculation OFF the engine dies at c=8 in CUDA graph capture: `cudaMalloc: operation not permitted when stream is capturing`. Located statically: `Pool(b).PreGrowForCapture(b, s.demand)` — the #1380 capture pre-grow — sits INSIDE `if (dbuf)` in both Qwen3.5 decode-graph drivers (`qwen3_5.cpp:10885/10907`, `:11439/11461`), and `dbuf = impl_->dbuf || spec_step` is false on the DEFAULT server, where `VT_ASYNC_EXECUTOR` is unset and no step is speculative. So the pre-grow is exactly the "path taken only when speculation is off" the issue names, by its absence. Spec [cudagraph-pregrow-nonspec.md](../../specs/cudagraph-pregrow-nonspec.md)
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: bug
GitHub: 2029
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:773`

### Frozen archive evidence

> | [#2029](https://github.com/mudler/vllm.cpp/issues/2029) | `ENG-CUDAGRAPH-BREAK` | With DFlash2 speculation OFF the engine dies at c=8 in CUDA graph capture: `cudaMalloc: operation not permitted when stream is capturing`. Located statically: `Pool(b).PreGrowForCapture(b, s.demand)` — the #1380 capture pre-grow — sits INSIDE `if (dbuf)` in both Qwen3.5 decode-graph drivers (`qwen3_5.cpp:10885/10907`, `:11439/11461`), and `dbuf = impl_->dbuf \|\| spec_step` is false on the DEFAULT server, where `VT_ASYNC_EXECUTOR` is unset and no step is speculative. So the pre-grow is exactly the "path taken only when speculation is off" the issue names, by its absence. Spec [cudagraph-pregrow-nonspec.md](../../specs/cudagraph-pregrow-nonspec.md) | bug |

## Resolution

-

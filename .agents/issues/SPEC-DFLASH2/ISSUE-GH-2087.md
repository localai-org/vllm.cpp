ID: ISSUE-GH-2087
Title: **At every `c > 1` the DFlash2 draft leaves the paged CUDA-graph fast path and re-attends the WHOLE batch's context every decode step.** `GPUModelRunner::propose_drafts_block` is the only production caller of `ForwardBlockLogitsWithDeviceKV` and passes one store per proposing row, so `P == num proposing rows`; the fast path is gated on `P == 1` (`qwen3_dflash.cpp:1577`) and everything above it falls to `:1888-1930`, whose own comment says it is "not capture-targeted". That fallback materializes `2 x L` `[C, kdim]` context buffers, then `ForwardWithCtxKVDev` (`:664`) allocates `[Ncomb = C + Tq]` query and output buffers per layer (`:792-794`, `:811`) and calls `vt::DFlashBlockAttention` (`:818`), whose CUDA grid is over ALL `Ncomb` rows (`cuda_ops.cu:2634`, `:2643`, `:2650`) — an attention output computed for every context row of every request and then discarded at `:820-827`. Per step, per layer: `sum_r (ctx_r + 1 + k)^2` attention pairs instead of the paged route's `(1+k) x C`. It enters at c=2, grows with c, and is the shape of the measured stall (ours 60.25 -> 63.3 tok/s from c=4 to c=8 where vLLM goes 64.25 -> 80.0). Spec [`specs/dflash2-batch-propose.md`](../../specs/dflash2-batch-propose.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: perf
GitHub: 2087
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:801`

### Frozen archive evidence

> | [#2087](https://github.com/mudler/vllm.cpp/issues/2087) | `SPEC-DFLASH2` | **At every `c > 1` the DFlash2 draft leaves the paged CUDA-graph fast path and re-attends the WHOLE batch's context every decode step.** `GPUModelRunner::propose_drafts_block` is the only production caller of `ForwardBlockLogitsWithDeviceKV` and passes one store per proposing row, so `P == num proposing rows`; the fast path is gated on `P == 1` (`qwen3_dflash.cpp:1577`) and everything above it falls to `:1888-1930`, whose own comment says it is "not capture-targeted". That fallback materializes `2 x L` `[C, kdim]` context buffers, then `ForwardWithCtxKVDev` (`:664`) allocates `[Ncomb = C + Tq]` query and output buffers per layer (`:792-794`, `:811`) and calls `vt::DFlashBlockAttention` (`:818`), whose CUDA grid is over ALL `Ncomb` rows (`cuda_ops.cu:2634`, `:2643`, `:2650`) — an attention output computed for every context row of every request and then discarded at `:820-827`. Per step, per layer: `sum_r (ctx_r + 1 + k)^2` attention pairs instead of the paged route's `(1+k) x C`. It enters at c=2, grows with c, and is the shape of the measured stall (ours 60.25 -> 63.3 tok/s from c=4 to c=8 where vLLM goes 64.25 -> 80.0). Spec [`specs/dflash2-batch-propose.md`](../../specs/dflash2-batch-propose.md) | perf |

## Resolution

-

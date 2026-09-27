ID: ISSUE-GH-521
Title: [`perf-fp8-alpha-fold.md`](../../specs/perf-fp8-alpha-fold.md) `:19`/`:211` claim the bf16-D lever "also applies to 35B-A3B" — it is INERT there: `GdnOutDType(dense_model=false)` is F32 on a MoE, contradicting the code's own comment at `qwen3_5.cpp:3617-3619`
Row: PERF-27B-LMHEAD-FP4
State: UNKNOWN
Kind: bug
GitHub: 521
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:120`

### Frozen archive evidence

> | [#521](https://github.com/mudler/vllm.cpp/issues/521) | `PERF-27B-LMHEAD-FP4` | [`perf-fp8-alpha-fold.md`](../../specs/perf-fp8-alpha-fold.md) `:19`/`:211` claim the bf16-D lever "also applies to 35B-A3B" — it is INERT there: `GdnOutDType(dense_model=false)` is F32 on a MoE, contradicting the code's own comment at `qwen3_5.cpp:3617-3619` | bug |

## Resolution

-

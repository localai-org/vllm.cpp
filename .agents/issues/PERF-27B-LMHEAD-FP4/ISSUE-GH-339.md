ID: ISSUE-GH-339
Title: 27B c1: every fp8 input projection asks for an f32 output, selecting the slower nvjet template family where vLLM emits bf16 (48 f32-out projections 18.51 ms vs 48 bf16-out 7.05 ms). The merged GDN `in_proj` arm is built DEFAULT OFF as `VT_GDN_FP8_IN_BF16`, spec [`perf-fp8-alpha-fold.md`](../../specs/perf-fp8-alpha-fold.md) §Attempt 4 — UNMEASURED: no committed gate loads the fp8 tower (`row/GATE-27B-FP8-TOWER-GOLDEN` builds that arm)
Row: PERF-27B-LMHEAD-FP4
State: UNKNOWN
Kind: perf
GitHub: 339
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:116`

### Frozen archive evidence

> | [#339](https://github.com/mudler/vllm.cpp/issues/339) | `PERF-27B-LMHEAD-FP4` | 27B c1: every fp8 input projection asks for an f32 output, selecting the slower nvjet template family where vLLM emits bf16 (48 f32-out projections 18.51 ms vs 48 bf16-out 7.05 ms). The merged GDN `in_proj` arm is built DEFAULT OFF as `VT_GDN_FP8_IN_BF16`, spec [`perf-fp8-alpha-fold.md`](../../specs/perf-fp8-alpha-fold.md) §Attempt 4 — UNMEASURED: no committed gate loads the fp8 tower (`row/GATE-27B-FP8-TOWER-GOLDEN` builds that arm) | perf |

## Resolution

-

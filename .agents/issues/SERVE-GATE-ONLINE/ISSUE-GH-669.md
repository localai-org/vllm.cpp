ID: ISSUE-GH-669
Title: Chunked prefill: our TTFT is budget-invariant because 1x4096 and 2x2048 are the same total work, NOT because the budget is ignored — composition now pinned, so no timing gate can re-litigate it; the pin's own 4096-token forward is the outlier (spec [`perf-chunked-prefill-budget-2026-08-13.md`](../../specs/perf-chunked-prefill-budget-2026-08-13.md))
Row: SERVE-GATE-ONLINE
State: UNKNOWN
Kind: verification
GitHub: 669
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:155`

### Frozen archive evidence

> | [#669](https://github.com/mudler/vllm.cpp/issues/669) | `SERVE-GATE-ONLINE` | Chunked prefill: our TTFT is budget-invariant because 1x4096 and 2x2048 are the same total work, NOT because the budget is ignored — composition now pinned, so no timing gate can re-litigate it; the pin's own 4096-token forward is the outlier (spec [`perf-chunked-prefill-budget-2026-08-13.md`](../../specs/perf-chunked-prefill-budget-2026-08-13.md)) | verification |

## Resolution

-

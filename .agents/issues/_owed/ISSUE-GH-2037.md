ID: ISSUE-GH-2037
Title: `EngineDeadError` promises "See stack trace (above)" (`include/vllm/v1/engine/core_client.h:63`) and the fatal handler prints only `e.what()` (`src/vllm/v1/engine/core_client.cpp:36-38`), so no trace is ever emitted. #1380 closed only because somebody instrumented `CudaBackend::Alloc` by hand; #2028 and #2029 both record the gap. Owed under `## Owed` in [cudagraph-pregrow-nonspec.md](../../specs/cudagraph-pregrow-nonspec.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 2037
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:776`

### Frozen archive evidence

> | [#2037](https://github.com/mudler/vllm.cpp/issues/2037) | — | `EngineDeadError` promises "See stack trace (above)" (`include/vllm/v1/engine/core_client.h:63`) and the fatal handler prints only `e.what()` (`src/vllm/v1/engine/core_client.cpp:36-38`), so no trace is ever emitted. #1380 closed only because somebody instrumented `CudaBackend::Alloc` by hand; #2028 and #2029 both record the gap. Owed under `## Owed` in [cudagraph-pregrow-nonspec.md](../../specs/cudagraph-pregrow-nonspec.md) | bug |

## Resolution

-

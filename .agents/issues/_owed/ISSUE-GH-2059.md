ID: ISSUE-GH-2059
Title: `.github/workflows/ci.yml:1598` sets `VT_POOL_BYPASS: "1"` for BOTH `sanitize-cpu` lanes, so the `DevicePool` free list, size-class ladder, best-fit borrow (#1922) and capture pre-grow (#1380) are unexecuted under ASan AND TSan. The stated justification is ASan's `detect_leaks`; ThreadSanitizer has no leak detector and gains nothing. MEASURED: under `-DVLLM_CPP_SANITIZE=thread` with the pool ENABLED, `test_qwen3_5_decode_graph_seam` is 10/10, 156 assertions, exit 0, zero TSan warnings. Found while repairing the #2047 red; owed under `## Owed` in [cudagraph-pregrow-nonspec.md](../../specs/cudagraph-pregrow-nonspec.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 2059
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:785`

### Frozen archive evidence

> | [#2059](https://github.com/mudler/vllm.cpp/issues/2059) | — | `.github/workflows/ci.yml:1598` sets `VT_POOL_BYPASS: "1"` for BOTH `sanitize-cpu` lanes, so the `DevicePool` free list, size-class ladder, best-fit borrow (#1922) and capture pre-grow (#1380) are unexecuted under ASan AND TSan. The stated justification is ASan's `detect_leaks`; ThreadSanitizer has no leak detector and gains nothing. MEASURED: under `-DVLLM_CPP_SANITIZE=thread` with the pool ENABLED, `test_qwen3_5_decode_graph_seam` is 10/10, 156 assertions, exit 0, zero TSan warnings. Found while repairing the #2047 red; owed under `## Owed` in [cudagraph-pregrow-nonspec.md](../../specs/cudagraph-pregrow-nonspec.md) | bug |

## Resolution

-

ID: ISSUE-GH-1361
Title: [`eng-cudagraph-break.md`](../../specs/eng-cudagraph-break.md) credited the W1 exit criterion to GB10 `sm_121a`, and the same file says it ran on `orin:gpu0`. Found by W5 ([#1335](https://github.com/mudler/vllm.cpp/issues/1335)) while reading `## Owed` to record what that stage did and did not discharge; record-only, no product behaviour. The G5 entry closed with "the seam's CUDA arm now runs on TWO architectures rather than one, sm_110 here and sm_121a on GB10 for the W1 exit criterion", while `## Work breakdown` W1 in the same file records the criterion as "measured on `orin:gpu0` through an `rc` lease, driver `12060`" — a Jetson AGX Orin, which is neither a GB10 nor `sm_121a`. Wrong in both halves: it named a device the measurement did not run on and an architecture nothing in this row has measured the criterion against. The two-architecture claim itself SURVIVES and only the attribution was wrong: the exit criterion (`cudaStreamEndCapture` then `cudaStreamBeginCapture` mid-forward with eager work between) ran on `orin:gpu0`, and G1 plus the unit suite ran on `thor:gpu0` at sm_110 for W3, W4 and W5. The criterion has NOT been re-measured on `thor`, for a structural reason rather than an omission: every migrated driver opens `kFull`, so nothing in the tree re-begins a capture mid-forward, and G1 exercises capture and replay rather than the re-begin. `sm_121a` on GB10 is OWED and not done — W5 could not take it because `dgx:gpu0` was held by another session for its whole window. This is the shape where a number quoted often starts being treated as measured, which is why it is filed rather than quietly reworded. FIXED IN FLOW by the W5 pull request, which states what ran where and moves `sm_121a` into `## Owed`
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: bug
GitHub: 1361
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:445`

### Frozen archive evidence

> | [#1361](https://github.com/mudler/vllm.cpp/issues/1361) | `ENG-CUDAGRAPH-BREAK` | [`eng-cudagraph-break.md`](../../specs/eng-cudagraph-break.md) credited the W1 exit criterion to GB10 `sm_121a`, and the same file says it ran on `orin:gpu0`. Found by W5 ([#1335](https://github.com/mudler/vllm.cpp/issues/1335)) while reading `## Owed` to record what that stage did and did not discharge; record-only, no product behaviour. The G5 entry closed with "the seam's CUDA arm now runs on TWO architectures rather than one, sm_110 here and sm_121a on GB10 for the W1 exit criterion", while `## Work breakdown` W1 in the same file records the criterion as "measured on `orin:gpu0` through an `rc` lease, driver `12060`" — a Jetson AGX Orin, which is neither a GB10 nor `sm_121a`. Wrong in both halves: it named a device the measurement did not run on and an architecture nothing in this row has measured the criterion against. The two-architecture claim itself SURVIVES and only the attribution was wrong: the exit criterion (`cudaStreamEndCapture` then `cudaStreamBeginCapture` mid-forward with eager work between) ran on `orin:gpu0`, and G1 plus the unit suite ran on `thor:gpu0` at sm_110 for W3, W4 and W5. The criterion has NOT been re-measured on `thor`, for a structural reason rather than an omission: every migrated driver opens `kFull`, so nothing in the tree re-begins a capture mid-forward, and G1 exercises capture and replay rather than the re-begin. `sm_121a` on GB10 is OWED and not done — W5 could not take it because `dgx:gpu0` was held by another session for its whole window. This is the shape where a number quoted often starts being treated as measured, which is why it is filed rather than quietly reworded. FIXED IN FLOW by the W5 pull request, which states what ran where and moves `sm_121a` into `## Owed` | bug |

## Resolution

-

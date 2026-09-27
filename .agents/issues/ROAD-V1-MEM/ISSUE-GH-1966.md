ID: ISSUE-GH-1966
Title: The #371 recurrent-state OOM guard is 48x under and passes a config that allocates 43.4 GiB, because `recurrent_state_bytes` counts placeholder layer names. Same root defect as [#1963](https://github.com/mudler/vllm.cpp/issues/1963), different code path; both fixed by `FIX-KV-GROUP-LAYER-COUNT` ([spec](../../specs/kv-group-layer-count.md))
Row: ROAD-V1-MEM
State: UNKNOWN
Kind: bug
GitHub: 1966
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:760`

### Frozen archive evidence

> | [#1966](https://github.com/mudler/vllm.cpp/issues/1966) | `ROAD-V1-MEM` | The #371 recurrent-state OOM guard is 48x under and passes a config that allocates 43.4 GiB, because `recurrent_state_bytes` counts placeholder layer names. Same root defect as [#1963](https://github.com/mudler/vllm.cpp/issues/1963), different code path; both fixed by `FIX-KV-GROUP-LAYER-COUNT` ([spec](../../specs/kv-group-layer-count.md)) | bug |

## Resolution

-

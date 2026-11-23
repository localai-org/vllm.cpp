ID: ISSUE-GH-835
Title: No row owns GPUDirect Storage, also called GDS or cuFile, for weight reads. `ENG-EXPERT-STREAM` W2 uses an `O_DIRECT` pool with aligned staging (`specs/expert-streaming.md:376`), which bypasses the page cache and still stages every expert through host memory. GPUDirect appears twice in the records and neither entry covers weights: `KV-MOONCAKE-STORE` names it for KV blocks over a fabric no box we own has, and `specs/lmcache-cpp-client-connector.md:305` marks GDS `NOT SCHEDULED` as an LMCache backend. The value differs by host, so a row must measure both paths before it claims a number
Row: -
State: UNKNOWN
Kind: feature
GitHub: 835
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:203`

### Frozen archive evidence

> | [#835](https://github.com/mudler/vllm.cpp/issues/835) | — | No row owns GPUDirect Storage, also called GDS or cuFile, for weight reads. `ENG-EXPERT-STREAM` W2 uses an `O_DIRECT` pool with aligned staging (`specs/expert-streaming.md:376`), which bypasses the page cache and still stages every expert through host memory. GPUDirect appears twice in the records and neither entry covers weights: `KV-MOONCAKE-STORE` names it for KV blocks over a fabric no box we own has, and `specs/lmcache-cpp-client-connector.md:305` marks GDS `NOT SCHEDULED` as an LMCache backend. The value differs by host, so a row must measure both paths before it claims a number | feature |

## Resolution

-

ID: ISSUE-GH-834
Title: No row owns router-lookahead prefetch for offloaded MoE experts. `ENG-EXPERT-STREAM` W3 copies router identifiers device to host and waits once per MoE layer (`specs/expert-streaming.md:377`), which is a synchronous stall. The only overlap work in that row is W6, and W6 runs `only if W3 trace shows wait dominance` and needs a separate accepted spike (`:380`). `ENG-WEIGHT-OFFLOAD` has a `PrefetchOffloader` arm, and it selects layers by position and never reads the router (`vllm/config/offload.py:48-76`). Prefetch is the lever that converts the per-layer fetch stall into an overlapped transfer, so the gap is recorded rather than left to be rediscovered
Row: -
State: UNKNOWN
Kind: feature
GitHub: 834
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:202`

### Frozen archive evidence

> | [#834](https://github.com/mudler/vllm.cpp/issues/834) | — | No row owns router-lookahead prefetch for offloaded MoE experts. `ENG-EXPERT-STREAM` W3 copies router identifiers device to host and waits once per MoE layer (`specs/expert-streaming.md:377`), which is a synchronous stall. The only overlap work in that row is W6, and W6 runs `only if W3 trace shows wait dominance` and needs a separate accepted spike (`:380`). `ENG-WEIGHT-OFFLOAD` has a `PrefetchOffloader` arm, and it selects layers by position and never reads the router (`vllm/config/offload.py:48-76`). Prefetch is the lever that converts the per-layer fetch stall into an overlapped transfer, so the gap is recorded rather than left to be rediscovered | feature |

## Resolution

-

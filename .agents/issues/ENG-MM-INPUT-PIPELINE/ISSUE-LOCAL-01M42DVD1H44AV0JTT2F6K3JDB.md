ID: ISSUE-LOCAL-01M42DVD1H44AV0JTT2F6K3JDB
Title: Multimodal usage omits the dense tower skip and repeats benchmark history
Row: ENG-MM-INPUT-PIPELINE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

At 33fb82b09 the Qwen3.5 dense production loader passes source.multimodal to LoadQwen3_5Dense and skips visual weights when image and video limits are zero. docs/reference/server.md and docs/guides/multimodal-input.md still list only the earlier loaders. The guide repeats superseded memory results already retained in docs/benchmarks/memory.md. Refresh these instructions and announce the shipped loading change without new benchmark or HTTP support claims.

## Resolution

-

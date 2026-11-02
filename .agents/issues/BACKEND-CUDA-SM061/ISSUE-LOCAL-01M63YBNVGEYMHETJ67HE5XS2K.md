ID: ISSUE-LOCAL-01M63YBNVGEYMHETJ67HE5XS2K
Title: Remove per-key block barriers from Pascal dense decode attention
Row: BACKEND-CUDA-SM061
State: OPEN
Kind: performance
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

P620 real TinyLlama trace spends 2647 ms in 2772 portable decode attention launches. The d64 model misses the existing warp-register decode kernel and pays a 256-thread shared-memory reduction and barriers for every key.

## Resolution

-

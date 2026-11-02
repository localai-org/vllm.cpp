ID: ISSUE-LOCAL-01M6434Z4NXA27S36QYE271NDG
Title: Preserve BF16 SiLU intermediate rounding on newer CUDA targets
Row: BACKEND-CUDA-SM061
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-30
Closed: -

## Problem

Split and merged SiLU omit the BF16 intermediate store when CUDA architecture is at least 75. Pinned primary scalar and packed kernels round this intermediate on every target, so the required exact upstream activation test exposes the gap.

## Resolution

-

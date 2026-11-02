ID: ISSUE-LOCAL-01M645FSTTEMK6E7QHQ183ZD3F
Title: Restore original cached RoPE dtype matrix in FP16 scalar port
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

Fresh Wave1 review found the cached RoPE port omitted pinned upstream F32 and BF16 combinations and retained a stale gamma-only comment.

## Resolution

-

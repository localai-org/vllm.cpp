ID: ISSUE-LOCAL-01M63MWPJR5Q5R2D24BDRS5RA5
Title: Legacy CUDA cannot select portable attention at public load
Row: BACKEND-CUDA-SM061
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

Public CUDA load on sm61 fails because FLASH_ATTN correctly requires sm80 and no legacy portable backend is registered. Existing portable paged attention kernels are unreachable from production model load on Pascal and Volta.

## Resolution

-

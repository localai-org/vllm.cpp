ID: ISSUE-LOCAL-01M63ZPB4JB2634AYK6XN4S6EH
Title: Honor explicit Pascal d128 attention disable override
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

The Pascal d128 hardware dispatch bypasses DecodeD128Enabled and ignores explicit VT_ATTN_DECODE_D128=0.

## Resolution

-

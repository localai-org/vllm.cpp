ID: ISSUE-LOCAL-01M63W80J6R0M1JXTQ5P1ENWGN
Title: Round legacy split SwiGLU activation to input width
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

The CPU split SwiGLU kernel and legacy merged SiLU kernel narrow SiLU to the BF16 gate width before multiplication, but the CUDA split kernel retains FP32. Dense original-format gate/up shards reach this shared split kernel and expose the mismatch on Pascal.

## Resolution

-

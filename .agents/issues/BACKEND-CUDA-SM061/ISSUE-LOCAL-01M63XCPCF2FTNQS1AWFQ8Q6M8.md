ID: ISSUE-LOCAL-01M63XCPCF2FTNQS1AWFQ8Q6M8
Title: Real-checkpoint CUDA gate undersizes the KV pool
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

The public ABI real-checkpoint gate sets 16 blocks of 16 tokens for a 256-token maximum context. The allocator reserves one null block, so the pool admits only 240 tokens and model loading fails before inference.

## Resolution

-

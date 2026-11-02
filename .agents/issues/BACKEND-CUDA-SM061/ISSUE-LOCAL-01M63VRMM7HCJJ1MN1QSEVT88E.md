ID: ISSUE-LOCAL-01M63VRMM7HCJJ1MN1QSEVT88E
Title: Preserve dense GGUF quantized projection shards
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

The dense GGUF loader expands merged QKV and gate-up weights to BF16 because the shared dense container cannot represent heterogeneous quantized shards. Pascal fitting-model validation needs original quant storage and existing production merged seams.

## Resolution

-

ID: ISSUE-LOCAL-01M653NJXDM6B5KE8VAA2VP93X
Title: Tile K-quant prefill for Pascal DP4A
Row: BACKEND-CUDA-SM061
State: OPEN
Kind: performance
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-30
Closed: -

## Problem

The generic K-quant prefill GEMM maps one warp per output row and re-reads each weight row once per prompt token. On the P620 the Q4_K prefill costs 412 ms and the Q6_K prefill 201 ms of a 1.25 s TinyLlama p16 trace. Add a tiled DP4A prefill lane that stages a weight tile once in shared memory and reuses it across the token tile, mirroring llama.cpp's Pascal DP4A MMQ config.

## Resolution

-

ID: ISSUE-LOCAL-01M63J6B9WK8Y1HB61TMVKA1ZF
Title: Legacy CUDA targets reject instructions and mixed batched output
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

CUDA 12.4 builds for sm60/sm61/sm70 fail on unguarded WMMA aliases, DP4A, nanosleep, EXL3 tensor instructions, and an incorrect runtime-query version guard. P620 registered BatchedMatmul with F32 inputs and BF16 output fails CUBLAS_STATUS_NOT_SUPPORTED. Existing CUDA SiLU omits the input-dtype intermediate rounding required by the pinned vLLM activation kernel and the CPU reference, producing two failing BF16-to-F32 test cases. Red logs are retained in pascal-volta-enable evidence.

## Resolution

-

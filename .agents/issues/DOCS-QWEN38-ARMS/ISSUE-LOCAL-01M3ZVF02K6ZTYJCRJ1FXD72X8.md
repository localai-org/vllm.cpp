ID: ISSUE-LOCAL-01M3ZVF02K6ZTYJCRJ1FXD72X8
Title: Correct Qwen3.8-27B quantized support guidance
Row: DOCS-QWEN38-ARMS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-03
Updated: 2026-10-03
Closed: -

## Problem

The model index calls block-wise FP8 CPU-only although CUDA is implemented and has a recorded GB10 token gate. The model guide implies every runnable arm is correctness-gated, including artifacts with failed or unrun gates.

## Resolution

-

ID: ISSUE-LOCAL-01M63XH2ED7MGFNBNWG8BKEEB6
Title: Honor NK layout for untied dense GGUF output heads
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

TinyLlama GGUF output.weight retains original quantized [vocab, hidden] NK layout but shared dense forward dispatches Matmul instead of MatmulBT, causing public completion to fail inner dimensions.

## Resolution

29 October 2026: dispatch untied NK heads through the existing MatmulBT seam.
The small public quantized-head regression and real TinyLlama generation pass.
Reverting the branch reproduces both failures. Evidence:
`.agents/evidence/pascal-volta-enable/untied-head-repair/`.

The issue stays OPEN until landing. Oracle and performance gates remain owed.

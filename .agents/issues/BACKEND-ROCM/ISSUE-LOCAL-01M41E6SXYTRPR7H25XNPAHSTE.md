ID: ISSUE-LOCAL-01M41E6SXYTRPR7H25XNPAHSTE
Title: bf16 residual fold corrupts EXL3 decode output (reverted)
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-03
Updated: 2026-10-03
Closed: 2026-10-03

## Problem

e1982b8aa folded CastF16/CastBf16 into the EXL3 Hadamard claiming bit-identical RN rounding; serving Qwen3.8-27B-EXL3-3.5bpw at that head produced degenerate greedy output while the parent and pre-fold images are coherent. Reverted at 72aa332ae.

## Resolution

2026-10-03: reverted the fold at 72aa332ae; the greedy 'Say hello' probe returns coherent output on the reverted build and the ROCm GEMV tier-3c gate is green. Root cause was the bf16-in Hadamard load arm not reproducing CastF16K<bf16> bit-exactly on the real residual stream.

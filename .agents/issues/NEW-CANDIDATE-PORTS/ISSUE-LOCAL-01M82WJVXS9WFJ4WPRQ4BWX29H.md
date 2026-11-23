ID: ISSUE-LOCAL-01M82WJVXS9WFJ4WPRQ4BWX29H
Title: olala keep-quant row cache thrashed on same-tensor row pairs (base-only key)
Row: NEW-CANDIDATE-PORTS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-23
Updated: 2026-11-23
Closed: 2026-11-23

## Problem

The first W2e keep-quant implementation cached the dequantized row in 4 slots keyed only on the tensor base pointer. The olala host ops alternate rows of the SAME tensor per element (ssm_in z/x projections, shift_k/shift_v, ssm_in_dyn B/C), so every element access collided in the single per-base slot and re-dequantized a full 1536-element Q6_K row per element read. The q6_k token gate exceeded 25 minutes without finishing (est. hours). Fix: 64 slots keyed on (base ^ row); rebuild finishes the gate in minutes. Found and fixed in the same flow per the in-flow bug rule.

## Resolution

2026-11-23: fixed in olala_ops.cpp QuantRowF32 — 64 slots keyed (base ^ row*golden-ratio-hash); perf profile showed 94% of cycles in DequantQ6_K before the fix; gate wall time 25+ min (unfixed) -> q6_k 168.6 s / q5_k_m 221.0 s / q8_0-safe2 129.8 s (fixed, Release). Evidence: docs/bench-evidence/new-candidate-ports-olala-w2e-20261122/token-gate-w2e-20261122.txt; gates PASS via RunArmGate in tests/vllm/models/test_olala_ops.cpp.

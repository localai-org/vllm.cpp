ID: ISSUE-LOCAL-01M5T8070WQ8RJYXZXD5ZWN5KZ
Title: Flint TT W2: audit the ttnn::matmul f32->tf32 operand truncation against the device-vs-CPU gap
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-25
Updated: 2026-10-25
Closed: -

## Problem

W0 recorded rel_rms 2.58e-1 device-vs-CPU on the Flint TT arm with a 1/8
greedy chain. W1 (row/TT-RMSNORM-W1) falsified the rms_norm attribution.
The standing substrate record (src/vt/tenstorrent/tenstorrent_ops.cpp,
commit b67b30fd lineage) says ttnn::matmul truncates f32 operands to tf32
on this pin (Blackhole); W2 must enumerate every matmul in
src/vllm/model_executor/models/flint_tt.cpp with its operand dtypes,
measure the truncation in isolation against a host reference, and either
fix the indicted site or kill the hypothesis with numbers and bisect the
Flint forward site-by-site (embed -> norms -> q/k/v/o -> mlp -> router ->
combine) to the first diverging op.

## Resolution

-

ID: ISSUE-LOCAL-01M5T8070WQ8RJYXZXD5ZWN5KZ
Title: Flint TT W2: audit the ttnn::matmul f32->tf32 operand truncation against the device-vs-CPU gap
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-25
Updated: 2026-10-25
Closed: 2026-10-25

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

2026-10-25 (branch row/FLINT-TT-W2, base 3090f069b): the hypothesis is
KILLED BY MEASUREMENT. (1) Static audit: flint_tt.cpp contains NO
f32-operand matmul — every activation operand is bf16; the tf32 operand
truncation has zero in-model sites. (2) The committed micro-probe
(test_flint_tt_w2_micro) shows the f32-staged and bf16-staged arms are
BIT-IDENTICAL at every shape (rel_rms 2.56-2.64e-2 vs host double,
K/M-independent), the multiply-only probe puts 2.39e-2 on a SINGLE product,
and small integers are bit-exact: the loss is operand-dtype-independent
substrate matmul noise (~23x the tf32 operand bound), not f32-to-tf32
operand truncation. (3) The layer-0 bisection (env-gated dumps in both
arms, commit 36b1c6868) shows divergence entering at the first post-norm
q/k/v projections (sums +3.7/-1.8/+8.4%, stream energies within 0.3%), a
dense-gate_proj outlier generator (20x energy, mostly suppressed by silu),
and ff re-converging (-0.4%): the model-level 2.579e-1 is inherited from
the post-attn bias, not from matmul operand truncation. Evidence:
docs/bench-evidence/flint-tt-w2-20261025/ (report.md, micro-all.log,
w0-after.log; W0 gate re-run 2/2 PASS, rel_rms 2.579e-01 unchanged, band
not re-tightened). The substrate matmul-noise defect and the dense
gate_proj outliers are named for the substrate row.


## Addendum 2026-10-26: the "gate_proj outlier generator" is retracted

The 20x-energy finding was a dump-tag artifact — the CPU dump block fuses
silu*u into g before printing, so the comparison was device pre-activation
against CPU post-activation. Apples-to-apples energies match at every
layer-0 stage (docs/bench-evidence/tt-matmul-noise-20261025/report.md, the
2026-10-26 addendum; instrument fixed in daf378deb).

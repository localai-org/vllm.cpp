ID: ISSUE-LOCAL-01M5TB3983MW41T4NPBZ5ZYF0E
Title: the ttnn::matmul per-op noise: characterize it and attempt an exact arm
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-25
Updated: 2026-10-26
Closed: 2026-10-26

## Problem

W2 (docs/bench-evidence/flint-tt-w2-20261025/report.md) measured a per-product rel_rms ~2.5e-2 on ttnn::matmul against host double, operand-dtype-independent (f32-staged == bf16-staged bit-identical), multiply-path dominated, integers exact — ~23x the tf32 operand bound. The mechanism is uncharacterized: no shape sweep, no layout/fp32_dest_acc_en axis, and the MatmulBT/bf16-out micro arms read uncorrelated (rel~1.2) in isolation yet behave in-model. W3 must map the noise against shape/layout/dest-acc, attempt exactness (composed exact arm, tt-metal-pin patch, or silicon-real verdict with an upstream-filing draft), and rerun the W0 harness honestly before/after.

## Resolution

CLOSED 2026-10-26 (W3, docs/bench-evidence/tt-matmul-noise-20261025/report.md).
Characterized: the noise is the pinned ttnn DEFAULT math fidelity (LoFi)
multiply truncation — K/M/N- and layout-independent (~2.3e-2 rel_rms flat on
the micro), per-product probe bit-exact, reproducing on a PLAIN ttnn matmul
with no vt staging (test_flint_tt_w3_plain). The exactness arm (HiFi4 +
approx off + fp32_dest_acc_en) measures rel_rms 3.3e-4, frac_bad 0 (~70x),
but does NOT move the W0 model number (first-token rel_rms 2.045e-1 shipped
vs 2.181e-1 HiFi4, chain 1/8 both), so the shipped arm is NOT flipped
(W1/W2 discipline); the arm stays selectable via
VT_TT_MATMUL_MATH_FIDELITY=hifi4. No upstream filing: the behavior is
documented math_fidelity semantics, not a defect. The W2 MatmulBT/bf16-out
"anomaly" is corrected as a test-side staging bug (correct-staging map rerun
agrees across all four arms). Two new precisely-named staging defects filed
as ISSUE-LOCAL-01M5TWFY79FWJN2BDPKYQZ3VXS (bf16-out readback at M=1;
order-dependent bf16-out staging corruption).

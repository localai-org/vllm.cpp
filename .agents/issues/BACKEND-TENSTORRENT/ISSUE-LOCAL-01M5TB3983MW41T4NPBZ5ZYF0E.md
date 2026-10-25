ID: ISSUE-LOCAL-01M5TB3983MW41T4NPBZ5ZYF0E
Title: the ttnn::matmul per-op noise: characterize it and attempt an exact arm
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

W2 (docs/bench-evidence/flint-tt-w2-20261025/report.md) measured a per-product rel_rms ~2.5e-2 on ttnn::matmul against host double, operand-dtype-independent (f32-staged == bf16-staged bit-identical), multiply-path dominated, integers exact — ~23x the tf32 operand bound. The mechanism is uncharacterized: no shape sweep, no layout/fp32_dest_acc_en axis, and the MatmulBT/bf16-out micro arms read uncorrelated (rel~1.2) in isolation yet behave in-model. W3 must map the noise against shape/layout/dest-acc, attempt exactness (composed exact arm, tt-metal-pin patch, or silicon-real verdict with an upstream-filing draft), and rerun the W0 harness honestly before/after.

## Resolution

-

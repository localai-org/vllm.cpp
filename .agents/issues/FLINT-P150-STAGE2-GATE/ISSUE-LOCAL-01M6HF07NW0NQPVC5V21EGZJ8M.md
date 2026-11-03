ID: ISSUE-LOCAL-01M6HF07NW0NQPVC5V21EGZJ8M
Title: The model-level device-attention gate at the fixed base, then the adoption decision
Row: FLINT-P150-STAGE2-GATE
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-03
Updated: 2026-11-03
Closed: -

## Problem

FLINT-P150-ATTENTION stage 2 STOPPED because the W0 model-level band gate was allocation-order-sensitive (neutral DBuf edits moved it 2.683e-1 to 7.224e-1). The blocker is cured: FLINT-TT-DENSE-MLP-STAGING (fd15425a1) fixed EnsureHostBytes to write the shadow's dtype; all four lifetime permutations now read 2.683e-1. Deliverable: a model-level gate case running the keep-quant paged recipe with VT_FLINT_TT_DEVICE_ATTN=1 asserting the same drift-tolerant band (rel_rms < 0.50, max_abs < 64.0) plus the device arm's own chain consistency; then the recorded adoption decision (default flip if the band holds, env-gated default-off with the measured gap if not -- stage 3 recorded device attention roughly neutral at 53.0 vs 55.2 s/step, so the numbers decide). Spec: .agents/specs/flint-p150-stage2-gate.md

## Resolution

-

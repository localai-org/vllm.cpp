ID: ISSUE-LOCAL-01M697FEEHXV9Y4CTYEJGRYHQR
Title: Flint TT dense-MLP gate/up staging produces allocation-order-dependent outliers on the P150
Row: FLINT-P150-ATTENTION
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: -

## Problem

The W2 record (docs/bench-evidence/flint-tt-w2-20261025/report.md section 3, reading 2) measured the dense gate_proj as an outlier generator on device: TT sum -14168 vs CPU +15.3, 20x the CPU stream energy, with silu mostly suppressing the damage (ff re-converges to -0.4%). The FLINT-P150-ATTENTION row re-hit this while wiring device attention: the full-model W0 band is ALLOCATION-ORDER-SENSITIVE — functionally neutral changes to DBuf lifetimes in flint_tt.cpp (scoping ao_dev/aoout_dev to the attention else-block, or staging three per-layer scalars at build) moved the host-shadow arm's first-token rel_rms between 2.683e-1 and 7.224e-1, and moved the device-attention arm between 4.825e-1 and 3.068 — with the attention legs themselves healthy (single-layer probe 64/64 per-head slices under 1/10 of the W2-derived band; layer-0 stage sums q/k/v/attn_out/ao match the host arm within 0.4%). Attention sits downstream of the same ln/dense-MLP staging every layer, so no model-level band gate can hold while the dense-MLP outliers persist. This blocks stage-2 adoption of FLINT-P150-ATTENTION (spec stop condition). Next lever, per W2: chase the dense gate/up STAGING with the W1 methodology, then re-run this row's model gate.

## Resolution

-

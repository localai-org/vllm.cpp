ID: ISSUE-LOCAL-01M6BQY53TJ3PPEVT7DT472HQP
Title: Flint TT dense gate/up staging produces allocation-order-dependent outliers on the P150
Row: FLINT-TT-DENSE-MLP-STAGING
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: -

## Problem

The Flint P150 device arm cannot pass a model-level band gate: functionally neutral DBuf lifetime changes in flint_tt.cpp moved the host-shadow arm's first-token rel_rms between 2.683e-1 and 7.224e-1, and the device-attention arm between 4.825e-1 and 3.068, with the attention legs themselves healthy (stage-1 probe 64/64 per-head slices under 1/10 of the W2 band, both window modes). The W2 report's own 'next site to chase' note names the dense gate/up staging as the suspect. Deliverable: attribute the outlier to the staging site with per-tensor rel_rms evidence at L0 and L46 under every DBuf lifetime permutation the fix permits, repair it, and verify the model-level band is allocation-order-INSENSITIVE after the fix -- the W0 band gate must hold under every permutation, not just the landed one.

## Resolution

-

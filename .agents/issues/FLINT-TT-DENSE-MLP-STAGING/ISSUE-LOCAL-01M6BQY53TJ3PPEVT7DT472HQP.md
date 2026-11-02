ID: ISSUE-LOCAL-01M6BQY53TJ3PPEVT7DT472HQP
Title: Flint TT dense gate/up staging produces allocation-order-dependent outliers on the P150
Row: FLINT-TT-DENSE-MLP-STAGING
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-02
Closed: 2026-11-02

## Problem

The Flint P150 device arm cannot pass a model-level band gate: functionally neutral DBuf lifetime changes in flint_tt.cpp moved the host-shadow arm's first-token rel_rms between 2.683e-1 and 7.224e-1, and the device-attention arm between 4.825e-1 and 3.068, with the attention legs themselves healthy (stage-1 probe 64/64 per-head slices under 1/10 of the W2 band, both window modes). The W2 report's own 'next site to chase' note names the dense gate/up staging as the suspect. Deliverable: attribute the outlier to the staging site with per-tensor rel_rms evidence at L0 and L46 under every DBuf lifetime permutation the fix permits, repair it, and verify the model-level band is allocation-order-INSENSITIVE after the fix -- the W0 band gate must hold under every permutation, not just the landed one.

## Resolution

RESOLVED 2026-11-02 (fix stage, this branch). The attribution narrowed the
poison to the down matmul's OUTPUT binding on a recycled pool block; the
three-matmul probe (same input slot, same weight, only the out block
differing: correct / WRONG / correct) pinned the vehicle, and the mechanism
is in `EnsureHostBytes` (src/vt/tenstorrent/tenstorrent_residency.cpp): the
host write width was picked from the slot's REGISTERED CAPACITY (`s->bytes`,
padded by the #1922 best-fit borrow) instead of the shadow's dtype, so a
bf16 activation whose DBuf borrowed a 2x block downloaded as f32 bytes and
the bf16 reader consumed them word-halved. The fix writes the shadow's own
dtype and demotes the capacity to a size check. Evidence:
`.agents/evidence/flint-tt-dense-mlp-staging-20261101/README.md` (the
slot-serve fix section) with the permutation matrix — all four staging-order
legs land at 2.683e-1 after the fix (the spread is gone) — and the standing
gates: attention probe 64/64 both window modes, test_flint_paged 208/208,
test_flint_gguf 184/184, test_flint_forward 110/110, W0 device case 18/18
in every leg.

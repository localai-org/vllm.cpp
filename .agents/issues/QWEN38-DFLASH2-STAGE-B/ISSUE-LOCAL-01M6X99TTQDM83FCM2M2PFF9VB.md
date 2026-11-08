ID: ISSUE-LOCAL-01M6X99TTQDM83FCM2M2PFF9VB
Title: DFlash2 stage B: the TT D2D capture lane clobbers the destination slot's recorded geometry, so residual taps read back pool residue
Row: QWEN38-DFLASH2-STAGE-B
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-08
Updated: 2026-11-08
Closed: -

## Problem

CopyDeviceDeviceIfCapture (tenstorrent_residency.cpp:1490) allocates a fresh src-shaped device tensor and installs it as the destination slot's shadow (:1558-1574) even when the destination already owns a persistent device shadow of equal numel but different geometry. The qwen3.5 dense decode aux taps write per-layer [1,H] clones into a persistent [S,H*taps] DBuf slot (qwen3_5.cpp:12430-12441 VT_TT_GDN_REPLAY_TAP comment); the captured trace then bakes reads at the slot's recorded [S,H*taps] geometry while the shadow holds a [1,H] tensor, so tap columns read back pool residue. DFlash2's drafter consumes exactly these taps, so the seam must be correct before stage B's verify capture. Stage B also owes the GDN two-buffer commit-or-restore and the exact-shape spec verify capture arm. Spec: .agents/specs/qwen38-dflash2-stage-b.md

## Resolution

- 2026-11-08 (implementer, row/QWEN38-DFLASH2-STAGE-B @ 49f04cb17): the three
  stage-B units shipped on the branch, host gates green.
  - Fix 1 (this issue's seam defect): CopyDeviceDeviceIfCapture serves the
    destination's RECORDED geometry when the dst slot holds an equal-numel
    device shadow; the src-shaped-install path survives unchanged for the
    no-shadow destinations (#2294/#1625). Red-before:
    docs/bench-evidence/qwen38-dflash2-stage-b-host/red-geometry-mismatch-d2d-copy.log
    (the consume-at-recorded-geometry matmul threw the a/b inner-dimension
    mismatch — the clobbered [1, H] record served). Green:
    green-focused-tests.log (5/5 with the #2294 and inertness-guard cases).
  - Which reshape arm shipped: the spec's named FALLBACK — ttnn::copy into
    the live dst shadow's storage segfaulted at the tt-metal pin, so the arm
    allocates a dst-shaped tensor, copies the reshaped source into it, and
    installs it with the recorded geometry preserved.
  - Fix 2: the dense decode driver stages GDN state shadows before a captured
    spec step (VT_TT_SPEC_STATE_SHADOW, default on); accept frees, reject
    restores at the recorded geometry; the restore declines on
    eviction/geometry change and joins the gdn_idx_changed reset lane. Red:
    red-gdn-spec-state-shadow-rollback.log (via the -DVT_TEST_GDN_SHADOW_STUBS
    reproduction build — the seam did not exist on the head); green in the
    focused set.
  - Arm 3: the (S, 9, spec=true) verify capture is the existing keyed-ring
    spec arm; the replay-tap probe now prints per-tap FNV checksums from the
    corrected seam. The spec shape consumes one qlen of the default
    VT_SPEC_GRAPH_MAX_QLENS=2 budget; the default did not move.
  - Host build repair: the branch's tt-metal shim pointed at the
    experimental spec-subtree include layout and the 17-arg
    chunk_gated_delta_rule; the local durable pin (a585e5744a8) carries the
    restored paths and the 18-arg symbol. Commit 9996ce94d aligns the branch
    with the pin it must link (host-build repair, not a pin advance).
  - PENDING: the P150 device gate (greedy token identity, traced spec arm vs
    eager) — no lease held this session; not waived. The checker battery and
    scripts/agent-preflight.sh --staged run recorded in the row evidence.

-

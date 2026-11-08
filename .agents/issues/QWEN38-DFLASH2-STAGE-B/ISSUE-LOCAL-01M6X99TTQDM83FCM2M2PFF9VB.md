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

-

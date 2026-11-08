ID: ISSUE-LOCAL-01M6Y0TC5D24YGF74H0TQ2VAN5
Title: kRmsNormGated committed-gate-shadow serves wrong bytes (max_abs 3.43251, max_rel 1452.01) — pre-existing, suite-context dependent
Row: BACKEND-TENSTORRENT-GDN
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-08
Updated: 2026-11-08
Closed: -

## Problem

The test case kTENSTORRENT kRmsNormGated serves a [256,6144] committed gate shadow at the batched-prefill [12288,128] geometry (tests/vt/test_tenstorrent_backend.cpp:3245, this row's W2 op family) fails with byte-identical tolerance blowout (max_abs=3.43251 max_rel=1452.01 tol=0.035) in isolation at BOTH row/QWEN38-DFLASH2-STAGE-B head 983eba149 AND that row's baseline 9996ce94d, on the P150 (/dev/tenstorrent/0, durable pin tt-metal-pin). Isolation evidence: docs/bench-evidence/qwen38-dflash2-stage-b-host/rms-head.log and rms-base.log. An earlier reconciliation record on the stage-B branch wrongly called it a branch regression from absence in the baseline failures-only log; the identical isolated result at the pre-branch SHA falsifies that. The defect (gate shadow served with wrong bytes at the [12288,128] batched-prefill geometry) predates the stage-B branch and is suite-context dependent — it passed inside the recorded baseline full-suite run and fails in isolation. Owed: bisection against an earlier base, a suite-context reproduction, and the shadow-serve fix.

## Resolution

-

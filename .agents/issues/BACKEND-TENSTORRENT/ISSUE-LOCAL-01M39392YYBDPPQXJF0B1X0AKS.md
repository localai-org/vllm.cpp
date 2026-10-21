ID: ISSUE-LOCAL-01M39392YYBDPPQXJF0B1X0AKS
Title: OPT-125m device arm throws kQkvSplit shape-contract failure on the TT backend (pre-existing on main)
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

test_opt_paged_engine with VLLM_CPP_OPT_MODEL_DIR=/mnt/models/opt-125m-bf16-st throws 'vt: tenstorrent kQkvSplit: out shapes must be [T, *]' on the TENSTORRENT device path, at tenstorrent_ops.cpp:1968 on main (control run 2026-09-24, P150, reset-hardened) and at the identical moved site on the split-stage-4 branch (:1454) — same exception, same 21 pre-checkpoint assertions passed, so the failure predates and is independent of the paged extraction. The 92/92 device suite passes on both trees; only the OPT checkpoint-backed leg hits it. This blocks the reviewer-named checkpoint-backed backend proof (the device-gated reachability detector for RAC/PA now rides the suite instead) and the OPT device lane until root-caused.

## Resolution

2026-10-21: FALSIFIED BY THE TREE. `c179fa248` ("tenstorrent kQkvSplit:
accept the rank-3 [T, H, Dh] outs the OPT caller passes", 2026-10-10) lifted
the rank-2-only restriction this issue names. The contract check in this
tree reads "tenstorrent kQkvSplit: out shapes must be [T, *] or [T, H, Dh]"
(`src/vt/tenstorrent/tenstorrent_ops.cpp:1654`), which is the check that
threw "out shapes must be [T, *]" on the control run this issue cites.

The commit carries the device evidence: the OPT-125M bf16 e2e on device
serves tokens and passes the STRICT gate, 6/6 prompts and 96/96 tokens
token-exact against the vLLM 0.25.0 oracle, all 9 OPT ops dispatched on the
device with 0 declines (`docs/bench-evidence/opt125m-tt-serve-20261010.md`).
Its red-first case is "kTENSTORRENT kQkvSplit accepts rank-3 [T, H, Dh] outs
(OPT shape)".

That commit closes the sibling record `ISSUE-LOCAL-01M4J8TY34AKMC4NFACBNSA2ED`
(the SIGSEGV that followed the rejection), which is already CLOSED. The
teardown SIGSEGV the commit records as a separate, unattributed defect is
not this issue's condition.


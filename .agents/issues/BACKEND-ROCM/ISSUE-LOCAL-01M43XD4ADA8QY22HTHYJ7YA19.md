ID: ISSUE-LOCAL-01M43XD4ADA8QY22HTHYJ7YA19
Title: GDN fused decode scan (GdnScanCoopFusedK qsl==nullptr) has no in-tree gate
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

GdnScanCoopFusedK serves DECODE by default (VT_GDN_SCAN_COOP on, VT_GDN_SCAN_ZSPLIT=4, qsl==nullptr, dk 64/128; landed 63e117e81 with no test in the commit). The branch's new byte-identity test covers only prefill mode (VT_GDN_SCAN_ZSPLIT_PREFILL); the only device test reaching GdnDecode uses DK=16 so fused_ok is false and it falls back to GdnScanCoopK. The served model runs 48 GDN layers per token through the untested decode branch -- recurring state, the exact profile of a delayed-onset corruption (see the decode-loop issue). Adversarial review finding (2026-10-04): add a decode-mode fused-vs-coop byte A/B at dk=128 with nonzero initial state, out AND final state compared.

## Resolution

2026-10-04: test_ops_gdn gained 'ROCm gdn_decode chained steps match the CPU
oracle within the state-drift bound' — 32 sequential decode steps on one
live state buffer for bf16 and f32 I/O, out and final state checked against
the CPU oracle each step. The ctest entry test_ops_gdn_rocm_zsplit1 pins
VT_GDN_SCAN_ZSPLIT=1 so the non-default decode split arm is exercised too.
State: resolved by this change; close on commit.

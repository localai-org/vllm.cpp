ID: ISSUE-LOCAL-01M61X3Q5N51S2AGHB5N8SS3HH
Title: CPU paged-attention regression: the kolibri1 kernel batch broke glm4-moe-lite goldens and the dtype-dispatch bit-identity contract
Row: FIX-PAGED-REGRESSION
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-28
Updated: 2026-10-28
Closed: -

## Problem

test_glm4_moe_lite_paged_engine and test_ops_paged_attn_dtype fail on main since the kolibri1 CPU paged-attention batch that landed 2026-10-19 13:50Z (draft commits 2cd6b8a14 FMA-contract, 3c4925def BFDOT K-dot, 6be7f0214, fa373385e, c975549b0, d63fb81fc; the 61d333e52 bf16-dot default landed 10-19). CI bracket: run 39200285700 (04:36Z) had glm4 failing only; run 39278997419 (15:34Z) failed both. Today both still fail: glm4 69/128 exact positions vs a STRICT oracle licensing 128/128 at gap 0; ops_paged_attn_dtype loses bit-identity across dispatch arms. The FMA contraction changes summation order in the SHARED kernel — prime suspect; exact culprit owed via bisect. The draft commits landed with failing goldens; the repair restores the contracts, never widens the assertions. Spec: .agents/specs/fix-paged-regression.md

## Resolution

-

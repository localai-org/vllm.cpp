ID: ISSUE-LOCAL-01M5AA1RRQYAT1JG3MDSKNEYAH
Title: test_kolibri1_spec_probe anchor stale since the FMA-contract re-anchor
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-19
Updated: 2026-10-19
Closed: 2026-10-19

## Problem

The spec probe's plain-chain anchor is 109726 (captured main 323f81ca6, pre-FMA-contract). The f32 production chain has been constant 33382 since the paged-attention FMA-contract re-anchor (decode bench re-anchor comment), so the probe's GATE (ii) CHECK(plain.chain.back()==kAnchorLastToken) has been red on main's f32 tier; evidenced 2026-10-19 by VT_CPU_KOLIBRI_BF16_DOT=0 (bit-identical to main production) printing CHECK( 33382 == 109726 ). Found during row MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm bf16-dot adoption; repaired in the same flow with per-tier anchors (f32 33382, bf16-dot default 11622).

## Resolution

2026-10-19: per-tier anchors landed with the bf16-dot adoption on row/kolibri-bf16-dot-cpu; the f32 leg (VT_CPU_KOLIBRI_BF16_DOT=0, bit-identical to main production) now passes GATE (ii) with anchor 33382 and 8/8 spec-vs-plain byte-identity; the bf16-dot default tier anchors 11622 (3/3 identical chains).

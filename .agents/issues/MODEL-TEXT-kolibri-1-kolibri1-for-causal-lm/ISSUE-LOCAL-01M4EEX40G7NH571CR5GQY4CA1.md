ID: ISSUE-LOCAL-01M4EEX40G7NH571CR5GQY4CA1
Title: kolibri1 CPU: name and land the next decode performance lever from the stage profile
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: enhancement
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: -

## Problem

Production decode is ~3.1 tok/s (8 threads, 16 GiB dequant cache) after two landed levers (NEON GEMM tier 2.07x, fp8 dequant cache 2.52x). The recorded stage profile (docs/bench-evidence/kolibri1-combined-neon-cache-20261008.md) still shows moe_glue ~0.25 s (residual per-expert dequant work), linear_gemm ~0.25-0.30 s, and unchanged attn_core/lm_head per forward. No measurement-first attribution of the remaining wall exists on the current tree, so the next lever is unnamed. Profile-first scope: reproduce the baseline, attribute the decode wall per stage at 4/8/16 threads, name the top lever from measurement, land it red-first with bit-exactness pinned (fingerprints 2.18646/26763, anchor 109726, budget invariance), or deliver a measured proposal if the lever is a policy choice (cache default/budget).

## Resolution

-

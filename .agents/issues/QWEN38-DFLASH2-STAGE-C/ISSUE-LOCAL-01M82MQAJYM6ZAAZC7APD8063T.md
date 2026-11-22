ID: ISSUE-LOCAL-01M82MQAJYM6ZAAZC7APD8063T
Title: Spec the P150 benchmark unit that turns the reference map into gated axes
Row: QWEN38-DFLASH2-STAGE-C
State: OPEN
Kind: docs
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-22
Updated: 2026-11-22
Closed: -

## Problem

The P150 reference map (PR #3593, docs/bench-evidence/p150-reference-map-20261122/reference-map.md) lists ten benchmark axes but no gated plan. Spec-first discipline requires a committed spec before any benchmark is published. Define the benchmark IDs, identical-workload definition per axis, denominator rule, correctness preconditions (DFlash2 blocked-on-W33, olala bf16 publishable-after-W2e arms, inkling2 W2e-equivalent arms), idle-host same-binary A/B discipline, recording recipe, and per-axis PENDING/FAILING/PUBLISHABLE stop conditions. Records-only; no public benchmark ID is added until a number actually lands.

## Resolution

-

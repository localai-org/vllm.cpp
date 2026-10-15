ID: ISSUE-LOCAL-01M4YWC05Y4CQCVH70FSDPEKZ7
Title: Kolibri1 CPU decode GEMV: 25.6 ms/step dispatch/sync overhead across 1301 pool kicks, and 8x in-kernel rate gap
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-15
Updated: 2026-10-15
Closed: -

## Problem

Measured attribution (VT_GEMV_PROF, t32 NEON decode, 32GiB dequant cache): the linear GEMV stage costs 105.8 ms wall per decode step (55% of the 180 ms forward), of which only 80.2 ms is worker math (per-worker 1.26 Gelem/s vs the 10.5 Gelem/s vbfdotq micro ideal) and 25.6 ms is pool dispatch/sync/idle spread over 1301 kicks/step (~20 us per kick). Routed experts are worst: 300 kicks/step, 17.0 ms/step dispatch+idle (38% of their wall). The kernel-side gap is structural: MatmulOneChunk is a deliberately scalar sequential-over-K f32 accumulate (bit-exact vs the reference), so even a perfect dispatch fix leaves ~70 ms/step on the table pending a numerics-policy decision. Restructures priced: batch 6-expert GEMVs per layer (save ~12-15 ms/step), batch q/k/v (save ~1.2 ms/step), eliminate per-call sync (save up to ~20 ms/step combined). Evidence: docs/bench-evidence/kolibri1-gemv-overhead.md

## Resolution

-

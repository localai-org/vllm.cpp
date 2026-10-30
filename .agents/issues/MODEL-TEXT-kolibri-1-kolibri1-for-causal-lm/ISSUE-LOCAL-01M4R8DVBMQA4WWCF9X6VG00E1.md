ID: ISSUE-LOCAL-01M4R8DVBMQA4WWCF9X6VG00E1
Title: Kolibri1 CPU decode GEMV stage: attribute math vs dispatch/sync overhead at t32 and price the batching restructures
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-12
Updated: 2026-10-12
Closed: -

## Problem

Production kolibri decode at t32+32GiB cache+NEON runs 175 ms/step while the linear GEMV stage aggregates only ~3.1 Gelem/s across 32 threads vs 10.5 Gelem/s single-thread (vbfdotq micro). The step looks serialization-dominated (pool kicks, per-call sync), not arithmetic- or bandwidth-bound. No per-GEMV-site attribution of math vs dispatch exists, and the candidate restructures (batch 6-expert GEMVs, batch q/k/v, eliminate per-call sync) are unpriced.

## Resolution

-

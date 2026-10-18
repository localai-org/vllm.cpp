ID: ISSUE-LOCAL-01M4NZA9B3DBKS3MD0M4TGRVES
Title: kolibri1 TT: device-side router op (kSigmoidBiasTopK) landed gate-verified but not default; no measured decode win
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-11
Updated: 2026-10-11
Closed: -

## Problem

vt::SigmoidBiasTopK removes the per-layer full-tail router-logits download, but the composite's extra eager launches per layer cost as much as the drain (decode 952 vs 924 ms/step median, within the run-to-run band; b2ii drift slightly worse: 4 vs 3 HARD flips). Default stays the host path; the op is opt-in via VT_KOLIBRI1_TT_DEVICE_ROUTER=1. The win needs a captured/graphed region to remove the launch overhead.

## Resolution

-

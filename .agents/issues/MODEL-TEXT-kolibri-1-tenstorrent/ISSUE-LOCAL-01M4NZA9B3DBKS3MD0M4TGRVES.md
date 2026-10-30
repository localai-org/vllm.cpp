ID: ISSUE-LOCAL-01M4NZA9B3DBKS3MD0M4TGRVES
Title: kolibri1 TT: device-side router op (kSigmoidBiasTopK) landed gate-verified but not default; no measured decode win
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-11
Updated: 2026-10-21
Closed: -

## Problem

vt::SigmoidBiasTopK removes the per-layer full-tail router-logits download, but the composite's extra eager launches per layer cost as much as the drain (decode 952 vs 924 ms/step median, within the run-to-run band; b2ii drift slightly worse: 4 vs 3 HARD flips). Default stays the host path; the op is opt-in via VT_KOLIBRI1_TT_DEVICE_ROUTER=1. The win needs a captured/graphed region to remove the launch overhead.

## Resolution

- CURRENT P150 NUMBER (2026-10-21, `row/P150-NUMBERS-REFRESH`, records-only):
  the per-step decode cost quoted above (decode 952 against 924 ms/step) is
  the 2026-10-11 A/B vintage, measured on a different tree under different
  host conditions, and it is not a like-for-like comparand for the current
  number. The current per-step decode cost on `origin/main` `3ce14f35c` is
  547.7 / 550.2 / 548.5 ms per decode step over three runs (median 548.5 ms,
  which is 0.548 s per token), with decode 1.82585 / 1.81767 / 1.82310 tok/s
  (median 1.823) over the bench's 64 incremental decode steps after a
  128-token prompt prefill. The issue's own verdict, that the device-router
  composite buys no measured decode win, is unchanged by this refresh. The
  tree carries no attribution for the change against the 2026-10-18 reading
  of the same bench (about 1556 ms/step), and the attribution stays OPEN.
  Evidence:
  `docs/bench-evidence/kolibri1-tt-p150-refresh-20261021.md`; issue
  ISSUE-LOCAL-01M5F68A30HNPG4V95BKTAQS79.

ID: ISSUE-LOCAL-01M4NS6D2GCD5WXBFN24XGXYQ4
Title: Generalize the B1 per-token stream bound to multi-token forwards; record the chunked-prefill evidence
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

The B1 per-step stream guard admits only one-token forwards; a 128-token whole-prompt prefill refuses at 1.40 GB against the 1.18 GB bound. Chunked prefill was measured (prefill 156.3 s -> 72.1 s) but chain equivalence is not adjudicable on this instrument (chains differ run to run on identical code).

## Resolution

- CURRENT P150 PREFILL NUMBER (2026-10-21, `row/P150-NUMBERS-REFRESH`,
  records-only): the prompt-prefill cost quoted above (prefill 156.3 s to
  72.1 s) is the 2026-10-11 chunked-prefill vintage, and it is not a
  like-for-like comparand for the current number. The current cost of the
  128-token prompt prefill on `origin/main` `3ce14f35c` is 146.97 / 131.11 /
  124.89 s over three runs (median 131.1 s), and the spread is the prompt
  forward's load-time variance: the decode phase of the same three runs
  sits inside a 0.16 s window. The bench prefills that 128-token prompt and
  then times 64 incremental decode steps, so the prefill number is the cost
  of the 128-token prompt alone and the decode tok/s figure (1.82585 /
  1.81767 / 1.82310, median 1.823) covers the decode phase alone. The tree
  carries no attribution for the change against the 2026-10-18 readings of
  the same bench (about 1556 ms/step, prefill 156.4 s), and the attribution
  stays OPEN. Evidence:
  `docs/bench-evidence/kolibri1-tt-p150-refresh-20261021.md`; issue
  ISSUE-LOCAL-01M5F68A30HNPG4V95BKTAQS79.

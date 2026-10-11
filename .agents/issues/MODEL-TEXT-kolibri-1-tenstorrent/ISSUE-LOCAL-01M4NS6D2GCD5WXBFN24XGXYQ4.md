ID: ISSUE-LOCAL-01M4NS6D2GCD5WXBFN24XGXYQ4
Title: Generalize the B1 per-token stream bound to multi-token forwards; record the chunked-prefill evidence
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

The B1 per-step stream guard admits only one-token forwards; a 128-token whole-prompt prefill refuses at 1.40 GB against the 1.18 GB bound. Chunked prefill was measured (prefill 156.3 s -> 72.1 s) but chain equivalence is not adjudicable on this instrument (chains differ run to run on identical code).

## Resolution

-

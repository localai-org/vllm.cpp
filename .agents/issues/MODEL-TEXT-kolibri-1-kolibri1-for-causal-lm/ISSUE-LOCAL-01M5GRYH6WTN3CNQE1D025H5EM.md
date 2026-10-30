ID: ISSUE-LOCAL-01M5GRYH6WTN3CNQE1D025H5EM
Title: Align Kolibri user guidance with the shipped CPU cache default
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: docs
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-22
Updated: 2026-10-22
Closed: 2026-10-22

## Problem

USAGE documents the obsolete 1 GiB floor and post-load memory budget. The model guide still says TT lacks routed experts. FEATURES duplicates detailed cache internals in a long table cell, and README news omits the new CPU default.

## Resolution

2026-10-22: corrected the public cache policy, Tenstorrent expert path, and BF16 oracle caveat against current source and evidence. Added README news and shortened the feature row. Documentation gates and 55 checker tests pass. No runtime or benchmark changes. Full preflight is limited by missing build tools and PyYAML in this environment.

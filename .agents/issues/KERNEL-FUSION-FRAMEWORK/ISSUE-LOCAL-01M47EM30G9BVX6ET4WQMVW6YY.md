ID: ISSUE-LOCAL-01M47EM30G9BVX6ET4WQMVW6YY
Title: kolibri1_forward hand-fuses add+RMSNorm outside vt::FusedChain; allowlist known-drift
Row: KERNEL-FUSION-FRAMEWORK
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-06
Updated: 2026-10-06
Closed: -

## Problem

kolibri1_forward.cpp carries 3 residual add+RMSNorm hand-call sites that do not route through the vt::FusedChain catalog, so scripts/check-fusion-consistency.py fails on origin/main as of e67a071a6. The kFusedAddRmsNorm{,Std} recipe exists; the migration is deferred work, so the honest state is a known-drift allowlist entry naming this issue.

## Resolution

-

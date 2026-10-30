ID: ISSUE-LOCAL-01M57TAF5TQKX6Y30WADFEP791
Title: VT_TT_SHADOW_DROP_TRACE is read in production but neither documented nor allowlisted
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-18
Updated: 2026-10-18
Closed: 2026-10-18

## Problem

Commit 93e37e703 (#3468) introduced the VT_TT_SHADOW_DROP_TRACE attribution probe in src/vt/tenstorrent/tenstorrent_residency.cpp but did not document it in docs/ENVIRONMENT.md or add it to scripts/env-doc-allowlist.txt. check-env-doc.py is therefore red on origin/main, and the agent-record CI job fails on every tree that contains the knob — including PR #3478 once its benchmark-index repair clears the other failure. Repair: the knob is a kernel-internal attribution probe gated to the literal '1', so per the checker's own rule it belongs on the allowlist.

## Resolution

2026-10-18: added VT_TT_SHADOW_DROP_TRACE to scripts/env-doc-allowlist.txt (kernel-internal probe, literal-1 gate). check-env-doc.py: ERROR on main -> OK on this tree. Rides PR #3478.

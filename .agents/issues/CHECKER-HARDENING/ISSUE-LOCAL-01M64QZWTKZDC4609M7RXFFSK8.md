ID: ISSUE-LOCAL-01M64QZWTKZDC4609M7RXFFSK8
Title: Four detection gaps found by campaign reviews: evidence-sha, registry multiplicity, CI wiring coverage, stderr-count
Row: CHECKER-HARDENING
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

Four fresh-review findings, none owned elsewhere: (1) nothing compares an evidence report's recorded artifact sha256 against the oracle pin's claim (FLINT-CPU-PARITY M3); (2) check-oracle-pins compares registry ids as sets (scripts/check-oracle-pins.py:559-585) so duplicate AGENTS.md registry rows are invisible (M4); (3) deleting a wired step from the agent-record job reds nothing -- check-test-registration pins only its own step and 3 named suites (FIX-EVIDENCE-LOG-GATE M5); (4) the kolibri1 NUMA fallback warning is enforced by count evidence only, no stderr-capturing gate (FIX-TT-NUMA-DEGRADE M1). Spec: .agents/specs/checker-hardening.md. One row, four independent commits, each with its own red-first capture. Any item needing semantics beyond the stated shape splits into its own row.

## Resolution

-

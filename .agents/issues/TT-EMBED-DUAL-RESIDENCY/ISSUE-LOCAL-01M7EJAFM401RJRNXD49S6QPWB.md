ID: ISSUE-LOCAL-01M7EJAFM401RJRNXD49S6QPWB
Title: check-agent-record treats [TT-ALLOC] #N allocator IDs in device stderr evidence logs as issue references
Row: TT-EMBED-DUAL-RESIDENCY
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-15
Closed: -

## Problem

scripts/check-agent-record.py on row/TT-EMBED-DUAL-RESIDENCY reports _intake-resolution errors for the GitHub-issue-number tokens 652/834/835/148/242/250/296/362/469 and their ISSUE-GH twins. These tokens are not issue references: they are device allocator IDs in committed stderr evidence ([TT-ALLOC] #N label=... lines in docs/bench-evidence/tt-embed-dual-residency-20261114/{q6kl-cap-1,q6kl-cap16-1}.stderr.log) and quoted checker diagnostics in preflight-staged-repair.log. The enforced property (references resolve to canonical records) is correct; the reference scanner has a false-positive class on evidence-log lines. Fix the scanner, not the evidence logs.

## Resolution

-

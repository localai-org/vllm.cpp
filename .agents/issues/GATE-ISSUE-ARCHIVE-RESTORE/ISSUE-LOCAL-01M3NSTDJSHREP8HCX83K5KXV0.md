ID: ISSUE-LOCAL-01M3NSTDJSHREP8HCX83K5KXV0
Title: the frozen-evidence contract is not enforced outside _intake
Row: GATE-ISSUE-ARCHIVE-RESTORE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-29
Updated: 2026-09-29
Closed: -

## Problem

validate_issue_record runs the frozen-evidence comparison (the quoted archive line must be byte-equal to the declared line, and must name the record's own GitHub number) only in the _intake branch. Row-owned and _owed records that carry a Frozen archive evidence block are never compared: any of the 822 non-intake blocks could drift from the archive, swap a URL, or quote another issue's row, and every gate stays green. Measured on the restored archive (#3351 data half): 831 records carry a block, 831 resolve under the #3350 comparison with the record directory as base, 831 quote a line carrying their own issue number, 0 violations, so enforcing the same rule outside _intake adds no new red today while closing the drift door. Absence of the block stays legal outside _intake (456 row-owned records legitimately have none); presence is not. Fix: hoist the comparison into a shared helper and apply it in the _owed and row-owned branches, and strip a trailing CR from the archived line so a CRLF Windows working copy compares equal to the LF committed blob.

## Resolution

-

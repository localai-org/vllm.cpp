ID: ISSUE-LOCAL-01M3NR65JDEV0NH28W5ZPQDPYC
Title: the frozen archive dropped 27 rows its own evidence records still cite
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

Commit 2e84a073b deleted .agents/completed/issue-index.md wholesale (913 lines) and e3539d994 restored a hand-picked 886-line copy that dropped 27 archived rows. Those 27 rows are exactly what 43 frozen-evidence records cite: 24 cite a line past the restored file's end, 19 cite a line that now holds a different row. No gate sees any of that today: the frozen-evidence comparison runs only in the _intake branch, whose 9 records all cite surviving lines, so all 43 stale quotes are row-owned records that validate untouched. But every one of those 43 quotes is byte-true to the pre-deletion archive, so no edit to any record can repair them against a truncated archive; the defect is in the archive, not in the quoting. Measured: re-inserting the deleted lines at the positions difflib reports leaves the file byte-identical to the pre-deletion archive except line 406, which keeps e3539d994's own deliberate link re-point; the 831-quote census moves from 350 byte-equal to 373, and check-agent-record stays green with unchanged row counts. With the #3350 relative-link comparison the restored archive resolves all 831 quotes, 0 failing. Fix: splice the 27 rows back at their original positions, re-pointing any link that does not resolve from .agents/completed/ (measured: none needed). Prerequisite for the stacked checker change that enforces the comparison outside _intake.

## Resolution

-

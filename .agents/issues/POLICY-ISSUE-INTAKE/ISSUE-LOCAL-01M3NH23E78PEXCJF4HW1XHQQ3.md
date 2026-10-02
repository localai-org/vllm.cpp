ID: ISSUE-LOCAL-01M3NH23E78PEXCJF4HW1XHQQ3
Title: frozen-evidence byte equality and record link resolution cannot both hold
Row: POLICY-ISSUE-INTAKE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-28
Updated: 2026-09-28
Closed: -

## Problem

An _intake record's Problem must quote the frozen archive line byte for byte, but check-agent-record's check_links requires every link in a record to resolve from the record's own directory. The archive lives at .agents/completed/issue-index.md, one level under .agents, so a link to a spec is spelled ../specs/x.md there; the record that QUOTES the row lives at .agents/issues/<owner>/, two levels down, so the same link must be spelled ../../specs/x.md from there. One string cannot satisfy both. Commit e3539d994 re-pointed the ISSUE-GH-1033 quote to the record-relative spelling to fix the dangling link, which broke the byte comparison: first divergence at column 2191 of archive line 350, archive 2237 chars against evidence 2240. Measured over all 831 records that carry a Frozen archive evidence block, 350 are byte-equal, 438 differ from the cited line ONLY by a relative link rebase, and 43 cite a line the archive no longer has (24 past its 886 lines after the delete and re-add, 19 naming a row that moved). Restoring the archive spelling makes check-links red instead. Fix: compare the quote against the cited line with each side's relative link targets resolved to the file they denote, so the check asks whether the quote IS the archived row rather than which directory holds the quote.

## Resolution

-

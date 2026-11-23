ID: ISSUE-LOCAL-01M56MAC9WPXPYV16XVTAH7KVS
Title: The last nine _intake records red every frozen-archive edit; triage them to their owning specs
Row: ENG-RECORD-CONFLICT-SURFACES
State: OPEN
Kind: record
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-18
Updated: 2026-10-18
Closed: -

## Problem

check-agent-record harvests issue references from the full content of every changed file. .agents/completed/issue-index.md is the frozen archive and every one of its rows is a citation, so ANY branch that edits the archive - which every archive-row repair must, because the record's quoted evidence has to stay byte-equal to the row - harvests the nine remaining intake identities the archive still cites (#148, #242, #250, #296, #362, #469, #652, #834, #835). check_canonical_issue_references rejects a reference that resolves to an _intake record with 'triage it to a canonical row or exactly owned _owed', so #3354 and #3367 red on nine errors apiece while their own diffs touch none of those rows. Measured on both branches at upstream/main 4b1a33856. The rejection is the designed exit and not a checker defect: roadmap-consistency.md requirements 42-45 reject citations to intake in changed-file content and in commit bodies 'until triage', and line 301 states the archive-evidence exclusion is scoped to the intake file's own evidence block, never to the archive itself. No archive edit has passed the rule since da3a8cfc7 introduced it; the last archive edit (617730b33) predates it topologically, which is why the lock went unnoticed. The burn-down is 594 records at cutover down to these nine, so triage is the designed completion. Each of the nine is moved atomically from _intake to _owed with exactly one committed spec reference under ## Owed, per roadmap-consistency.md line 235; no row, date, state, problem or resolution is invented. Owners are chosen by a surface the issue itself names or by the spec that owns the mechanism the issue measures.

## Resolution

-

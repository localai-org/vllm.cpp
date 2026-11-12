ID: ISSUE-LOCAL-01M3TZZWZBKWSJKGSV6NYQEQW5
Title: The #1033 row, its record and its spec cite check_table_shapes at a checker line the restructure moved - four instances convert to the symbol convention
Row: ENG-RECORD-CONFLICT-SURFACES
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-01
Updated: 2026-10-01
Closed: -

## Problem

The citation gate resolves every `check-agent-record.py` line citation in the tree against the tracked checker, and at `fce36733b` the checker is 1270 lines while the gate lists eighteen dangling instances. Four of them are ONE claim: `check_table_shapes` is cited at line 1292 from the frozen archive row for #1033, twice more from ISSUE-GH-1033 (its Title and the Frozen archive evidence quote), and once from `.agents/specs/gate-issue-index-table-shape.md`.

The claim was true when written and is false against every checker the tree has since carried. The symbol sits at line 1105 on main, and the reviewed restructure that a merge resolved away defines it at line 2028, so no single number is correct in both trees and a re-point would land a number the next restore moves again. The repair adopts the repository's own symbol convention (`citation-anchor-freshness.md`, option 4) and cites `scripts/check-agent-record.py::check_table_shapes`, which resolves in both trees.

The archive row and the record's Title and its quote move together and stay byte-equal, because the frozen-evidence contract compares them; the spec carries the same conversion. Red-first: the gate lists the four instances before the repair and none of them after, with the pre-existing remainder (the #3354 family and the pending-restore family) untouched.

## Resolution

-

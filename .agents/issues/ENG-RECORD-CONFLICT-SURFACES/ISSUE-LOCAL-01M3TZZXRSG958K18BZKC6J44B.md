ID: ISSUE-LOCAL-01M3TZZXRSG958K18BZKC6J44B
Title: Seven checker-line citations the restructure and two retirements left dangling - owed_issues(), the duplicate-issue refusal, check_row_contracts and the canonical-reference pass
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

Seven of the citation gate's dangling instances sit outside both the #3354 family and the checker-restore family. They need three different repairs, and two of them name instruments the tree has retired.

Symbol conversions (the target exists in both the current and the reviewed-restructure checker): `scripts/check-agent-record.py::check_row_contracts` is where the `CLAIM-*` owner requirement lives, cited at line 1785 from `.agents/claims/CLAIM-MODEL-DSV4-EXL3.md`.

Retired instruments, repaired as retirement prose rather than instrument swaps: the record checker's backlog reader `owed_issues()`, cited at lines 2002-2017 and 2012-2014 from `.agents/specs/ltx25-completion-scope.md`, was removed by the local-issue-authority cutover (`a9f6186c2`), and its successor in `scripts/issue_records.py` reads stable issue identities rather than the bare numbers those measurements counted, so naming the successor would change the subject; and the duplicate-issue refusal, cited at lines 1437-1441 and 1437-1442 from two specs, was removed by W6 (`7dc2ef1ea`) together with the live issue index it protected.

One status correction: the canonical-reference pass cited at lines 2216-2222 from this row's own ISSUE-LOCAL-01M2TR7N06G9EF8J2QVWVM49W7 exists only in the restructured checker; the pre-restructure file on main resolves no canonical references at all.

Red-first: seven instances before, none after.

## Resolution

-

ID: ISSUE-LOCAL-01M3SQTG14MVDB5MWCQPB418QM
Title: check-agent-record red: restored checker's records drifted on main
Row: ENG-RECORD-CONFLICT-SURFACES
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

The 40990825d merge dropped the restructured check-agent-record checker; main carried the old monolith against tests written for the restructured API (35 test failures). The tree also drifted under the restored gate: 3 specs listed row-owned issue IDs under ## Owed, one issue carried the invalid State PARTIAL, 10 stale record-anchor citations in engine-matrix.md, 6 stale checker-line citations in tracked specs, and one test dereferenced the tracked evidence 'latest' symlink directory. Reconciled in row/agent-record-reconcile.

## Resolution

-

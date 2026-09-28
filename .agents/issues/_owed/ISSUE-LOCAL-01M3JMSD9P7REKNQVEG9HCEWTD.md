ID: ISSUE-LOCAL-01M3JMSD9P7REKNQVEG9HCEWTD
Title: Restore the record checker lost in the 40990825d merge and repair the tree to its gate
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-28
Updated: 2026-09-28
Closed: -

## Problem

Merge 40990825d resolved scripts/check-agent-record.py to the pre-restructure parent, silently dropping the reviewed restructured checker (fence-aware link extraction, claim-state consistency from #3099, record-anchor ratchet, canonical roadmap issue-row refusal, derived inventory counts) while keeping its new module files. The record suite at main is red (72F/83P pre-repair) and the checker gate no longer enforces AGENTS.md obligations. Tree records written under the lax checker also violate the restored gate: 7 claims lack lifecycle annotations, roadmap_v1 stores GitHub issue table rows, 3 issue files reference non-canonical row IDs, 43 record citations are stale or broken, and the model-inventory constants predate the deliberate row closures.

## Resolution

-
Listed under `## Owed` in [`fix-test-agent-record-imports.md`](../../specs/fix-test-agent-record-imports.md)

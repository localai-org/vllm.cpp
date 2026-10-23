ID: ISSUE-LOCAL-01M3Q0N3Y2C29368C30S90JXGR
Title: The #2317 row and its record cite check-agent-record.py line 1973, retired by W6 - the citation gate answers that the line does not exist
Row: ENG-RECORD-CONFLICT-SURFACES
State: OPEN
Kind: record
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-29
Updated: 2026-09-29
Closed: -

## Problem

The frozen archive is a file of anchors, not only words: rows cite script lines, and tests/scripts/test_agent_record.py::test_every_tracked_checker_line_citation_still_resolves resolves every citation the tree carries against the tracked scripts. When a cited line moves or retires, the citation dangles and the gate goes red at the stale reference. While restoring the 27 archive rows (GATE-ISSUE-ARCHIVE-RESTORE), row 898 - the #2317 row for ENG-RECORD-CONFLICT-SURFACES - came back carrying scripts/check-agent-record.py line 1973, and the citation gate red on it the same hour. Measured at the archive tip: the checker is 1253 lines, so line 1973 is 745 past end-of-file, and the gate lists three subfailures for it - the archive row, plus the record Title and quote block of ISSUE-GH-2317, which quote it too. git show df024dce4:scripts/check-agent-record.py is 2254 lines and line 1972 opens INDEX_PREAMBLE, a multiline constant that pinned the archive preamble so it could not drift without a deliberate edit on both sides - the row was true when written; 1973 is the second line of that constant. W6 (7dc2ef1ea) retired the live index and the constant with it: the checker today has no INDEX_PREAMBLE and no union prose, the preamble sentence lives in the archive itself, and nothing gates it. The correct target does not exist, so a re-point would lie. Fixed in the same flow, red-first against the citation gate: all three instances reworded to keep the factual claim and retire the dead anchor - the sentence now says the expected preamble was once frozen in the checker and was retired with the index in W6, with no line number. No citation-shaped spelling enters the text, because the gate also resolves ranges, and the repair is one sentence on both sides so the quote stays byte-equal to the row.

## Resolution

-

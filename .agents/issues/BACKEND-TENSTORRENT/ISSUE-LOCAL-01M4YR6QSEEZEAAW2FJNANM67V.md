ID: ISSUE-LOCAL-01M4YR6QSEEZEAAW2FJNANM67V
Title: Qualify the historical Kolibri-1 TT capture study
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: docs
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-15
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

docs/gap-kolibri1-tt-captured-decode.md presents nondeterministic prototype runs and their performance inference without noting the memo/residency fixes in ad8d9b45b or the golden indexing correction in 50d425bec. Preserve measurements as historical evidence and distinguish repeatability from oracle correctness.

## Resolution

2026-10-21: FALSIFIED BY THE TREE. The qualification landed as
`ba85e3878` ("docs: qualify historical Kolibri TT measurements (#3456)"),
which carries the spec `.agents/specs/docs-kolibri-tt-study.md` with its
"Implementation evidence (15 October 2026)" section.

The study now names what this issue asks it to name.
`docs/gap-kolibri1-tt-captured-decode.md` opens with a "Status after the 12
October 2026 fixes" section: it states that the study records the 11 October
prototype, before the decode fixes in `ad8d9b45b` and the golden-index
correction in `50d425bec`; it links the follow-up record that attributes the
nondeterminism to memo eviction and stale weight-view shadows; and it
separates repeatability from correctness -- "Repeatability does not establish
oracle correctness. After the golden-index correction, the recorded gate still
fails in both arms, with two identical runs per arm. The timings below remain
historical measurements, not current performance evidence."

The three timing rows stay byte-identical to the spec's base, and the spec
records its source checks against `kolibri1_tt_forward.cpp:810-822`,
`tenstorrent_residency.cpp:665-693`, and `test_kolibri1_tt_b2ii.cpp:1137-1147`
with the corrected teacher-forced prefix at `:1233-1241`.


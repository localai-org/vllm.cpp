ID: ISSUE-GH-230
Title: `test_agent_record.py`: 7 issue-intake tests never run, and error when they do
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: 230
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-07
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

The frozen archive entry quoted below records the claim this issue made:
`tests/scripts/test_agent_record.py` carried seven issue-intake tests that
never ran, and that errored when they were run by hand. The GitHub copy is
not readable from this checkout (`GET repos/localai-org/vllm.cpp/issues/230`
returns 404), so the archived line is the only surviving statement of the
claim.

Archive: `.agents/completed/issue-index.md:168`

### Frozen archive evidence

> | [#230](https://github.com/mudler/vllm.cpp/issues/230) | — | `test_agent_record.py`: 7 issue-intake tests never run, and error when they do | bug |

## Resolution

2026-10-21: FALSIFIED BY THE TREE, and the tree said so before this
session. Commit `bd4b7b60e` (2026-09-20) moved this record out of
`_intake` into `BACKEND-TENSTORRENT` with the falsification in its own
body: "The tree falsifies it: tests/scripts/test_agent_record.py exists and
runs green (104/104), including the intake tests the issue said never ran."
The record never received that resolution, so the claim stayed open against
a tree that had already answered it.

Re-verified on this tree (`origin/main` at `3ce14f35c`, 2026-10-21):
`python3 -m unittest tests.scripts.test_agent_record.CanonicalIssueReferenceTests -v`
runs 10 tests, 10 OK, 0 failures. The three `_intake` ownership cases the
issue names run and pass:
`test_git_discovery_includes_tests_and_rejects_intake_ownership`,
`test_every_reference_form_to_intake_fails_canonical_ownership`, and
`test_intake_debt_is_reported_without_a_baseline_or_ratchet`. The class is a
plain `unittest.TestCase` whose methods carry the `test_` prefix, so the
runner collects it. Both halves of the claim -- "never run" and "error when
they do" -- no longer hold.

The record is promoted from `METADATA_ONLY` to `FULL` in this change so the
closure carries a canonical Problem: the frozen archive entry stays quoted
below, and the prose above it only restates that entry.


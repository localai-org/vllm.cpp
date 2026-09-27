ID: ISSUE-GH-1262
Title: `281b4bc76c0e` is on `main` carrying `Assisted-by: AGENT:claude-opus-5 CLI`, which is missing the bracketed `[TOOL]` the grammar requires. Measured with the checker itself from `origin/main` `27d5432f9`: `--range '281b4bc76~1..origin/main'` is `rc=1` with that one offender and `--range '281b4bc76..origin/main'` is `rc=0`, so the blast radius is one commit carrying one error. It CANNOT be repaired, because correcting a landed message rewrites `main`, and it does NOT clear itself: the main lane walks `LAST_GREEN..head` and `LAST_GREEN` advances only on a green run, so every later push re-walks a range that still contains it -- the same property `ci.yml:74` relies on to make a cancelled run lossless makes an unrepairable red permanent. The branch commits were all correct; `squash_merge_commit_message = PR_BODY` landed the pull request body, which still held the pre-repair value, and the guard that reads the body (`ci.yml:626-635`, #848) was `pending` at merge time because the runner pool was saturated -- it did not fail, it never ran. FIXED by an enumerated exception keyed on the full commit oid AND the exact rendered error string, printed on every run that applies it. `--cutover` was measured and REJECTED as the instrument: `--cutover 281b4bc76` does not excuse `281b4bc76` at all, because `merge-base --is-ancestor X X` succeeds and the cutover commit is checked strictly; naming its child does excuse it and drops 2986 ancestors to the marker-only check, waiving defects nobody has read; and it is a value that can be moved to hide the next red, which is the failure mode `AGENTS.md` §"Changing the rules or a checker" exists to prevent. Spec [`fix-trailer-lane-cutover.md`](../../specs/fix-trailer-lane-cutover.md)
Row: FIX-TRAILER-LANE-CUTOVER
State: UNKNOWN
Kind: bug
GitHub: 1262
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:415`

### Frozen archive evidence

> | [#1262](https://github.com/mudler/vllm.cpp/issues/1262) | `FIX-TRAILER-LANE-CUTOVER` | `281b4bc76c0e` is on `main` carrying `Assisted-by: AGENT:claude-opus-5 CLI`, which is missing the bracketed `[TOOL]` the grammar requires. Measured with the checker itself from `origin/main` `27d5432f9`: `--range '281b4bc76~1..origin/main'` is `rc=1` with that one offender and `--range '281b4bc76..origin/main'` is `rc=0`, so the blast radius is one commit carrying one error. It CANNOT be repaired, because correcting a landed message rewrites `main`, and it does NOT clear itself: the main lane walks `LAST_GREEN..head` and `LAST_GREEN` advances only on a green run, so every later push re-walks a range that still contains it -- the same property `ci.yml:74` relies on to make a cancelled run lossless makes an unrepairable red permanent. The branch commits were all correct; `squash_merge_commit_message = PR_BODY` landed the pull request body, which still held the pre-repair value, and the guard that reads the body (`ci.yml:626-635`, #848) was `pending` at merge time because the runner pool was saturated -- it did not fail, it never ran. FIXED by an enumerated exception keyed on the full commit oid AND the exact rendered error string, printed on every run that applies it. `--cutover` was measured and REJECTED as the instrument: `--cutover 281b4bc76` does not excuse `281b4bc76` at all, because `merge-base --is-ancestor X X` succeeds and the cutover commit is checked strictly; naming its child does excuse it and drops 2986 ancestors to the marker-only check, waiving defects nobody has read; and it is a value that can be moved to hide the next red, which is the failure mode `AGENTS.md` §"Changing the rules or a checker" exists to prevent. Spec [`fix-trailer-lane-cutover.md`](../../specs/fix-trailer-lane-cutover.md) | bug |

## Resolution

-

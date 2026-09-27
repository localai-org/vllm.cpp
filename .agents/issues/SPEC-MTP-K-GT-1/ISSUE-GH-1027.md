ID: ISSUE-GH-1027
Title: DUPLICATE of [#1022](https://github.com/mudler/vllm.cpp/issues/1022), and FIXED on `origin/main` by [#1025](https://github.com/mudler/vllm.cpp/pull/1025) before this row landed. Filed while merging `origin/main` into `SPEC-MTP-K-GT-1` ([#81](https://github.com/mudler/vllm.cpp/issues/81)): `.agents/issue-index.md` listed [#995](https://github.com/mudler/vllm.cpp/issues/995) TWICE on `origin/main` @ `45b022cdc`, because `332aed738` (#996) and `45b022cdc` (#997) each appended a row for the same issue without seeing the other and the path carries `merge=union`, so `check-agent-record` and `test_agent_record` were RED there and on every branch merging it. The filing claimed the file "cannot be made green by any edit to it" because `check-agent-record` refuses the duplicate while `check-issue-index-append-only` refuses the deletion that resolves it. **That premise was WRONG**, and the reason is worth keeping: `check-issue-index-append-only.py:47-56` diffs `merge-base(origin/main, HEAD)..HEAD`, not the file's own history, so the two checkers conflict only while the duplicate sits in the MERGE BASE. #1025 merged the two rows by key ON MAIN, which moved the merge base, and a branch that then takes main's version and appends removes nothing at all. Both checkers are green on one tree with neither weakened. The union driver still produces the WRONG result automatically here: `git merge-tree` against `origin/main` re-added the deleted row and yielded TWO `#995` rows again, which is why the keyed-record rule (take the complete target-branch version, re-apply the scoped edit) applies rather than the driver's output. Owed only a duplicate-close on GitHub, which this flow had no authority for; listed under `## Owed` in [`mtp-k-gt-1.md`](../../specs/mtp-k-gt-1.md)
Row: SPEC-MTP-K-GT-1
State: UNKNOWN
Kind: bug
GitHub: 1027
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:280`

### Frozen archive evidence

> | [#1027](https://github.com/mudler/vllm.cpp/issues/1027) | `SPEC-MTP-K-GT-1` | DUPLICATE of [#1022](https://github.com/mudler/vllm.cpp/issues/1022), and FIXED on `origin/main` by [#1025](https://github.com/mudler/vllm.cpp/pull/1025) before this row landed. Filed while merging `origin/main` into `SPEC-MTP-K-GT-1` ([#81](https://github.com/mudler/vllm.cpp/issues/81)): `.agents/issue-index.md` listed [#995](https://github.com/mudler/vllm.cpp/issues/995) TWICE on `origin/main` @ `45b022cdc`, because `332aed738` (#996) and `45b022cdc` (#997) each appended a row for the same issue without seeing the other and the path carries `merge=union`, so `check-agent-record` and `test_agent_record` were RED there and on every branch merging it. The filing claimed the file "cannot be made green by any edit to it" because `check-agent-record` refuses the duplicate while `check-issue-index-append-only` refuses the deletion that resolves it. **That premise was WRONG**, and the reason is worth keeping: `check-issue-index-append-only.py:47-56` diffs `merge-base(origin/main, HEAD)..HEAD`, not the file's own history, so the two checkers conflict only while the duplicate sits in the MERGE BASE. #1025 merged the two rows by key ON MAIN, which moved the merge base, and a branch that then takes main's version and appends removes nothing at all. Both checkers are green on one tree with neither weakened. The union driver still produces the WRONG result automatically here: `git merge-tree` against `origin/main` re-added the deleted row and yielded TWO `#995` rows again, which is why the keyed-record rule (take the complete target-branch version, re-apply the scoped edit) applies rather than the driver's output. Owed only a duplicate-close on GitHub, which this flow had no authority for; listed under `## Owed` in [`mtp-k-gt-1.md`](../../specs/mtp-k-gt-1.md) | bug |

## Resolution

-

ID: ISSUE-GH-1263
Title: No operator-side check reads the pull request BODY before a squash merge. The CI guard that does (`ci.yml:626-635`, #848) runs the same checker over `$PR_BODY` with `--filled`, but it is not a precondition of merging: on #1257 it was still `pending` when the merge went ahead, so the malformed body of [#1262](https://github.com/mudler/vllm.cpp/issues/1262) landed unread. What is missing is one local call, `gh pr view <N> --json body --jq .body | check-commit-trailers.py --message-file - --filled`, in the operator's own shell rather than in a queue. NOT fixed in the #1262 flow because it changes an operator procedure and adds a gate command rather than a checker rule, so it owes its own red-first evidence and its own reviewer, and because the lane clears without it. Owned under `## Owed` in [`fix-trailer-lane-cutover.md`](../../specs/fix-trailer-lane-cutover.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1263
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:416`

### Frozen archive evidence

> | [#1263](https://github.com/mudler/vllm.cpp/issues/1263) | — | No operator-side check reads the pull request BODY before a squash merge. The CI guard that does (`ci.yml:626-635`, #848) runs the same checker over `$PR_BODY` with `--filled`, but it is not a precondition of merging: on #1257 it was still `pending` when the merge went ahead, so the malformed body of [#1262](https://github.com/mudler/vllm.cpp/issues/1262) landed unread. What is missing is one local call, `gh pr view <N> --json body --jq .body \| check-commit-trailers.py --message-file - --filled`, in the operator's own shell rather than in a queue. NOT fixed in the #1262 flow because it changes an operator procedure and adds a gate command rather than a checker rule, so it owes its own red-first evidence and its own reviewer, and because the lane clears without it. Owned under `## Owed` in [`fix-trailer-lane-cutover.md`](../../specs/fix-trailer-lane-cutover.md) | bug |

## Resolution

-

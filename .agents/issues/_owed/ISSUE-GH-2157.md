ID: ISSUE-GH-2157
Title: `scripts/check-commit-trailers.py:463` walks `rev-list --reverse base..head` with no `--no-merges` and no parent-count test, so a plain `git merge origin/main` on a task branch reds `commit-protocol-tag` with three findings per merge commit — while the SAME job skips merge commits 50 lines earlier (`.github/workflows/ci.yml:873`, "they are not authored content"), so one job carries two opposite rules and only one of them is written down. Measured: `--range a0f12b727..d05723f8e` is rc=1 with 9 findings across 3 merge OIDs, while the same range's 3 NON-merge commits all pass, which is the isolating control; CI agrees on PR #2134 (job 98706339787) over `0d8962500cc1`, two parents and a 0-byte body. Nothing reaches `main`: `squash_merge_commit_message = PR_BODY` means a branch merge commit never becomes a landed message, so the cost is a red gate plus a forced branch rewrite on every branch that syncs — which AGENTS.md § Landing work instructs as the routine response to a rejected push. AGENTS.md is SILENT on merge commits (`grep -rn 'merges included'` returns nothing against a positive control returning 33), so this is a gap rather than a policy. NOT FIXED: changing the walk is a semantic gate change owing its own row, spec, red-before and green-after, and the developer chose on 2026-08-28 to authorize `row/*` force-push instead. #1136 (CLOSED) records the same mechanism as one PR's review finding and owns no repair; #581 is the forge's merges on `main`; #467 is preflight not running the checker; #406 is trailer-block LOCATION and leaves this shape red on purpose. Owed under `## Owed` in [`fix-trailer-lane-cutover.md`](../../specs/fix-trailer-lane-cutover.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 2157
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:855`

### Frozen archive evidence

> | [#2157](https://github.com/mudler/vllm.cpp/issues/2157) | — | `scripts/check-commit-trailers.py:463` walks `rev-list --reverse base..head` with no `--no-merges` and no parent-count test, so a plain `git merge origin/main` on a task branch reds `commit-protocol-tag` with three findings per merge commit — while the SAME job skips merge commits 50 lines earlier (`.github/workflows/ci.yml:873`, "they are not authored content"), so one job carries two opposite rules and only one of them is written down. Measured: `--range a0f12b727..d05723f8e` is rc=1 with 9 findings across 3 merge OIDs, while the same range's 3 NON-merge commits all pass, which is the isolating control; CI agrees on PR #2134 (job 98706339787) over `0d8962500cc1`, two parents and a 0-byte body. Nothing reaches `main`: `squash_merge_commit_message = PR_BODY` means a branch merge commit never becomes a landed message, so the cost is a red gate plus a forced branch rewrite on every branch that syncs — which AGENTS.md § Landing work instructs as the routine response to a rejected push. AGENTS.md is SILENT on merge commits (`grep -rn 'merges included'` returns nothing against a positive control returning 33), so this is a gap rather than a policy. NOT FIXED: changing the walk is a semantic gate change owing its own row, spec, red-before and green-after, and the developer chose on 2026-08-28 to authorize `row/*` force-push instead. #1136 (CLOSED) records the same mechanism as one PR's review finding and owns no repair; #581 is the forge's merges on `main`; #467 is preflight not running the checker; #406 is trailer-block LOCATION and leaves this shape red on purpose. Owed under `## Owed` in [`fix-trailer-lane-cutover.md`](../../specs/fix-trailer-lane-cutover.md) | bug |

## Resolution

-

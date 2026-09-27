ID: ISSUE-GH-2184
Title: Four `ci.yml` line anchors in [`fix-trailer-lane-cutover.md`](../../specs/fix-trailer-lane-cutover.md) resolved to code that does not support the sentence citing them, because the strict trailer walk MOVED from `agent-record` (#863) to `commit-protocol-tag` (`ci.yml:818`) after the spec was written: the prose survived the move and the numbers did not. `:596-623` landed on a GPU-mutex comment and `pending_args`; `:626-635` on a bare `fi` and a `--pr-base` continuation. NOT a guess to repair — spec line 44 names its own job in the sentence ("in `commit-protocol-tag`"), so it is right-job/wrong-number, and the other two claims each map to a unique construct. Repointed at `6f02680bb` to `:899-927` (LAST_GREEN `:899`, base `:917-920`, walk `:927`), `:935-939` (the `--filled` body guard), `:924`/`:927` (the only two `--range`-alone calls) and `agent-integration.py:106-110` (tightened; `--cutover` is on `:108`). No checker can see this class: `check-symbol-anchors` resolves SYMBOLS, not whether a line range supports a claim, so the pointer lands on plausible code and the reader finds nothing to contradict them. Fixed in flow in PR #2159
Row: FIX-TRAILER-LANE-CUTOVER
State: UNKNOWN
Kind: bug
GitHub: 2184
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:856`

### Frozen archive evidence

> | [#2184](https://github.com/mudler/vllm.cpp/issues/2184) | `FIX-TRAILER-LANE-CUTOVER` | Four `ci.yml` line anchors in [`fix-trailer-lane-cutover.md`](../../specs/fix-trailer-lane-cutover.md) resolved to code that does not support the sentence citing them, because the strict trailer walk MOVED from `agent-record` (#863) to `commit-protocol-tag` (`ci.yml:818`) after the spec was written: the prose survived the move and the numbers did not. `:596-623` landed on a GPU-mutex comment and `pending_args`; `:626-635` on a bare `fi` and a `--pr-base` continuation. NOT a guess to repair — spec line 44 names its own job in the sentence ("in `commit-protocol-tag`"), so it is right-job/wrong-number, and the other two claims each map to a unique construct. Repointed at `6f02680bb` to `:899-927` (LAST_GREEN `:899`, base `:917-920`, walk `:927`), `:935-939` (the `--filled` body guard), `:924`/`:927` (the only two `--range`-alone calls) and `agent-integration.py:106-110` (tightened; `--cutover` is on `:108`). No checker can see this class: `check-symbol-anchors` resolves SYMBOLS, not whether a line range supports a claim, so the pointer lands on plausible code and the reader finds nothing to contradict them. Fixed in flow in PR #2159 | bug |

## Resolution

-

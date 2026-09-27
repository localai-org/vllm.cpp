ID: ISSUE-GH-911
Title: A `file:line` anchor into a file the row is ITSELF editing is stale by default, and spec BODIES are checked by nothing: `check-agent-record.py`'s `MATRIX_PATHS` (`:521`, `:529-530`) covers the five matrices, `feature-matrix.md` and `specs/model-family-inventory.md`, leaving 4772 line-carrying citations across 315 `.agents/specs/*.md` unexamined (positive control: 2314 line-less `.cpp` mentions match the same shape). `ltx25-prompt-adaln.md` shipped EIGHT stale repo-local anchors across two repair commits, moved by its own `020381676` and by `98f8e046d` (#658), then SEVEN more that were correct at `00613767d` and wrong at the merge of `origin/main`, because `0785cfc4d` (#882) added 70 lines to `ltx2_video.cpp` and 306 to `test_ltx2_video.cpp` ahead of every one. The obvious checker is a TAUTOLOGY — reading the span out of the file it validates reports 27 of 27 fresh on the same tree where reading the spans against their CLAIMS finds seven stale. Remedy is already in use and unwritten: `path:NN @ <sha>` for a historical claim, claim-sourced uniqueness re-derivation for a live one, re-run after the merge. Narrower than [#632](https://github.com/mudler/vllm.cpp/issues/632) on surface and sharper on mechanism. Listed under `## Owed` in [`ltx25-prompt-adaln.md`](../../specs/ltx25-prompt-adaln.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 911
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:243`

### Frozen archive evidence

> | [#911](https://github.com/mudler/vllm.cpp/issues/911) | — | A `file:line` anchor into a file the row is ITSELF editing is stale by default, and spec BODIES are checked by nothing: `check-agent-record.py`'s `MATRIX_PATHS` (`:521`, `:529-530`) covers the five matrices, `feature-matrix.md` and `specs/model-family-inventory.md`, leaving 4772 line-carrying citations across 315 `.agents/specs/*.md` unexamined (positive control: 2314 line-less `.cpp` mentions match the same shape). `ltx25-prompt-adaln.md` shipped EIGHT stale repo-local anchors across two repair commits, moved by its own `020381676` and by `98f8e046d` (#658), then SEVEN more that were correct at `00613767d` and wrong at the merge of `origin/main`, because `0785cfc4d` (#882) added 70 lines to `ltx2_video.cpp` and 306 to `test_ltx2_video.cpp` ahead of every one. The obvious checker is a TAUTOLOGY — reading the span out of the file it validates reports 27 of 27 fresh on the same tree where reading the spans against their CLAIMS finds seven stale. Remedy is already in use and unwritten: `path:NN @ <sha>` for a historical claim, claim-sourced uniqueness re-derivation for a live one, re-run after the merge. Narrower than [#632](https://github.com/mudler/vllm.cpp/issues/632) on surface and sharper on mechanism. Listed under `## Owed` in [`ltx25-prompt-adaln.md`](../../specs/ltx25-prompt-adaln.md) | bug |

## Resolution

-

ID: ISSUE-GH-829
Title: A squash of a multi-commit PR repeats the trailer block once per commit; the repetition is harmless because `git interpret-trailers --parse` reads only the trailing block, and the real defect is GitHub's `---------` separator falling between the block and the appended `Co-authored-by:`, which orphans it. Fixed by setting `squash_merge_commit_message = PR_BODY` (spec [`squash-trailers.md`](../../specs/squash-trailers.md))
Row: GATE-SQUASH-TRAILERS
State: UNKNOWN
Kind: bug
GitHub: 829
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:209`

### Frozen archive evidence

> | [#829](https://github.com/mudler/vllm.cpp/issues/829) | `GATE-SQUASH-TRAILERS` | A squash of a multi-commit PR repeats the trailer block once per commit; the repetition is harmless because `git interpret-trailers --parse` reads only the trailing block, and the real defect is GitHub's `---------` separator falling between the block and the appended `Co-authored-by:`, which orphans it. Fixed by setting `squash_merge_commit_message = PR_BODY` (spec [`squash-trailers.md`](../../specs/squash-trailers.md)) | bug |

## Resolution

-

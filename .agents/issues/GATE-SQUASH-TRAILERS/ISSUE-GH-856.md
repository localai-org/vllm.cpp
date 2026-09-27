ID: ISSUE-GH-856
Title: `pr-size` aborts on every pull request: `.agents/issue-index.md` was never added to `PROJECT_RECORD_FILES` when #840 created it, and classification is a hard error by design, so every PR appending an index row fails the gate (spec [`squash-trailers.md`](../../specs/squash-trailers.md))
Row: GATE-SQUASH-TRAILERS
State: UNKNOWN
Kind: bug
GitHub: 856
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:212`

### Frozen archive evidence

> | [#856](https://github.com/mudler/vllm.cpp/issues/856) | `GATE-SQUASH-TRAILERS` | `pr-size` aborts on every pull request: `.agents/issue-index.md` was never added to `PROJECT_RECORD_FILES` when #840 created it, and classification is a hard error by design, so every PR appending an index row fails the gate (spec [`squash-trailers.md`](../../specs/squash-trailers.md)) | bug |

## Resolution

-

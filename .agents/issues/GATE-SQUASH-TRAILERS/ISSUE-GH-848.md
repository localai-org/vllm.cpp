ID: ISSUE-GH-848
Title: `PR_BODY` makes the pull request body the landed commit message, and nothing validated it: `.github/pull_request_template.md` carried no trailer block, so a PR opened from the template lands a commit with no trailers at all, caught only after the merge (spec [`squash-trailers.md`](../../specs/squash-trailers.md))
Row: GATE-SQUASH-TRAILERS
State: UNKNOWN
Kind: bug
GitHub: 848
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:210`

### Frozen archive evidence

> | [#848](https://github.com/mudler/vllm.cpp/issues/848) | `GATE-SQUASH-TRAILERS` | `PR_BODY` makes the pull request body the landed commit message, and nothing validated it: `.github/pull_request_template.md` carried no trailer block, so a PR opened from the template lands a commit with no trailers at all, caught only after the merge (spec [`squash-trailers.md`](../../specs/squash-trailers.md)) | bug |

## Resolution

-

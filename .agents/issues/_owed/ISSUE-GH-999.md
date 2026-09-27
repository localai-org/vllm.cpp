ID: ISSUE-GH-999
Title: `scripts/check-commit-style.py` `validate_range` still raises `range base must be an ancestor of range head`, so the merge-base repair `GATE-FORK-ANCESTRY` (#773) applied to `check-commit-trailers.py` never reached it: the checker aborts before reading a commit on any branch cut before the last merge of `main`. Found auditing the guards for #998, and it is why that row reports a SKIP rather than dropping the guard and letting both checkers run. Owed by [`gate-preflight-skip-report.md`](../../specs/gate-preflight-skip-report.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 999
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:276`

### Frozen archive evidence

> | [#999](https://github.com/mudler/vllm.cpp/issues/999) | — | `scripts/check-commit-style.py` `validate_range` still raises `range base must be an ancestor of range head`, so the merge-base repair `GATE-FORK-ANCESTRY` (#773) applied to `check-commit-trailers.py` never reached it: the checker aborts before reading a commit on any branch cut before the last merge of `main`. Found auditing the guards for #998, and it is why that row reports a SKIP rather than dropping the guard and letting both checkers run. Owed by [`gate-preflight-skip-report.md`](../../specs/gate-preflight-skip-report.md) | bug |

## Resolution

-

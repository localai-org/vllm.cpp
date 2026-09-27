ID: ISSUE-GH-726
Title: `audit-live-rows --check` calls every in-flight row ABANDONED in CI: the `agent-record` job fetches only `main`, so no `row/*` ref exists and the `IN-FLIGHT` verdict is unreachable — any PR moving a row to `ACTIVE` before its code lands fails deterministically, spec [`gate-audit-branch-evidence.md`](../../specs/gate-audit-branch-evidence.md)
Row: GATE-AUDIT-BRANCH-EVIDENCE
State: UNKNOWN
Kind: bug
GitHub: 726
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:76`

### Frozen archive evidence

> | [#726](https://github.com/mudler/vllm.cpp/issues/726) | `GATE-AUDIT-BRANCH-EVIDENCE` | `audit-live-rows --check` calls every in-flight row ABANDONED in CI: the `agent-record` job fetches only `main`, so no `row/*` ref exists and the `IN-FLIGHT` verdict is unreachable — any PR moving a row to `ACTIVE` before its code lands fails deterministically, spec [`gate-audit-branch-evidence.md`](../../specs/gate-audit-branch-evidence.md) | bug |

## Resolution

-

ID: ISSUE-GH-998
Title: `scripts/agent-preflight.sh` prints `All gates green.` while a guarded block never ran: the trailer block is gated on `git merge-base --is-ancestor origin/main HEAD`, and when that is false the two gates vanish from the report with no output. Fired three times in one session, twice because `origin/main` is a remote-tracking ref every worktree of the checkout shares and another worktree advanced it MID-RUN, so the `ok` count fell from 76 to 74 and the banner did not change. Same shape in the `Committed range` block, whose unresolvable-ref arm drops three more gates. A green banner over a block that never executed is a FALSE REPORT, which the file's own `audit-live-rows` comment already calls "the one unacceptable outcome", spec [`gate-preflight-skip-report.md`](../../specs/gate-preflight-skip-report.md)
Row: GATE-PREFLIGHT-SKIP-REPORT
State: UNKNOWN
Kind: bug
GitHub: 998
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:275`

### Frozen archive evidence

> | [#998](https://github.com/mudler/vllm.cpp/issues/998) | `GATE-PREFLIGHT-SKIP-REPORT` | `scripts/agent-preflight.sh` prints `All gates green.` while a guarded block never ran: the trailer block is gated on `git merge-base --is-ancestor origin/main HEAD`, and when that is false the two gates vanish from the report with no output. Fired three times in one session, twice because `origin/main` is a remote-tracking ref every worktree of the checkout shares and another worktree advanced it MID-RUN, so the `ok` count fell from 76 to 74 and the banner did not change. Same shape in the `Committed range` block, whose unresolvable-ref arm drops three more gates. A green banner over a block that never executed is a FALSE REPORT, which the file's own `audit-live-rows` comment already calls "the one unacceptable outcome", spec [`gate-preflight-skip-report.md`](../../specs/gate-preflight-skip-report.md) | bug |

## Resolution

-

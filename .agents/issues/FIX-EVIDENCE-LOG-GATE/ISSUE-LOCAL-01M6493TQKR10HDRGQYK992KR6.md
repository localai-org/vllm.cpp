ID: ISSUE-LOCAL-01M6493TQKR10HDRGQYK992KR6
Title: Evidence dirs silently lose their raw logs: global *.log gitignore drops them at git add, no gate checks evidence contents
Row: FIX-EVIDENCE-LOG-GATE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

.gitignore:34 ignores *.log globally, so every bench-evidence dir that captured raw logs committed only its README/report: git add silently skipped the logs and no gate checks evidence-dir contents. Confirmed on main 2026-10-29: TT-KOLIBRI1-READBACK, FIX-PAGED-REGRESSION, and FIX-TT-NUMA-DEGRADE evidence dirs contain no raw logs, and their worktrees were removed post-merge, taking the untracked logs with them — the raw measurement records for three landed rows are unrecoverable (the reviews verified the logs at review time). Found by the FLINT-CPU-PARITY reviewer (report lists 9 logs, zero in tree); that row's logs survived and are being rescued onto its branch. Spec: .agents/specs/fix-evidence-log-gate.md. Design: whitelist docs/bench-evidence/**/*.log + a checker (check-evidence-refs.py, agent-record lane + preflight) that fails when an evidence README references a file missing from the tree. Red-before: the checker fails on main today.

## Resolution

-

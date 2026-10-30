ID: ISSUE-LOCAL-01M6493TQKR10HDRGQYK992KR6
Title: Evidence dirs silently lose their raw logs: global *.log gitignore drops them at git add, no gate checks evidence contents
Row: FIX-EVIDENCE-LOG-GATE
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: 2026-10-29

## Problem

.gitignore:34 ignores *.log globally, so every bench-evidence dir that captured raw logs committed only its README/report: git add silently skipped the logs and no gate checks evidence-dir contents. Confirmed on main 2026-10-29: TT-KOLIBRI1-READBACK, FIX-PAGED-REGRESSION, and FIX-TT-NUMA-DEGRADE evidence dirs contain no raw logs, and their worktrees were removed post-merge, taking the untracked logs with them — the raw measurement records for three landed rows are unrecoverable (the reviews verified the logs at review time). Found by the FLINT-CPU-PARITY reviewer (report lists 9 logs, zero in tree); that row's logs survived and are being rescued onto its branch. Spec: .agents/specs/fix-evidence-log-gate.md. Design: whitelist docs/bench-evidence/**/*.log + a checker (check-evidence-refs.py, agent-record lane + preflight) that fails when an evidence README references a file missing from the tree. Red-before: the checker fails on main today.

## Resolution

Landed 2026-10-29 as 007c67b6a (PR #3522, squash of row/FIX-EVIDENCE-LOG-GATE). The whitelist !docs/bench-evidence/**/*.log is in .gitignore (verified on git 2.54: nested evidence logs tracked, root/scratch logs still ignored) and scripts/check-evidence-refs.py is wired into the agent-record CI job (ci.yml:704) and preflight CHECKERS/SUITES. Red-before: 46 entries / 41 distinct filenames across NINE evidence dirs referencing untracked logs (the reviewer corrected the implementer's 47). Review MERGEABLE (6 claims; M1/M2/M4/M6 RED prove the checker, whitelist, and tracked-check load-bearing; M5 characterized the CI-wiring gap -- recorded debt, candidate wiring-coverage checker). Evidence loss notes dated 2026-10-30 on nine dirs; the raw logs for the pre-gate rows are unrecoverable (recorded, not papered over).

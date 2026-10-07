ID: ISSUE-LOCAL-01M4AGYRTVCYJNKYMVKEPF8VX6
Title: VT_KOLIBRI1_PROFILE env var is undocumented (env-doc gate red on row/kolibri-perf)
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-07
Updated: 2026-10-07
Closed: -

## Problem

Commit f45d4cd33 (perf(kolibri-1): thread the fp8-block dequant across the CPU pool and profile the forward) added the VT_KOLIBRI1_PROFILE stage profiler to src/vllm/model_executor/models/kolibri1_forward.cpp (read at line 85) without documenting it in docs/ENVIRONMENT.md or listing it in scripts/env-doc-allowlist.txt. scripts/check-env-doc.py and tests/scripts/test_check_env_doc.py therefore fail on this branch (found while running the agent-preflight gate during the test_kolibri1_dequant scale-grid repair; the failure reproduces at HEAD bc7a9264c with the repair stashed). The pre-existing stale gitignored build/ directory in the worktree additionally inflated the FetchContent site count in test_check_test_registration (6 vs 3 allowlisted sites) and has been moved out of the worktree.

## Resolution

-

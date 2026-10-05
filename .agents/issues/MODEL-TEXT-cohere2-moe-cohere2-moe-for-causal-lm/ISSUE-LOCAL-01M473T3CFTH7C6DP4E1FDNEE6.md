ID: ISSUE-LOCAL-01M473T3CFTH7C6DP4E1FDNEE6
Title: main-level preflight strata: the enumerated gates red on pristine main
Row: MODEL-TEXT-cohere2-moe-cohere2-moe-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-06
Updated: 2026-10-06
Closed: -

## Problem

Several gates red on pristine origin/main redden every PR: (1) test_gpu_lock_one_truth::test_coordination_names_the_one_lock_and_not_the_retired_one — .agents/coordination.md's GPU scheduling section still named the retired flock /tmp/gpu mutex and never the canonical GPU_LOCK default truth; (2) audit-live-rows --check — this row is ACTIVE with its work landed on main (651ac08ee, 9f619494e, 1d36ebc8d) but no commit message or row/<ID> branch carries the full row ID, so the audit reads the row as ABANDONED and the gate goes red on pristine main; the record repair cites the landed commits; (3) test_rocprof_attach_preflight — setUpClass compiles the pinned rocm-systems x86-only ptrace controller (PTRACE_GETREGS/SETREGS, user_regs_struct rax/rsp/rip) and fails to build on aarch64 hosts, erroring the whole class instead of skipping with a named reason; (4) role-undeclared — the preflight role gate is ON by default and a session without scripts/agent-role.py claim is a red gate; the fix there is declaring a role, not weakening the gate. check-oracle-pins verified green on current origin/main (21 oracles pinned).

## Resolution

-

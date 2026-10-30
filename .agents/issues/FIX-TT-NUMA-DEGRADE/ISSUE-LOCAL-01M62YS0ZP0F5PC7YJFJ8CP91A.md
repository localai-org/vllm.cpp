ID: ISSUE-LOCAL-01M62YS0ZP0F5PC7YJFJ8CP91A
Title: kolibri1 NUMA interleave refusal reds every test that constructs PolicyGuard on restricted runners
Row: FIX-TT-NUMA-DEGRADE
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: 2026-10-29

## Problem

CI runners (ubuntu image rollout) refuse set_mempolicy(MPOL_INTERLEAVE) -- kolibri1_numa.h:142 VT_CHECK throws, reding test_kolibri1, test_kolibri1_bf16_arm, test_kolibri1_w2, test_kolibri1_numa (CI run 40712283192 job 122900454192, first seen 2026-10-28T05:37Z run 40639035531). Approved semantics (developer, 2026-10-28): warn + default-policy fallback. Knob: unset = best-effort fallback, 0 = off, 1 = strict (throw by name). Spec: .agents/specs/fix-tt-numa-degrade.md

## Resolution

Landed 2026-10-29 as db1d71c9b (PR #3519, squash of row/FIX-TT-NUMA-DEGRADE). The guard degrades: set_mempolicy(MPOL_INTERLEAVE) refusal now emits one named warning (errno + 'interleave refused by the host; continuing with the default policy') and constructs a no-op guard; knob VT_KOLIBRI1_NUMA_INTERLEAVE is three-valued (unset/other = best-effort, 0 = off, 1 = strict throw-by-name); the dtor never touches the policy after a fallback (mutation-probed: a restoring dtor would issue writes from knob-off inert guards and clobber an outer active guard). Four suites green under forced SetMempolicy failure with exactly one warning per guard (4/63/1420/4); strict arm throws the old message by name. Review MERGEABLE (6 claims, 5 mutations). Recorded gap: the fallback warning is enforced by count evidence only — no in-tree gate captures stderr (candidate for a stderr-count gate).

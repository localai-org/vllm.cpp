# SPEC — FIX-TT-NUMA-DEGRADE — degrade the kolibri1 NUMA interleave refusal on hosts that refuse set_mempolicy

- Owning row: FIX-TT-NUMA-DEGRADE
- Kind: bug (environment-class CI red, product behavior change gated behind an env knob)
- Issue: created in the same flow (row-scoped local issue)

## Problem

The ubuntu runner image rollout changed what the CI sandbox permits:
`set_mempolicy(MPOL_INTERLEAVE)` is refused on the runners. Every test that
constructs `kolibri1_numa::PolicyGuard` throws at
`src/vllm/model_executor/models/kolibri1_numa.h:142` and reds —
`test_kolibri1`, `test_kolibri1_bf16_arm`, `test_kolibri1_w2`,
`test_kolibri1_numa` (CI run 40712283192, job 122900454192; first seen
2026-10-28T05:37Z run 40639035531). This is the same environment-class as the
/tmp-tmpfs finding: the tree is fine, the ground moved.

## Approved semantics (developer, 2026-10-28)

Warn + default-policy fallback — NOT refusal, NOT a silent skip:

1. `PolicyGuard` ctor: when `SetMempolicy(MPOL_INTERLEAVE)` fails, emit one
   named warning to stderr (the errno, the phrase `interleave refused by the
   host; continuing with the default policy`, and the knob name) and construct
   as a no-op guard (`active_ = false`). Nothing to restore (the failed call
   changed nothing), so the dtor must not touch the policy.
2. Knob semantics (`VT_KOLIBRI1_NUMA_INTERLEAVE`):
   - unset → best-effort: try interleave, fall back with the warning (NEW
     default; this is what fixes CI);
   - `0` → off (unchanged);
   - `1` → strict: require interleave, throw by name on refusal (the old
     behavior, for a host that wants the guarantee). Anyone setting a value
     other than `0`/`1` today gets best-effort; document it.
3. A test may assert the degraded mode (e.g. the router bias cache still
   serves, results identical) but must SKIP with a named message when the
   host actually granted interleave (nothing to test) — a skip that says why,
   never a vacuous pass.

## Scope

`kolibri1_numa.h` only (the guard + the knob), plus the tests that construct
the guard, plus `docs/ENVIRONMENT.md` (env-doc checker owns a row for the
changed knob semantics). No allocation-path change: when the fallback fires,
the code runs exactly the pre-NUMA-row path (default policy).

## Red-before

CI: the four named suites red on runners (evidence above). Local reproduction:
not available (this host grants interleave) — the red-before is the CI record;
the mutation below is the local proof the gate is load-bearing.

## Tests

- Local green: all four suites pass on this host (which grants interleave) —
  they exercise the strict path.
- Mutation (reviewer's obligation, implementer captures once): patch
  `SetMempolicy` to fail unconditionally in a scratch copy → the suites must
  still pass, each printing the named warning exactly once per guard (a count
  of zero = the warning is unreachable; more than one per guard = noisy).
  Restore byte-for-byte.
- The `1` strict arm: with the env at `1` and the patched-failing
  `SetMempolicy`, the guard must throw by name (the old message).

## Gates

check-agent-record, commit style, trailers, env-doc checker, preflight (named
host dotnet failure). CI: the four suites green on the PR.

## Stop conditions

If the fallback fires but results diverge from the strict arm (they must not —
the policy affects page placement, not arithmetic), STOP: that would mean a
hidden dependency on interleave in the numerics, which is a different bug.

## Outcome

Landed 2026-10-29 as db1d71c9b (PR #3519). The knob is three-valued (unset/other = best-effort fallback with one named warning; "0" = off; "1" = strict throw-by-name); the dtor never touches the policy after a fallback. Verified by review (6 claims, 5 mutations): identical assertion counts under forced fallback (5/12498/1608/234), exactly one warning per guard (4/63/1420/4), the strict arm throws the old message, and the M3 probe shows a restoring dtor would write policy from knob-off inert guards (9 set_mempolicy calls vs 4) — active_=false is load-bearing. The doctest 2.5.2 SUBCASE finding (a return silently suppresses sibling SUBCASEs) is recorded in the evidence README — it shaped the TEST_CASE split. Left open: the fallback warning is enforced by count evidence only (no stderr-capturing gate — candidate); check-env-doc enforces presence, not semantics drift (review-enforced). Both ubuntu-runner-class reds (tmpfs, set_mempolicy) now carry the same lesson: gate the environment, name the fallback.

## Git integration

One pull request, row/FIX-TT-NUMA-DEGRADE, spec first.

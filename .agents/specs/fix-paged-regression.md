# SPEC — `FIX-PAGED-REGRESSION` — repair the CPU paged-attention regression from the kolibri1 kernel batch

- Owning row: `FIX-PAGED-REGRESSION`
- Kind: bug (correctness regression on main)
- Issue: created in the same flow (row-scoped local issue)

## Problem

`test_glm4_moe_lite_paged_engine` and `test_ops_paged_attn_dtype` fail on
`main`. CI bracket:

- run 39200285700 (2026-10-19T04:36Z): glm4 failing (1 failure)
- run 39278997419 (2026-10-19T15:34Z): both failing (2 failures)

Between those: the 13:50Z batch `2cd6b8a14` (FMA-contract the paged-attention
scalar body), `3c4925def` (BFDOT K-dot arm), `6be7f0214`, `fa373385e`,
`c975549b0`, `d63fb81fc` — several committed with `draft(...)` subjects.
`61d333e52` (bf16-dot as the aarch64 default) carries an author date of
17:39Z the same day. Several of these landed with the goldens already red.

Failure evidence (run 40712283192, job 122900454192):

- `test_ops_paged_attn_dtype.cpp:304`: the hoisted dtype dispatch is NOT
  bit-identical — "decode c1 [q=f32 kv=f32 out=f32] first differing storage
  byte at index 0: got 3 want 4", likewise f16/bf16 KV arms.
- `test_glm4_moe_lite_paged_engine.cpp:226`: 69/128 exact positions against
  a STRICT oracle that licenses 128/128 with teacher-forced gap 0 mnats;
  1/8 prompts token-exact.

Also on the same batch's authority: the FMA contraction changes summation
order in the SHARED paged kernel — the leading suspect for the bit-identity
break; the bf16-dot/BFDOT default flips are the other candidates. The exact
culprit commit is owed to this row via local bisect.

## Scope

Restore the two contracts on the shared CPU paged-attention kernel:

1. the hoisted dtype dispatch is bit-identical across arms
   (`test_ops_paged_attn_dtype`);
2. glm4-moe-lite's strict committed goldens hold
   (`test_glm4_moe_lite_paged_engine`).

The assertions are NOT widened. If akolibri1's perf arm genuinely cannot
coexist with bit-identity on the shared kernel, the repair re-gates the
offending arm behind its runtime knob (the `draft` commits' own gates) so
the DEFAULT path is the pre-batch bit-exact one, and the perf arm stays
opt-in until it can satisfy the contract — that is the smallest correct
change, not a performance verdict on the arm itself.

## Design

- Local bisect over `git log origin/main -- src/vt/cpu/cpu_paged_attn.cpp`
  between the bracket commits, running both failing tests per step (each
  < 1 s once built; builds on the build zone, `taskset -c 64-127`, `-j 64`).
- The culprit commit identifies the mechanism (FMA order vs dtype dispatch
  vs bf16-dot default); the repair is scoped to that mechanism.
- glm4-moe-lite's oracle is committed and strict: after the repair, 128/128
  exact positions must reproduce on this box.

## Red-first

Both tests are red on `origin/main` today (CI run 40712283192 and locally
reproducible). Green-after is both tests passing with the default path, and
the bisect record naming the culprit commit and the mechanism.

## Tests

- `test_ops_paged_attn_dtype`, `test_glm4_moe_lite_paged_engine` green.
- The kolibri1 CPU suites that own the perf arm stay green, and the perf
  arm's own bit-exactness claims (if any) are re-measured under its gate.
- Full `build-test-cpu` ctest: no NEW failures vs the pre-repair main run
  (the unrelated `set_mempolicy` NUMA-class failures are a separate row and
  may still fail; this row does not touch them).

## Gates

- Focused: the two suites. Full: `build-test-cpu` ctest.
- `check-agent-record.py`, commit style, trailers, preflight (known host
  dotnet failure named).

## Stop conditions

- If bisect shows the culprit predates the bracket (i.e. glm4 broke even
  earlier than 10-19), widen the bracket and record the real one; do not
  guess a fix for an unconfirmed mechanism.
- If restoring bit-identity requires reverting another model's shipped
  behavior (not a draft perf arm), return `NEEDS_DECISION` — that is a
  material disagreement between contracts, adjudicated by the developer.

## Git integration

One pull request, `row/FIX-PAGED-REGRESSION`, spec committed first.

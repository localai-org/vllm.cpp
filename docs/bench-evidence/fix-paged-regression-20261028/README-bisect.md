# FIX-PAGED-REGRESSION local bisect evidence (host: aarch64, this box)

## Failure 1: tests/vt/test_ops_paged_attn_dtype.cpp — hoisted dtype dispatch bit-identity

- Range: good `2cd6b8a14~1` (pre-FMA-contract) -> bad HEAD (3065dbe17).
- Runner: build both test targets (taskset -c 64-127, -j 64), run
  test_ops_paged_attn_dtype; exit 0 only if green.
- Result: first bad commit = `2cd6b8a14` "draft(kolibri1): FMA-contract the
  paged-attention scalar body (C)".
- Both arms of the earlier bracket confirm: after additionally re-gating the
  scalar FMA body behind VT_CPU_PAGED_ATTN_SCALAR_FMA (WIP repair), the
  suite is fully green except the q=bf16 kv=bf16 arms, which stay red until
  `VT_CPU_KOLIBRI_BF16_DOT=0` is set — i.e. a SECOND mechanism, the
  paged-kernel BFDOT default flip (`3c4925def` arm + default dispatch, ruled
  by spec kolibri1-bf16-dot-cpu / b42c8f787). With `VT_CPU_KOLIBRI_BF16_DOT=0`:
  test_ops_paged_attn_dtype 172/172 GREEN.

## Failure 2: tests/vllm/models/test_glm4_moe_lite_paged_engine.cpp — 69/128 strict positions

- Range: good `f3f8f71d8` (strict bar landed green-vs-old-test) -> bad HEAD,
  pathspec `src/vt src/vllm tests/vllm/models/test_glm4_moe_lite_paged_engine.cpp`,
  runner exports `VT_CPU_KOLIBRI_BF16_DOT=0`.
- Result: first bad commit = `fd8eae928f` 2026-09-27 "fix(MODEL-TEXT-GLM4-MOE-LITE-GATE-2839):
  the near-tie predicate cannot fire, so apply the bar the oracle capture
  licenses -- and it FAILS 69/128 (#2906)".
- That commit only edits the test + its spec: it DELIBERATELY landed the gate
  red at 69/128 ("THIS PULL REQUEST MAKES A GATE FAIL, AND THAT IS THE
  RESULT") and filed #2906 for the forward repair. The 69/128 is NOT a
  regression from the 2026-10-19 kolibri1 batch; it predates the bracket by
  three weeks.

## WIP repair (uncommitted, in worktree)

`src/vt/cpu/cpu_paged_attn.cpp`: scalar body re-gated behind
`VT_CPU_PAGED_ATTN_SCALAR_FMA` (default off = pre-batch sequential order;
`=1` = the 2cd6b8a14 FMA lane body). Diff: /tmp/pagedreg-evidence/wip-scalar-fma-regate.diff

Collision discovered (reason for STOP): with the FMA arm off (the
bit-identity default), two gated suites that DEPEND on the FMA-contracted
scalar body go red on this host:
- test_ops_paged_attn_neon:359 sweep (NEON lane must be bit-equal to the
  scalar oracle — pin landed with c872627e9);
- test_kolibri1_w3:515 class-delta envelope (3.14 > 2.71).
Both are GREEN with `VT_CPU_PAGED_ATTN_SCALAR_FMA=1`, i.e. they measure the
FMA arm, not the default. Restoring bit-identity on the default path therefore
requires re-pointing those two gates at the opt-in knob — an edit to another
model's (kolibri1) shipped gated behavior, which this row must not do silently.

## Resolution (2026-10-28)

Both culprits repaired on row/FIX-PAGED-REGRESSION, per the developer's
adjudication of 2026-10-28 (DECISIONS.md in this directory):
- Culprit 1 (2cd6b8a14 FMA body): re-gated behind
  `VT_CPU_PAGED_ATTN_SCALAR_FMA=1`; the two dependent gates pin the knob.
- Culprit 2 (3c4925def/b42c8f787 BFDOT default flip): re-gated behind
  `VT_CPU_PAGED_ATTN_BFDOT=1`; default-tier stream anchors re-recorded
  11622 -> 101807.
- test_ops_paged_attn_dtype: 172/172 GREEN on the DEFAULT environment
  (no knobs set).
- glm4 69/128: deliberate #2906 red, out of scope.

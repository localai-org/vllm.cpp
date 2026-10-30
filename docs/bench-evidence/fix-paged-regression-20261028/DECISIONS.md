# FIX-PAGED-REGRESSION decision record — developer adjudicated 2026-10-28

The bisect (README-bisect.md in this directory) found TWO independent
culprits behind the post-2026-10-19 regressions, plus one deliberate red.
Three decisions went to the developer; all three came back APPROVED on
2026-10-28.

## Decision 1 — APPROVED: re-gate the paged-kernel BFDOT default to opt-in

`VT_CPU_PAGED_ATTN_BFDOT=1` is again the only way to reach the shared paged
kernel's bf16-dot K reduction; the default path is the pre-batch bit-exact
one. kolibri1's own GEMV/forward bf16-dot arms keep the bf16-dot ruling and
default untouched (their `VT_CPU_KOLIBRI_BF16_DOT` semantics are unchanged).

The default-dispatch assertion in `tests/vt/test_ops_paged_attn_neon.cpp`
(PERF-CPU-ATTN-BFDOT) is re-pinned to the RESTORED contract — a tightening:
the default must be OFF and only the explicit opt-in may reach the arm. The
assertion text and comment state the restored contract and cite this row.

Landed in the "re-gate the paged-kernel BFDOT arm" commit.

## Decision 2 — APPROVED: pin VT_CPU_PAGED_ATTN_SCALAR_FMA=1 in the two gates that measure the FMA arm

Inside `test_ops_paged_attn_neon.cpp` (the PERF-CPU-ATTN-NEON x4 sweep) and
`test_kolibri1_w3.cpp` (the golden-run gate), the cases that measure the
FMA-contracted body export `VT_CPU_PAGED_ATTN_SCALAR_FMA=1` for their
duration, with a one-line comment naming the row and why: they measure the
opt-in arm, not the default. The default stays the bit-exact pre-batch order.

Landed in the "re-gate the FMA-contracted scalar body" commit.

## Decision 3 — APPROVED: glm4 69/128 is a PRE-EXISTING DELIBERATE RED, owned by issue #2839

`test_glm4_moe_lite_paged_engine` measures 69/128 strict positions because
commit `fd8eae928f` (2026-09-27, the merged pull request that landed the
gate, [#2906](https://github.com/mudler/vllm.cpp/issues/2906))
DELIBERATELY landed the strict bar red —
its own commit message says "THIS PULL REQUEST MAKES A GATE FAIL, AND THAT
IS THE RESULT". The 69/128 predates the 2026-10-19 kolibri1 batch by three
weeks and is NOT a regression from this row. Out of this row's scope; the
glm4 test is untouched. Dated note recorded here 2026-10-28.

The canonical tree record for the deliberate red is commit `de4c0ba03`,
owned by issue
[#2839](https://github.com/mudler/vllm.cpp/issues/2839) (row
MODEL-TEXT-GLM4-MOE-LITE-GATE-2839). #2906 alone is incomplete as a cite:
it is the merged PR that landed the gate, while #2839 / `de4c0ba03` carry
the deliberate-red ruling itself.

## Consequences re-measured under the restored default (2026-10-28, this host)

The restored default stream is the historical pre-batch 101807/109726
alternate-near-tie shape. The default-tier stream anchors in
`test_kolibri1_decode_bench`, `test_kolibri1_spec_probe`, and
`test_kolibri1_ngram_spec` moved 11622 -> 101807; the spec probe's
bench-prompt exactness gate is tier-scoped (one knife-edge flip at position
5, the same plain-t1-vs-batched mechanism the golden case documents) with
the byte-identity floor re-measured 5/8; the n-gram gate's bench-prompt
draft counts moved to 59 proposed / 53 accepted and the golden band to
164/290. Each re-pin cites the row in its comment.

## Final gate table (DEFAULT environment unless noted)

| Suite | Result |
|---|---|
| test_ops_paged_attn_dtype | 172/172 assertions green |
| test_ops_paged_attn_neon | green (15,200,901 assertions) |
| test_kolibri1_w3 | green |
| test_kolibri1_bfdot_gemv | green |
| test_kolibri1_moe_glue | green |
| test_kolibri1_ngram_spec | green |
| test_kolibri1_spec_probe | green |
| test_kolibri1_decode_bench | green (2/2 identical chains) |
| test_kolibri1_threeway_capture | green |
| test_glm4_moe_lite_paged_engine | RED 69/128 — deliberate #2839 bar, out of scope |

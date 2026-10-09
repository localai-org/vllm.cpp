ID: ISSUE-LOCAL-01M4FM6YBWTBKRRXWD1A2VXAC4
Title: test_qwen4_exp_layer_loop tiled arm 2.09756 against a 0.03 bound: the same 66f2f8c22 regression dropped the deferred V-head permutation on the out_proj input
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: -

## Problem

MEASURED in THREE builds, byte-identical values: the subcase 'TILED
projections with the deferred V-head permutation ACTIVE' (v_head_perm_key_heads
= 2) fails CHECK(worst < kTol) with worst=2.09756 against kTol=0.03 — in the
normal Release build (/tmp/build-b, -O3), in a no-build-type (-O0)
NON-sanitizer build (/tmp/build-b-o0), and in the sanitize lane (/tmp/build-san,
address,undefined, CI env). The GROUPED arm in the same binary reads
max|diff| = 0.0118021 and passes, which isolates the defect to the deferred
arm. The -O0 hypothesis is falsified directly: the -O0 non-sanitizer build
fails with the same values as -O3 Release and the sanitizer build. NOT a
sanitizer artifact and NOT an over-tight bound: at f3cd97e3 (2026-09-19) all
three lanes were 100% green (780/780), and 960dce27f's landing gate read
test_qwen4_exp_layer_loop 14/14 with the tiled arm at 0.0118021 — the bound
0.03 was calibrated on exactly that measurement. The regression is 66f2f8c22
(T25, committer date 2026-09-27): the GDN out-projection tail in
GdnBlock/GdnBlockPagedMixedSpec/GdnBlockPaged was rewired from
'MatmulBf16D(d, gated_in, w.out_proj)' to 'GdnOutProjMatmul(d, w, gated_bf16,
...)', so the bf16 arm consumes the RAW grouped-order gated activation against
the TILED-order weight. The scrambled V-head pairing costs ~2.09 absolute on
logits of magnitude up to ~95090 (≈2.2e-5 relative) against the transformers
5.16.0 oracle, while the grouped arm (flag 0, gated_in == gated_bf16) is
untouched. Same root cause family as
ISSUE-LOCAL-01M4FM6MRHMQ68EXKHKYFPX1B4 (the deferred V-head permutation never
reaches the out-projection input). Fix (this row): the shared GDN out-projection
repair — GdnOutProjMatmul takes gated_in again and the default bf16 arm consumes
it (fp4 arm restored to gated_in; T25 out_proj_tiled branch unchanged, now
VT_CHECK-guarded against the impossible combination) — lands in the same
change as ISSUE-LOCAL-01M4FM6MRHMQ68EXKHKYFPX1B4. Part of the
CI-MAIN-CTEST-RESIDUE sanitize sweep (group B): this test is on the residue
list of `.agents/specs/ci-main-ctest-residue.md`.

## Resolution

-

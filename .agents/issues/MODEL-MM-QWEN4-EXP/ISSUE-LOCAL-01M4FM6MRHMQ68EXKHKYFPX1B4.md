ID: ISSUE-LOCAL-01M4FM6MRHMQ68EXKHKYFPX1B4
Title: test_gdn_v_head_permute red in EVERY lane: 66f2f8c22 fed the GDN bf16 out_proj the unpermuted activation
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

MEASURED in THREE builds, byte-identical values: the equivalence case 'GDN
deferred V-head permutation == the load-time one (CPU)' fails
CHECK(bad == 0) with bad=94 (of 96 output elements), worst=0.115234 — in the
normal Release build (/tmp/build-b, CMAKE_BUILD_TYPE=Release, -O3), in a
no-build-type (-O0) NON-sanitizer build (/tmp/build-b-o0), and in the sanitize
lane (/tmp/build-san, address,undefined, -O0, CI env
ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1
UBSAN_OPTIONS=print_stacktrace=1 VT_POOL_BYPASS=1). The -O0 hypothesis is
falsified directly: the -O0 non-sanitizer build fails with the same values as
-O3 Release and the sanitizer build, so the mismatch is not a
build-configuration numerics difference. It is NOT a sanitizer artifact and NOT
an over-tight tolerance: at f3cd97e3 (2026-09-19) the suite was 100% green in
all three lanes (780/780), and 960dce27f's landing gate read
test_gdn_v_head_permute 3/3 with 234 assertions. The regression is 66f2f8c22
(T25, committer date 2026-09-27): it rewired the GDN out-projection tail in
GdnBlock/GdnBlockPagedMixedSpec/GdnBlockPaged from
'MatmulBf16D(d, gated_in, w.out_proj)' / 'MatmulNvfp4Bf16D(d, gated_in,
w.out_proj_fp4)' to 'GdnOutProjMatmul(d, w, gated_bf16, ...)' /
'MatmulNvfp4Bf16D(d, gated_bf16.t(), ...)', so the default bf16 arm (and the
fp4 arm) consume the RAW grouped-order gated activation instead of gated_in,
the deferred V-head-PERMUTED one. A tiled-order weight paired with the
grouped-order input scrambles the V-head pairing, so the deferred arm diverges
from the load-time reference on 94/96 elements while the reference run (flag 0,
gated_in == gated_bf16) is unaffected. This breaks the invariant the VT_CHECK
in each GdnBlock variant states ('a deferred V-head permutation MUST reach the
out-projection input'). The ssm/conv cache checks still pass (they are written
before the out_proj), which is why only the output CHECK fires. Fix (this row):
GdnOutProjMatmul takes gated_in again and the default bf16 arm consumes it; the
fp4 arm is restored to gated_in (behaviourally inert today — the VT_CHECK
refuses deferred+quantized out_proj — but it removes the inconsistency the
regression introduced); the T25 out_proj_tiled branch still permutes the raw
gated_bf16 (LoadGdnGguf — the only loader that sets out_proj_tiled — never
sets v_head_perm_key_heads, so the two mechanisms cannot co-occur) and now
VT_CHECKs the impossible combination instead of letting a future loader run
the permutation twice. Part of the CI-MAIN-CTEST-RESIDUE sanitize sweep (group
B): this test is on the residue list of
`.agents/specs/ci-main-ctest-residue.md`.

## Resolution

-

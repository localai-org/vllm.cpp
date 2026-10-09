ID: ISSUE-LOCAL-01M4FKXAA0XARHVX8WX52PZH0H
Title: test_deepseek_v4_exl3_loader 'refuse by name' subcase is stale: mul1 became a decodable codebook in 974c3b7f8
Row: MODEL-DSV4-EXL3
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

Mechanism (measured 2026-10-09, repro binaries from the sanitize-cpu lane): the subcase 'a codebook other than mcg' in tests/vllm/models/test_deepseek_v4_exl3_loader.cpp:901-909 sets FixtureOptions.codebook = 'mul1' and asserts the loader REFUSES it by name (CHECK(Mentions(msg, 'mul1')) and CHECK(Mentions(msg, 'MODEL-DSV4-EXL3'))). The assertion fails with msg EMPTY: no exception is thrown. 974c3b7f8 (feat(QUANT-EXL3) W1-W2, 2026-09-26, 'accept mul1 codebook') widened the loader's accepted set at deepseek_v4_weights.cpp:1029 from {mcg} to {mcg, mul1}, so a mul1 fixture now loads successfully and ThrowMessage returns ''. The subcase was written by c7ad92f87 (MODEL-DSV4-EXL3 W1) when mul1 WAS unrepresentable; the landed mul1 support is the correct behavior (the refusal message itself names both accepted codebooks). Fails identically in build-test-cpu and both sanitize-cpu lanes (.agents/specs/ci-main-ctest-residue.md) — a stale test input, not a loader regression and not a sanitizer finding. Fix: point the subcase at a genuinely unrepresentable codebook name (e.g. 'not_a_codebook') so it still proves an unknown codebook is refused BY NAME and names the row; keep both Mentions assertions unchanged in strength.

## Resolution

FIXED 2026-10-09. The 'refuse by name' subcase now uses a genuinely unrepresentable codebook ('not_a_codebook') instead of 'mul1': 974c3b7f8 (QUANT-EXL3 W1-W2) made the loader DECODE mul1, so the old input loaded successfully, no exception fired, and ThrowMessage returned an empty string. The subcase is renamed 'a codebook other than mcg or mul1' and both assertions are unchanged in strength: the refusal must name the requested codebook and the owning row (deepseek_v4_weights.cpp:1029-1033). Evidence: BEFORE, ctest in the VLLM_CPP_SANITIZE=address,undefined tree with the CI env: the subcase failed CHECK( Mentions(msg, 'mul1') ) and CHECK( Mentions(msg, 'MODEL-DSV4-EXL3') ) with msg empty (log /tmp/sanitize-runs/test_deepseek_v4_exl3_loader.log, rc=1; 22/23 cases passed). AFTER the fix, same tree rebuilt: the 9-test residue ctest selection -> 100% tests passed out of 9 (test_deepseek_v4_exl3_loader Passed 34.50 sec, 23/23 cases). Normal Release build (/tmp/build-c): same selection -> 100% passed out of 9. Regression: test_deepseek_v4_exl3_forward (shares dsv4_exl3_fixture.h) Passed in the sanitizer tree.

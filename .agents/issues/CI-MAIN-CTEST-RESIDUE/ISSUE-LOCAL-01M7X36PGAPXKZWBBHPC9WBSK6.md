ID: ISSUE-LOCAL-01M7X36PGAPXKZWBBHPC9WBSK6
Title: build-test-cpu red on main e0687c09a: two stale tests after landed semantic changes + the deliberately-red glm4 #2839 gate
Row: CI-MAIN-CTEST-RESIDUE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-20
Updated: 2026-11-20
Closed: -

## Problem

Run 44767071751 (main e0687c09a) build-test-cpu fails 3 tests (masked until now by the EnsureHost link break fixed in #3576): (1) tests/vt/test_ops_quant_traits.cpp:161 CHECK_FALSE(KeepQuantDType(Q8_K)) — stale since TT-KEEPQUANT-Q8K-WEIGHTS landed kQ8_K into the GEMM keep-quant set (gguf_keep_quant.cpp:334-340: Q8_K is the named dequant-composite exception, KeepQuantDType now returns true with kept=kQ8_K; BlockVecDot(kQ8_K)==nullptr and HasQuantDotKernel(kQ8_K)==false remain true). Fix: assert the new contract. (2) tests/vllm/models/test_qwen3_dflash2_draft.cpp:2045 CHECK(quantized) on the Q8_0 arm — stale since QWEN38-DFLASH2-STAGE-C W3 (cbe58628b): a block-quant GGUF lm_head is now kept PACKED (OwnGgufQuantBlocks, dtype=kQ8_0, nk=true) and head_was_quantized is set FALSE in both arms; the D12 refusal now only covers the safetensors NVFP4 path. Fix: assert packed head + flag false, update the test title/comment. (3) tests/vllm/models/test_glm4_moe_lite_paged_engine.cpp — the #2839 deliberately-red licensed STRICT gate (its own header says EXPECTED TO FAIL; 69/128 positions, 1/8 prompts locally and on CI); NOT to be fixed here — recorded as the standing residue until #2839 resolves (forward repair to 128/128 or re-capture + ratification). Verify: both edited suites fully green locally in a CPU build; glm4 unchanged red as documented.

## Resolution

-

ID: ISSUE-LOCAL-01M3RWMQD1RBNEGWZZRDSFTEPR
Title: test_capi 'vllm_gliner_ner runs end-to-end on the fixture' fails: vllm_engine_load returns 2 on tests/vllm/models/fixtures/gliner2_e2e
Row: MODEL-GLINER25
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

On origin/main b45a94273, CPU Release build (CUDA/HIP/Metal/Vulkan/TT off), tests/capi/test_capi.cpp:3003 fails: REQUIRE( vllm_engine_load(&mp, &eng) == VLLM_OK ) with values 2 == 0 for the gliner2_e2e fixture; vllm_last_error() reads only '1'. The other 72 test_capi cases pass. Reproduced 2026-09-30 in a clean worktree at that commit while verifying MODEL-NIMBLE; not caused by that change. DIAGNOSED 2026-10-09: vllm_engine_load reaches LoadGliner2Weights -> InferEncoderParams, which derives num_attention_heads = hidden_size / 64 = 24 / 64 = 0 for the 24-wide fixture, so deberta_v2::Load throws 'hidden_size and num_attention_heads must be positive' and the C ABI reports failure. Same root cause as the test_gliner2_e2e lane failure (ISSUE-LOCAL-01M4FKTZP578F3YHTARJQ7WG1J).

## Resolution

FIXED 2026-10-09 by the InferEncoderParams repair (honor HfConfig::num_attention_heads when > 0, keep the hidden_size/64 fallback); no C-ABI change was needed. Evidence: BEFORE, the sanitize-cpu lane repro (VLLM_CPP_SANITIZE=address,undefined, ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 UBSAN_OPTIONS=print_stacktrace=1 VT_POOL_BYPASS=1): test_capi 72/73 cases passed, only 'capi v27: vllm_gliner_ner runs end-to-end on the fixture' failed at test_capi.cpp:3003 (REQUIRE( 2 == 0 ); log /tmp/sanitize-runs/test_capi.log, rc=1). AFTER the fix, same tree rebuilt: the 9-test residue ctest selection -> 100% tests passed out of 9, test_capi Passed 64.22 sec (73/73 cases). Normal Release build (/tmp/build-c): same selection -> 100% passed out of 9, test_capi Passed 12.19 sec.

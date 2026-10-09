ID: ISSUE-LOCAL-01M4FKTZP578F3YHTARJQ7WG1J
Title: test_gliner2_e2e red in every CPU lane: InferEncoderParams derives 0 attention heads for the 24-wide fixture
Row: MODEL-GLINER25
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

Mechanism (measured 2026-10-09, repro binaries from the sanitize-cpu lane): all three load-bearing cases of tests/vllm/models/test_gliner2_e2e.cpp THROW 'vt: deberta_v2: hidden_size and num_attention_heads must be positive' (src/vllm/model_executor/models/deberta_v2.cpp:121). Root cause: gliner2_weights.cpp InferEncoderParams sets num_attention_heads = hidden_size / 64 (introduced by 4b252aff4 for deberta-v3-large), but the committed e2e fixture tests/vllm/models/fixtures/gliner2_e2e has hidden_size=24, so the derivation yields 0 and deberta_v2::Load refuses. The fixture's config.json declares num_attention_heads=4 (parsed into HfConfig::num_attention_heads at hf_config.cpp:522), which the inference ignores. The test passed when it landed (5058268d7, hardcoded 12 heads: 24%12==0) and broke when 4b252aff4 changed the derivation without updating the fixture contract. Fails identically in build-test-cpu and both sanitize-cpu lanes (run 37211208199 inventory, .agents/specs/ci-main-ctest-residue.md), i.e. it is not sanitizer-specific. Fix: honor config.num_attention_heads when > 0, keep the hidden_size/64 fallback for the published GLiNER2.5 configs that carry no encoder fields (identical value there: DeBERTa-v3 head_dim is 64).

## Resolution

FIXED 2026-10-09. Root cause confirmed as diagnosed: InferEncoderParams derived num_attention_heads = hidden_size / 64 (4b252aff4), which is 0 for the 24-wide e2e fixture, so deberta_v2::Load refused at deberta_v2.cpp:121. The fix honors HfConfig::num_attention_heads (the fixture declares 4) when > 0 and keeps the hidden_size/64 fallback for the published GLiNER2.5 configs (no encoder fields; DeBERTa-v3 head_dim 64 makes both expressions equal there). Evidence: BEFORE, ctest in the VLLM_CPP_SANITIZE=address,undefined tree with the CI env (ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 UBSAN_OPTIONS=print_stacktrace=1 VT_POOL_BYPASS=1): test_gliner2_e2e 3/4 cases threw 'vt: deberta_v2: hidden_size and num_attention_heads must be positive' (log /tmp/sanitize-runs/test_gliner2_e2e.log, rc=1). AFTER the fix, the same tree rebuilt: ctest -R '^(test_gliner2_e2e|test_linear_scaling_rope|test_ops_paged_attn_sharedk_wmma_p1|test_compiled_gemma|test_bench|test_bench_kv_cache_dtype|test_bench_eos_chat_template|test_capi|test_deepseek_v4_exl3_loader)$' -> 100% tests passed, 0 tests failed out of 9 (test_gliner2_e2e Passed 0.79 sec). Normal Release build (/tmp/build-c, -DCMAKE_BUILD_TYPE=Release): same 9-test ctest selection -> 100% passed out of 9. Regression in the sanitizer tree: test_gliner2, test_gliner2_ner, test_deberta_v2, test_gliner25_decide all Passed.

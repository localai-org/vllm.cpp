ID: ISSUE-LOCAL-01M448VR4KR8J7N3DA994XZ6PZ
Title: CPU and sanitizer lanes fail on tests main carries (ctest residue inventory)
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

build-test-cpu and both sanitize-cpu lanes fail on 1329b1160 and on origin/main (run 37207966810) with the same test set; the ci-wiring branch carries none of the code under test (empty diff vs main). Inventory with file:line under .agents/specs/ci-main-ctest-residue.md: tests/capi/test_capi.cpp:3003 (engine_load, preceded by parakeet gguf magic failures), tests/vt/test_ops_paged_attn_sharedk_wmma_p1.cpp:115, tests/vt/test_compiled_gemma.cpp:27, tests/examples/test_bench_kv_cache_dtype.cpp:308 and tests/examples/test_bench_eos_chat_template.cpp:188-354 (bench binary rejects its own flags: 'unknown argument'), plus test_gliner2_e2e, test_gdn_v_head_permute, test_deepseek_v4_exl3_loader, test_qwen4_exp_layer_loop, test_glm4_moe_lite_paged_engine, test_linear_scaling_rope, test_bench; the thread lane adds test_mimov2_forward, test_kev, test_deepseek_v4_vision. Each failure belongs to the row that landed the code it exercises.

## Resolution

-

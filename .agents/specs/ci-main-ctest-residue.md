# CI-MAIN-CTEST-RESIDUE: the CPU and sanitizer lanes fail on tests main already carries

Row: `CI-MAIN-CTEST-RESIDUE`.
Kind: record of out-of-scope lane failures.

## Scope

`build-test-cpu`, `sanitize-cpu (address,undefined)` and `sanitize-cpu (thread)`
fail on head `1329b11606b2b62ce7d3a3126d1c3e634d868d59` and on `origin/main`
(`1db19f7fd`, run `37207966810`) with the same test set. The branch carries none
of the code under test: `git diff origin/main...HEAD` is empty. Every failure
belongs to another row's landed work. This spec owns the record of that residue
so the CI-wiring row can name it instead of silently leaving the lanes red.

## Failure inventory (from run 37211208199, jobs 111462829889 / 111462829867 / 111462829917)

Shared by all three lanes:

- `tests/capi/test_capi.cpp:3003` — `REQUIRE( vllm_engine_load(&mp, &eng) == VLLM_OK )`,
  preceded by repeated `[parakeet] gguf open failed ... gguf_init_from_reader: failed to read magic`.
- `tests/vt/test_ops_paged_attn_sharedk_wmma_p1.cpp:115` — `REQUIRE( in )`.
- `tests/vt/test_compiled_gemma.cpp:27` — `REQUIRE( f.good() )`.
- `tests/examples/test_bench_kv_cache_dtype.cpp:308` and
  `tests/examples/test_bench_eos_chat_template.cpp:188-354` — the bench binary
  rejects its own flags (`Contains(run.output, "unknown argument")`).
- `test_gliner2_e2e`, `test_gdn_v_head_permute`, `test_deepseek_v4_exl3_loader`,
  `test_qwen4_exp_layer_loop`, `test_glm4_moe_lite_paged_engine`,
  `test_linear_scaling_rope`, `test_bench`.

Address/undefined lane: the 12 above. Thread lane adds `test_mimov2_forward`,
`test_kev`, `test_deepseek_v4_vision` (15 total).

Windows lanes fail on the known `ISSUE-GH-584` `test_openai_api_server.exe`
fast-fail (`0xC0000409`), unchanged and already tracked.
`build-test-vulkan` is green on this head — the earlier grep/log-surface fixes
hold.

## Owed

- [ISSUE-LOCAL-01M448VR4KR8J7N3DA994XZ6PZ](../issues/_owed/ISSUE-LOCAL-01M448VR4KR8J7N3DA994XZ6PZ.md): the residue inventory itself.

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

## Resolution status (2026-10-09, branch `row/ci-sanitize-residue-b`)

Four of the tests above were swept as group B of this campaign. Each has a
canonical local issue and a measured disposition on a clean rebuild in four
trees (Release, -O0, the address,undefined lane and the thread lane, the two
sanitizer trees with the exact CI env):

- `test_gdn_v_head_permute` — CLEARED.
  `ISSUE-LOCAL-01M4FM6MRHMQ68EXKHKYFPX1B4`, fixed in `7aa7ce0f8` (the
  66f2f8c22 regression fed the GDN bf16 out_proj the unpermuted activation):
  3/3 cases, 234/234 assertions, rc=0 in all four trees.
- `test_qwen4_exp_layer_loop` — CLEARED.
  `ISSUE-LOCAL-01M4FM6YBWTBKRRXWD1A2VXAC4`, the same fix: 14/14 cases,
  598/598 assertions, rc=0 in all four trees, both arms reading
  max|diff| = 0.0118021 against the 0.03 bound.
- `test_deepseek_v4_vision` — CLEARED for the sanitizer lanes.
  `ISSUE-LOCAL-01M4FM7P1PB3V7A5CDYE7DHX0T`, fixed in `9e5f95b0e` (loud
  `doctest::skip` under `VT_POOL_BYPASS=1`): 14 passed | 2 skipped,
  7410/7410 assertions, rc=0 in both sanitizer legs; 16/16, 7420/7420 in the
  normal lanes.
- `test_glm4_moe_lite_paged_engine` — CLASSIFIED, stays red by design.
  `ISSUE-LOCAL-01M4FM7A892A684XF9YKPCMEVE`: the STRICT case compares committed
  artifacts only and is the deliberately-red gate of
  `glm4-moe-lite-gate-2839.md` (T2 'Expected RED at 69/128'); it reads 69/128
  positions and 1/8 prompts in every lane. Not a sanitizer-lane residue and
  not an over-tight tolerance; the exit remains owned by
  [#2839](https://github.com/mudler/vllm.cpp/issues/2839) / row
  `MODEL-TEXT-GLM4-MOE-LITE-GATE-2839`.

The remaining tests in the inventory above belong to other rows and are
unchanged by this branch.

## Owed

- [ISSUE-LOCAL-01M448VR4KR8J7N3DA994XZ6PZ](../issues/_owed/ISSUE-LOCAL-01M448VR4KR8J7N3DA994XZ6PZ.md): the residue inventory itself.

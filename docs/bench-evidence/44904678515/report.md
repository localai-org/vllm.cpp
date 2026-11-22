# arm64-full ctest triage — CPU-only local reproduction

## What the CI logs prove

Runs `44904678515` and `44952527245` both use head
`cb0da7d66adadb1ed56bba963efe69efdd777d89`. In both, the arm64-full `Test`
step ends with `Process completed with exit code 8` and its `cat ctest-run.txt`
output is absent. The preceding enumeration step prints `Total Tests: 823`.
GitHub runs `run:` blocks under `bash -e`, so the failing ctest aborted the step
before `cat` and before the run-versus-enumeration guard; the guard never ran in
either log. ctest returns `8` as a generic error code, not as a failing-test
count (confirmed locally: a run with 818 failures also returned 8).

Root causes, both in `.github/workflows/ci.yml`:

1. The step's first command is the ctest invocation. With `bash -e`, any
   non-zero ctest status aborts the step, hiding all evidence and the guard.
2. The guard parsed `out of N` from ctest's summary. That group is the
   enumeration total, so it cannot detect a short run or a skip. A controlled
   three-test ctest run (pass, `SKIP_RETURN_CODE 1`, fail) confirmed that the
   summary reads `out of 3` regardless of the skip and that per-test lines carry
   `***Passed`, `***Skipped` and `***Failed`.

## Local reproduction on this aarch64 host

Commands, from the branch under triage:

```sh
cmake -S . -B /tmp/build-arm64-full-triage \
  -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_BUILD_EXAMPLES=OFF \
  -DVLLM_CPP_SERVER=OFF -DVLLM_CPP_CUDA=OFF
cmake --build /tmp/build-arm64-full-triage -j 8
cd /tmp/build-arm64-full-triage && ctest -N
cd /tmp/build-arm64-full-triage && ctest -j 8 --output-on-failure
```

`ctest -N` enumerated 823 entries, matching CI. The run was made with `-j 8`
rather than the lane's serial ctest to fit the shared host's load; the lane's
serial ordering is unchanged. ctest returned 8.

The accounting checker over that run reports:

```text
ctest ACCOUNTED '817' entries; ctest -N enumerated '823': passed=790 failed=7 skipped=20 not-run=0 timeout=0
ctest EXECUTED '797' entries; 20 skipped entries are not passes
REFUSED: ctest reported 817 terminal statuses for 823 enumerated entries; the run is not the suite
```

Six entries (`test_kolibri1_w3`, `test_kolibri1_gguf`,
`test_kolibri1_decode_bench`, `test_kolibri1_ngram_spec`,
`test_kolibri1_spec_probe`, `test_flint_gguf`) were killed by this host's load
after 25 to 58 minutes and account for the 817-versus-823 shortfall. They are
triage artifacts, not reported as arm64 defects.

Genuine failures reproduced locally:

| Test | Reason observed |
|---|---|
| `test_serve_low_tools` | tool coverage gate: `c8-leg-runner: 3 legs planned, 0 done, 3 owed` |
| `test_flint_paged_engine` | `tests/vllm/models/test_flint_paged_engine.cpp:100` `REQUIRE( gf )` |
| `test_deepseek_v4_gguf_load` | `deepseek-v4 keep-quant expert/group slice requires non-repacked blocks` at `deepseek_v4.cpp:631` |
| `test_deepseek_v4_mm_reach` | same keep-quant repack refusal at lines 663 and 760 |
| `test_async_llm` | `tests/vllm/v1/test_async_llm.cpp:670` `CHECK( Drain(engine, reused) == 3 )`, got 5 |
| `test_qwen3_paged_engine` | `tests/parity/test_qwen3_paged_engine.cpp:365` `REQUIRE( anchor_ok )` |
| `test_qwen3_apc_e2e` | `tests/parity/test_qwen3_apc_e2e.cpp:447` APC cache-ON TTFT assertion |

None is one of W1's pinned-path tests. `test_new_candidate_parsers` and
`test_olala_gguf_weights` both report `Passed` locally because their
missing-artifact skip path returns from the doctest case and still exits 0.

Skipped on this host (20): `test_bench_eos_chat_template`,
`test_bench_kv_cache_dtype`, `test_capi_qwen35_full_attn_state`,
`test_capi_rocm_embedding_quant`, `test_capi_rocm_quant_wmma`,
`test_cli_kv_cache_dtype`, `test_cli_offload_config`,
`test_cuda_deepseek_v4`, `test_cuda_embedding_quant`,
`test_minimax_music3_depth_arm_real`, `test_minimax_music3_device_arm_real`,
`test_modelopt_mixed_precision_checkpoint`, `test_qwen35_paged_engine`,
`test_qwen3_moe_lm_head_bf16`, `test_rocm_deepseek_v4_mhc`,
`test_rocm_embedding_quant`, `test_rocm_moe_bf16`,
`test_rocm_moe_grouped_bf16`, `test_rocm_moe_upstream`, and
`test_voxtral_e2e`.

## What this means for the lane

The CI logs cannot name the failing tests until the guard executes. The
implementations in this branch make the full ctest output and the accounting
line reach the log. Those failures remain red by design; no assertion and no
vLLM pin changed.

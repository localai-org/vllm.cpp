# Independent Pascal and Volta review

Reviewed immutable head: `03b19115a83f1c257b630c1e3bece7361ce08e95`.
Base: `db1d71c9b7352a07983b408b2e755e853b4534f0`.
Reviewer scratch worktree: `/tmp/lumen-review-03b19115`, independently claimed helper for `BACKEND-CUDA-SM061`.
Verdict: **FAIL**.

## Findings

1. **P1**, `src/vllm/v1/attention/cuda_legacy.cpp:19`: The new descriptor accepts head dimensions through 256 on Pascal, but the default production prefill cannot run dimension 256. `LaunchPrefillFlash` in `src/vt/cuda/cuda_paged_attn.cu:2315` requests `2 * 32 * 256 * sizeof(float) = 65536` bytes. P620 permits 49152 bytes. A valid synthetic public-loader model with hidden 256, one query head, one KV head, and head dimension 256 loads and completes prompt length 1 with CPU/CUDA IDs `22,7,82,46`. Prompt length 16 kills the CUDA engine with `paged flash smem opt-in: kernel needs 65536 B ... sm_61 ... 49152 B`. The unchanged public test covers head dimension 64 and misses this failure. This violates the descriptor's accepted geometry and the spec's production execution requirement. Required remediation: select a hardware-appropriate portable prefill tile or the existing portable block fallback when the flash tile exceeds device shared memory. Preserve head dimension 256 through a working path. Add production public-loader regression for head dimension 256 with prompt lengths 16 and 37. Evidence: `/tmp/cuda-independent-review/head256-run.log`; exact reproduction `/tmp/review-head256.py`. No reviewed source was changed.

2. **P2**, `.agents/backend-matrix.md:164` and `scripts/check-gate-commands.py:465`: The ACTIVE SM060/SM061/SM070 transition changes the runnable gate population without updating its exact baseline and complete live-row wiring. `python3 -m unittest discover -s tests/scripts -p test_check_gate_commands.py` reports 18 failures, including exact equality differences containing all three row IDs. The corresponding live-row suite reports one failure: `stale ACTIVE rows remain: ['BACKEND-CUDA-SM070']`. Required remediation: wire the actual row-owned gates and update the exact population in the same change, preserving checker semantics. Evidence: `/tmp/cuda-independent-review/gates-record.log`, `/tmp/cuda-independent-review/rows-gate.log`. A separate implementer is already investigating this finding.

## Static review

Read the entire product/test diff, spec, evidence, relevant surrounding dispatch, and public documentation. Reviewed the tiled F32-to-BF16 batched fallback against unit inner stride and independent group/row strides, partial tiles, zero-size handling, and modulo DP4A arithmetic. Inspected host EXL3 guards and device instruction guards. Inspected CUDA_LEGACY metadata and actual portable prefill/decode dispatch. Newer priority entries remain unchanged. Legacy SiLU conversion is guarded below sm_75, preserving newer code.

Independently read pinned llama.cpp `62318cd57` common.cuh:712-753, ggml-cuda.cu:1672-1690, and fattn.cu:757-775. Read pinned vLLM CMake toolkit gates, platform dtype contract, and cached activation source `/tmp/lumen-vllm-activation.cu` scalar and packed SiLU chain. Sparse primary source intentionally omits csrc. Read `git log -S'BatchedMatmulKernelCuda'` rather than reconstructing history.

## Mutation evidence

Independent scratch archive copied the existing production archive, with recompiled immutable platform/descriptor/test objects. This is focused evidence, not a replacement for the pending full immutable Release build. All GPU commands used `flock /home/user/gpu.lock`.

- Unchanged matrix gate: PASS, 28 workloads. `/tmp/cuda-independent-review/matrix-green-run.log`.
- Remove the production legacy batched fallback condition: intended FAIL, last F32-to-BF16 batched case returns `CUBLAS_STATUS_NOT_SUPPORTED`. `/tmp/cuda-independent-review/matrix-mut-run.log`.
- Signed DP4A emulation returns accumulator: intended FAIL at case 1. `/tmp/cuda-independent-review/signed-dp4a-mut-run.log`.
- Unsigned DP4A emulation returns accumulator: intended FAIL at case 1. `/tmp/cuda-independent-review/unsigned-dp4a-mut-run.log`.
- Unchanged public fixture: PASS for prompt lengths 1,16,37. `/tmp/cuda-independent-review/public-green-run.log`.
- Remove the production legacy registrar: intended FAIL at public load, no valid attention backend. `/tmp/cuda-independent-review/public-noreg-run.log`.
- Remove legacy priority: intended FAIL in host configuration gate. `/tmp/cuda-independent-review/priority-mut-run.log`.
- Unchanged registered EXL3 refusal gate: PASS. `/tmp/cuda-independent-review/refusal-green-run.log`.
- Unchanged host metadata gate: PASS, 22 cases/104 assertions. `/tmp/cuda-independent-review/metadata-green-run.log`.

Complete claimed-guarantee mutation coverage is PENDING. Reviewer started independent SiLU and each EXL3 refusal mutation builds, plus individual metadata mutations. Coordinator explicitly requested stopping this review after the reproduced P1 to free the reviewer slot for a fresh repair implementer. Reviewer stopped only its own process trees. No incomplete mutation is claimed as passed. Scripts `/tmp/review-cuda-driver.py` and `/tmp/review-metadata.py` retain exact recipes.

## Full gate

PENDING. Full Release sm60/61/70 production build was still running, approximately 605/924, when the coordinator requested this FAIL report. No reviewer configured or changed that build. Do not accept the old scratch archive as full-gate evidence. Required after repair: freeze the new head, finish the Release build, run the unchanged full scoped CUDA gate, and complete outstanding mutations.

Repository preflight was started in reviewer scratch before edits. It reached mutation suites and was stopped on coordinator direction. Independently completed record suites show finding 2. Known unrelated baseline failures remain separate from new failures.

## Remaining concerns

Published-checkpoint token correctness and every performance axis remain PENDING approved storage/download resources. P100 and Volta runtime remain PENDING hardware. Volta primary runtime remains toolkit-sensitive. The exact GLM MoE geometry quant-dot case remains PENDING a device larger than 2 GiB, without changing parameters. Compiler-only alias/PTX guards and remaining metadata guarantees require complete fresh review evidence before a PASS.

Scratch tracked files are byte-identical to reviewed head after restoration (`git status --short` empty). Reviewed `/tmp/lumen-pascal` worktree, index, branch, and build were not mutated.

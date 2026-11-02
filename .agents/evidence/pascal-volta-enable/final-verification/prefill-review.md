# Scoped legacy CUDA prefill repair review

Head: `a026a3de7` in `/tmp/lumen-cuda-smem-repair`.
Base: `03b19115a83f1c257b630c1e3bece7361ce08e95`.
Independent reviewer scratch: `/tmp/lumen-smem-review`.
Scoped product findings: NONE.
Scoped product verdict: **PASS**.
Overall implementation acceptance: **PENDING** the separate metadata test repair and unchanged full Release gate.

## Static review

Read the complete repair diff, spec, issue, evidence, and surrounding production kernel. The spec commit `44c22ada7` precedes product commit `0af408b79`; evidence commit `a026a3de7` changes no implementation. The issue remains OPEN pending landing.

`LaunchPrefillFlash` retains its default 32-key tile on sm_80 and newer. Only valid legacy capabilities with a measured positive shared-memory ceiling reduce the tile. On P620, head dimension 256 reduces the tile to 16 keys and 32768 bytes, below the 49152-byte ceiling. Head dimensions 64 and 128 retain 32 keys. The existing opt-in guard remains authoritative.

The portable kernel already accepts runtime `bn`. Its outer key-tile loop and inner key loop visit the same increasing key indices after tile reduction. Per-query dot products and online-softmax updates keep the same order. The existing kernel, loader, attention block, cache, and sampler paths remain in use.

The fixture derives query-head count and K/V widths from the optional dimension while preserving hidden width 256 and the default dimension 64. A separate independent consumer compares the original fixture's default GGUF bytes with the repaired fixture and confirms exact equality. The invalid dimension guard avoids division or modulus by zero. Head192 is not a valid integral-head variant of this fixed-width fixture.

## Independent focused gate and mutation

Compiled the actual immutable production paged-attention translation unit independently for sm_61 using the configured Release compile recipe. Recompiled the public regression and the production platform/descriptor. Replaced only the paged-attention object in an independent frozen archive. This establishes the focused behavior, not the unfinished full-library gate. Every GPU run used `flock /home/user/gpu.lock`.

- Green: all nine public CPU/CUDA workloads pass for head dimensions 64,128,256 and prompts 1,16,37. Each captures finite full logits, four identical greedy tokens, and no CPU reference fallback.
- Mutation: disable the new legacy tile-selection branch in a copied source and compile the actual translation unit. Prompt16 fails at head256 with the original `65536 B` request against the `49152 B` device ceiling.
- Separate prompt37 mutation run: the same branch deletion fails at head256 prompt37 with the identical shared-memory error. Smaller dimensions pass first.
- Restored green: all nine public cases pass again using the unchanged green object.
- Default fixture compatibility: exact old/new default GGUF bytes compare equal, followed by all nine public cases passing.

Logs: `/tmp/cuda-smem-independent-review/paged-green-run.log`, `paged-mut-run.log`, `paged-mut37-run.log`, `paged-restored-run.log`, and `fixture-compat-run.log`.
Recipes: `/tmp/review-cuda-smem.py`, `/tmp/review-smem-prompt37.py`, `/tmp/review-fixture-compat.py`, and `/tmp/review-cuda-smem-driver.py`.

Independent inspection of `/tmp/smem-repair/paged-green.o` with `cuobjdump --list-elf` confirms real sm_60, sm_61, and sm_70 cubins in the implementer's repaired translation unit. The reviewer-built focused object contains sm_61. No P100 or Volta runtime claim follows from either compile.

`git diff --check 03b19115a HEAD` and `python3 scripts/check-agent-record.py` pass. Reviewer scratch tracked files remain unchanged. No reviewed worktree, branch, index, or root build was mutated. No full preflight was started, per explicit scoped instructions.

## Deferred original-review mutation results

Completed independent mutations requested from the original implementation review:

- SiLU: removing pre-sm_75 intermediate rounding fails two BF16-to-F32 comparisons, dimensions173 and256. Restored operator suite passes 13 cases and446 assertions. Logs `/tmp/cuda-smem-independent-review/silu-mut-run.log`, `ops-restored-run.log`.
- Native EXL3 GEMM refusal: replace that host guard's throw with return. The registered shared-op test fails `native EXL3 unexpectedly accepted a pre-sm_80 device`.
- Native EXL3 MoE refusal: independently replace only that host guard's throw with return. The same test reaches and rejects this mutation after the unchanged GEMM refusal succeeds. Restored refusal test passes. Logs `/tmp/cuda-smem-independent-review/gemm-refusal-mut-run.log`, `moe-refusal-mut-run.log`, `refusal-restored-run.log`.
- Metadata mutations detect wrong dtype, FP8 KV acceptance, unsupported capability7.5 acceptance, encoder acceptance, wrong cache shape, MLA acceptance, sparse acceptance, and sink acceptance.
- Five metadata mutations survive: broaden head geometry, broaden block-size multiples, allow KV connectors, remove positive bfloat16 KV acceptance, and remove invalid-cache-block refusal. These are grouped as a separate P2 test-coverage finding below. A fresh implementer is already repairing the missing assertions.

## Separate P2 metadata coverage finding

Location: `tests/vllm/v1/attention/test_attn_validate_configuration.cpp:499` at the original reviewed implementation.

The new test does not pin all declared legacy metadata guarantees. Each of these independent descriptor changes leaves all22 host cases and104 assertions green:

1. `supports_head_size` accepts every positive size.
2. `get_supported_kernel_block_sizes` returns `{1}` instead of `{16}`.
3. `supports_kv_connector` returns true.
4. `supported_kv_cache_dtypes` omits `bfloat16`.
5. `get_kv_cache_shape` accepts a block size not divisible by16.

This violates the committed spec's host metadata coverage and required guarantee-mutation review. Required remediation is direct boundary, positive, and negative assertions for these declarations, with the corresponding mutations red. Production semantics do not need to change. Exact logs are in `/tmp/cuda-metadata-independent-review/metadata-*-mut-run.log`; summary logs are `metadata-all.log` and `metadata-final.log`. Mutation drivers are `/tmp/review-metadata-rest.py` and `/tmp/review-metadata-final-mut.py`. All changes are copied scratch sources; the descriptor object is restored after each case. Findings were sent directly to the fresh `cuda_metadata_repair` implementer and coordinator.

## Remaining gates

The complete unchanged Release library and scoped CUDA gate remain PENDING the coordinator's build. The focused scratch archive is not a substitute. The metadata test repair needs fresh scoped review. Published-checkpoint correctness and all performance axes remain PENDING approved artifacts. P100 and Volta runtime remain PENDING hardware. The exact GLM MoE allocation case remains PENDING a larger device. No scope here closes those obligations.

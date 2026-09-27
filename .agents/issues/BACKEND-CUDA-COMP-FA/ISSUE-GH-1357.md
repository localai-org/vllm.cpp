ID: ISSUE-GH-1357
Title: `src/vllm/platforms/cuda.cpp::CudaPlatform::supports_fa2_attention` returns `true` unconditionally for every CUDA device, and is consumed at `src/vllm/model_executor/models/qwen3_5.cpp:5163` to gate `fa2_prefill`/`fa2_decode` and therefore `attn_dt` — bf16 FlashAttention-2 versus the f32 graph-captured fallback, on the default decode path. `CMakeLists.txt:169` defaults `VLLM_CPP_CUDA_ARCHITECTURES` to `121a` alone and `cmake/CudaArchFeatures.cmake:349` narrows FA2 to the intersection with the `fa2` row `8.0,8.6,8.7,8.9,12.0a,12.1a`, so a default build carries FA2 cubins for ONE architecture while the predicate promises FA2 to all of them; run it on an sm_86 card and the model takes a path with no SASS for the device. Same class of claim that made vLLM select an unrunnable FlashAttention on a GB10 ([#1332](https://github.com/mudler/vllm.cpp/issues/1332)), where requesting `FLASHINFER` generates text and exits 0 while the default resolves `FLASH_ATTN` and dies at the first attention call with `cudaErrorUnsupportedPtxVersion`. Fixable here in a way upstream cannot manage, because the answer already exists in the build: `vt_cuda_feature_archs(VT_FA2_ARCHS "fa2")` (`CMakeLists.txt:499`) computes the compiled FA2 arch set and `CMakeLists.txt:2279` passes that SAME variable to `vt_cuda_set_source_gencode` to emit the `-gencode` flags, so a `configure_file` manifest is DERIVED from the flag computation rather than tracking it. A hand-written list would be `CUDA_SUPPORTED_ARCHS` again and is refused. Spec [`cuda-compiled-arch-manifest.md`](../../specs/cuda-compiled-arch-manifest.md)
Row: BACKEND-CUDA-COMP-FA
State: UNKNOWN
Kind: bug
GitHub: 1357
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:446`

### Frozen archive evidence

> | [#1357](https://github.com/mudler/vllm.cpp/issues/1357) | `BACKEND-CUDA-COMP-FA` | `src/vllm/platforms/cuda.cpp::CudaPlatform::supports_fa2_attention` returns `true` unconditionally for every CUDA device, and is consumed at `src/vllm/model_executor/models/qwen3_5.cpp:5163` to gate `fa2_prefill`/`fa2_decode` and therefore `attn_dt` — bf16 FlashAttention-2 versus the f32 graph-captured fallback, on the default decode path. `CMakeLists.txt:169` defaults `VLLM_CPP_CUDA_ARCHITECTURES` to `121a` alone and `cmake/CudaArchFeatures.cmake:349` narrows FA2 to the intersection with the `fa2` row `8.0,8.6,8.7,8.9,12.0a,12.1a`, so a default build carries FA2 cubins for ONE architecture while the predicate promises FA2 to all of them; run it on an sm_86 card and the model takes a path with no SASS for the device. Same class of claim that made vLLM select an unrunnable FlashAttention on a GB10 ([#1332](https://github.com/mudler/vllm.cpp/issues/1332)), where requesting `FLASHINFER` generates text and exits 0 while the default resolves `FLASH_ATTN` and dies at the first attention call with `cudaErrorUnsupportedPtxVersion`. Fixable here in a way upstream cannot manage, because the answer already exists in the build: `vt_cuda_feature_archs(VT_FA2_ARCHS "fa2")` (`CMakeLists.txt:499`) computes the compiled FA2 arch set and `CMakeLists.txt:2279` passes that SAME variable to `vt_cuda_set_source_gencode` to emit the `-gencode` flags, so a `configure_file` manifest is DERIVED from the flag computation rather than tracking it. A hand-written list would be `CUDA_SUPPORTED_ARCHS` again and is refused. Spec [`cuda-compiled-arch-manifest.md`](../../specs/cuda-compiled-arch-manifest.md) | bug |

## Resolution

-

ID: ISSUE-GH-2260
Title: **CUDA has no keep-quant kernel for IQ2_XS or IQ4_XS, so the GLM-5.3-Flash artifact FITS `dgx:gpu0` and does not RUN there — the expert GEMM falls back to the CPU, and the fused seam throws.** Found reviewing [#2256](https://github.com/mudler/vllm.cpp/pull/2256), which lands the two CPU keep-quant `vec_dot` kernels and thereby flips the artifact's 82 IQ2_XS and 3 IQ4_XS tensors from `kExpandBf16` to `kKeepQuant`, taking resident cost 426.72 -> 101.14 GiB. `IsCudaKeepQuantSupported` admits ten Q8_K-family encodings (IQ2_XXS, IQ3_XXS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_S, IQ1_S, IQ1_XXXS) and neither of these two, while `DeviceKeepQuantSupported` returns `true` for CUDA on its `default:` arm regardless, on the recorded ground that CUDA falls back to the CPU kernel for anything it lacks. So `MatmulBTQuantGroupedKernelCuda` round-trips every grouped expert GEMM to the host cores behind a full `cudaStreamSynchronize`, and `MoeGateUpSwiGLUGroupedCuda` THROWS `gate/up must be the SAME CUDA keep-quant dtype` because `MergedGemm` selects the fused op on device registration alone with no dtype predicate. NOT reachable today — `glm5_next_moe.cpp` is W5's host reference and `laguna.cpp` is the only model on the fused seam — so #2256 breaks nothing; it becomes live when AGENTS.md's `vt::MergedGemmGroup` routing lands in W5b ([#2241](https://github.com/mudler/vllm.cpp/issues/2241)) / W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)), and a 101 GiB-resident model then throws at first forward. Three options in the issue: port the two CUDA kernels, keep expanding these two on CUDA (honest, but the artifact stops fitting), or refuse by name at load instead of throwing with the model resident. Owning rows `QUANT-GGUF-IQ2_XS` and `QUANT-GGUF-IQ4_XS` in [`quantization-matrix.md`](../../quantization-matrix.md), both carrying the disclosure in place; also carried as **O19** under `## Owed` in [`specs/glm5-next-flash.md`](../../specs/glm5-next-flash.md). Campaign [#1998](https://github.com/mudler/vllm.cpp/issues/1998)
Row: QUANT-GGUF-IQ2_XS
State: UNKNOWN
Kind: bug
GitHub: 2260
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:881`

### Frozen archive evidence

> | [#2260](https://github.com/mudler/vllm.cpp/issues/2260) | `QUANT-GGUF-IQ2_XS` | **CUDA has no keep-quant kernel for IQ2_XS or IQ4_XS, so the GLM-5.3-Flash artifact FITS `dgx:gpu0` and does not RUN there — the expert GEMM falls back to the CPU, and the fused seam throws.** Found reviewing [#2256](https://github.com/mudler/vllm.cpp/pull/2256), which lands the two CPU keep-quant `vec_dot` kernels and thereby flips the artifact's 82 IQ2_XS and 3 IQ4_XS tensors from `kExpandBf16` to `kKeepQuant`, taking resident cost 426.72 -> 101.14 GiB. `IsCudaKeepQuantSupported` admits ten Q8_K-family encodings (IQ2_XXS, IQ3_XXS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_S, IQ1_S, IQ1_XXXS) and neither of these two, while `DeviceKeepQuantSupported` returns `true` for CUDA on its `default:` arm regardless, on the recorded ground that CUDA falls back to the CPU kernel for anything it lacks. So `MatmulBTQuantGroupedKernelCuda` round-trips every grouped expert GEMM to the host cores behind a full `cudaStreamSynchronize`, and `MoeGateUpSwiGLUGroupedCuda` THROWS `gate/up must be the SAME CUDA keep-quant dtype` because `MergedGemm` selects the fused op on device registration alone with no dtype predicate. NOT reachable today — `glm5_next_moe.cpp` is W5's host reference and `laguna.cpp` is the only model on the fused seam — so #2256 breaks nothing; it becomes live when AGENTS.md's `vt::MergedGemmGroup` routing lands in W5b ([#2241](https://github.com/mudler/vllm.cpp/issues/2241)) / W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)), and a 101 GiB-resident model then throws at first forward. Three options in the issue: port the two CUDA kernels, keep expanding these two on CUDA (honest, but the artifact stops fitting), or refuse by name at load instead of throwing with the model resident. Owning rows `QUANT-GGUF-IQ2_XS` and `QUANT-GGUF-IQ4_XS` in [`quantization-matrix.md`](../../quantization-matrix.md), both carrying the disclosure in place; also carried as **O19** under `## Owed` in [`specs/glm5-next-flash.md`](../../specs/glm5-next-flash.md). Campaign [#1998](https://github.com/mudler/vllm.cpp/issues/1998) | bug |

## Resolution

-

ID: ISSUE-GH-1593
Title: **`KV-FP8` W2 and W3: the CUDA fp8 KV store, its paged-attention read, and the runner integration.** W1 landed the CPU half (`vt::ReshapeAndCacheFp8`, the read dequant in CPU paged attention, `vllm::v1::ParseCacheDType`) and left W2/W3/W4 `later`. The issue is now the critical path of benchmark campaign [#1574](https://github.com/mudler/vllm.cpp/issues/1574), whose subject `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` declares `kv_cache_quant_algo: "FP8"` and carries ZERO `k_scale`/`v_scale` tensors, so every published profile serves it with `--kv-cache-dtype fp8` and no cell can be served correctly without this. **W2 IS LANDED HERE**: the CUDA fp8-e4m3 store (`src/vt/cuda/cuda_cache.cu`), the fp8 dequant on the CUDA paged-attention read (`src/vt/cuda/cuda_paged_attn.cu` `LoadKv` + `LaunchPagedFp8`), the removal of the two W1 device-class refusals that made the CUDA arm unreachable however well it was registered, and a named CPU-or-CUDA refusal for the READ because it rides ADDITIVE `PagedAttentionArgs` fields on an op `kMETAL`/`kROCM` already register for the FLOAT path — without which an fp8 cache would be read as that backend's float dtype and return silent garbage. Gate `tests/vt/test_cuda_fp8_kv_cache.cpp`, RED-first on the provider-routing case. **The device half of that gate is UNEXECUTED and the CUDA TUs are UNCOMPILED**: the implementing session had no `nvcc` and no device, and says so under `## Owed` in [fp8-kv-cache.md](../../specs/fp8-kv-cache.md) together with the reachability debt — nothing calls the fp8 KV path from a production entry point on either backend, which is **W3's** wiring (half-sized KV blocks, `--kv-cache-dtype` threading, the checkpoint scale path including this checkpoint's scales-absent case). W3, W4, the Metal/ROCm arms and fp8_e5m2 remain owed
Row: KV-FP8
State: UNKNOWN
Kind: feature
GitHub: 1593
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:570`

### Frozen archive evidence

> | [#1593](https://github.com/mudler/vllm.cpp/issues/1593) | `KV-FP8` | **`KV-FP8` W2 and W3: the CUDA fp8 KV store, its paged-attention read, and the runner integration.** W1 landed the CPU half (`vt::ReshapeAndCacheFp8`, the read dequant in CPU paged attention, `vllm::v1::ParseCacheDType`) and left W2/W3/W4 `later`. The issue is now the critical path of benchmark campaign [#1574](https://github.com/mudler/vllm.cpp/issues/1574), whose subject `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` declares `kv_cache_quant_algo: "FP8"` and carries ZERO `k_scale`/`v_scale` tensors, so every published profile serves it with `--kv-cache-dtype fp8` and no cell can be served correctly without this. **W2 IS LANDED HERE**: the CUDA fp8-e4m3 store (`src/vt/cuda/cuda_cache.cu`), the fp8 dequant on the CUDA paged-attention read (`src/vt/cuda/cuda_paged_attn.cu` `LoadKv` + `LaunchPagedFp8`), the removal of the two W1 device-class refusals that made the CUDA arm unreachable however well it was registered, and a named CPU-or-CUDA refusal for the READ because it rides ADDITIVE `PagedAttentionArgs` fields on an op `kMETAL`/`kROCM` already register for the FLOAT path — without which an fp8 cache would be read as that backend's float dtype and return silent garbage. Gate `tests/vt/test_cuda_fp8_kv_cache.cpp`, RED-first on the provider-routing case. **The device half of that gate is UNEXECUTED and the CUDA TUs are UNCOMPILED**: the implementing session had no `nvcc` and no device, and says so under `## Owed` in [fp8-kv-cache.md](../../specs/fp8-kv-cache.md) together with the reachability debt — nothing calls the fp8 KV path from a production entry point on either backend, which is **W3's** wiring (half-sized KV blocks, `--kv-cache-dtype` threading, the checkpoint scale path including this checkpoint's scales-absent case). W3, W4, the Metal/ROCm arms and fp8_e5m2 remain owed | feature |

## Resolution

-

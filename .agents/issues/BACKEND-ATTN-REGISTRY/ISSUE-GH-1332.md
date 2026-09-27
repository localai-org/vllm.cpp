ID: ISSUE-GH-1332
Title: The attention-backend selector routes nothing, and a capability check is not a runnability check. `vllm::v1::SelectAttentionBackendName` (`src/vllm/v1/attention/registry.cpp`) mirrors vLLM's priority walk faithfully, and its result reaches only `attn_backend_names_`, a `VT_ATTN_SELECT_LOG` print and `CheckKvCacheShape` in `runner.cpp::GpuModelRunner::InitializeKvCache`; `dense_attn::AttnBlock` calls `vt::PagedAttention` unconditionally and the real arm choice is an env-flag + shape + dtype ladder in `src/vt/cuda/cuda_paged_attn.cu`. Deleting the whole selector would leave every emitted token identical, which is `.agents/reachability.md`'s "unselected branch" and "unpassed parameter" at once. Separately, measured on a GB10 (capability 12,1): vLLM selects `FLASH_ATTN` because `supports_compute_capability` is `capability >= (8,0)`, while the shipped FA2 binary carries `sm_80` SASS plus `compute_80` PTX alone, so every launch needs a driver JIT that fails with `cudaErrorUnsupportedPtxVersion` — `grep -rn get_arch_list vllm/` returns zero hits, so vLLM never asks what its own fatbins contain. We have written the same class of check: `src/vllm/platforms/cuda.cpp::CudaPlatform::supports_fa2_attention` returns true for every CUDA device while `CMakeLists.txt` defaults `VLLM_CPP_CUDA_ARCHITECTURES` to `121a` alone. The invariant: no backend may be declared valid on the strength of a property of the DEVICE alone; every predicate naming a compute capability must be paired with one naming the binary. M0 (reconcile) and M1 (the `validate_configuration` capability surface) land in [`attn-validate-configuration.md`](../../specs/attn-validate-configuration.md); the compiled-arch manifest, the launch probe, the dispatch wiring and the ABI override remain owed there under `## Owed`
Row: BACKEND-ATTN-REGISTRY
State: UNKNOWN
Kind: bug
GitHub: 1332
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:450`

### Frozen archive evidence

> | [#1332](https://github.com/mudler/vllm.cpp/issues/1332) | `BACKEND-ATTN-REGISTRY` | The attention-backend selector routes nothing, and a capability check is not a runnability check. `vllm::v1::SelectAttentionBackendName` (`src/vllm/v1/attention/registry.cpp`) mirrors vLLM's priority walk faithfully, and its result reaches only `attn_backend_names_`, a `VT_ATTN_SELECT_LOG` print and `CheckKvCacheShape` in `runner.cpp::GpuModelRunner::InitializeKvCache`; `dense_attn::AttnBlock` calls `vt::PagedAttention` unconditionally and the real arm choice is an env-flag + shape + dtype ladder in `src/vt/cuda/cuda_paged_attn.cu`. Deleting the whole selector would leave every emitted token identical, which is `.agents/reachability.md`'s "unselected branch" and "unpassed parameter" at once. Separately, measured on a GB10 (capability 12,1): vLLM selects `FLASH_ATTN` because `supports_compute_capability` is `capability >= (8,0)`, while the shipped FA2 binary carries `sm_80` SASS plus `compute_80` PTX alone, so every launch needs a driver JIT that fails with `cudaErrorUnsupportedPtxVersion` — `grep -rn get_arch_list vllm/` returns zero hits, so vLLM never asks what its own fatbins contain. We have written the same class of check: `src/vllm/platforms/cuda.cpp::CudaPlatform::supports_fa2_attention` returns true for every CUDA device while `CMakeLists.txt` defaults `VLLM_CPP_CUDA_ARCHITECTURES` to `121a` alone. The invariant: no backend may be declared valid on the strength of a property of the DEVICE alone; every predicate naming a compute capability must be paired with one naming the binary. M0 (reconcile) and M1 (the `validate_configuration` capability surface) land in [`attn-validate-configuration.md`](../../specs/attn-validate-configuration.md); the compiled-arch manifest, the launch probe, the dispatch wiring and the ABI override remain owed there under `## Owed` | bug |

## Resolution

-

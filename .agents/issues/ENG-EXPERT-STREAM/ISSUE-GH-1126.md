ID: ISSUE-GH-1126
Title: `CudaBackend` never overrides `Backend::DeviceMemoryInfo`, and the seam's own comment says it does: `include/vt/backend.h:79-83` reads "ROCm/CUDA override with hipMemGetInfo/cudaMemGetInfo" while only `src/vt/rocm/rocm_backend.hip:338-345` does, and `cudaMemGetInfo` is called NOWHERE in the repository. The comment is corrected in prose by [#1123](https://github.com/mudler/vllm.cpp/issues/1123); the capability is this issue. The consequence is not only a comment: `Gemma4MoE` is the seam's only consumer, `FreeBytes` returns false on an absent probe (`src/vllm/model_executor/models/gemma4_moe.cpp:439-447`) and `MakeRoom` refuses on unknown by design (`:494-506`), so on EVERY CUDA device the device-expert LRU (`kMaxSlots = 24`, `kHeadroom = 1.5 GiB`) admits nothing and falls back to host H2D permanently, silently. That polarity is right for that call site; the defect is the missing probe. #1123 therefore probed `cudaMemGetInfo` in `CudaPlatform` (which already includes `<cuda_runtime.h>` and already probes attributes at registration) and carried the total on `ResidencyPolicy`, touching nothing Gemma4 reads, because adding the override wakes another model's residency policy and that needs its own measurement. Measured on `dgx:gpu0`: total 128,452,956,160 (119.631 GiB), free 113.677 GiB, against `nvidia-smi` answering `[N/A], [N/A], [N/A]`. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md)
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: bug
GitHub: 1126
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:344`

### Frozen archive evidence

> | [#1126](https://github.com/mudler/vllm.cpp/issues/1126) | `ENG-EXPERT-STREAM` | `CudaBackend` never overrides `Backend::DeviceMemoryInfo`, and the seam's own comment says it does: `include/vt/backend.h:79-83` reads "ROCm/CUDA override with hipMemGetInfo/cudaMemGetInfo" while only `src/vt/rocm/rocm_backend.hip:338-345` does, and `cudaMemGetInfo` is called NOWHERE in the repository. The comment is corrected in prose by [#1123](https://github.com/mudler/vllm.cpp/issues/1123); the capability is this issue. The consequence is not only a comment: `Gemma4MoE` is the seam's only consumer, `FreeBytes` returns false on an absent probe (`src/vllm/model_executor/models/gemma4_moe.cpp:439-447`) and `MakeRoom` refuses on unknown by design (`:494-506`), so on EVERY CUDA device the device-expert LRU (`kMaxSlots = 24`, `kHeadroom = 1.5 GiB`) admits nothing and falls back to host H2D permanently, silently. That polarity is right for that call site; the defect is the missing probe. #1123 therefore probed `cudaMemGetInfo` in `CudaPlatform` (which already includes `<cuda_runtime.h>` and already probes attributes at registration) and carried the total on `ResidencyPolicy`, touching nothing Gemma4 reads, because adding the override wakes another model's residency policy and that needs its own measurement. Measured on `dgx:gpu0`: total 128,452,956,160 (119.631 GiB), free 113.677 GiB, against `nvidia-smi` answering `[N/A], [N/A], [N/A]`. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md) | bug |

## Resolution

-

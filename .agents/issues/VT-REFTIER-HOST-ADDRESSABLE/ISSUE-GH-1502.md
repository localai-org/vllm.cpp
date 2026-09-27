ID: ISSUE-GH-1502
Title: **`docs/ENVIRONMENT.md` described `VT_ADOPT_DEVICE_BYTES` as Vulkan-only and said it has "No effect on CUDA/CPU/Metal", and [`cffe59b02`](https://github.com/mudler/vllm.cpp/commit/cffe59b02) ([#1477](https://github.com/mudler/vllm.cpp/issues/1477)) made both halves false.** That change moved `ReferenceTierEligible` off `UnifiedMemory()` onto `Backend::DeviceMemoryIsHostAddressable()` and added truthful overrides so no backend lost the reference tier, so `MetalBackend` now answers `MetalContext::unified_memory()` and `RocmBackend` answers its `unified_memory_`. The weight loader gates the lever on exactly that predicate, at both `AdoptDeviceBytesAsHost` branches in `src/vllm/model_executor/models/qwen3_5_weights.cpp`, so the lever ACTS on Apple silicon and on an integrated ROCm part. **The correction is not "add two backend names".** Every number in that row is GB10 through Vulkan, and nobody has measured the lever on either new arm, so the row now separates the backends it is MEASURED on from the backends that merely satisfy the predicate — reach and measurement are different claims and the row read as if the measurement covered the reach. CUDA and CPU stay inert and are unchanged: neither overrides the default `false`, which `tests/vllm/platforms/test_platform.cpp` pins for GB10, and the CPU backend answering `UnifiedMemory() == true` while the narrower predicate stays `false` is the whole reason the two properties are separate. The MEASUREMENT on Metal and integrated ROCm stays owed and is listed under `## Owed` in [`vt-reference-tier-host-addressable.md`](../../specs/vt-reference-tier-host-addressable.md); it needs an Apple-silicon box or an integrated AMD part
Row: VT-REFTIER-HOST-ADDRESSABLE
State: UNKNOWN
Kind: documentation
GitHub: 1502
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:580`

### Frozen archive evidence

> | [#1502](https://github.com/mudler/vllm.cpp/issues/1502) | `VT-REFTIER-HOST-ADDRESSABLE` | **`docs/ENVIRONMENT.md` described `VT_ADOPT_DEVICE_BYTES` as Vulkan-only and said it has "No effect on CUDA/CPU/Metal", and [`cffe59b02`](https://github.com/mudler/vllm.cpp/commit/cffe59b02) ([#1477](https://github.com/mudler/vllm.cpp/issues/1477)) made both halves false.** That change moved `ReferenceTierEligible` off `UnifiedMemory()` onto `Backend::DeviceMemoryIsHostAddressable()` and added truthful overrides so no backend lost the reference tier, so `MetalBackend` now answers `MetalContext::unified_memory()` and `RocmBackend` answers its `unified_memory_`. The weight loader gates the lever on exactly that predicate, at both `AdoptDeviceBytesAsHost` branches in `src/vllm/model_executor/models/qwen3_5_weights.cpp`, so the lever ACTS on Apple silicon and on an integrated ROCm part. **The correction is not "add two backend names".** Every number in that row is GB10 through Vulkan, and nobody has measured the lever on either new arm, so the row now separates the backends it is MEASURED on from the backends that merely satisfy the predicate — reach and measurement are different claims and the row read as if the measurement covered the reach. CUDA and CPU stay inert and are unchanged: neither overrides the default `false`, which `tests/vllm/platforms/test_platform.cpp` pins for GB10, and the CPU backend answering `UnifiedMemory() == true` while the narrower predicate stays `false` is the whole reason the two properties are separate. The MEASUREMENT on Metal and integrated ROCm stays owed and is listed under `## Owed` in [`vt-reference-tier-host-addressable.md`](../../specs/vt-reference-tier-host-addressable.md); it needs an Apple-silicon box or an integrated AMD part | documentation |

## Resolution

-

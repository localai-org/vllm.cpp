ID: ISSUE-GH-1746
Title: **`apply_logits_processors` hands every ABI logits-processor callback a raw `cudaMalloc` pointer on CUDA/GB10, because it gates the staging bounce on `Backend::UnifiedMemory()` where the question is `Backend::DeviceMemoryIsHostAddressable()`.** `src/vllm/v1/sample/logits_processor/builtin.cpp` sets `host = logits.data` when the WIDE predicate holds, and `CudaBackend` answers it `pageable_memory_access && integrated`, which is true on GB10 over allocations `CudaBackend::Alloc` takes from `cudaMalloc`; CUDA never overrides the narrow predicate, so it keeps the base `false` from `include/vt/backend.h`, whose own comment says a backend must opt in "because being wrong here hands a device pointer to a host memcpy and segfaults". This is the [#844](https://github.com/mudler/vllm.cpp/issues/844) / [#1435](https://github.com/mudler/vllm.cpp/issues/1435) / [#960](https://github.com/mudler/vllm.cpp/issues/960) class in a second location, and `src/vt/op_provider.cpp` warns about it in the same tree. The CPU suite could not see it: on `Device{kCPU,0}` both predicates are true and the pointer really is host memory, so the wrong one reads correct. FIXED IN FLOW: the predicate narrows, the `else` staging arm is unchanged, and a new own-executable test carries the GB10 pair on a fake backend — `UnifiedMemory()` true, `DeviceMemoryIsHostAddressable()` false — entering through `Sampler::forward` rather than through the function, so the reachability mutation of the one production call site turns it red. A second case pins that a backend answering both predicates true keeps the zero-copy in-place wrap, so the fix does not become "always stage". Spec [`logits-processor-host-addressable.md`](../../specs/logits-processor-host-addressable.md)
Row: V1-LOGITSPROC-HOST-ADDRESSABLE
State: UNKNOWN
Kind: bug
GitHub: 1746
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:654`

### Frozen archive evidence

> | [#1746](https://github.com/mudler/vllm.cpp/issues/1746) | `V1-LOGITSPROC-HOST-ADDRESSABLE` | **`apply_logits_processors` hands every ABI logits-processor callback a raw `cudaMalloc` pointer on CUDA/GB10, because it gates the staging bounce on `Backend::UnifiedMemory()` where the question is `Backend::DeviceMemoryIsHostAddressable()`.** `src/vllm/v1/sample/logits_processor/builtin.cpp` sets `host = logits.data` when the WIDE predicate holds, and `CudaBackend` answers it `pageable_memory_access && integrated`, which is true on GB10 over allocations `CudaBackend::Alloc` takes from `cudaMalloc`; CUDA never overrides the narrow predicate, so it keeps the base `false` from `include/vt/backend.h`, whose own comment says a backend must opt in "because being wrong here hands a device pointer to a host memcpy and segfaults". This is the [#844](https://github.com/mudler/vllm.cpp/issues/844) / [#1435](https://github.com/mudler/vllm.cpp/issues/1435) / [#960](https://github.com/mudler/vllm.cpp/issues/960) class in a second location, and `src/vt/op_provider.cpp` warns about it in the same tree. The CPU suite could not see it: on `Device{kCPU,0}` both predicates are true and the pointer really is host memory, so the wrong one reads correct. FIXED IN FLOW: the predicate narrows, the `else` staging arm is unchanged, and a new own-executable test carries the GB10 pair on a fake backend — `UnifiedMemory()` true, `DeviceMemoryIsHostAddressable()` false — entering through `Sampler::forward` rather than through the function, so the reachability mutation of the one production call site turns it red. A second case pins that a backend answering both predicates true keeps the zero-copy in-place wrap, so the fix does not become "always stage". Spec [`logits-processor-host-addressable.md`](../../specs/logits-processor-host-addressable.md) | bug |

## Resolution

-

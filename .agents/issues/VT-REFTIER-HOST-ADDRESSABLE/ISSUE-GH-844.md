ID: ISSUE-GH-844
Title: **The portable CPU reference tier gates on `Backend::UnifiedMemory()`, which is not the property a host kernel needs, so a device tensor reaches a host kernel and the process gets SIGSEGV under a banner reading `correct but slow`.** The eligibility predicate is `ReferenceTierEligible` in `src/vt/op_provider.cpp`; grep the symbol, because this index is append-only and a line number here would be permanent. It asks `Backend::UnifiedMemory()`. The question a host kernel actually asks is `Backend::DeviceMemoryIsHostAddressable()`, whose own comment in `include/vt/backend.h` records the refutation: CUDA on GB10 reports unified memory because host and device address the same physical RAM, yet a plain `cudaMalloc` pointer is still not host-dereferenceable, and its default is `false` because "being wrong here hands a device pointer to a host memcpy and segfaults". `CudaBackend::Alloc` calls `cudaMalloc`, so every CUDA op without a native kernel installed the CPU host kernel and crashed. Measured twice on GB10: `vt::QuantFp8Static` on `sm_110` ([#960](https://github.com/mudler/vllm.cpp/issues/960)), whose spec `## Outcome` relocated the one registration and recorded that the CLASS stayed open; and `vt::MatmulFp8BlockScaled` on a CUTLASS-less CUDA build ([#1435](https://github.com/mudler/vllm.cpp/issues/1435)), which exits 139 with `test cases: 0 assertions: 0`. Fixed by asking the narrow predicate, and by `MetalBackend` and `RocmBackend` answering it truthfully so neither loses the tier: both answers are equal to what those backends report today, so only the CUDA cell moves. The refusal now names why the portable tier did not run, and the banner names its precondition instead of asserting correctness. Reproduced without a GPU in `tests/vt/test_reference_tier.cpp` by a fake backend carrying the GB10 property pair. Spec [`vt-reference-tier-host-addressable.md`](../../specs/vt-reference-tier-host-addressable.md)
Row: VT-REFTIER-HOST-ADDRESSABLE
State: UNKNOWN
Kind: bug
GitHub: 844
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:517`

### Frozen archive evidence

> | [#844](https://github.com/mudler/vllm.cpp/issues/844) | `VT-REFTIER-HOST-ADDRESSABLE` | **The portable CPU reference tier gates on `Backend::UnifiedMemory()`, which is not the property a host kernel needs, so a device tensor reaches a host kernel and the process gets SIGSEGV under a banner reading `correct but slow`.** The eligibility predicate is `ReferenceTierEligible` in `src/vt/op_provider.cpp`; grep the symbol, because this index is append-only and a line number here would be permanent. It asks `Backend::UnifiedMemory()`. The question a host kernel actually asks is `Backend::DeviceMemoryIsHostAddressable()`, whose own comment in `include/vt/backend.h` records the refutation: CUDA on GB10 reports unified memory because host and device address the same physical RAM, yet a plain `cudaMalloc` pointer is still not host-dereferenceable, and its default is `false` because "being wrong here hands a device pointer to a host memcpy and segfaults". `CudaBackend::Alloc` calls `cudaMalloc`, so every CUDA op without a native kernel installed the CPU host kernel and crashed. Measured twice on GB10: `vt::QuantFp8Static` on `sm_110` ([#960](https://github.com/mudler/vllm.cpp/issues/960)), whose spec `## Outcome` relocated the one registration and recorded that the CLASS stayed open; and `vt::MatmulFp8BlockScaled` on a CUTLASS-less CUDA build ([#1435](https://github.com/mudler/vllm.cpp/issues/1435)), which exits 139 with `test cases: 0 assertions: 0`. Fixed by asking the narrow predicate, and by `MetalBackend` and `RocmBackend` answering it truthfully so neither loses the tier: both answers are equal to what those backends report today, so only the CUDA cell moves. The refusal now names why the portable tier did not run, and the banner names its precondition instead of asserting correctness. Reproduced without a GPU in `tests/vt/test_reference_tier.cpp` by a fake backend carrying the GB10 property pair. Spec [`vt-reference-tier-host-addressable.md`](../../specs/vt-reference-tier-host-addressable.md) | bug |

## Resolution

-

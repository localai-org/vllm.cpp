ID: ISSUE-GH-1748
Title: **`CpuBackend` never opts in to `DeviceMemoryIsHostAddressable()`, so every reader of that predicate gets the conservative `false` on the one device where it is trivially true.** `src/vt/cpu/cpu_backend.cpp` overrides `UnifiedMemory()` to `true` and leaves the narrow predicate at the `include/vt/backend.h` default, although `CpuBackend::Alloc` returns ordinary aligned host memory, `Copy` is `std::memcpy`, and the class comment two lines above says "Host and device memory are the SAME allocation here". Found while grounding [#1746](https://github.com/mudler/vllm.cpp/issues/1746). Three readers: `ReferenceTierEligible` and `ReferenceTierRefusalReason` (`src/vt/op_provider.cpp`) never reach it, because both return earlier on `device == DeviceType::kCPU` — which is why the wrong answer stayed invisible; the direct-upload adoption in `src/vllm/model_executor/models/qwen3_5_weights.cpp` (two sites) returns early on it and therefore never adopts on CPU, and whether adoption is even meaningful there is NOT established and is part of what this issue owes; and `apply_logits_processors` reads it from #1746 onward, so CPU takes a staging bounce of `[n, vocab]` f32 down and back per step, charged only to a request that registered a processor. NOT fixed in flow: the one-line override flips the qwen3_5 residency path and needs its own red-before test and its own measurement, and a crash-class correctness repair must not carry an unmeasured residency change. Owed under [`logits-processor-host-addressable.md`](../../specs/logits-processor-host-addressable.md) `## Owed`
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1748
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:655`

### Frozen archive evidence

> | [#1748](https://github.com/mudler/vllm.cpp/issues/1748) | — | **`CpuBackend` never opts in to `DeviceMemoryIsHostAddressable()`, so every reader of that predicate gets the conservative `false` on the one device where it is trivially true.** `src/vt/cpu/cpu_backend.cpp` overrides `UnifiedMemory()` to `true` and leaves the narrow predicate at the `include/vt/backend.h` default, although `CpuBackend::Alloc` returns ordinary aligned host memory, `Copy` is `std::memcpy`, and the class comment two lines above says "Host and device memory are the SAME allocation here". Found while grounding [#1746](https://github.com/mudler/vllm.cpp/issues/1746). Three readers: `ReferenceTierEligible` and `ReferenceTierRefusalReason` (`src/vt/op_provider.cpp`) never reach it, because both return earlier on `device == DeviceType::kCPU` — which is why the wrong answer stayed invisible; the direct-upload adoption in `src/vllm/model_executor/models/qwen3_5_weights.cpp` (two sites) returns early on it and therefore never adopts on CPU, and whether adoption is even meaningful there is NOT established and is part of what this issue owes; and `apply_logits_processors` reads it from #1746 onward, so CPU takes a staging bounce of `[n, vocab]` f32 down and back per step, charged only to a request that registered a processor. NOT fixed in flow: the one-line override flips the qwen3_5 residency path and needs its own red-before test and its own measurement, and a crash-class correctness repair must not carry an unmeasured residency change. Owed under [`logits-processor-host-addressable.md`](../../specs/logits-processor-host-addressable.md) `## Owed` | bug |

## Resolution

-

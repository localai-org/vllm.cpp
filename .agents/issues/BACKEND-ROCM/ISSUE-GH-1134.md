ID: ISSUE-GH-1134
Title: The `VT_ATTN_DECODE_D128=1` ctest registration added by #767 cannot show the ROCm `d=128` decode arm REACHED the new kernel, and is empty off ROCm. `RegisteredDevices()` (`tests/vt/test_backend_cross_device.cpp:84-96`) enumerates `{kCUDA, kMETAL, kVULKAN, kXPU, kROCM}` and excludes `kCPU`, so on a CPU-only runner — which is what CI has — the new "Qwen3 geometry (bf16, GQA 2, head_dim 128)" case reports 1 test case, 0 assertions, exit 0, for BOTH registrations. On ROCm hardware the case's only backend assertion is `OpProviderStats::declines == 0`, and `OpProviderStats` counts at PROVIDER granularity, so it is identical with the flag set and unset; the NMSE bound passes on either kernel because the arm is correctness-complete. The two compose: there is no machine in this project on which the flag-ON registration distinguishes itself from the flag-OFF one. Disclosed in [`specs/rocm-decode-attn-d128.md`](../../specs/rocm-decode-attn-d128.md) §4, its `## Owed` section and its result banner, and §9 stop condition 2 is left OPEN rather than claimed discharged. Closing it needs a kernel-selection counter in `src/vt/rocm/rocm_paged_attn.hip` asserted to DIFFER between the two registrations; the CPU-runner half wants `kCPU` in `RegisteredDevices()` or a non-zero-assertion floor per [#463](https://github.com/mudler/vllm.cpp/issues/463). Not a duplicate of #463 (that is the unset-weights-env-var shape and does not describe the `declines` granularity half), #785 (a kernel that never LAUNCHES behind a dead `#if`, a code defect not a coverage one) or #900 (same family, LTX-2.5 subject)
Row: BACKEND-ROCM
State: UNKNOWN
Kind: bug
GitHub: 1134
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:336`

### Frozen archive evidence

> | [#1134](https://github.com/mudler/vllm.cpp/issues/1134) | `BACKEND-ROCM` | The `VT_ATTN_DECODE_D128=1` ctest registration added by #767 cannot show the ROCm `d=128` decode arm REACHED the new kernel, and is empty off ROCm. `RegisteredDevices()` (`tests/vt/test_backend_cross_device.cpp:84-96`) enumerates `{kCUDA, kMETAL, kVULKAN, kXPU, kROCM}` and excludes `kCPU`, so on a CPU-only runner — which is what CI has — the new "Qwen3 geometry (bf16, GQA 2, head_dim 128)" case reports 1 test case, 0 assertions, exit 0, for BOTH registrations. On ROCm hardware the case's only backend assertion is `OpProviderStats::declines == 0`, and `OpProviderStats` counts at PROVIDER granularity, so it is identical with the flag set and unset; the NMSE bound passes on either kernel because the arm is correctness-complete. The two compose: there is no machine in this project on which the flag-ON registration distinguishes itself from the flag-OFF one. Disclosed in [`specs/rocm-decode-attn-d128.md`](../../specs/rocm-decode-attn-d128.md) §4, its `## Owed` section and its result banner, and §9 stop condition 2 is left OPEN rather than claimed discharged. Closing it needs a kernel-selection counter in `src/vt/rocm/rocm_paged_attn.hip` asserted to DIFFER between the two registrations; the CPU-runner half wants `kCPU` in `RegisteredDevices()` or a non-zero-assertion floor per [#463](https://github.com/mudler/vllm.cpp/issues/463). Not a duplicate of #463 (that is the unset-weights-env-var shape and does not describe the `declines` granularity half), #785 (a kernel that never LAUNCHES behind a dead `#if`, a code defect not a coverage one) or #900 (same family, LTX-2.5 subject) | bug |

## Resolution

-

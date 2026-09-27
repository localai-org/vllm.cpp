ID: ISSUE-GH-955
Title: sm_110 (Jetson Thor) `ctest` baseline at `2daa3287f`: 468 passed / 2 skipped / **15 red** of 485. Four are the build honestly refusing what the arch lacks — `built without the vendored FlashAttention-2` on MLA prefill. Two hardcode GB10: `test_platform:307` asserts `is_device_capability_family(120)`, and `test_op_parity:2487` replays a dgx-captured `output_cbor_sha256` in a case that names itself dgx-only and runs anyway. `test_capi`s SIGSEGV and `test_linear_method`s un-run MXFP4 fused path are already red on GB10 ([#907](https://github.com/mudler/vllm.cpp/issues/907)). Six are the FP8 fallback crash split out as [#960](https://github.com/mudler/vllm.cpp/issues/960). The one substantive standing finding is `test_ops_moe_grouped:1144`, `NVFP4 block8-vs-block16 M=8 K=4096 N=4096 bitdiff=15/32768` — `marlin-nvfp4` IS enabled for `[110]`, so that is a live kernel disagreeing with itself, not an absent feature. Recipe and table in [`environment.md`](../../environment.md)
Row: BACKEND-CUDA-SM110
State: UNKNOWN
Kind: bug
GitHub: 955
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:263`

### Frozen archive evidence

> | [#955](https://github.com/mudler/vllm.cpp/issues/955) | `BACKEND-CUDA-SM110` | sm_110 (Jetson Thor) `ctest` baseline at `2daa3287f`: 468 passed / 2 skipped / **15 red** of 485. Four are the build honestly refusing what the arch lacks — `built without the vendored FlashAttention-2` on MLA prefill. Two hardcode GB10: `test_platform:307` asserts `is_device_capability_family(120)`, and `test_op_parity:2487` replays a dgx-captured `output_cbor_sha256` in a case that names itself dgx-only and runs anyway. `test_capi`s SIGSEGV and `test_linear_method`s un-run MXFP4 fused path are already red on GB10 ([#907](https://github.com/mudler/vllm.cpp/issues/907)). Six are the FP8 fallback crash split out as [#960](https://github.com/mudler/vllm.cpp/issues/960). The one substantive standing finding is `test_ops_moe_grouped:1144`, `NVFP4 block8-vs-block16 M=8 K=4096 N=4096 bitdiff=15/32768` — `marlin-nvfp4` IS enabled for `[110]`, so that is a live kernel disagreeing with itself, not an absent feature. Recipe and table in [`environment.md`](../../environment.md) | bug |

## Resolution

-

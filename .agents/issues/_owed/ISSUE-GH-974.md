ID: ISSUE-GH-974
Title: The FP8 W8A8 resident helpers move weight bytes host->device without `vllm::load_stats::AddDeviceUpload` and without the post-upload `AdoptDeviceBytesAsHost`, while every other resident-weight helper in the same file performs both: `ResidentWeight` (`src/vllm/model_executor/models/qwen3_5.cpp:1009,1016 @ c7cb59fbb`) and `ResidentNvfp4` (`:1106,1111,1116,1121`), whose own comment states the obligation against [#150](https://github.com/mudler/vllm.cpp/issues/150). Affects `ResidentFp8` (`:1458`, now `dense_fp8_gemm.h`), `ResidentFp8Qkv` (`:1555`) and `ResidentFp8Qkvz` (`:3440`). Two unmeasured consequences: load accounting is short by the whole fp8 tower, and its device pages are never re-tagged, which is the shape of the GB10 weight-residency penalty on the 27B decode gap's largest attributed bucket. Found while extracting those entry points in [#940](https://github.com/mudler/vllm.cpp/issues/940) and deliberately NOT fixed there: a byte-identity gate cannot see a device -behaviour change hidden in a move. Listed under `## Owed` in [`vt-fp8-shared-seam.md`](../../specs/vt-fp8-shared-seam.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 974
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:271`

### Frozen archive evidence

> | [#974](https://github.com/mudler/vllm.cpp/issues/974) | — | The FP8 W8A8 resident helpers move weight bytes host->device without `vllm::load_stats::AddDeviceUpload` and without the post-upload `AdoptDeviceBytesAsHost`, while every other resident-weight helper in the same file performs both: `ResidentWeight` (`src/vllm/model_executor/models/qwen3_5.cpp:1009,1016 @ c7cb59fbb`) and `ResidentNvfp4` (`:1106,1111,1116,1121`), whose own comment states the obligation against [#150](https://github.com/mudler/vllm.cpp/issues/150). Affects `ResidentFp8` (`:1458`, now `dense_fp8_gemm.h`), `ResidentFp8Qkv` (`:1555`) and `ResidentFp8Qkvz` (`:3440`). Two unmeasured consequences: load accounting is short by the whole fp8 tower, and its device pages are never re-tagged, which is the shape of the GB10 weight-residency penalty on the 27B decode gap's largest attributed bucket. Found while extracting those entry points in [#940](https://github.com/mudler/vllm.cpp/issues/940) and deliberately NOT fixed there: a byte-identity gate cannot see a device -behaviour change hidden in a move. Listed under `## Owed` in [`vt-fp8-shared-seam.md`](../../specs/vt-fp8-shared-seam.md) | bug |

## Resolution

-

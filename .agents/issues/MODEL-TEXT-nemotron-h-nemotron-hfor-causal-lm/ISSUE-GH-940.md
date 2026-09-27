ID: ISSUE-GH-940
Title: The FP8 W8A8 linear path is not a shared seam: `ResidentFp8`, `MatmulFp8CutlassD` and `MatmulFp8CutlassPreQuantD` lived in the anonymous namespace of `src/vllm/model_executor/models/qwen3_5.cpp` (`:1458`, `:1495`, `:1517` @ `c7cb59fbb`), so a second model could reach them only by copying them — the hand-rolled parallel path AGENTS.md §"Shared seams" forbids. NVFP4 already had `dense_nvfp4_gemm.h` + `compressed_tensors/schemes/nvfp4.h`; FP8 had neither half. Forced by `MODEL-NEMOTRON-H` ([#517](https://github.com/mudler/vllm.cpp/issues/517)), whose 46 FP8 W8A8 mamba `in_proj`/`out_proj` projections are 36.6% of decode bytes and 27.6% of GEMM FLOPs, and whose `in_proj` produces the fused `zxbcdt` the conv and the SSD scan consume — so that block cannot be split and has no device path at all without the seam. Extracted to `dense_fp8_gemm.h` + `quantization/fp8.h` with Qwen3.5 byte-identity as the gate; spec [`vt-fp8-shared-seam.md`](../../specs/vt-fp8-shared-seam.md)
Row: MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 940
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:270`

### Frozen archive evidence

> | [#940](https://github.com/mudler/vllm.cpp/issues/940) | `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` | The FP8 W8A8 linear path is not a shared seam: `ResidentFp8`, `MatmulFp8CutlassD` and `MatmulFp8CutlassPreQuantD` lived in the anonymous namespace of `src/vllm/model_executor/models/qwen3_5.cpp` (`:1458`, `:1495`, `:1517` @ `c7cb59fbb`), so a second model could reach them only by copying them — the hand-rolled parallel path AGENTS.md §"Shared seams" forbids. NVFP4 already had `dense_nvfp4_gemm.h` + `compressed_tensors/schemes/nvfp4.h`; FP8 had neither half. Forced by `MODEL-NEMOTRON-H` ([#517](https://github.com/mudler/vllm.cpp/issues/517)), whose 46 FP8 W8A8 mamba `in_proj`/`out_proj` projections are 36.6% of decode bytes and 27.6% of GEMM FLOPs, and whose `in_proj` produces the fused `zxbcdt` the conv and the SSD scan consume — so that block cannot be split and has no device path at all without the seam. Extracted to `dense_fp8_gemm.h` + `quantization/fp8.h` with Qwen3.5 byte-identity as the gate; spec [`vt-fp8-shared-seam.md`](../../specs/vt-fp8-shared-seam.md) | bug |

## Resolution

-

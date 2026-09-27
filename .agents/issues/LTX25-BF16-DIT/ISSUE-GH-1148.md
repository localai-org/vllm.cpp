ID: ISSUE-GH-1148
Title: `PlanDit` refused every DiT checkpoint carrying neither `U8` nor `F8_E4M3` (`src/vllm/model_executor/models/ltx2_loader.cpp:414-418` @ `c83b96934`), so `ltx-2.5-22b-dev-transformer-bf16.safetensors` — the FULL model, 42,018,190,584 bytes, 4349 tensors, 4059 BF16 / 290 F32, ZERO `_scale` names, measured from its own header on 2026-08-17 — could not be read at all. Upstream's pipeline table (`packages/ltx-pipelines/CLAUDE.md:17-30` @ `fd4ded7f`) names that model for `TI2VidOneStagePipeline`, `T2AOneStagePipeline`, `TI2VidTwoStagesPipeline`, `TI2VidTwoStagesHQPipeline`, `A2VidPipelineTwoStage` and `KeyframeInterpolationPipeline`, so `one_stage`, `t2a_one_stage`, `res2s_two_stage` and `a2vid_two_stage` were all landed and all runnable only against a DISTILLED checkpoint, which is a different sampling regime that renders plausibly. The refusal's own advice was unreachable: it said "use the L2 path" and `Ltx2LoadDitFromSafetensors` IS the L2 path, calling `PlanDit` on its first line, as do `Ltx2ParseDitParamsFromCheckpoint`, `Ltx2StreamDitToDevice` and `Ltx2RebindDitLoras`. Upstream has no third quant state — `_DTYPE_CASTABLE` (`single_gpu_model_builder.py:51-57`) lists float32/float64/float16/bfloat16 and calls uint8-NVFP4 and float8 "quantized payloads", so unquantized is the BASELINE the two arms are exceptions to — and `MaterializeDitTensor` already carried a `BF16` branch, so the wall was the one `if` and nothing behind it. Fixed by `Ltx2DitQuant::kNone`, with the refusal that survives naming the dtypes the file holds. Spec [`ltx25-bf16-dit.md`](../../specs/ltx25-bf16-dit.md)
Row: LTX25-BF16-DIT
State: UNKNOWN
Kind: bug
GitHub: 1148
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:353`

### Frozen archive evidence

> | [#1148](https://github.com/mudler/vllm.cpp/issues/1148) | `LTX25-BF16-DIT` | `PlanDit` refused every DiT checkpoint carrying neither `U8` nor `F8_E4M3` (`src/vllm/model_executor/models/ltx2_loader.cpp:414-418` @ `c83b96934`), so `ltx-2.5-22b-dev-transformer-bf16.safetensors` — the FULL model, 42,018,190,584 bytes, 4349 tensors, 4059 BF16 / 290 F32, ZERO `_scale` names, measured from its own header on 2026-08-17 — could not be read at all. Upstream's pipeline table (`packages/ltx-pipelines/CLAUDE.md:17-30` @ `fd4ded7f`) names that model for `TI2VidOneStagePipeline`, `T2AOneStagePipeline`, `TI2VidTwoStagesPipeline`, `TI2VidTwoStagesHQPipeline`, `A2VidPipelineTwoStage` and `KeyframeInterpolationPipeline`, so `one_stage`, `t2a_one_stage`, `res2s_two_stage` and `a2vid_two_stage` were all landed and all runnable only against a DISTILLED checkpoint, which is a different sampling regime that renders plausibly. The refusal's own advice was unreachable: it said "use the L2 path" and `Ltx2LoadDitFromSafetensors` IS the L2 path, calling `PlanDit` on its first line, as do `Ltx2ParseDitParamsFromCheckpoint`, `Ltx2StreamDitToDevice` and `Ltx2RebindDitLoras`. Upstream has no third quant state — `_DTYPE_CASTABLE` (`single_gpu_model_builder.py:51-57`) lists float32/float64/float16/bfloat16 and calls uint8-NVFP4 and float8 "quantized payloads", so unquantized is the BASELINE the two arms are exceptions to — and `MaterializeDitTensor` already carried a `BF16` branch, so the wall was the one `if` and nothing behind it. Fixed by `Ltx2DitQuant::kNone`, with the refusal that survives naming the dtypes the file holds. Spec [`ltx25-bf16-dit.md`](../../specs/ltx25-bf16-dit.md) | bug |

## Resolution

-

ID: ISSUE-GH-1005
Title: LTX-2.5 text-to-audio (`T2AOneStagePipeline`, `t2a_one_stage.py:43`, `__call__` at `:109` @ `fd4ded7f`) is absent, and the three blockers are not the ones a reader would guess. (a) `Ltx2DitForward` refuses a one-stream call at `src/vllm/model_executor/models/ltx2_dit.cpp:765 @ 332aed738`, citing a weight contract that describes a checkpoint T2A never loads — upstream reads the ordinary AudioVideo FILE through `LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP` (`model_configurator.py:228-239`). (b) The same message advises `enabled=false` as the substitute, and it is NOT: `run_v2a = run_ax and (video is not None and vx.numel() > 0)` (`transformer.py:269`) tests PRESENCE, not `enabled`, so a disabled-but-present video stream still feeds v2a cross attention and still returns a finished waveform; our port mirrors that polarity at `src/vllm/model_executor/models/ltx2_dit.cpp:251 @ 332aed738`. (c) The engine has NO guided denoiser at all — `git grep -n 'guid\|cfg_scale' src/vllm/multimodal/ltx2_video.cpp` is 0 against 66 for `ltx2` in the same file as the control — while T2A defaults to `cfg_scale=7.0` and `stg_scale=1.0`, i.e. THREE forwards per step. Spec [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md)
Row: LTX25-T2A-ONE-STAGE
State: UNKNOWN
Kind: feature
GitHub: 1005
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:300`

### Frozen archive evidence

> | [#1005](https://github.com/mudler/vllm.cpp/issues/1005) | `LTX25-T2A-ONE-STAGE` | LTX-2.5 text-to-audio (`T2AOneStagePipeline`, `t2a_one_stage.py:43`, `__call__` at `:109` @ `fd4ded7f`) is absent, and the three blockers are not the ones a reader would guess. (a) `Ltx2DitForward` refuses a one-stream call at `src/vllm/model_executor/models/ltx2_dit.cpp:765 @ 332aed738`, citing a weight contract that describes a checkpoint T2A never loads — upstream reads the ordinary AudioVideo FILE through `LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP` (`model_configurator.py:228-239`). (b) The same message advises `enabled=false` as the substitute, and it is NOT: `run_v2a = run_ax and (video is not None and vx.numel() > 0)` (`transformer.py:269`) tests PRESENCE, not `enabled`, so a disabled-but-present video stream still feeds v2a cross attention and still returns a finished waveform; our port mirrors that polarity at `src/vllm/model_executor/models/ltx2_dit.cpp:251 @ 332aed738`. (c) The engine has NO guided denoiser at all — `git grep -n 'guid\\|cfg_scale' src/vllm/multimodal/ltx2_video.cpp` is 0 against 66 for `ltx2` in the same file as the control — while T2A defaults to `cfg_scale=7.0` and `stg_scale=1.0`, i.e. THREE forwards per step. Spec [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md) | feature |

## Resolution

-

ID: ISSUE-GH-1092
Title: The LTX-2.5 **video** denoise loop runs one UNGUIDED forward per step: `Ltx2PhaseRecipe::video_guidance` is set by every recipe (`src/vllm/model_executor/models/ltx2_pipeline.cpp:1069 @ b5756ea8c`) and read by nothing, so a `pipeline_kind = one_stage` render ignores `cfg_scale = 3.0`, `stg_scale = 1.0`, `rescale_scale = 0.7` and `modality_scale = 3.0` and denoises along a different trajectory than `ti2vid_one_stage.py:221-226 @ fd4ded7f`, which builds a `FactoryGuidedDenoiser` from exactly those. `allow_guidance_override` (`ltx2_pipeline.h:534`) is dead the same way. Positive control for the grep: the same command for `audio_guidance` returns the T2A consumer at `ltx2_video.cpp:3527`. Blocks four more pipelines on one missing seam (`a2vid_two_stage.py:230`, `ti2vid_two_stages.py:248`, `ti2vid_two_stages_hq.py:271`, `keyframe_interpolation.py:232`). Spec [`ltx25-guided-video.md`](../../specs/ltx25-guided-video.md)
Row: LTX25-GUIDED-VIDEO
State: UNKNOWN
Kind: bug
GitHub: 1092
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:329`

### Frozen archive evidence

> | [#1092](https://github.com/mudler/vllm.cpp/issues/1092) | `LTX25-GUIDED-VIDEO` | The LTX-2.5 **video** denoise loop runs one UNGUIDED forward per step: `Ltx2PhaseRecipe::video_guidance` is set by every recipe (`src/vllm/model_executor/models/ltx2_pipeline.cpp:1069 @ b5756ea8c`) and read by nothing, so a `pipeline_kind = one_stage` render ignores `cfg_scale = 3.0`, `stg_scale = 1.0`, `rescale_scale = 0.7` and `modality_scale = 3.0` and denoises along a different trajectory than `ti2vid_one_stage.py:221-226 @ fd4ded7f`, which builds a `FactoryGuidedDenoiser` from exactly those. `allow_guidance_override` (`ltx2_pipeline.h:534`) is dead the same way. Positive control for the grep: the same command for `audio_guidance` returns the T2A consumer at `ltx2_video.cpp:3527`. Blocks four more pipelines on one missing seam (`a2vid_two_stage.py:230`, `ti2vid_two_stages.py:248`, `ti2vid_two_stages_hq.py:271`, `keyframe_interpolation.py:232`). Spec [`ltx25-guided-video.md`](../../specs/ltx25-guided-video.md) | bug |

## Resolution

-

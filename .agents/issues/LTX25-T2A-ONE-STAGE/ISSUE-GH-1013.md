ID: ISSUE-GH-1013
Title: LTX-2.5 `one_stage` denoised from ZEROS. `OneStagePhase` (`src/vllm/model_executor/models/ltx2_pipeline.cpp:1066 @ 332aed738`) left `Ltx2PhaseRecipe::noise_scale` at the struct default of 0.0, and 0.0 is not "no extra noise": `Ltx2GaussianNoise` is `latent + noise_scale * (noise - latent)` (`:218 @ 332aed738`), so the state stayed exactly as `create_initial_state` wrote it, which with no initial latent is all zeros. Upstream `ModalitySpec.noise_scale` defaults to 1.0 (`ltx-pipelines/utils/types.py:110 @ fd4ded7f`) and `TI2VidOneStagePipeline.__call__` constructs BOTH specs without it (`ti2vid_one_stage.py:233-239`); the two neighbouring recipes set it explicitly, which is what made the omission legible. No gate saw it because every end-to-end test loads `distilled_two_stage`, and a zero-initialized denoise still returns a finite clip of the right size, frame count and sample rate. FOUND and FIXED in flow by row `LTX25-T2A-ONE-STAGE`, whose `t2a_one_stage` rows are built FROM `OneStageRecipe` and would have inherited it. `dmd2` leaves the same field at 0.0 and is deliberately NOT corrected by analogy (its source is vLLM-Omni, not checked out here); that half is listed under `## Owed` in [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md)
Row: LTX25-T2A-ONE-STAGE
State: UNKNOWN
Kind: bug
GitHub: 1013
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:301`

### Frozen archive evidence

> | [#1013](https://github.com/mudler/vllm.cpp/issues/1013) | `LTX25-T2A-ONE-STAGE` | LTX-2.5 `one_stage` denoised from ZEROS. `OneStagePhase` (`src/vllm/model_executor/models/ltx2_pipeline.cpp:1066 @ 332aed738`) left `Ltx2PhaseRecipe::noise_scale` at the struct default of 0.0, and 0.0 is not "no extra noise": `Ltx2GaussianNoise` is `latent + noise_scale * (noise - latent)` (`:218 @ 332aed738`), so the state stayed exactly as `create_initial_state` wrote it, which with no initial latent is all zeros. Upstream `ModalitySpec.noise_scale` defaults to 1.0 (`ltx-pipelines/utils/types.py:110 @ fd4ded7f`) and `TI2VidOneStagePipeline.__call__` constructs BOTH specs without it (`ti2vid_one_stage.py:233-239`); the two neighbouring recipes set it explicitly, which is what made the omission legible. No gate saw it because every end-to-end test loads `distilled_two_stage`, and a zero-initialized denoise still returns a finite clip of the right size, frame count and sample rate. FOUND and FIXED in flow by row `LTX25-T2A-ONE-STAGE`, whose `t2a_one_stage` rows are built FROM `OneStageRecipe` and would have inherited it. `dmd2` leaves the same field at 0.0 and is deliberately NOT corrected by analogy (its source is vLLM-Omni, not checked out here); that half is listed under `## Owed` in [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md) | bug |

## Resolution

-

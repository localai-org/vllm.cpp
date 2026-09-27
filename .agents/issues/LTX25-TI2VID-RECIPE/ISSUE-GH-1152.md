ID: ISSUE-GH-1152
Title: `Ltx2PipelineRecipe::allow_request_latents` (`include/vllm/model_executor/models/ltx2_pipeline.h:705`) is WRITTEN by every recipe and READ by nothing — the "a parameter no caller passes" shape AGENTS.md `## Nothing lands dead` names. Measured at `c83b96934`: `grep -rn allow_request_latents src include examples` minus the declaration returns FIVE lines and all five are assignments (`ltx2_pipeline.cpp:1264` false, `:1345` false, `:1473` true, `:1604` false). Positive control, the field declared one line above and set in the same blocks: `allow_request_sigmas` returns its assignments PLUS a real reader at `src/vllm/multimodal/ltx2_video.cpp:3476`, so the grep is well-formed and the absence is the finding. Consequence: `Res2sTwoStageRecipe` carries `true` where every other recipe carries `false` and nothing can tell the difference — no upstream `__call__` among these pipelines takes an initial-latent parameter, so `false` is what the signatures support and the `true` looks like an oversight, but it is unfalsifiable while nothing reads the field. The tests assert the VALUES, so they gate the record against itself and cannot see that nothing consumes it, which is the tautology shape [#911](https://github.com/mudler/vllm.cpp/issues/911) recorded on the anchor checker. Two closes: give it a reader (a refusal on a request supplying a latent to a recipe whose upstream signature has none) or delete it and its assertions — deleting is defensible, since no request surface carries a latent at all so the refusal could never fire either. Found by row `LTX25-TI2VID-RECIPE` while deriving the same field for a sixth recipe; not fixed in flow because both closes touch five landed recipes and one deletes gated assertions. Listed under `## Owed` in [`ltx25-ti2vid-recipe.md`](../../specs/ltx25-ti2vid-recipe.md)
Row: LTX25-TI2VID-RECIPE
State: UNKNOWN
Kind: bug
GitHub: 1152
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:357`

### Frozen archive evidence

> | [#1152](https://github.com/mudler/vllm.cpp/issues/1152) | `LTX25-TI2VID-RECIPE` | `Ltx2PipelineRecipe::allow_request_latents` (`include/vllm/model_executor/models/ltx2_pipeline.h:705`) is WRITTEN by every recipe and READ by nothing — the "a parameter no caller passes" shape AGENTS.md `## Nothing lands dead` names. Measured at `c83b96934`: `grep -rn allow_request_latents src include examples` minus the declaration returns FIVE lines and all five are assignments (`ltx2_pipeline.cpp:1264` false, `:1345` false, `:1473` true, `:1604` false). Positive control, the field declared one line above and set in the same blocks: `allow_request_sigmas` returns its assignments PLUS a real reader at `src/vllm/multimodal/ltx2_video.cpp:3476`, so the grep is well-formed and the absence is the finding. Consequence: `Res2sTwoStageRecipe` carries `true` where every other recipe carries `false` and nothing can tell the difference — no upstream `__call__` among these pipelines takes an initial-latent parameter, so `false` is what the signatures support and the `true` looks like an oversight, but it is unfalsifiable while nothing reads the field. The tests assert the VALUES, so they gate the record against itself and cannot see that nothing consumes it, which is the tautology shape [#911](https://github.com/mudler/vllm.cpp/issues/911) recorded on the anchor checker. Two closes: give it a reader (a refusal on a request supplying a latent to a recipe whose upstream signature has none) or delete it and its assertions — deleting is defensible, since no request surface carries a latent at all so the refusal could never fire either. Found by row `LTX25-TI2VID-RECIPE` while deriving the same field for a sixth recipe; not fixed in flow because both closes touch five landed recipes and one deletes gated assertions. Listed under `## Owed` in [`ltx25-ti2vid-recipe.md`](../../specs/ltx25-ti2vid-recipe.md) | bug |

## Resolution

-

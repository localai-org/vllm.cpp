ID: ISSUE-GH-1117
Title: `A2VidPipelineTwoStage` (`a2vid_two_stage.py:53` @ `fd4ded7f`) has no recipe row, so `pipeline_kind = a2vid_two_stage` gets the generic table refusal (`src/vllm/model_executor/models/ltx2_pipeline.cpp:1328-1332`) naming the pair rather than the missing machinery. [#922](https://github.com/mudler/vllm.cpp/issues/922) is CLOSED and closed the audio CONDITIONING, not the recipe: a supplied take rides `distilled_two_stage`, which [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md):438-446 already records as a different trajectory. Measured at `daeff67f2`: `git grep -n '"a2vid' -- src include tests docs examples` returns TWO hits, both upstream anchors inside `Fail`-message assertions (`tests/vllm/multimodal/test_ltx2_video.cpp:4363,:4427`), against a control of 4 for `"one_stage"` in `include/` alone. Four differences from the recipe it rides, each read at the pin: stage 1 is CFG/STG/modality-guided and caller-configured (`:230-240`, fed from `utils/args.py:947-1006`, `--a2v-guidance-scale` defaulting to `video_guider.modality_scale` = 3.0 at `utils/constants.py:54,:64`) where `distilled_two_stage` fixes `allow_guidance_override = false`; stage 1's schedule is scheduler-derived (`:225-227`) against our fixed `DistilledSigmas()`; stage 1 is plain Euler (`:229-258` passes no `stepper`, `utils/blocks.py:526-527`) against our `kEulerAncestral` on 2.5; and the AUDIO guider is the DEFAULT positive-only one (`:237-239`, `ltx-core components/guiders.py:200-210`) rather than the params table's cfg-7.0 row. Two non-schedule facts that must not be guessed: `--audio-path` is `required=True` (`:312-317`), and the distilled LoRA rides stage 2 ALONE (`:114` against `:107`) with `--distilled-lora` `required=True` (`utils/args.py:1140-1153`). Unblocked by `Ltx2GuidedDenoise` landing at `daeff67f2` (#1092/#1102), which [`ltx25-guided-video.md`](../../specs/ltx25-guided-video.md) `## Owed` names this arm against. Spec [`ltx25-a2vid-recipe.md`](../../specs/ltx25-a2vid-recipe.md)
Row: LTX25-A2VID-RECIPE
State: UNKNOWN
Kind: enhancement
GitHub: 1117
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:334`

### Frozen archive evidence

> | [#1117](https://github.com/mudler/vllm.cpp/issues/1117) | `LTX25-A2VID-RECIPE` | `A2VidPipelineTwoStage` (`a2vid_two_stage.py:53` @ `fd4ded7f`) has no recipe row, so `pipeline_kind = a2vid_two_stage` gets the generic table refusal (`src/vllm/model_executor/models/ltx2_pipeline.cpp:1328-1332`) naming the pair rather than the missing machinery. [#922](https://github.com/mudler/vllm.cpp/issues/922) is CLOSED and closed the audio CONDITIONING, not the recipe: a supplied take rides `distilled_two_stage`, which [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md):438-446 already records as a different trajectory. Measured at `daeff67f2`: `git grep -n '"a2vid' -- src include tests docs examples` returns TWO hits, both upstream anchors inside `Fail`-message assertions (`tests/vllm/multimodal/test_ltx2_video.cpp:4363,:4427`), against a control of 4 for `"one_stage"` in `include/` alone. Four differences from the recipe it rides, each read at the pin: stage 1 is CFG/STG/modality-guided and caller-configured (`:230-240`, fed from `utils/args.py:947-1006`, `--a2v-guidance-scale` defaulting to `video_guider.modality_scale` = 3.0 at `utils/constants.py:54,:64`) where `distilled_two_stage` fixes `allow_guidance_override = false`; stage 1's schedule is scheduler-derived (`:225-227`) against our fixed `DistilledSigmas()`; stage 1 is plain Euler (`:229-258` passes no `stepper`, `utils/blocks.py:526-527`) against our `kEulerAncestral` on 2.5; and the AUDIO guider is the DEFAULT positive-only one (`:237-239`, `ltx-core components/guiders.py:200-210`) rather than the params table's cfg-7.0 row. Two non-schedule facts that must not be guessed: `--audio-path` is `required=True` (`:312-317`), and the distilled LoRA rides stage 2 ALONE (`:114` against `:107`) with `--distilled-lora` `required=True` (`utils/args.py:1140-1153`). Unblocked by `Ltx2GuidedDenoise` landing at `daeff67f2` (#1092/#1102), which [`ltx25-guided-video.md`](../../specs/ltx25-guided-video.md) `## Owed` names this arm against. Spec [`ltx25-a2vid-recipe.md`](../../specs/ltx25-a2vid-recipe.md) | enhancement |

## Resolution

-

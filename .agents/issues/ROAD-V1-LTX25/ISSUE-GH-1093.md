ID: ISSUE-GH-1093
Title: `TI2VidTwoStagesPipeline` (`ti2vid_two_stages.py:61` @ `fd4ded7f`) has no recipe row, no refusal and no `Ltx2UnportedPipelineFeature` marker, so asking for it gets the generic table refusal (`ltx2_pipeline.cpp:1328-1332`) naming the pair rather than the missing machinery. It is NOT the `distilled_two_stage` we ship: stage 1 is CFG-guided on the FULL model (`ti2vid_two_stages.py:247-259`) where `distilled.py:265-266` uses `SimpleDenoiser`; the distilled LoRA rides stage 2 ALONE (`:151`); stage-1 sigmas are scheduler-derived (`:243-245`) against our fixed `DistilledSigmas()` (`ltx2_pipeline.cpp:1163`); and `distilled.py:94-107` has no `distilled_lora` parameter at all. It is also NOT `TI2VidTwoStagesHQPipeline` (`ti2vid_two_stages_hq.py:59`), which [#921](https://github.com/mudler/vllm.cpp/issues/921) owns and which puts the LoRA on BOTH stages at separate strengths. Blocked on (a) a guided VIDEO denoise loop: the guidance arithmetic IS ported and generic (`ltx2_pipeline.cpp:439-524`) but its only consumer is the audio-only T2A loop (`ltx2_t2a.cpp:367`, `video=nullptr` at `:332-334`), and the joint loop is single-forward (`ltx2_video.cpp:3036-3040`); (b) TWO checkpoints absent from the NAS, `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (`--distilled-lora` is `required=True`, `utils/args.py:1146`) and the full `ltx-2.5-22b-dev-transformer-bf16.safetensors` stage 1 runs. Control: `find /mnt/nas_share/checkpoints -iname '*lora*'` returns nothing while `-name '*.safetensors'` returns the 8 LTX-2.5 files we hold, every transformer among them a `-distilled-` build. Side finding: `one_stage` writes `video_guidance` at `ltx2_pipeline.cpp:1069` and nothing reads it. Already under `## Owed` in [`ltx25-resolution-envelope.md`](../../specs/ltx25-resolution-envelope.md) as "not separately filed"; filed now because an umbrella row cannot say what THIS arm is blocked on
Row: ROAD-V1-LTX25
State: UNKNOWN
Kind: feature
GitHub: 1093
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:321`

### Frozen archive evidence

> | [#1093](https://github.com/mudler/vllm.cpp/issues/1093) | `ROAD-V1-LTX25` | `TI2VidTwoStagesPipeline` (`ti2vid_two_stages.py:61` @ `fd4ded7f`) has no recipe row, no refusal and no `Ltx2UnportedPipelineFeature` marker, so asking for it gets the generic table refusal (`ltx2_pipeline.cpp:1328-1332`) naming the pair rather than the missing machinery. It is NOT the `distilled_two_stage` we ship: stage 1 is CFG-guided on the FULL model (`ti2vid_two_stages.py:247-259`) where `distilled.py:265-266` uses `SimpleDenoiser`; the distilled LoRA rides stage 2 ALONE (`:151`); stage-1 sigmas are scheduler-derived (`:243-245`) against our fixed `DistilledSigmas()` (`ltx2_pipeline.cpp:1163`); and `distilled.py:94-107` has no `distilled_lora` parameter at all. It is also NOT `TI2VidTwoStagesHQPipeline` (`ti2vid_two_stages_hq.py:59`), which [#921](https://github.com/mudler/vllm.cpp/issues/921) owns and which puts the LoRA on BOTH stages at separate strengths. Blocked on (a) a guided VIDEO denoise loop: the guidance arithmetic IS ported and generic (`ltx2_pipeline.cpp:439-524`) but its only consumer is the audio-only T2A loop (`ltx2_t2a.cpp:367`, `video=nullptr` at `:332-334`), and the joint loop is single-forward (`ltx2_video.cpp:3036-3040`); (b) TWO checkpoints absent from the NAS, `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (`--distilled-lora` is `required=True`, `utils/args.py:1146`) and the full `ltx-2.5-22b-dev-transformer-bf16.safetensors` stage 1 runs. Control: `find /mnt/nas_share/checkpoints -iname '*lora*'` returns nothing while `-name '*.safetensors'` returns the 8 LTX-2.5 files we hold, every transformer among them a `-distilled-` build. Side finding: `one_stage` writes `video_guidance` at `ltx2_pipeline.cpp:1069` and nothing reads it. Already under `## Owed` in [`ltx25-resolution-envelope.md`](../../specs/ltx25-resolution-envelope.md) as "not separately filed"; filed now because an umbrella row cannot say what THIS arm is blocked on | feature |

## Resolution

-

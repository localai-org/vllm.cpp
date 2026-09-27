ID: ISSUE-GH-1096
Title: `KeyframeInterpolationPipeline` (`keyframe_interpolation.py` @ `fd4ded7f`) is absent with no `Ltx2UnportedPipelineFeature` marker, and its conditioning building block IS served: `Ltx2ConditionVideoByKeyframe` (`ltx2_conditioning.h:172`, `.cpp:530`) is reached from `include/vllm.h:935` through `vllm_c.cpp:1635,:1646` to `ltx2_video.cpp:2745-2751`, mutation-proven at [`ltx25-token-append.md`](../../specs/ltx25-token-append.md):270. Blocked on (a) no multi-keyframe request surface: the ABI carries two scalar slots, `first_frame` and `last_frame` (`include/vllm.h:934-935`), the engine request two paths and one blob (`video_engine.h:89-93`), and the indices are hard-coded (`latent_idx=0` at `ltx2_video.cpp:2699`, `frame_idx=frames-1` at `:2750`); the CLI exposes only `--first-frame` (`examples/ltx2_gen/main.cpp:269`) and the server only a first frame (`video_engine.cpp:365-372`), against upstream's repeatable `--image PATH FRAME_IDX STRENGTH [CRF]` (`utils/args.py:805-817`, expanded per keyframe at `utils/helpers.py:343-367`). `num_generated_keyframes` is a DIFFERENT feature (model-invented interior slots, `ltx2_video.cpp:1328-1354`) and must not be mistaken for it. (b) A per-sigma guided denoiser: ours is one struct per PHASE (`ltx2_pipeline.h:526-527`, assigned `ltx2_pipeline.cpp:1069-1070`) and audio-only, with `git grep "build_from_sigma|GuiderFactory|per_sigma"` returning 0 against a `sigma` control of ~10 lines in the same header; upstream resolves guiders per step from sigma (`utils/denoisers.py:304-361`, `ltx-core/components/guiders.py:294-342`). REJECTS the audit's "pure porting, no missing checkpoint": `--distilled-lora` is `required=True` on the parser this pipeline uses (`utils/args.py:1146`, selected at `keyframe_interpolation.py:301`, consumed `:111-122`) and stage 1 runs the full `-dev-` transformer; neither file is on the NAS. It needs no IC-LoRA, which is the half of that framing that holds
Row: ROAD-V1-LTX25
State: UNKNOWN
Kind: feature
GitHub: 1096
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:324`

### Frozen archive evidence

> | [#1096](https://github.com/mudler/vllm.cpp/issues/1096) | `ROAD-V1-LTX25` | `KeyframeInterpolationPipeline` (`keyframe_interpolation.py` @ `fd4ded7f`) is absent with no `Ltx2UnportedPipelineFeature` marker, and its conditioning building block IS served: `Ltx2ConditionVideoByKeyframe` (`ltx2_conditioning.h:172`, `.cpp:530`) is reached from `include/vllm.h:935` through `vllm_c.cpp:1635,:1646` to `ltx2_video.cpp:2745-2751`, mutation-proven at [`ltx25-token-append.md`](../../specs/ltx25-token-append.md):270. Blocked on (a) no multi-keyframe request surface: the ABI carries two scalar slots, `first_frame` and `last_frame` (`include/vllm.h:934-935`), the engine request two paths and one blob (`video_engine.h:89-93`), and the indices are hard-coded (`latent_idx=0` at `ltx2_video.cpp:2699`, `frame_idx=frames-1` at `:2750`); the CLI exposes only `--first-frame` (`examples/ltx2_gen/main.cpp:269`) and the server only a first frame (`video_engine.cpp:365-372`), against upstream's repeatable `--image PATH FRAME_IDX STRENGTH [CRF]` (`utils/args.py:805-817`, expanded per keyframe at `utils/helpers.py:343-367`). `num_generated_keyframes` is a DIFFERENT feature (model-invented interior slots, `ltx2_video.cpp:1328-1354`) and must not be mistaken for it. (b) A per-sigma guided denoiser: ours is one struct per PHASE (`ltx2_pipeline.h:526-527`, assigned `ltx2_pipeline.cpp:1069-1070`) and audio-only, with `git grep "build_from_sigma\|GuiderFactory\|per_sigma"` returning 0 against a `sigma` control of ~10 lines in the same header; upstream resolves guiders per step from sigma (`utils/denoisers.py:304-361`, `ltx-core/components/guiders.py:294-342`). REJECTS the audit's "pure porting, no missing checkpoint": `--distilled-lora` is `required=True` on the parser this pipeline uses (`utils/args.py:1146`, selected at `keyframe_interpolation.py:301`, consumed `:111-122`) and stage 1 runs the full `-dev-` transformer; neither file is on the NAS. It needs no IC-LoRA, which is the half of that framing that holds | feature |

## Resolution

-

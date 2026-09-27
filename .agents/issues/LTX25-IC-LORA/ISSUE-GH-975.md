ID: ISSUE-GH-975
Title: The reference-image / reference-video arm of `ltx-2.5` is still refused, and BOTH reasons the refusal ever gave are now false: the IC-LoRA metadata ([#923](https://github.com/mudler/vllm.cpp/issues/923) reads it at load, `iclora_utils.py:30-49`) and the token-APPEND machinery ([#930](https://github.com/mudler/vllm.cpp/issues/930) built it in `c7cb59fbb`; the LAST-frame keyframe is SERVED on it). Two causes remain. (1) The reference CLIP has no pixel path: upstream reads it at `height // scale` by `width // scale` (`iclora_utils.py:116-117`), refuses a target the factor does not divide (`:112-115`), keeps frame 0 then every Nth frame (`temporal_subsample`, `:87-89`, called at `:144`) and encodes the clip (`:145-148`), while this engine's only pixel-to-latent route encodes ONE frame at the phase's own resolution and nothing reads `ref_video_dir` (`src/vllm/multimodal/video_engine.cpp:375`). `Ltx2ConvVideoEncode` already takes a `frame_count`, so the encoder is not the gap. (2) The reference item is a STAGE-1 item and stage 2 must run UNFUSED: `ic_lora.py:108` gives stage 1 `loras=tuple(loras)` and the reference conditioning (`:269-278`, `:377-402`), `:119` gives stage 2 `loras=()` and `:314-321` gives it `combined_image_conditionings` with no reference item — and this engine holds ONE DiT, fused at load, that every phase runs. `Ltx2LatentState` having no attention-mask field is NOT the reason either: the default arm builds no mask (`iclora_utils.py:159-160`, `:168-169`). Listed under `## Owed` in [`ltx25-ic-lora.md`](../../specs/ltx25-ic-lora.md)
Row: LTX25-IC-LORA
State: UNKNOWN
Kind: feature
GitHub: 975
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:268`

### Frozen archive evidence

> | [#975](https://github.com/mudler/vllm.cpp/issues/975) | `LTX25-IC-LORA` | The reference-image / reference-video arm of `ltx-2.5` is still refused, and BOTH reasons the refusal ever gave are now false: the IC-LoRA metadata ([#923](https://github.com/mudler/vllm.cpp/issues/923) reads it at load, `iclora_utils.py:30-49`) and the token-APPEND machinery ([#930](https://github.com/mudler/vllm.cpp/issues/930) built it in `c7cb59fbb`; the LAST-frame keyframe is SERVED on it). Two causes remain. (1) The reference CLIP has no pixel path: upstream reads it at `height // scale` by `width // scale` (`iclora_utils.py:116-117`), refuses a target the factor does not divide (`:112-115`), keeps frame 0 then every Nth frame (`temporal_subsample`, `:87-89`, called at `:144`) and encodes the clip (`:145-148`), while this engine's only pixel-to-latent route encodes ONE frame at the phase's own resolution and nothing reads `ref_video_dir` (`src/vllm/multimodal/video_engine.cpp:375`). `Ltx2ConvVideoEncode` already takes a `frame_count`, so the encoder is not the gap. (2) The reference item is a STAGE-1 item and stage 2 must run UNFUSED: `ic_lora.py:108` gives stage 1 `loras=tuple(loras)` and the reference conditioning (`:269-278`, `:377-402`), `:119` gives stage 2 `loras=()` and `:314-321` gives it `combined_image_conditionings` with no reference item — and this engine holds ONE DiT, fused at load, that every phase runs. `Ltx2LatentState` having no attention-mask field is NOT the reason either: the default arm builds no mask (`iclora_utils.py:159-160`, `:168-169`). Listed under `## Owed` in [`ltx25-ic-lora.md`](../../specs/ltx25-ic-lora.md) | feature |

## Resolution

-

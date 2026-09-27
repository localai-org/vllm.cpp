ID: ISSUE-GH-1095
Title: `DubItPipeline` (`dubit.py` @ `fd4ded7f`) is absent with no `Ltx2UnportedPipelineFeature` marker, and its gap is narrower than the silence suggests. `Ltx2ConditionAudioByReference` is ported (`ltx2_conditioning.h:308`, `ltx2_conditioning.cpp:615-622`), gated (`test_ltx2_vae.cpp:2926`, call at `:2948`) and UNDRIVEN: zero call sites in `src/` or `examples/`, its only other `src/` appearance being inside the refusal string at `ltx2_video.cpp:1993,:1997`. Control: the siblings `Ltx2ConditionVideoByLatentIndex` (`ltx2_video.cpp:2698`) and `Ltx2ConditionVideoByKeyframe` (`:2749`) DO have production call sites, so the grep finds both polarities. Reference audio is already refused by name (`ltx2_video.cpp:1991-2004`, reached from `vllm_c.cpp:1643` and `video_engine.cpp:377-380`). Blocked on (a) the negative RoPE shift, `positions = positions - aud_dur - 0.04` (`dubit.py:351-353`, applied for both stages from `:266-272`): our `Ltx2ConditionAudioByReference` applies no shift, and the ONE ported temporal shift (`ltx2_conditioning.cpp:596-601`) clamps `std::max(0.0, ...)` so it is structurally incapable of producing a negative position; (b) the Dub-It IC-LoRA, `ltx-2.3-22b-ic-lora-dubit-0.9.safetensors` from `Lightricks/LTX-2.3-22b-IC-LoRA-DubIt` (`MODELS-LTX-2.3.md:44`), required exactly once at `dubit.py:364-365` and absent from the NAS; (c) reference-audio ingestion end to end. REJECTS an audit claim in the same breath: `Ltx2AudioPatchify` is NOT undriven - it runs at `ltx2_video.cpp:2595` on every render; its `Ltx2CreateAudioLatentState` call site (`ltx2_conditioning.cpp:484`) is the undriven one. Record drift found while measuring: [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md):466-472 cites the gate at `test_ltx2_vae.cpp:2412,:2431` where it now sits at `:2926,:2948`, and [`ltx25-ic-lora.md`](../../specs/ltx25-ic-lora.md):339 says the audio VAE encoder has no load path, which #922 made false
Row: ROAD-V1-LTX25
State: UNKNOWN
Kind: feature
GitHub: 1095
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:323`

### Frozen archive evidence

> | [#1095](https://github.com/mudler/vllm.cpp/issues/1095) | `ROAD-V1-LTX25` | `DubItPipeline` (`dubit.py` @ `fd4ded7f`) is absent with no `Ltx2UnportedPipelineFeature` marker, and its gap is narrower than the silence suggests. `Ltx2ConditionAudioByReference` is ported (`ltx2_conditioning.h:308`, `ltx2_conditioning.cpp:615-622`), gated (`test_ltx2_vae.cpp:2926`, call at `:2948`) and UNDRIVEN: zero call sites in `src/` or `examples/`, its only other `src/` appearance being inside the refusal string at `ltx2_video.cpp:1993,:1997`. Control: the siblings `Ltx2ConditionVideoByLatentIndex` (`ltx2_video.cpp:2698`) and `Ltx2ConditionVideoByKeyframe` (`:2749`) DO have production call sites, so the grep finds both polarities. Reference audio is already refused by name (`ltx2_video.cpp:1991-2004`, reached from `vllm_c.cpp:1643` and `video_engine.cpp:377-380`). Blocked on (a) the negative RoPE shift, `positions = positions - aud_dur - 0.04` (`dubit.py:351-353`, applied for both stages from `:266-272`): our `Ltx2ConditionAudioByReference` applies no shift, and the ONE ported temporal shift (`ltx2_conditioning.cpp:596-601`) clamps `std::max(0.0, ...)` so it is structurally incapable of producing a negative position; (b) the Dub-It IC-LoRA, `ltx-2.3-22b-ic-lora-dubit-0.9.safetensors` from `Lightricks/LTX-2.3-22b-IC-LoRA-DubIt` (`MODELS-LTX-2.3.md:44`), required exactly once at `dubit.py:364-365` and absent from the NAS; (c) reference-audio ingestion end to end. REJECTS an audit claim in the same breath: `Ltx2AudioPatchify` is NOT undriven - it runs at `ltx2_video.cpp:2595` on every render; its `Ltx2CreateAudioLatentState` call site (`ltx2_conditioning.cpp:484`) is the undriven one. Record drift found while measuring: [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md):466-472 cites the gate at `test_ltx2_vae.cpp:2412,:2431` where it now sits at `:2926,:2948`, and [`ltx25-ic-lora.md`](../../specs/ltx25-ic-lora.md):339 says the audio VAE encoder has no load path, which #922 made false | feature |

## Resolution

-

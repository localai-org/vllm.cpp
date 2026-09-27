ID: ISSUE-GH-927
Title: LTX-2.5 accepts and NEVER READS seven typed fields that MiniMax-H3 reads: `audio_vae_config_path`, `video_vae_config_path`, `tokenizer_path`, `encoder_max_layers` on the load side, and `flow_shift`, `audio_flow_shift`, `task` per generation. `CheckKnownExtras` / `CheckUnservedExtras` (`ltx2_video.cpp:314-356 @ 5a0ffe9e`) exist to refuse an unserved knob by name and cover the `extras` MAPS only, never the typed fields. Sharpest case: `docs/USAGE.md` shows an H3-shaped recipe passing `--audio-vae-config`, and against LTX-2.5 that file is accepted and never opened, because the engine takes the config from the checkpoint `__metadata__` (`ltx2_video.cpp:920-921 @ 5a0ffe9e`) — so a JSON that disagrees is silently overridden. `audio_flow_shift` is even validated positive (`video_api.cpp:236`) before being dropped, which makes the request look served. Not fixed in flow because a blanket refusal needs a per-field serve/refuse/not-applicable decision, the distinction [#758](https://github.com/mudler/vllm.cpp/issues/758) records this project getting wrong. Found while surveying for [#922](https://github.com/mudler/vllm.cpp/issues/922). Listed under `## Owed` in [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md) §9
Row: -
State: UNKNOWN
Kind: bug
GitHub: 927
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:251`

### Frozen archive evidence

> | [#927](https://github.com/mudler/vllm.cpp/issues/927) | — | LTX-2.5 accepts and NEVER READS seven typed fields that MiniMax-H3 reads: `audio_vae_config_path`, `video_vae_config_path`, `tokenizer_path`, `encoder_max_layers` on the load side, and `flow_shift`, `audio_flow_shift`, `task` per generation. `CheckKnownExtras` / `CheckUnservedExtras` (`ltx2_video.cpp:314-356 @ 5a0ffe9e`) exist to refuse an unserved knob by name and cover the `extras` MAPS only, never the typed fields. Sharpest case: `docs/USAGE.md` shows an H3-shaped recipe passing `--audio-vae-config`, and against LTX-2.5 that file is accepted and never opened, because the engine takes the config from the checkpoint `__metadata__` (`ltx2_video.cpp:920-921 @ 5a0ffe9e`) — so a JSON that disagrees is silently overridden. `audio_flow_shift` is even validated positive (`video_api.cpp:236`) before being dropped, which makes the request look served. Not fixed in flow because a blanket refusal needs a per-field serve/refuse/not-applicable decision, the distinction [#758](https://github.com/mudler/vllm.cpp/issues/758) records this project getting wrong. Found while surveying for [#922](https://github.com/mudler/vllm.cpp/issues/922). Listed under `## Owed` in [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md) §9 | bug |

## Resolution

-

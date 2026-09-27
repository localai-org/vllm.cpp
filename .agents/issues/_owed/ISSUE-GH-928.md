ID: ISSUE-GH-928
Title: `/v1/videos` forwards NO per-generation extra to any engine: `ParseVideoRequest` stores every metadata key into `VideoRequest::metadata` (`video_api.cpp:104-111`) and `VideoGenParamsFromRequest` (`video_engine.cpp:349-385`) never assigns `gen.extras`, so the map arrives EMPTY on every HTTP request. LTX-2.5 image-to-video is therefore unreachable over the route entirely — it is served only at `image_crf=0` and absent resolves the checkpoint's CRF 18 and refuses (`ltx2_video.cpp:1378-1382 @ 5a0ffe9e`), so the one value that works cannot be sent — and the keys are dropped silently, returning 200 rather than a 400 naming them. `audio_path` and its two window knobs ([#922](https://github.com/mudler/vllm.cpp/issues/922)) are ABI- and CLI-only for the same reason. Not fixed in flow because forwarding `metadata` wholesale turns every bookkeeping key into an unknown-extra 400 and breaks the pass-through `video_api.h:66-67` promises; the contract needs deciding. `test_video_engine.cpp:380-430` pins the typed fields and asserts nothing about `extras`, which is why this survived. Listed under `## Owed` in [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md) §9
Row: -
State: UNKNOWN
Kind: bug
GitHub: 928
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:252`

### Frozen archive evidence

> | [#928](https://github.com/mudler/vllm.cpp/issues/928) | — | `/v1/videos` forwards NO per-generation extra to any engine: `ParseVideoRequest` stores every metadata key into `VideoRequest::metadata` (`video_api.cpp:104-111`) and `VideoGenParamsFromRequest` (`video_engine.cpp:349-385`) never assigns `gen.extras`, so the map arrives EMPTY on every HTTP request. LTX-2.5 image-to-video is therefore unreachable over the route entirely — it is served only at `image_crf=0` and absent resolves the checkpoint's CRF 18 and refuses (`ltx2_video.cpp:1378-1382 @ 5a0ffe9e`), so the one value that works cannot be sent — and the keys are dropped silently, returning 200 rather than a 400 naming them. `audio_path` and its two window knobs ([#922](https://github.com/mudler/vllm.cpp/issues/922)) are ABI- and CLI-only for the same reason. Not fixed in flow because forwarding `metadata` wholesale turns every bookkeeping key into an unknown-extra 400 and breaks the pass-through `video_api.h:66-67` promises; the contract needs deciding. `test_video_engine.cpp:380-430` pins the typed fields and asserts nothing about `extras`, which is why this survived. Listed under `## Owed` in [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md) §9 | bug |

## Resolution

-

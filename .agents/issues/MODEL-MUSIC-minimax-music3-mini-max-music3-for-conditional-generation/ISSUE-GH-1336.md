ID: ISSUE-GH-1336
Title: `POST /v1/audio/speech`: [#1315](https://github.com/mudler/vllm.cpp/issues/1315) routed the KNOBS and the REFUSALS through the `Owner(key)` lookup that sees both placements, and left the eight CONTENT keys on the bare `json` handle (`speech_api.cpp:126,127,133,134,136,138,140,141,144,145,175,176` @ `7d21644d7`): `model`, `input`, `text`, `language`, `lyrics`, `description`, `prompt`, `reference_audio`. So nesting a content key under `extra_params` still DROPPED it, and that DEFEATED three refusals: `{"extra_params":{"text":"hello"}}` returned 200 instead of `minimax_music3_speech.cpp:440-446`, whose refusal ends with the words "rather than having it silently dropped"; `{"extra_params":{"language":"en"}}` returned 200 instead of `:456-460`; and `{"extra_params":{"reference_audio":"data:…"}}` dropped the clip, so a family whose `requires_reference_audio()` is true answered `400 "reference_audio … is required"` (`api_server.cpp:511-516`) to a caller who had supplied one. NOT a regression, because the pre-#1315 code read those keys from the top level only as well. What let it survive an implementation and a first review was the FALSE structural claim #1315 left at `speech_api.cpp:110-113` — "there is no longer a handle on one placement only", when `Owner` needs `json` as its fallback and cannot remove it. FIXED IN FLOW while repairing the review of [#1328](https://github.com/mudler/vllm.cpp/pull/1328): the content keys route through `Owner`, the comment claims a convention the tests pin rather than an impossibility, and nested `text`/`language`/`reference_audio` are RED first at the parser and at the registered route. #925's boundary is unchanged and a negative control in BOTH placements pins it. Spec [§17.7](../../specs/minimax-music3.md)
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 1336
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:456`

### Frozen archive evidence

> | [#1336](https://github.com/mudler/vllm.cpp/issues/1336) | `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation` | `POST /v1/audio/speech`: [#1315](https://github.com/mudler/vllm.cpp/issues/1315) routed the KNOBS and the REFUSALS through the `Owner(key)` lookup that sees both placements, and left the eight CONTENT keys on the bare `json` handle (`speech_api.cpp:126,127,133,134,136,138,140,141,144,145,175,176` @ `7d21644d7`): `model`, `input`, `text`, `language`, `lyrics`, `description`, `prompt`, `reference_audio`. So nesting a content key under `extra_params` still DROPPED it, and that DEFEATED three refusals: `{"extra_params":{"text":"hello"}}` returned 200 instead of `minimax_music3_speech.cpp:440-446`, whose refusal ends with the words "rather than having it silently dropped"; `{"extra_params":{"language":"en"}}` returned 200 instead of `:456-460`; and `{"extra_params":{"reference_audio":"data:…"}}` dropped the clip, so a family whose `requires_reference_audio()` is true answered `400 "reference_audio … is required"` (`api_server.cpp:511-516`) to a caller who had supplied one. NOT a regression, because the pre-#1315 code read those keys from the top level only as well. What let it survive an implementation and a first review was the FALSE structural claim #1315 left at `speech_api.cpp:110-113` — "there is no longer a handle on one placement only", when `Owner` needs `json` as its fallback and cannot remove it. FIXED IN FLOW while repairing the review of [#1328](https://github.com/mudler/vllm.cpp/pull/1328): the content keys route through `Owner`, the comment claims a convention the tests pin rather than an impossibility, and nested `text`/`language`/`reference_audio` are RED first at the parser and at the registered route. #925's boundary is unchanged and a negative control in BOTH placements pins it. Spec [§17.7](../../specs/minimax-music3.md) | bug |

## Resolution

-

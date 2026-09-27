ID: ISSUE-GH-644
Title: LTX-2.5 FULL PORT campaign: close every refused conditioning arm. Row 0 `LTX25-PROMPT-ADALN` (spec [`ltx25-prompt-adaln.md`](../../specs/ltx25-prompt-adaln.md)) restored `use_prompt_adaln_single`, which the loader cleared unconditionally so every render dropped the timestep-conditioned half of the prompt K/V modulation. Row 1 `LTX25-IMAGE-COND` (spec [`ltx25-image-conditioning.md`](../../specs/ltx25-image-conditioning.md)) builds the video VAE ENCODER's load path — `Ltx2VideoVaeEncoderKeyRules` existed nowhere in the tree — and serves an image at latent frame 0 at `crf = 0`; keyframe / reference / non-zero-CRF stay refused by name
Row: ROAD-V1-LTX25
State: UNKNOWN
Kind: feature
GitHub: 644
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:140`

### Frozen archive evidence

> | [#644](https://github.com/mudler/vllm.cpp/issues/644) | `ROAD-V1-LTX25` | LTX-2.5 FULL PORT campaign: close every refused conditioning arm. Row 0 `LTX25-PROMPT-ADALN` (spec [`ltx25-prompt-adaln.md`](../../specs/ltx25-prompt-adaln.md)) restored `use_prompt_adaln_single`, which the loader cleared unconditionally so every render dropped the timestep-conditioned half of the prompt K/V modulation. Row 1 `LTX25-IMAGE-COND` (spec [`ltx25-image-conditioning.md`](../../specs/ltx25-image-conditioning.md)) builds the video VAE ENCODER's load path — `Ltx2VideoVaeEncoderKeyRules` existed nowhere in the tree — and serves an image at latent frame 0 at `crf = 0`; keyframe / reference / non-zero-CRF stay refused by name | feature |

## Resolution

-

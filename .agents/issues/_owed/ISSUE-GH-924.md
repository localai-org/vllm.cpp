ID: ISSUE-GH-924
Title: LTX-2.5 `RetakePipeline` (`retake.py:53,151`) regenerates a chosen time region of an existing video and is not served; the recipe table refuses the kind by name at `ltx2_pipeline.cpp:1131-1136`, which is the correct current state. Judged a SEPARATE row from audio-to-video (#922) after reading both upstream files: they share the audio VAE encoder and nothing else. Retake needs `TemporalRegionMask` (`noise_mask_cond.py:10-47`, zero hits in this tree, two coordinate conventions), video-file ingestion plus video VAE encode (`helpers.py:165-233`), and the audio VAE DECODER on the output side, where A2Vid returns the caller's waveform untouched (`a2vid_two_stage.py:301-303`). Their latent-length policies also disagree — A2Vid truncates only (`a2vid_two_stage.py:202`), Retake truncates or zero-pads (`helpers.py:149-162`) — so one shared helper would be wrong for one of them. Listed under `## Owed` in [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md)
Row: -
State: UNKNOWN
Kind: feature
GitHub: 924
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:250`

### Frozen archive evidence

> | [#924](https://github.com/mudler/vllm.cpp/issues/924) | — | LTX-2.5 `RetakePipeline` (`retake.py:53,151`) regenerates a chosen time region of an existing video and is not served; the recipe table refuses the kind by name at `ltx2_pipeline.cpp:1131-1136`, which is the correct current state. Judged a SEPARATE row from audio-to-video (#922) after reading both upstream files: they share the audio VAE encoder and nothing else. Retake needs `TemporalRegionMask` (`noise_mask_cond.py:10-47`, zero hits in this tree, two coordinate conventions), video-file ingestion plus video VAE encode (`helpers.py:165-233`), and the audio VAE DECODER on the output side, where A2Vid returns the caller's waveform untouched (`a2vid_two_stage.py:301-303`). Their latent-length policies also disagree — A2Vid truncates only (`a2vid_two_stage.py:202`), Retake truncates or zero-pads (`helpers.py:149-162`) — so one shared helper would be wrong for one of them. Listed under `## Owed` in [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md) | feature |

## Resolution

-

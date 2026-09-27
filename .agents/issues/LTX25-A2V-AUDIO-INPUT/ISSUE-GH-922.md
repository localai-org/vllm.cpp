ID: ISSUE-GH-922
Title: LTX-2.5 audio-to-video (`A2VidPipelineTwoStage`, `a2vid_two_stage.py:53,143`) is absent: `vllm_video_params` carries no field or extra that accepts a driving waveform and `ltx2-gen` has no `--audio-path`, so nothing turns a file on disk into the audio latent the DiT's audio stream consumes. Distinct from reference-audio conditioning, which is correctly refused by name at `ltx2_video.cpp:1348-1355 @ 5a0ffe9e`; the two share one blocking dependency, the audio VAE ENCODER load path (`ltx2_loader.cpp:1295-1300` materializes `audio_vae.decoder.` only). The analysis half is already ported and unreached — `Ltx2AudioEncoderForward` (`ltx2_audio_vae.cpp:1114`), `Ltx2WaveformToLogMel` (`:1019`), `Ltx2SlaneyMelFilterbank` (`:970`) — and the engine applies ONE `phase.noise_scale` to both streams (`ltx2_video.cpp:1708-1712 @ 5a0ffe9e`) where upstream's `ModalitySpec` carries `noise_scale` and `frozen` per modality (`utils/types.py:99-112`). Spec [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md)
Row: LTX25-A2V-AUDIO-INPUT
State: UNKNOWN
Kind: feature
GitHub: 922
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:249`

### Frozen archive evidence

> | [#922](https://github.com/mudler/vllm.cpp/issues/922) | `LTX25-A2V-AUDIO-INPUT` | LTX-2.5 audio-to-video (`A2VidPipelineTwoStage`, `a2vid_two_stage.py:53,143`) is absent: `vllm_video_params` carries no field or extra that accepts a driving waveform and `ltx2-gen` has no `--audio-path`, so nothing turns a file on disk into the audio latent the DiT's audio stream consumes. Distinct from reference-audio conditioning, which is correctly refused by name at `ltx2_video.cpp:1348-1355 @ 5a0ffe9e`; the two share one blocking dependency, the audio VAE ENCODER load path (`ltx2_loader.cpp:1295-1300` materializes `audio_vae.decoder.` only). The analysis half is already ported and unreached — `Ltx2AudioEncoderForward` (`ltx2_audio_vae.cpp:1114`), `Ltx2WaveformToLogMel` (`:1019`), `Ltx2SlaneyMelFilterbank` (`:970`) — and the engine applies ONE `phase.noise_scale` to both streams (`ltx2_video.cpp:1708-1712 @ 5a0ffe9e`) where upstream's `ModalitySpec` carries `noise_scale` and `frozen` per modality (`utils/types.py:99-112`). Spec [`ltx25-a2v-audio-input.md`](../../specs/ltx25-a2v-audio-input.md) | feature |

## Resolution

-

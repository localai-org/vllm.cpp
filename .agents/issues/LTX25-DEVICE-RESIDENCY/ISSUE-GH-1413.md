ID: ISSUE-GH-1413
Title: The LTX-2.5 render phase table [#1408](https://github.com/mudler/vllm.cpp/pull/1408) lands for [#1010](https://github.com/mudler/vllm.cpp/issues/1010) is written by the SUCCESS PATH ONLY — `WritePhaseLog` sits after `im.trace.completed = true` at `src/vllm/multimodal/ltx2_video.cpp:4655-4658 @ 4f3c24380` — so a render that is killed, aborted by a lease governor, or still running writes no table at all. That is the population the campaign actually has: [#1375](https://github.com/mudler/vllm.cpp/issues/1375) is `ABORT[92] PROJECTED OVERRUN`, `child exit=-15`, 0 frames; [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) rung 1 is `EXIT=137`, 0 frames; rung 2 is `EXIT=1`, 0 frames. **And nothing is emitted WHILE a render runs**: `PhaseLog::Open`/`Close` print nothing and `VLLM_RENDER_PHASE_LOG_STDERR` fires inside `WriteJson`, i.e. on the success path again, so between `ltx2-gen: family=...` and `wrote N frames` a 2.5-hour render is silent and working is byte-identical to hung. **The unit that costs the wall has no counter**: `denoise` is one leaf covering ~2.7 h, against #1375's measured ~162 s per DiT forward at 60 structural forwards (30 steps x 2 CFG legs — `cfg_scale != 1.0` forces the unconditional branch at `ltx2_pipeline.cpp:521-523`). External sampling is NOT the fallback and was tried: #1375 records `phase=OTHER` throughout because `eu-stack` unwinds ZERO frames inside the `rc` worker container, and its own text names an in-process phase marker as the way to attribute the 162 s. Owed: a stderr line on every phase open and close, so the LAST LINE PRINTED names the phase in flight, plus one tick per DiT forward carrying phase, step `k/N`, cumulative forward index, elapsed and seconds-since-previous — on the shipped default, because the failure happened on default settings and `VT_H3_PROGRESS` (`minimax_h3.cpp:776-793`) is the opt-in shape that is exactly why no LTX-2.5 run has one. Precondition for attributing #1375, which caps render resolution. Stage W0-live in [`ltx25-device-residency.md`](../../specs/ltx25-device-residency.md)
Row: LTX25-DEVICE-RESIDENCY
State: UNKNOWN
Kind: feature
GitHub: 1413
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:485`

### Frozen archive evidence

> | [#1413](https://github.com/mudler/vllm.cpp/issues/1413) | `LTX25-DEVICE-RESIDENCY` | The LTX-2.5 render phase table [#1408](https://github.com/mudler/vllm.cpp/pull/1408) lands for [#1010](https://github.com/mudler/vllm.cpp/issues/1010) is written by the SUCCESS PATH ONLY — `WritePhaseLog` sits after `im.trace.completed = true` at `src/vllm/multimodal/ltx2_video.cpp:4655-4658 @ 4f3c24380` — so a render that is killed, aborted by a lease governor, or still running writes no table at all. That is the population the campaign actually has: [#1375](https://github.com/mudler/vllm.cpp/issues/1375) is `ABORT[92] PROJECTED OVERRUN`, `child exit=-15`, 0 frames; [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) rung 1 is `EXIT=137`, 0 frames; rung 2 is `EXIT=1`, 0 frames. **And nothing is emitted WHILE a render runs**: `PhaseLog::Open`/`Close` print nothing and `VLLM_RENDER_PHASE_LOG_STDERR` fires inside `WriteJson`, i.e. on the success path again, so between `ltx2-gen: family=...` and `wrote N frames` a 2.5-hour render is silent and working is byte-identical to hung. **The unit that costs the wall has no counter**: `denoise` is one leaf covering ~2.7 h, against #1375's measured ~162 s per DiT forward at 60 structural forwards (30 steps x 2 CFG legs — `cfg_scale != 1.0` forces the unconditional branch at `ltx2_pipeline.cpp:521-523`). External sampling is NOT the fallback and was tried: #1375 records `phase=OTHER` throughout because `eu-stack` unwinds ZERO frames inside the `rc` worker container, and its own text names an in-process phase marker as the way to attribute the 162 s. Owed: a stderr line on every phase open and close, so the LAST LINE PRINTED names the phase in flight, plus one tick per DiT forward carrying phase, step `k/N`, cumulative forward index, elapsed and seconds-since-previous — on the shipped default, because the failure happened on default settings and `VT_H3_PROGRESS` (`minimax_h3.cpp:776-793`) is the opt-in shape that is exactly why no LTX-2.5 run has one. Precondition for attributing #1375, which caps render resolution. Stage W0-live in [`ltx25-device-residency.md`](../../specs/ltx25-device-residency.md) | feature |

## Resolution

-

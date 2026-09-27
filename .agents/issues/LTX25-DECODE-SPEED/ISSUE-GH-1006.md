ID: ISSUE-GH-1006
Title: LTX-2.5 render speed has never been attributed on any axis: `docs/BENCHMARKS.md` carries one LTX line, under `## Open gaps`. The shipped video VAE decode is the **CPU reference arm** and production executes it — `src/vllm/multimodal/ltx2_video.cpp:3258 @ 332aed738` calls `Ltx2VideoDecodeStreaming`, reaching `Ltx2ConvVideoDecode` via `ltx2_video_vae_tiled.cpp:113,369 @ 332aed738` — while the file itself says `src/vllm/model_executor/models/ltx2_video_vae.cpp:46-49 @ 332aed738` "no memory or throughput number should be taken from it". One 448x256/25f decode is ~7.25 TFLOP over 42 convs (COMPUTED from the LTX-2.5 conv VAE config in the checkpoint header) against a measured wall of 2681.02 s — ~2.7 GFLOP/s. **That wall is ONE sample on a CONTENDED box** and carries its conditions: its own source records a full 405-target build and a 416-test `ctest -j 6` running in the same window ([`ltx25-tiled-decode.md`](../../specs/ltx25-tiled-decode.md):377-380) at `uptime` load average 30.2 (`:454`), with no idle-host same-binary repeat. It is also pre-[#1008](https://github.com/mudler/vllm.cpp/issues/1008): the f64 accumulation it measured landed out at `d1b0ea3a8`, and the f32 arm has never been timed. Owning row for the ranked levers in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md)
Row: LTX25-DECODE-SPEED
State: UNKNOWN
Kind: feature
GitHub: 1006
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:283`

### Frozen archive evidence

> | [#1006](https://github.com/mudler/vllm.cpp/issues/1006) | `LTX25-DECODE-SPEED` | LTX-2.5 render speed has never been attributed on any axis: `docs/BENCHMARKS.md` carries one LTX line, under `## Open gaps`. The shipped video VAE decode is the **CPU reference arm** and production executes it — `src/vllm/multimodal/ltx2_video.cpp:3258 @ 332aed738` calls `Ltx2VideoDecodeStreaming`, reaching `Ltx2ConvVideoDecode` via `ltx2_video_vae_tiled.cpp:113,369 @ 332aed738` — while the file itself says `src/vllm/model_executor/models/ltx2_video_vae.cpp:46-49 @ 332aed738` "no memory or throughput number should be taken from it". One 448x256/25f decode is ~7.25 TFLOP over 42 convs (COMPUTED from the LTX-2.5 conv VAE config in the checkpoint header) against a measured wall of 2681.02 s — ~2.7 GFLOP/s. **That wall is ONE sample on a CONTENDED box** and carries its conditions: its own source records a full 405-target build and a 416-test `ctest -j 6` running in the same window ([`ltx25-tiled-decode.md`](../../specs/ltx25-tiled-decode.md):377-380) at `uptime` load average 30.2 (`:454`), with no idle-host same-binary repeat. It is also pre-[#1008](https://github.com/mudler/vllm.cpp/issues/1008): the f64 accumulation it measured landed out at `d1b0ea3a8`, and the f32 arm has never been timed. Owning row for the ranked levers in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) | feature |

## Resolution

-

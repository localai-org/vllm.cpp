ID: ISSUE-GH-1904
Title: **The LTX-2.5 video VAE hand-rolls `DevBuf` instead of the shared `dense_attn::DBuf` device-buffer seam.** W5 ([#1007](https://github.com/mudler/vllm.cpp/issues/1007)) added `DevBuf` at `src/vllm/model_executor/models/ltx2_video_vae.cpp:145-170` — move-deleted RAII over `vt::Backend::Alloc`/`Copy`/`Free` with a `Download` helper — which is a second copy of `vllm::dense_attn::DBuf` (`include/vllm/model_executor/models/dense_device_glue.h:109`), the same object with the same constructor shape and the same `.t()`/`.Download()` surface. `AGENTS.md` `## Shared seams` forbids a hand-written parallel path and no exception is recorded. The difference is not cosmetic: `DBuf` draws from the shared `DevicePool` (`device_pool.h:71`) so a block is reused, while `DevBuf` calls `Alloc`/`Free` directly and `Conv3dThroughSeam` builds three to four of them PER CONVOLUTION, so a decode performs a driver `Free` per operand per convolution on the one path the pool exists to serve. NOT fixed in the flow that found it, and the reason is a behaviour change rather than time: `DBuf` resolves `platforms::GetPlatform(device.type)` through `ResolveDevicePoolPolicy` (`dense_device_glue.h:88-105`) and THROWS for a device type whose platform was never registered, so the switch makes a registered platform a new precondition of a decode that has none today. The audit of whether any current caller reaches the video VAE on such a device is part of the issue. Found by `LTX25-VAE-DEVICE-RESIDENCY` while porting the decode onto a resident volume ([#1451](https://github.com/mudler/vllm.cpp/issues/1451)); listed under `## Owed` in [`ltx25-vae-device-residency.md`](../../specs/ltx25-vae-device-residency.md)
Row: LTX25-VAE-DEVICE-RESIDENCY
State: UNKNOWN
Kind: enhancement
GitHub: 1904
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:735`

### Frozen archive evidence

> | [#1904](https://github.com/mudler/vllm.cpp/issues/1904) | `LTX25-VAE-DEVICE-RESIDENCY` | **The LTX-2.5 video VAE hand-rolls `DevBuf` instead of the shared `dense_attn::DBuf` device-buffer seam.** W5 ([#1007](https://github.com/mudler/vllm.cpp/issues/1007)) added `DevBuf` at `src/vllm/model_executor/models/ltx2_video_vae.cpp:145-170` — move-deleted RAII over `vt::Backend::Alloc`/`Copy`/`Free` with a `Download` helper — which is a second copy of `vllm::dense_attn::DBuf` (`include/vllm/model_executor/models/dense_device_glue.h:109`), the same object with the same constructor shape and the same `.t()`/`.Download()` surface. `AGENTS.md` `## Shared seams` forbids a hand-written parallel path and no exception is recorded. The difference is not cosmetic: `DBuf` draws from the shared `DevicePool` (`device_pool.h:71`) so a block is reused, while `DevBuf` calls `Alloc`/`Free` directly and `Conv3dThroughSeam` builds three to four of them PER CONVOLUTION, so a decode performs a driver `Free` per operand per convolution on the one path the pool exists to serve. NOT fixed in the flow that found it, and the reason is a behaviour change rather than time: `DBuf` resolves `platforms::GetPlatform(device.type)` through `ResolveDevicePoolPolicy` (`dense_device_glue.h:88-105`) and THROWS for a device type whose platform was never registered, so the switch makes a registered platform a new precondition of a decode that has none today. The audit of whether any current caller reaches the video VAE on such a device is part of the issue. Found by `LTX25-VAE-DEVICE-RESIDENCY` while porting the decode onto a resident volume ([#1451](https://github.com/mudler/vllm.cpp/issues/1451)); listed under `## Owed` in [`ltx25-vae-device-residency.md`](../../specs/ltx25-vae-device-residency.md) | enhancement |

## Resolution

-

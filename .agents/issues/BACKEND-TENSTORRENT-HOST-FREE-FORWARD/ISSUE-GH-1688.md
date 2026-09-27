ID: ISSUE-GH-1688
Title: **`ReshapeAndCacheKernel` latched `VT_TT_HOST_FREE_DECODE` in a function-local `static`, so after the R5 default flip the documented opt-out `VT_TT_HOST_FREE_DECODE=0` silently did not reach that path.** Found by the fresh review of [PR #1630](https://github.com/mudler/vllm.cpp/pull/1630) at head `450a524b0`. `tenstorrent_device.h` documents `HostFreeDecodeEnabled()` as "No function-local static caching: tests toggle this env per case in one process", and the flip converted eleven call sites to a live read — `EnsureDevice2D`, `RmsNormKernel`, `PreferDeviceRope`, `CopyDeviceDeviceIfCapture`, `MemsetDeviceIfCapture`, `WarmRopeCosSin`, `WarmPagedKvShadow`, `WarmRacIdx`, `WarmPaMeta`, `WarmDecodePos`, `CaptureDecodePosAdvance` — but left `tenstorrent_ops.cpp:2130` a `static const bool`. The polarity flip is what made it bite: pre-flip the latch cached the default-OFF state so only the opt-IN could be defeated, and that was set before the process started; post-flip it caches default-ON, so the opt-out that the flip tells operators to use never arrives at RAC once any decode has run. The suite could not see it — the flip's own `support_static_graph_mode` round trip and the `setenv(...,"0")` inertness guard both prove the contract elsewhere and neither reaches `ReshapeAndCacheKernel`. FIXED IN FLOW: the `static` is dropped, matching every other converted site. NOT fixed as a test: `ReshapeAndCacheKernel` needs a real Blackhole device, so every case reaching it is behind `TenstorrentPresent()` and skips on every `rc` fleet host; the owed `thalia` case is recorded under [`tenstorrent-host-free-forward.md`](../../specs/tenstorrent-host-free-forward.md) `## Owed`
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: UNKNOWN
Kind: bug
GitHub: 1688
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:626`

### Frozen archive evidence

> | [#1688](https://github.com/mudler/vllm.cpp/issues/1688) | `BACKEND-TENSTORRENT-HOST-FREE-FORWARD` | **`ReshapeAndCacheKernel` latched `VT_TT_HOST_FREE_DECODE` in a function-local `static`, so after the R5 default flip the documented opt-out `VT_TT_HOST_FREE_DECODE=0` silently did not reach that path.** Found by the fresh review of [PR #1630](https://github.com/mudler/vllm.cpp/pull/1630) at head `450a524b0`. `tenstorrent_device.h` documents `HostFreeDecodeEnabled()` as "No function-local static caching: tests toggle this env per case in one process", and the flip converted eleven call sites to a live read — `EnsureDevice2D`, `RmsNormKernel`, `PreferDeviceRope`, `CopyDeviceDeviceIfCapture`, `MemsetDeviceIfCapture`, `WarmRopeCosSin`, `WarmPagedKvShadow`, `WarmRacIdx`, `WarmPaMeta`, `WarmDecodePos`, `CaptureDecodePosAdvance` — but left `tenstorrent_ops.cpp:2130` a `static const bool`. The polarity flip is what made it bite: pre-flip the latch cached the default-OFF state so only the opt-IN could be defeated, and that was set before the process started; post-flip it caches default-ON, so the opt-out that the flip tells operators to use never arrives at RAC once any decode has run. The suite could not see it — the flip's own `support_static_graph_mode` round trip and the `setenv(...,"0")` inertness guard both prove the contract elsewhere and neither reaches `ReshapeAndCacheKernel`. FIXED IN FLOW: the `static` is dropped, matching every other converted site. NOT fixed as a test: `ReshapeAndCacheKernel` needs a real Blackhole device, so every case reaching it is behind `TenstorrentPresent()` and skips on every `rc` fleet host; the owed `thalia` case is recorded under [`tenstorrent-host-free-forward.md`](../../specs/tenstorrent-host-free-forward.md) `## Owed` | bug |

## Resolution

-

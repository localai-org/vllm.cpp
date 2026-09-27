ID: ISSUE-GH-1839
Title: **The engine's call to `Music3SelectDepthArm` (`minimax_music3_speech.cpp:638`) is reachable but not gated, and [#1131](https://github.com/mudler/vllm.cpp/issues/1131) no longer covers it: #1131 named both device-arm twins and row `MUSIC3-DIT-ARM-REACH` closes it with only the DiT half.** Deleting the two-line call leaves `test_minimax_music3_ar` 37/37 · 640/640 and `test_minimax_music3_speech` 9/9 · 223/223 green ([`minimax-music3.md`](../../specs/minimax-music3.md) §19.5 carries the mutation and the binary hashes). Two things stop an existing gate from seeing it, and the SECOND is the one that matters: `--speech-device 1` is refused by name on a CPU-only build before a queue exists, AND §19.6's "device path TAKEN" leg rides `test_minimax_music3_ar`, whose observable `Music3DepthDeviceForwardCount()` is a counter §19.5 itself records as unreachable from production — its only readers are the tests written for it (`test_minimax_music3_ar.cpp:1325,1351,1583,1589,1753,1758`). The instrument that WOULD answer the call site is `ar.depth_staging`, emitted at `minimax_music3_llm.cpp:582` and read by nothing. NOT FIXED IN FLOW and the reason is precise: closing it needs the shipped engine on a real accelerator against the 28.5 GB checkpoint inside an `rc` lease, which is a second GPU leg and a second gate file, not a repair to the row in flight. It is closable by exactly `MUSIC3-DIT-ARM-REACH`'s method — a `gpu;checkpoint;music3`-labelled parity gate entering through `include/vllm.h` with `device = 1`, exiting 77 without its preconditions, asserting `ar.depth_staging` `calls == 1` with the host bucket absent — and that row's `thor:gpu0` job `f63f60e8-957a-4062-92f8-54e5bbb49d92` already FIRED `ar.depth_staging` once without asserting it, so the instrument is known live on the real path. Owed under `## Owed` in [`minimax-music3.md`](../../specs/minimax-music3.md) §19.7
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 1839
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:691`

### Frozen archive evidence

> | [#1839](https://github.com/mudler/vllm.cpp/issues/1839) | `MUSIC3-DEPTH-DEVICE` | **The engine's call to `Music3SelectDepthArm` (`minimax_music3_speech.cpp:638`) is reachable but not gated, and [#1131](https://github.com/mudler/vllm.cpp/issues/1131) no longer covers it: #1131 named both device-arm twins and row `MUSIC3-DIT-ARM-REACH` closes it with only the DiT half.** Deleting the two-line call leaves `test_minimax_music3_ar` 37/37 · 640/640 and `test_minimax_music3_speech` 9/9 · 223/223 green ([`minimax-music3.md`](../../specs/minimax-music3.md) §19.5 carries the mutation and the binary hashes). Two things stop an existing gate from seeing it, and the SECOND is the one that matters: `--speech-device 1` is refused by name on a CPU-only build before a queue exists, AND §19.6's "device path TAKEN" leg rides `test_minimax_music3_ar`, whose observable `Music3DepthDeviceForwardCount()` is a counter §19.5 itself records as unreachable from production — its only readers are the tests written for it (`test_minimax_music3_ar.cpp:1325,1351,1583,1589,1753,1758`). The instrument that WOULD answer the call site is `ar.depth_staging`, emitted at `minimax_music3_llm.cpp:582` and read by nothing. NOT FIXED IN FLOW and the reason is precise: closing it needs the shipped engine on a real accelerator against the 28.5 GB checkpoint inside an `rc` lease, which is a second GPU leg and a second gate file, not a repair to the row in flight. It is closable by exactly `MUSIC3-DIT-ARM-REACH`'s method — a `gpu;checkpoint;music3`-labelled parity gate entering through `include/vllm.h` with `device = 1`, exiting 77 without its preconditions, asserting `ar.depth_staging` `calls == 1` with the host bucket absent — and that row's `thor:gpu0` job `f63f60e8-957a-4062-92f8-54e5bbb49d92` already FIRED `ar.depth_staging` once without asserting it, so the instrument is known live on the real path. Owed under `## Owed` in [`minimax-music3.md`](../../specs/minimax-music3.md) §19.7 | bug |

## Resolution

-

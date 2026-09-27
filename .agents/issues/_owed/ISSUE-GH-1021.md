ID: ISSUE-GH-1021
Title: LTX-2.5 **DiT device staging takes 450 s — 7.5 minutes — and no record names the phase**. MEASURED on `dgx.casa` (GB10, boot id `03717c9d-63c8-4652-a8fe-a63d012c5718`, build `0e1bee42f`, under `$HOME/gpu.lock`), per-PID at a nominal 2 s over 192 samples: CUDA compute-app footprint 4.22 -> **35.20 GiB**, GPU utilization mean **0.2%** with zero in **164/192** samples, CPU **0.15 cores of 20**, `Anonymous` +0.01 GiB. Neither GPU-bound nor CPU-bound. **The RATE is not one number and this row records the spread rather than the slower figure.** Recomputed from the counters: rung 1 stages 31723 MiB in 450 s = **70.5 MiB/s**; rung 2 stages the same ~32 GiB (`capp_mib` 4322 -> 36396 = 32074 MiB) in **251 s = 127.8 MiB/s** — same host, same boot id, same build, a **1.81x spread** that is itself unattributed, because the sampler recorded no system-wide load column. An earlier draft said "~52 MiB/s", which is the PLATEAU divided by the WHOLE 700 s run rather than by the staging window, and which contradicts its own inputs in both directions (450 s x 52 MiB/s = 22.9 GiB against a recorded 35.54 GiB plateau). Any row taking this lever must measure the rate itself rather than inherit either figure. The shape is `src/vllm/model_executor/models/ltx2_loader.cpp:738-756 @ 332aed738`: ~3,504 tensors, each a raw `cudaMalloc` (`backend.Alloc` at `:747`, `src/vt/cuda/cuda_backend.cu:77-81 @ 332aed738`) followed by a full `backend.Synchronize(queue)` at `:749`, serialized against the host read. The plateau at 36396 MiB = 35.54 GiB lands within 1% of the 35.32 GiB the loader contract predicts. Sampler CSV not retrievable ([#1040](https://github.com/mudler/vllm.cpp/issues/1040)). Same loop as [#1016](https://github.com/mudler/vllm.cpp/issues/1016); a row should take both. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md)
Row: -
State: UNKNOWN
Kind: feature
GitHub: 1021
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:293`

### Frozen archive evidence

> | [#1021](https://github.com/mudler/vllm.cpp/issues/1021) | — | LTX-2.5 **DiT device staging takes 450 s — 7.5 minutes — and no record names the phase**. MEASURED on `dgx.casa` (GB10, boot id `03717c9d-63c8-4652-a8fe-a63d012c5718`, build `0e1bee42f`, under `$HOME/gpu.lock`), per-PID at a nominal 2 s over 192 samples: CUDA compute-app footprint 4.22 -> **35.20 GiB**, GPU utilization mean **0.2%** with zero in **164/192** samples, CPU **0.15 cores of 20**, `Anonymous` +0.01 GiB. Neither GPU-bound nor CPU-bound. **The RATE is not one number and this row records the spread rather than the slower figure.** Recomputed from the counters: rung 1 stages 31723 MiB in 450 s = **70.5 MiB/s**; rung 2 stages the same ~32 GiB (`capp_mib` 4322 -> 36396 = 32074 MiB) in **251 s = 127.8 MiB/s** — same host, same boot id, same build, a **1.81x spread** that is itself unattributed, because the sampler recorded no system-wide load column. An earlier draft said "~52 MiB/s", which is the PLATEAU divided by the WHOLE 700 s run rather than by the staging window, and which contradicts its own inputs in both directions (450 s x 52 MiB/s = 22.9 GiB against a recorded 35.54 GiB plateau). Any row taking this lever must measure the rate itself rather than inherit either figure. The shape is `src/vllm/model_executor/models/ltx2_loader.cpp:738-756 @ 332aed738`: ~3,504 tensors, each a raw `cudaMalloc` (`backend.Alloc` at `:747`, `src/vt/cuda/cuda_backend.cu:77-81 @ 332aed738`) followed by a full `backend.Synchronize(queue)` at `:749`, serialized against the host read. The plateau at 36396 MiB = 35.54 GiB lands within 1% of the 35.32 GiB the loader contract predicts. Sampler CSV not retrievable ([#1040](https://github.com/mudler/vllm.cpp/issues/1040)). Same loop as [#1016](https://github.com/mudler/vllm.cpp/issues/1016); a row should take both. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) | feature |

## Resolution

-

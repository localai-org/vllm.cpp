ID: ISSUE-GH-1375
Title: First end-to-end per-forward cost for the FULL 21.004 B LTX-2.5 DiT on GB10, measured on run `20260819T150230Z` with binary `0a43a750` built from [`7b9e207b1`](https://github.com/mudler/vllm.cpp/commit/7b9e207b1) (#1252). At 1024x576/25f (2304 latent tokens) the governor resolved **7 forward starts from the GPU busy/idle edge counter** and measured `per_forward ~162.0 s` with `first_dit = 481.5 s`, so the recipe's fixed 60 forwards (30 steps x 2 CFG legs, `ltx2_pipeline.cpp:521-529`) project **10 803 s against the rung's 7 153 s budget** and the rung was refused rather than run to the wall. The same lease then COMPLETED 768x448/25f (1344 tokens) in 2990 s, so the ceiling is geometry against lease length, not a defect. TWO instrument facts belong with the number, because both have already caused a wrong reading: `gpu_edges=0` means the GPU never went idle long enough to sample an edge (SATURATED), not that no work ran — this rung sampled 85% of 3191 samples above 50% utilisation; and `eu-stack` resolves no frames in the rc worker container, so phase attribution came from the cpu%/rss signature rather than from symbols. Owned by the LTX-2.5 row; spec [`ltx-2-5.md`](../../specs/ltx-2-5.md)
Row: MODEL-DIFFUSION-LTX25
State: UNKNOWN
Kind: measurement
GitHub: 1375
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:467`

### Frozen archive evidence

> | [#1375](https://github.com/mudler/vllm.cpp/issues/1375) | `MODEL-DIFFUSION-LTX25` | First end-to-end per-forward cost for the FULL 21.004 B LTX-2.5 DiT on GB10, measured on run `20260819T150230Z` with binary `0a43a750` built from [`7b9e207b1`](https://github.com/mudler/vllm.cpp/commit/7b9e207b1) (#1252). At 1024x576/25f (2304 latent tokens) the governor resolved **7 forward starts from the GPU busy/idle edge counter** and measured `per_forward ~162.0 s` with `first_dit = 481.5 s`, so the recipe's fixed 60 forwards (30 steps x 2 CFG legs, `ltx2_pipeline.cpp:521-529`) project **10 803 s against the rung's 7 153 s budget** and the rung was refused rather than run to the wall. The same lease then COMPLETED 768x448/25f (1344 tokens) in 2990 s, so the ceiling is geometry against lease length, not a defect. TWO instrument facts belong with the number, because both have already caused a wrong reading: `gpu_edges=0` means the GPU never went idle long enough to sample an edge (SATURATED), not that no work ran — this rung sampled 85% of 3191 samples above 50% utilisation; and `eu-stack` resolves no frames in the rc worker container, so phase attribution came from the cpu%/rss signature rather than from symbols. Owned by the LTX-2.5 row; spec [`ltx-2-5.md`](../../specs/ltx-2-5.md) | measurement |

## Resolution

-

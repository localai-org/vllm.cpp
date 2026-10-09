ID: ISSUE-LOCAL-01M3VQ4MPSG8NNZZ9B6R8AK28R
Title: gfx1101 decode latency: f32-query attention misses fast arms, GDN scan reads state twice
Row: BACKEND-ROCM
State: CLOSED
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-01
Updated: 2026-10-01
Closed: 2026-10-01

## Problem

Qwen3.5-EXL3 decode on gfx1101 ran 33.7ms/token with the GPU idle ~60%: (1) prefill/decode attention for the model's f32-query/bf16-KV dtype mix missed every fast arm and fell to PagedAttnOnline at 2.7ms/call; (2) GdnScanCoopK read the state row twice per token (349us/call); (3) the dense decode graph and SharedK WMMA prefill stay gated to gfx1100 while gfx1101 carries identical hardware.

## Resolution

Landed on pp-exl3-recon-lt: f32-query SharedK prefill arm, GdnScanCoopFusedK (349us -> 11.4us/call), gfx1101 WMMA prefill admission, decode graph verified capturing on gfx1101 (62 replays). 11.05s -> 3.10s prefill A/B, 29.6 -> 40.4 tok/s decode. Fresh review PASS after repair commit b2028bd57; residual items recorded in spec Outcome.

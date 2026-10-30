ID: ISSUE-LOCAL-01M5WCPZC4TWR9GZVJHH73E1XE
Title: Flint TT keep-quant device arm: stage quantized weights resident on the P150
Row: MODEL-TEXT-FLINT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-26
Updated: 2026-10-26
Closed: -

## Problem

The Flint TT device arm (flint_tt.cpp) expands every GGUF k-quant to bf16 at load (DequantGgufRowToBf16) and then cannot resident the routed-expert tier: 47 MoE layers x 256 experts x (3x2048x512+512x2048) bf16 is ~70.5 GiB against a 32 GiB P150, so the tier STREAMS from host bytes per layer (37.7 MB/layer) with only a 3.606 GiB resident slice. The keep-quant machinery proven on Qwen3.8 (gguf_keep_quant.cpp admission; tenstorrent_keepquant.cpp int8-dot/grouped decodes, enc_sel 0-13) is not wired into the Flint arm at all, so a quantized container pays the full bf16 expansion and the streaming wall the row exists to remove. Wanted: a keep-quant loader arm (admission per DeviceKeepQuantSupported(kTENSTORRENT), refusal by dtype name outside it), TT staging of raw blocks in [N,K] disk order, matmul sites routed through vt::MatmulBT for block-typed weights, the budget math re-derived for quantized residency (expert tier RESIDENT in a few GiB), and an honest device-vs-CPU distance gate on flint-Q4_K_M.

## Resolution

-

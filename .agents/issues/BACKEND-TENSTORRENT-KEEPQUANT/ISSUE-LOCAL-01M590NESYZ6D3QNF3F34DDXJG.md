ID: ISSUE-LOCAL-01M590NESYZ6D3QNF3F34DDXJG
Title: the tt-metal pin advance moved f32 ttnn::sum onto the tf32 FPU tile-reduce — the grouped keep-quant exact-f32 decode dot drifts past its envelope
Row: BACKEND-TENSTORRENT-KEEPQUANT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-19
Updated: 2026-10-19
Closed: -

## Problem

After the tt-metal stack advance (252ca04f8, tt_umd 0.9.12 era /tmp/umdtrial-install, built 2026-10-09) the P=1 grouped decode envelope test fails deterministically on the P150: Q4_K N=1024 K=5120 bad=154/1024, worst_rel 0.1307 (test_tenstorrent_backend.cpp:9696; 3/3 identical runs). KEEPQUANT product code is unchanged since the pin advance. The decode (P==1) arm computes the dot as broadcast multiply + f32 ttnn::sum (tenstorrent_keepquant.cpp, both the fused and the chunk tail), and the drift is BIT-IDENTICAL across the fused arm, VT_TT_KEEPQUANT_FUSED=0, and VT_TT_KEEPQUANT_MM_CHAIN=1 — the shared ttnn::sum is the defective stage, not a matmul arm. On the new pin f32 ttnn::sum lands on the tf32 FPU tile-reduce (the truncation class 8ab35b966 documented for ttnn::matmul), so the exact-f32 decode claim no longer holds.

## Resolution

FIXED in this repair: both exact-dot sites route through ttnn::mean(dim, keep_dim) - K. The pin's mean documents its default as the accurate full-fp32 SFPU path (generic_reductions.hpp). Q4_K decode worst_rel 0.1307 -> 1.7e-4, bad 154/1024 -> 0/1024, focused arm green 2026-10-19. No assertion re-pinned.

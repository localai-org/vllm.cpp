ID: ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ
Title: 27B serve: TT_FATAL writes during trace capture from CaptureSafeReshape tiled reshape
Row: BACKEND-TENSTORRENT-QWEN35
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-28
Updated: 2026-09-28
Closed: -

## Problem

At main 8b5435bb0, the Qwen3.8-27B-Q4_K_M served arm (2x128/32 c2, both VT_TT_KEEPQUANT_INT8DOT=0 and =1) crashes in Qwen3_5DenseDecodeGraph::Step during trace capture: TT_FATAL 'Writes are not supported during trace capture' (tt-metal fd_mesh_command_queue.cpp:826). Chain: EnsureDevice2D -> CaptureSafeReshape -> ttnn::reshape (tiled) -> ReshapeViewTiledProgramFactory::create_program_artifacts -> ttnn::to_device host write mid-capture. Logs: /tmp/int8dot-leg0.log, /tmp/int8dot-leg1.log (2026-09-28). Suspects: the capture-warmup redesign #3321 or the GDN state-binding fix #3327 changed the capture shape; or the pinned tt-metal (9161e8fdb27+4) tiled-reshape path now materializes at artifact creation. The 27B serve arm has not run since those landed; the 9B did. The 92/92 suite does not cover this shape.

## Resolution

-

ID: ISSUE-LOCAL-01M69GMT8WCQZMDNMEF8HH96D3
Title: CPU decode: row-parallel kernels degenerate to one thread at T=1
Row: QUANT-GGUF-CPU-THREADPOOL
State: CLOSED
Kind: performance
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: 2026-11-01

## Problem

At decode (T=1) every kernel that splits work over ROWS hands ParallelForRows nr==1, which short-circuits to the inline caller path, so the 17 pool workers spin while the caller runs the whole 10240-element activation, the rope rotation and the q/k/v split single-threaded. A 32-token greedy request over Nemotron-3.5-Bolt-E4B-GSQ-RCO-IQ3_M on 18 threads spends 2.20 s of 6.7-7.0 s wall in those inline bodies (GeluAndMul 1.21 s at 370 us/call, PagedAttention 0.30 s, RopeNeox 0.22 s, Add 0.17 s, QkvSplit 0.12 s); the same kernels are at per-thread parity with llama.cpp, so this is a parallelism defect, not a kernel defect.

## Resolution

2026-11-01. `kGeluAndMul`, `kRopeNeox`, `kQkvSplit` and `kAdd` now hand
`ParallelForRows` a flat output-element range, so at T=1 the pool receives 18
chunks instead of one inline row. `kRmsNorm` keeps the row split because its
per-row reduction order must not move. Same-binary interleaved A/B on the AVX2
host at 1-minute load 9.8-12.4: 6.355 s to 5.495 s wall (1.157x) and 114.08 s
to 98.78 s CPU (1.155x); all twelve greedy runs are byte-identical and a
`verify.py` capture differs from the pre-change one in zero fields outside
`id` and `created`. Ten review mutations made every seam, extent and boundary
RED; the rope q/k boundary was inert at `Hq == Hk == 1`, so a GQA case and four
determinism-battery entries were added. The rope case holds 32 guard words on
each side of q and of k, so the two head mutants now fail as assertions (233
and 156 of 1,142) instead of corrupting the heap. Evidence:
`evidence/gsq-lumen-w5-20261101/` in the serving deployment. The remaining
inline time (`kPagedAttention` about 0.30 s and `kRmsNorm` about 0.14 s per
32-token request) is a separate gap with its own change.

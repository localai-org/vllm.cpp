ID: ISSUE-LOCAL-01M69GMT8WCQZMDNMEF8HH96D3
Title: CPU decode: row-parallel kernels degenerate to one thread at T=1
Row: QUANT-GGUF-CPU-THREADPOOL
State: OPEN
Kind: performance
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: -

## Problem

At decode (T=1) every kernel that splits work over ROWS hands ParallelForRows nr==1, which short-circuits to the inline caller path, so the 17 pool workers spin while the caller runs the whole 10240-element activation, the rope rotation and the q/k/v split single-threaded. A 32-token greedy request over Nemotron-3.5-Bolt-E4B-GSQ-RCO-IQ3_M on 18 threads spends 2.20 s of 6.7-7.0 s wall in those inline bodies (GeluAndMul 1.21 s at 370 us/call, PagedAttention 0.30 s, RopeNeox 0.22 s, Add 0.17 s, QkvSplit 0.12 s); the same kernels are at per-thread parity with llama.cpp, so this is a parallelism defect, not a kernel defect.

## Resolution

-

ID: ISSUE-LOCAL-01M3WVXG0PV5F0MDB70VK9T9YX
Title: Qwen3.8-27B EXL3 3.5bpw decode to 60 tok/s on gfx1100
Row: BACKEND-ROCM
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-01
Updated: 2026-10-01
Closed: -

## Problem

vllm.cpp serves Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw on the RX 7900 XTX at ~1 tok/s decode (2026-09-29 baseline on main ee622a899: 1.007 tok/s decode-only, kExl3Gemm general kernel, no GEMV arm reach). Branch rocm-gfx11-exl3-perf carries the gfx11 EXL3 GEMV arms; the gap to the ~60 tok/s target (memory-bound ceiling ~70-80 tok/s at 3.5 bpw over ~24 GB weights) is unmeasured at branch head.

## Resolution

-

ID: ISSUE-LOCAL-01M652EAJ85CNWG65FGPTZ8742
Title: Add Pascal Q4_K decode and weight-reusing prefill warp mapping
Row: BACKEND-CUDA-SM061
State: CLOSED
Kind: performance
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-30
Closed: 2026-10-30

## Problem

Pascal Q4_K single-row decode ran the generic Q8_K GEMM (868 ms of a 1.25 s TinyLlama trace), and the generic K-quant prefill mapped consecutive warps to different output rows so each weight row was re-read once per prompt token. Add a 32-lane DP4A Q4_K decode lane and map prefill warps to share one weight row through L1.

## Resolution

2026-10-30: Added the Pascal Q4_K 32-lane DP4A decode lane (868 ms to 406 ms, 2.1x) and the weight-reusing prefill warp mapping (+8% on TinyLlama p16/t16). Focused quant suite passes in the default and rollback configurations; the real-model gate is exact on six of eight axes and records the two near-ties. Evidence: .agents/evidence/pascal-volta-enable/q4k-decode/.

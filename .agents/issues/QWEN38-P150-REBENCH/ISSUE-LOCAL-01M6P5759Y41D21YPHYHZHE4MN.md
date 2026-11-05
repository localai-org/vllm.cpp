ID: ISSUE-LOCAL-01M6P5759Y41D21YPHYHZHE4MN
Title: The qwen3.8-27B P150 re-bench: the first current-head measurement since the 09-14 end-to-end pass
Row: QWEN38-P150-REBENCH
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-05
Updated: 2026-11-05
Closed: -

## Problem

Qwen3.8-27B Q4_K_M completed end-to-end on the P150 on 2026-09-14 (RC=0, TTFT 1389s, TPOT 29.7s, 0.03 tok/s output) -- the only measurement that exists. Every perf fix since then applies to that path (pool match, NEON q4_K/q6_K dots, sliding decomposition, router total order, slot-serve cure) and none has been re-measured on the P150. Deliverable: the same bench (vllm-bench via AsyncLLM, VT_TT_KEEPQUANT_INT8DOT=1 mandatory -- the default lm_head arm OOMs), current head, governor sweep (performance + schedutil), the 09-14 comparison with the intervening fixes named. Regression vs 09-14 beyond run variance = STOP + bisect bracket. Spec: .agents/specs/qwen38-p150-rebench.md

## Resolution

-

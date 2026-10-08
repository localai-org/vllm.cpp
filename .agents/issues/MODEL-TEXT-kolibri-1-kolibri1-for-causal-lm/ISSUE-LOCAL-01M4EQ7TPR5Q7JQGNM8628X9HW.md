ID: ISSUE-LOCAL-01M4EQ7TPR5Q7JQGNM8628X9HW
Title: NEON paged-attention kernel for the shared CPU seam vt::PagedAttention
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: enhancement
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: -

## Problem

kolibri-1 spends ~20% of its decode wall in attn_core through the shared scalar paged-attention seam (src/vt/cpu/cpu_paged_attn.cpp). A NEON (aarch64) kernel in that shared seam benefits every CPU-served model using paged attention, not just kolibri-1. Scope: per-head Q.K^T dot products, f32 softmax, .V accumulation over bf16/f32 (and quantized ApplyKvCacheQuant) KV, sliding-window/causal masking, GQA grouping; aarch64-gated behind VT_CPU_PAGED_ATTN_NEON with the scalar path retained as reference and non-aarch64 path; red-first unit sweep against the scalar oracle plus cross-model paged-engine regression.

## Resolution

2026-10-08 (branch `row/kolibri-neon-attn`): the lane landed behind
`VT_CPU_PAGED_ATTN_NEON`, default OFF. attn_core halves at t32 and t8 (4.9->2.3,
5.3->2.5 s per 64-step decode run); decode tok/s +6% (t8) under load, idle-host
A/B owed. Gates with the lane ON: W3 900/900, ARGMAX 141/145 (4 near-tie, 0
hard), kolibri battery green, unit sweep (14.9M assertions) green, real-operand
replay within 1 bf16 ulp. The decode-bench strict token anchor flips on a
0.197-nat near-tie with the lane ON; the default flip is blocked on the idle
A/B and a near-tie adjudication of that anchor. Evidence:
docs/bench-evidence/kolibri1-neon-paged-attn-20261008.md. Issue stays OPEN
until the default-flip decision lands or the lane is accepted opt-in as the
final state.


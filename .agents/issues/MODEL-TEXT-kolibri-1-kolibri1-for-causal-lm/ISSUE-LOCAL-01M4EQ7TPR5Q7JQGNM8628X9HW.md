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

-

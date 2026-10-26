ID: ISSUE-LOCAL-01M5W2QQBJ32HRB29G1KVQJS77
Title: Flint CPU paged-engine serving arm (MODEL-TEXT-FLINT)
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

Flint serves on CPU only through the single-sequence reference forward: ForwardFlintForCausalLM refuses num_reqs>1 by name and the model never reaches the LLMEngine/paged path, so there is no continuous batching and no honest CPU speed number. The shared paged infra already expresses what Flint needs: per-head attention sinks (cpu_paged_attn.cpp attn_sink), sliding-window paged attention (AttentionWindow), and the hybrid full/SWA KV split (MakeFlintKVCache). Missing is the model-side glue: a paged forward arm over vt ops (QK head-norm, NeoX RoPE on sliding layers only, sinks on every layer, window on sliding layers, sigmoid top-k renormalized MoE with ungated shared expert), plus gates, bench evidence and usage docs.

## Resolution

-

ID: ISSUE-LOCAL-01M41VCM4Z0JEFQHBFS5VT0GYA
Title: Port Kolibri-1 (kolibri1, Kolibri1ForCausalLM) to the CPU path
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-03
Updated: 2026-10-03
Closed: -

## Problem

The model registry has no kolibri1 support. Aleph-Alpha/Kolibri-1 (Kolibri1ForCausalLM, model_type kolibri1) is downloaded at /mnt/models/Aleph-Alpha/Kolibri-1 and the primary oracle plugin aleph-alpha-inference (pinned 049a6a7bd240) implements it on vLLM. This row adds the CPU-only port: config parse + validation, two-group KV cache (full + sliding-window 513), 32-shard FP8-block safetensors weight loader, sigmoid-logit-add MoE router (384 experts, top-6, shared expert, norm_topk_prob=false), hybrid SWA/full attention with qk-norm and RNoPE, sandwich norms, and tokenizer integration, token-exact against a reference. Spec: .agents/specs/kolibri-1-cpu.md

## Resolution

-

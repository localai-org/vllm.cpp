ID: ISSUE-LOCAL-01M3S23H9B25EFPFJN75Y1XBWY
Title: Cohere2MoeForCausalLM (North): port the MoE text model with interleaved sliding window
Row: MODEL-TEXT-cohere2-moe-cohere2-moe-for-causal-lm
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

Cohere2MoeForCausalLM (vLLM a7c23ac96d (the parity pin) vllm/model_executor/models/cohere2_moe.py, CohereLabs/North-Mini-Code-1.0) is not registered. The commandr registry refuses use_qk_norm and sliding_window by name because CohereForCausalLM has neither. Cohere2Moe needs: per-layer sliding window (layer_types, window sliding_window+1) with GPT-J RoPE on sliding and prefix-dense layers only, RMSNorm when rms_norm_eps is set, the parallel attention+MLP block, a dense prefix MLP (first_k_dense_replace / mlp_layer_types), a sigmoid top-k router without renormalization (norm_topk_prob False), optional shared experts with average/sum combination, tied embeddings with logit_scale.

## Resolution

-

ID: ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7
Title: Port MiMoV2ForCausalLM (text-only LLM arm)
Row: MODEL-TEXT-mimo-v2-mi-mo-v2-for-causal-lm
State: OPEN
Kind: MODEL-TEXT
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-26
Updated: 2026-09-26
Closed: -

## Problem

MiMoV2ForCausalLM (model_type: mimo_v2) is not registered, loaded, or forwarded. The checkpoint vcruz305/MiMo-V2.6-Flash-RL-EXL3 ships modeling_mimo_v2.py with auto_map. The architecture combines: hybrid full-attention + sliding-window (128) per-layer pattern with per-layer KV head split (4 vs 8), partial_rotary_factor 0.334, attention_sink_bias (SWA layers only), attention_value_scale 0.707, v_head_dim != head_dim (128 vs 192), sigmoid+noaux_tc MoE (256 experts, 8/tok, no shared experts), moe_layer_freq per-layer MLP type switch (layer 0 dense), and 3 nextn predict layers (MTP). No mimo code exists in the tree.

## Resolution

Port MiMoV2ForCausalLM as a new model (mimo_v2.h/cpp/weights.cpp/registry.cpp) reusing existing vt:: ops for RMSNorm, partial RoPE, paged attention, sigmoid+noaux_tc MoE routing, and MTP. New code: hybrid attention layer dispatch (full vs SWA per hybrid_layer_pattern), attention_sink_bias parameter, attention_value_scale post-projection scalar, moe_layer_freq per-layer MLP type switch. Multimodal arms (vision/audio) are out of scope for this row.

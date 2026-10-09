ID: ISSUE-LOCAL-01M3SQMXYKT2RV2D6MHCY2FCAC
Title: qwen3_5 dense vision: hardcoded 27B/35B tower geometry breaks the 4B checkpoint
Row: MODEL-QWEN35-EXL3
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: 2026-09-30

## Problem

Qwen3_5FamilyVisionConfig in qwen3_5_weights.cpp hardcodes the 27B/35B vision tower (depth 27, hidden_size 1152, intermediate_size 4304) and never reads config.json's vision_config. A Qwen3.5-4B checkpoint ships a depth-24, hidden-1024 tower, so LoadQwen3VLVisionWeights walks blocks.0..26 and fatals on model.visual.blocks.24.norm1.weight. Observed serving UnstableLlama/Qwen3.5-4B-EXL3-6.00bpw: 'server: fatal: vt: qwen3-vl vision: tensor not found: model.visual.blocks.24.norm1.weight'. Fix: parse the checkpoint's vision_config (hidden_size, num_heads, depth, intermediate_size, out_hidden_size, patch_size, temporal_patch_size, spatial_merge_size, num_position_embeddings, in_channels, deepstack_visual_indexes) with the family constants as fallback.

## Resolution

Qwen3_5FamilyVisionConfig now reads config.json vision_config (all tower fields with 27B/35B constants as fallback; out_hidden_size prefers the declared value over text hidden_size). Rebuilt vllm-cpp:rocm-gfx1101-exl3 via docker/Dockerfile.rocm, restarted vllm-qwen35-exl3 on UnstableLlama/Qwen3.5-4B-EXL3-6.00bpw: vision tower loaded (no tensor-not-found), server listening on :8420, /v1/chat/completions returned a valid reasoning-model response (63 tokens).

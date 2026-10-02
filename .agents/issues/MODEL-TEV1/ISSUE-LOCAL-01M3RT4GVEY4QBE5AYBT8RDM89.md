ID: ISSUE-LOCAL-01M3RT4GVEY4QBE5AYBT8RDM89
Title: Qwen3.5 dense/MoE vision loader ignores vision_config and refuses every sub-9B checkpoint
Row: MODEL-TEV1
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: 2026-09-30

## Problem

Qwen3_5FamilyVisionConfig (src/vllm/model_executor/models/qwen3_5_weights.cpp) returns hardcoded 27B/35B tower geometry (depth 27, hidden 1152, 16 heads, intermediate 4304, 2304 position embeddings) and never reads the checkpoint's vision_config. A Qwen3.5 *ForConditionalGeneration checkpoint whose tower is smaller therefore fails at load. Measured on main b45a94273, CPU: togethercomputer/Tev1-0.8B-experimental @6bb2dff1 (vision depth 12) aborts with 'qwen3-vl vision: tensor not found: model.visual.blocks.12.norm1.weight', and togethercomputer/Tev1-4B-experimental @0b7becf0 (depth 24) aborts at model.visual.blocks.24.norm1.weight. The documented docs/models/tev1.md run command cannot start. vLLM builds Qwen3_VisionTransformer from config.vision_config, so the geometry must come from the config.

## Resolution

2026-09-30: Qwen3_5FamilyVisionConfig now reads every Qwen3_VisionTransformer field from config.vision_config (qwen3_vl.py:536-628 @ 5559679229) and keeps the 27B/35B values only as the fallback for an absent key. Red then green: tests/vllm/models/test_qwen3_5_dense_vision.cpp case qwen3_5_dense_vision_config_reads_a_smaller_checkpoint_tower failed 8 assertions (27B values returned) before the fix and passes after; that file had no CMake target since 3edb37da9 and is now registered, 8/8 cases pass; test_qwen3_5_moe_vision 7/7 unchanged. Real weights, CPU: Tev1-0.8B @6bb2dff1 and Tev1-4B @0b7becf0 both load and answer /v1/chat/completions; argmax matches HF transformers 5.3.0 bf16 on 3/3 prompts each.

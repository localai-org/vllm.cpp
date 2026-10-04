ID: ISSUE-LOCAL-01M3QEHEDB23V1VF0VWWDR4DM9
Title: The Qwen3.5 dense loader materializes the whole vision tower for a text-only serve, where the Qwen3-VL, MuseGlimmer and DeepSeek-V4 loaders skip it (#607 L3)
Row: ENG-MM-INPUT-PIPELINE
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-29
Updated: 2026-09-29
Closed: -

## Problem

L3 of #607 (tower skip) landed for Qwen3-VL (src/vllm/model_executor/models/qwen3_vl.cpp:463), MuseGlimmer (muse_glimmer_weights.cpp:803), the clip projector and DeepSeek-V4 (deepseek_v4_mm.cpp:93,122), but NOT for the Qwen3.5 dense arm. LoadQwen3_5Dense (src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:1008) read model.visual.* whenever the checkpoint index named them, so `--language-model-only` (which sets every modality limit to 0 at the chat seam and is sugar over --limit-mm-per-prompt) still paid for the tower: 333 bf16 tensors, measured as host_copy 7.621 GiB vs 6.763 GiB at load on Qwen3.6-27B dense, same binary both arms (docs/FEATURES.md, 2026-09-27). Upstream builds a zero-limit model with the tower uninitialised, so a skipped read is the mirror, not a new policy. The engine already has the contract: SkipTowerForModalities (include/vllm/model_executor/models/interfaces.h:66, src/vllm/model_executor/models/interfaces.cpp:9). The gap is reaching it from LoadQwen3_5Dense and reporting the skip through LoadedModel::skipped_towers(). Found while enumerating the dense arm for the per-modality limit surface; the fix is the same shape the other four loaders use. The pre-existing test file tests/vllm/models/test_qwen3_5_dense_vision.cpp (602 lines) is registered in NO CMake target on main, so the new case cannot run until that registration lands with it.

## Resolution

-

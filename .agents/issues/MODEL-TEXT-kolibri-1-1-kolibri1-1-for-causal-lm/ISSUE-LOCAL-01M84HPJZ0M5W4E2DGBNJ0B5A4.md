ID: ISSUE-LOCAL-01M84HPJZ0M5W4E2DGBNJ0B5A4
Title: Kolibri-1.1 (kolibri1_1) assessment: enumeration, delta vs kolibri1, oracle pins
Row: MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-23
Updated: 2026-11-23
Closed: -

## Problem

Aleph Alpha released Kolibri-1.1 (Kolibri1_1ForCausalLM / kolibri1_1) on 2026-11-23: same pretrained base and MoE as Kolibri-1, new sparse attention (top-2048 keys + 128 recent + 8 sink; 16-token block pool of 16384; NVFP4-routed lightning indexer; indexers 4/14/19/29, reindexers reuse the preceding indexer's pool), per-layer KV sharing (kv_source_layers, shared-KV layers carry only a Q projection), and a dense FFN prefix (layers 0-1, intermediate 3584). This unit is ASSESSMENT ONLY: GGUF walk + KV diff, delta analysis, oracle pinning, effort estimate, refusal verification. Findings already recorded: (a) HF-path refusal-by-name is CORRECT (registry exact-match on Kolibri1ForCausalLM; indexed_attention layer-type refused by name); (b) GGUF path is a trap - llama.cpp's converter writes general.architecture=kolibri1 for 1.1 files, so kolibri1_gguf_weights.cpp would claim a Kolibri-1.1 GGUF; it must refuse one carrying the indexer KVs by name (guard owed in wave U1); (c) no gateable kolibri1_1 oracle exists on CPU today (llama.cpp PR #32891 unmerged; plugin v2.0.0 CUDA-only; transformers lacks kolibri1_1 in main at v5.24.0).

## Resolution

-

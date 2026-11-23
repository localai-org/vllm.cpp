ID: ISSUE-LOCAL-01M84HPYRT3BNNF61BKGNRQ740
Title: kolibri1 GGUF arm must refuse a Kolibri-1.1 GGUF by name (arch-string collision)
Row: MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-23
Updated: 2026-11-23
Closed: -

## Problem

llama.cpp's Kolibri-1.1 converter (conversion/kolibri1.py in PR ggml-org/llama.cpp#32891, unmerged 2026-11-23) maps Kolibri1_1ForCausalLM to GGUF general.architecture = 'kolibri1' - the SAME arch string our GGUF arm claims (kKolibri1GgufArch, kolibri1_gguf_weights.cpp:25, checked at :186 and :222; model_loader.cpp:1297 routes it to Kolibri1HfConfigFromGguf). Kolibri1HfConfigFromGguf reads only kolibri1 KVs and synthesizes architectures={'Kolibri1ForCausalLM'}, so a Kolibri-1.1 GGUF (official Aleph-Alpha/Kolibri-1.1-GGUF Q4_K_M) enters the kolibri1 arm: layer_types come off sliding_window_pattern and call the indexed layers 'full_attention', the new indexer/KV-source KVs are silently ignored, and the shared-KV layers' missing attn_k/v tensors surface at best as an incidental missing-tensor load error rather than a refusal naming the missing kolibri1_1 support. The collision makes the current tree either refuse with a misleading message or, on a file that carries the dropped tensors, silently run the wrong architecture. Required: Kolibri1HfConfigFromGguf (or IsKolibri1Gguf) refuses a kolibri1 GGUF that carries the Kolibri-1.1 indexer KVs (kolibri1.attention.indexer_top_k, the indexer-types array, or the KV-source map) with a message naming the missing kolibri1_1 support, per AGENTS.md 'Refuse an unimplemented arm with a message that names the missing part'. Gated by a red-first test over a fixture GGUF header carrying the indexer KV. Found during the MODEL-TEXT-kolibri-1-1 assessment (this same row); the guard lands with the port row's wave U1.

## Resolution

-

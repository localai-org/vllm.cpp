ID: ISSUE-LOCAL-01M5HJRRHBYMRX4G4JXXKCBFJS
Title: Kolibri-1 GGUF: the published Hob-forge files carry routing and tokenizer metadata that no llama.cpp can load, so the arm needs its own converter
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-22
Updated: 2026-10-22
Closed: -

## Problem

The Hob-forge/Kolibri-1-GGUF files (Q2_K 28,634,017,760 B sha256 de2fac52..., Q4_K_M 47,454,113,472 B sha256 c2ac1301..., repo 40eaa5cf6cfca3afaf8b8832bac962652edb0bde) are NOT llama.cpp-conformant, and the divergence is not cosmetic. Read from the files' own metadata: kolibri1.expert_gating_func = 5, while llama.cpp's enum is {NONE=0, SOFTMAX=1, SIGMOID=2, SOFTMAX_WEIGHT=3, SQRT_SOFTPLUS=4} (src/llama-hparams.h) and gguf-py's is {1,2,4}. llama.cpp's Kolibri-1 model READS that key (src/models/kolibri1.cpp:9 ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func)) and passes it to build_moe_ffn, whose switch ends in 'default: GGML_ABORT("fatal error")' (src/llama-graph.cpp:2218). So llama.cpp at b11765 ABORTS on the first MoE layer: the file cannot be loaded at all, and it cannot be the floor oracle. Two further divergences are silent rather than fatal: kolibri1.expert_weights_norm = False against the config's norm_topk_prob: true, which skips the 'if (norm_w)' renormalisation of the selected routing weights (the expert SELECTION is top-k on the biased logits and is unchanged, but the MoE output is an un-normalised weighted sum); and tokenizer.ggml.pre = kolibri1, a pre-type that does not exist upstream - llama.cpp maps Kolibri-1's tokenizer to qwen2 by sha256 (PR #30673), and master still has 63 pre-types with no KOLIBRI1. By contrast the FLINT GGUF is conformant field-for-field (tokenizer.ggml.pre = flint, the pre-type PR #30928 adds; expert_gating_func = 2 = SIGMOID; expert_weights_norm = True; expert_shared_count = 2; leading_dense_block_count = 2; attention.rope_pattern == the sliding pattern), because it was written by llama.cpp's own converter. CONSEQUENCE: the Kolibri-1 GGUF arm must produce its OWN container with llama.cpp's converter from the upstream bf16 safetensors (the glm5next precedent: our script is the only writer, and the container is ours with its own sha256), or it must patch the three keys and record the result as our container. Until then the arm has no floor and the file's routing cannot be trusted.

## Resolution

-

## Resolution

- 2026-10-23 (the developer's ruling): **ROUTE A** — gate the published
  Hob-forge container as published, interpreting its two non-llama.cpp KVs in
  our loader: `tokenizer.ggml.pre = kolibri1` maps to our qwen2-classic
  pattern (mistral.rs implements the same pre-tokenizer at
  `mistralrs-core/src/gguf/gguf_tokenizer.rs:364`), and
  `expert_gating_func = 5` is NOT consulted for this architecture (the
  kolibri1 routing is fixed: sigmoid, raw unscaled top-k weights, per
  kolibri1.py:126-142; mistral.rs's `build_kolibri1` never reads the key
  either). The file is NOT modified and its sha256 stays the published one.
  The token gate against the primary oracle adjudicates the interpretation.
  mistral.rs is the working reference implementation and the floor for the
  published container. Ettore produces a comparison quantization separately
  (`/mnt/models/KOLIBRI1-GGUF-HANDOFF.md`). The earlier "the file cannot be
  loaded by any implementation" reading is corrected: it cannot be loaded by
  LLAMA.CPP; mistral.rs loads and runs it (benchmarked at 8.0 tok/s on the
  Q4_K_XL).

ID: ISSUE-LOCAL-01M5YRQ1SWC93MDG52QZGDFBY4
Title: FlintHfConfigFromGguf never maps flint.context_length, so paged-engine serving sizes its buffers at max_model_len 0 and segfaults
Row: MODEL-TEXT-FLINT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-27
Updated: 2026-10-27
Closed: 2026-10-27

## Problem

FlintHfConfigFromGguf (flint_gguf_weights.cpp:262) sets every architecture field EXCEPT max_position_embeddings; every other GGUF config builder (qwen3_gguf_weights.cpp:440, qwen3_5:1019, qwen4_exp:110, muse_glimmer:426) maps <arch>.context_length onto it. The engine resolves max_model_len = 0 => config.max_position_embeddings = 0, and the runner's InputBatch sizes token_ids_cpu rows with it — the first add_request writes through a zero-size row and SEGFAULTS (vllm-bench and any FromModelDir serve on the published flint-Q4_K_M.gguf; found by the W-c/W-d waves of MODEL-TEXT-FLINT). The reference/token arms never noticed: they never build an engine.

## Resolution

Fixed 2026-10-27 in the same flow (flint_gguf_weights.cpp): c.max_position_embeddings = OptInt(gguf, flint.context_length, 0), mirroring qwen3/qwen3_5/qwen4_exp/muse_glimmer; the FromModelDir serve then resolves a real max_model_len (auto-fit 4064 on the default pool) and the first add_request no longer segfaults. Red-first: vllm-bench on the published Q4_K_M segfaulted in InputBatch::add_request before the fix; serves after.

ID: ISSUE-LOCAL-01M66NS2QCG5C0JN8Y88SQDPD3
Title: kolibri1 GGUF serving segfaults: the config mapping never reads kolibri1.context_length
Row: KOLIBRI1-GGUF-ARM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-30
Closed: 2026-10-30

## Problem

vllm-bench/vllm-server on a kolibri1 GGUF (APEX I-Micro) segfaults in v1::InputBatch::add_request: Kolibri1HfConfigFromGguf (kolibri1_gguf_weights.cpp:214) never sets HfConfig::max_position_embeddings, so ResolveMaxModelLen (model_loader.cpp:2056) returns 0 and the engine's token buffer is zero-length. The file carries kolibri1.context_length=262144; the flint GGUF mapping (flint_gguf_weights.cpp:386) reads the same key. Row KOLIBRI1-GGUF-ARM, discovered during the throughput legs.

## Resolution

Landed 2026-10-30 as 42022facc (PR #3527). Root cause: Kolibri1HfConfigFromGguf never set max_position_embeddings (zero-length token buffer in InputBatch::add_request) though the file carries kolibri1.context_length = 262144. Fixed via OptInt reading the key (kolibri1_gguf_weights.cpp:168-172, :238). Red-first: CHECK(0 == 262144) captured by the implementer and independently reproduced by the fresh reviewer after the fix landed; serving verified end-to-end through vllm-cli.

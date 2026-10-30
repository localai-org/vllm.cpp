ID: ISSUE-LOCAL-01M66NS2QCG5C0JN8Y88SQDPD3
Title: kolibri1 GGUF serving segfaults: the config mapping never reads kolibri1.context_length
Row: KOLIBRI1-GGUF-ARM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-30
Closed: -

## Problem

vllm-bench/vllm-server on a kolibri1 GGUF (APEX I-Micro) segfaults in v1::InputBatch::add_request: Kolibri1HfConfigFromGguf (kolibri1_gguf_weights.cpp:214) never sets HfConfig::max_position_embeddings, so ResolveMaxModelLen (model_loader.cpp:2056) returns 0 and the engine's token buffer is zero-length. The file carries kolibri1.context_length=262144; the flint GGUF mapping (flint_gguf_weights.cpp:386) reads the same key. Row KOLIBRI1-GGUF-ARM, discovered during the throughput legs.


## Resolution

Fixed in this row's branch: `Kolibri1HfConfigFromGguf` now reads
`kolibri1.context_length` into `HfConfig::max_position_embeddings` (new
`OptInt` helper beside `ReqInt`; the flint GGUF mapping reads the same key),
with a red-first CHECK in the config-map case of `test_kolibri1_gguf`
(`CHECK(0 == 262144)` red, green after the fix). Serving re-verified:
`vllm-cli` completes greedy decode on the APEX I-Micro artifact, and the
engine auto-fits max_model_len 262144 -> 4064 against the KV pool.

-

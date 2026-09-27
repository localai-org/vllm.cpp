ID: ISSUE-GH-1963
Title: At ctx=32768 `--max-num-seqs 32` our engine consumes ~108 GB during load and never serves; vLLM and SGLang both serve there. Root cause found and fixed by `FIX-KV-GROUP-LAYER-COUNT` ([spec](../../specs/kv-group-layer-count.md)): thirty-three of thirty-four registries publish ONE placeholder name per KV group, `KVBytesPerBlock` reads `layer_names.size()` as the layer count, and `ResolveNumBlocks` arm 2 therefore divides an absolute `--kv-cache-memory` budget by ONE layer's page while the runner allocates one buffer per layer — measured 8.5 GiB allocated for a 1 GiB budget on the 27B
Row: ROAD-V1-MEM
State: UNKNOWN
Kind: bug
GitHub: 1963
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:759`

### Frozen archive evidence

> | [#1963](https://github.com/mudler/vllm.cpp/issues/1963) | `ROAD-V1-MEM` | At ctx=32768 `--max-num-seqs 32` our engine consumes ~108 GB during load and never serves; vLLM and SGLang both serve there. Root cause found and fixed by `FIX-KV-GROUP-LAYER-COUNT` ([spec](../../specs/kv-group-layer-count.md)): thirty-three of thirty-four registries publish ONE placeholder name per KV group, `KVBytesPerBlock` reads `layer_names.size()` as the layer count, and `ResolveNumBlocks` arm 2 therefore divides an absolute `--kv-cache-memory` budget by ONE layer's page while the runner allocates one buffer per layer — measured 8.5 GiB allocated for a 1 GiB budget on the 27B | bug |

## Resolution

-

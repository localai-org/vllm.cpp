ID: ISSUE-GH-1196
Title: The server's pooling path builds `EngineParams` from 8 of the engine flags and drops the rest, `--device` included. Found while closing [#1135](https://github.com/mudler/vllm.cpp/issues/1135), which is the same shape over one flag. The `if (pooling_model)` block in `src/vllm/entrypoints/openai/server_main.cpp` sets `block_size`, `num_blocks`, `gpu_memory_utilization`, `kv_cache_memory_bytes`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens` and `enable_prefix_caching`; #1135 adds `offload_config` and `weight_residency`. STILL DROPPED: `--device` (an embedding server started with `--device cuda` runs the accelerator-first probe instead of the named device, so an explicitly named ABSENT device does not fail loudly there — the one with a user-visible consequence today), `--scheduling-policy`, `--kv-transfer-config`, `--speculative-config`, `--enable-jump-forward`/`--disable-jump-forward`, and the multimodal per-modality limits. Each is accepted on the command line, reported by nothing and honoured by nothing; the shape predates [#1110](https://github.com/mudler/vllm.cpp/issues/1110). NOT fixed in the #1135 flow: each dropped flag is a separate behaviour with its own dispatch test surface, and the pooling dispatch block is `ARCH-ONE-SURFACE ROW 6`'s surface rather than `ENG-RESIDENCY-CONFIG`'s. Closing it means one construction of `EngineParams` shared by both branches, or an explicit refusal per flag the pooling path cannot honour. Listed under `## Owed` in [`weight-residency-config.md`](../../specs/weight-residency-config.md)
Row: SERVE-POOLING-ENDPOINTS
State: UNKNOWN
Kind: bug
GitHub: 1196
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:376`

### Frozen archive evidence

> | [#1196](https://github.com/mudler/vllm.cpp/issues/1196) | `SERVE-POOLING-ENDPOINTS` | The server's pooling path builds `EngineParams` from 8 of the engine flags and drops the rest, `--device` included. Found while closing [#1135](https://github.com/mudler/vllm.cpp/issues/1135), which is the same shape over one flag. The `if (pooling_model)` block in `src/vllm/entrypoints/openai/server_main.cpp` sets `block_size`, `num_blocks`, `gpu_memory_utilization`, `kv_cache_memory_bytes`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens` and `enable_prefix_caching`; #1135 adds `offload_config` and `weight_residency`. STILL DROPPED: `--device` (an embedding server started with `--device cuda` runs the accelerator-first probe instead of the named device, so an explicitly named ABSENT device does not fail loudly there — the one with a user-visible consequence today), `--scheduling-policy`, `--kv-transfer-config`, `--speculative-config`, `--enable-jump-forward`/`--disable-jump-forward`, and the multimodal per-modality limits. Each is accepted on the command line, reported by nothing and honoured by nothing; the shape predates [#1110](https://github.com/mudler/vllm.cpp/issues/1110). NOT fixed in the #1135 flow: each dropped flag is a separate behaviour with its own dispatch test surface, and the pooling dispatch block is `ARCH-ONE-SURFACE ROW 6`'s surface rather than `ENG-RESIDENCY-CONFIG`'s. Closing it means one construction of `EngineParams` shared by both branches, or an explicit refusal per flag the pooling path cannot honour. Listed under `## Owed` in [`weight-residency-config.md`](../../specs/weight-residency-config.md) | bug |

## Resolution

-

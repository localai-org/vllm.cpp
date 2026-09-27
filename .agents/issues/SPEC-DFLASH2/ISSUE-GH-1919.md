ID: ISSUE-GH-1919
Title: **The DFlash2 draft context store is hard-capped at 4096 slots, so a >4K-token prompt kills EngineCore and every later request on that server gets `[request submitted to a stopped AsyncLLM]`.** `kDflashMaxCtxSlots = 4096` (`src/vllm/model_executor/models/qwen3_dflash.cpp:1006`) sizes `max_pages` regardless of `--max-model-len`, so the engine advertises 12288, admits the request, and then throws `AppendContextKVDevice: paged store capacity exceeded` from inside the EngineCore step. Upstream has no private store and no private cap: the DFlash draft's context K/V goes into the engine's own paged KV cache (`vllm/model_executor/models/qwen3_dflash.py:604-620` at pin `5559679229`), whose block tables are `cdiv(max_model_len, block_size)` (`vllm/v1/worker/gpu/model_runner.py:426,444`); and where a speculator cannot serve a request it emits an EMPTY draft and lets the target run alone (`vllm/v1/spec_decode/ngram_proposer.py:156-159`, `suffix_decoding.py:59-62`), never raising. Repair: size the store from `max_model_len + num_query_per_req` under a per-request byte budget, fall back to the non-speculative path for a request that outgrows it, and announce the effective speculative context once at startup — wave spec [dflash2-ctx-store-capacity.md](../../specs/dflash2-ctx-store-capacity.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1919
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:728`

### Frozen archive evidence

> | [#1919](https://github.com/mudler/vllm.cpp/issues/1919) | `SPEC-DFLASH2` | **The DFlash2 draft context store is hard-capped at 4096 slots, so a >4K-token prompt kills EngineCore and every later request on that server gets `[request submitted to a stopped AsyncLLM]`.** `kDflashMaxCtxSlots = 4096` (`src/vllm/model_executor/models/qwen3_dflash.cpp:1006`) sizes `max_pages` regardless of `--max-model-len`, so the engine advertises 12288, admits the request, and then throws `AppendContextKVDevice: paged store capacity exceeded` from inside the EngineCore step. Upstream has no private store and no private cap: the DFlash draft's context K/V goes into the engine's own paged KV cache (`vllm/model_executor/models/qwen3_dflash.py:604-620` at pin `5559679229`), whose block tables are `cdiv(max_model_len, block_size)` (`vllm/v1/worker/gpu/model_runner.py:426,444`); and where a speculator cannot serve a request it emits an EMPTY draft and lets the target run alone (`vllm/v1/spec_decode/ngram_proposer.py:156-159`, `suffix_decoding.py:59-62`), never raising. Repair: size the store from `max_model_len + num_query_per_req` under a per-request byte budget, fall back to the non-speculative path for a request that outgrows it, and announce the effective speculative context once at startup — wave spec [dflash2-ctx-store-capacity.md](../../specs/dflash2-ctx-store-capacity.md) | bug |

## Resolution

-

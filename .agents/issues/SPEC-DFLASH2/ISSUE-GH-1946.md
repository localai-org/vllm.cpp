ID: ISSUE-GH-1946
Title: **The DFlash2 draft uploaded a SECOND device copy of the target's embedding table — BF16 `[248320, 5120]` = 2,542,796,800 B (2.543 GB) — because `ResidentWeight` caches its upload on the `OwnedTensor` and the draft held its own.** W9 ([#1849](https://github.com/mudler/vllm.cpp/issues/1849)) made both HOST reads borrow-first and scoped itself to the host in its own comment at `src/vllm/entrypoints/model_loader.cpp:358-360`; the `if (!w.d_dev)` guard at `include/vllm/model_executor/models/dense_attn_block.h:191` is per-tensor, so two `OwnedTensor`s meant two `d_dev` allocations of identical bytes whatever the host residency was. Upstream rebinds the MODULE by reference instead (`vllm/v1/worker/gpu/spec_decode/dflash/utils.py:64-74 @ b389ac29465b33f9e9c534df221ea3c129e9793f`, `del draft_inner.embed_tokens; draft_inner.embed_tokens = target_embed`) and holds one, which our own MTP lane already mirrors (`Qwen3_5MTPModel` points at the target's tensor) and the DFlash lane did not. GB10 is unified memory, so the second copy is 2.543 GB of the same 119 GiB the KV pool comes out of. Fixed in flow: the draft and the target now share ONE `OwnedTensor`, rebound at the one `LoadedEngine` constructor all three draft loaders cross. The `lm_head` half stays owed to the parent spec's `## Owed` O3. See [the embed device dedup spec](../../specs/dflash2-embed-device-dedup.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1946
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:738`

### Frozen archive evidence

> | [#1946](https://github.com/mudler/vllm.cpp/issues/1946) | `SPEC-DFLASH2` | **The DFlash2 draft uploaded a SECOND device copy of the target's embedding table — BF16 `[248320, 5120]` = 2,542,796,800 B (2.543 GB) — because `ResidentWeight` caches its upload on the `OwnedTensor` and the draft held its own.** W9 ([#1849](https://github.com/mudler/vllm.cpp/issues/1849)) made both HOST reads borrow-first and scoped itself to the host in its own comment at `src/vllm/entrypoints/model_loader.cpp:358-360`; the `if (!w.d_dev)` guard at `include/vllm/model_executor/models/dense_attn_block.h:191` is per-tensor, so two `OwnedTensor`s meant two `d_dev` allocations of identical bytes whatever the host residency was. Upstream rebinds the MODULE by reference instead (`vllm/v1/worker/gpu/spec_decode/dflash/utils.py:64-74 @ b389ac29465b33f9e9c534df221ea3c129e9793f`, `del draft_inner.embed_tokens; draft_inner.embed_tokens = target_embed`) and holds one, which our own MTP lane already mirrors (`Qwen3_5MTPModel` points at the target's tensor) and the DFlash lane did not. GB10 is unified memory, so the second copy is 2.543 GB of the same 119 GiB the KV pool comes out of. Fixed in flow: the draft and the target now share ONE `OwnedTensor`, rebound at the one `LoadedEngine` constructor all three draft loaders cross. The `lm_head` half stays owed to the parent spec's `## Owed` O3. See [the embed device dedup spec](../../specs/dflash2-embed-device-dedup.md) | bug |

## Resolution

-

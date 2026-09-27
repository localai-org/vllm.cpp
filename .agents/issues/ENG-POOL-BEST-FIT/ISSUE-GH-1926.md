ID: ISSUE-GH-1926
Title: The other shape-keyed process-lifetime caches on the forward path, each keyed by an EXACT token count and each holding memory for every distinct count ever seen: `row_idx_by_t` (`dense_attn_block.h`, host, retained deliberately because a captured graph bakes its address), `DenseAlignFor` and its second copy in `qwen3_5.cpp` (four device buffers per distinct `M`, never freed), `MoeFusedResident`/`MoeBf16Resident` `tok_map`, and `cuda_matmul.cu`'s `heurs`/`plans` (whose `plans` values hold cuBLASLt descriptors that are never destroyed). Found while measuring #1922: with the `ENG-POOL-BEST-FIT` fix in — and equally with the pool switched off entirely under `VT_POOL_BYPASS=1`, the clean control — the same twelve requests still grow the heap by ~390 KB on a model whose hidden size is 32. An order of magnitude under the pool, and each needs its own decision (size class instead of exact count, a bound with eviction, or preallocation at `max_num_batched_tokens`); `row_idx_by_t` additionally needs the capture lifetime in the answer, so none of them is a one-line change. It is the floor under any memory-steady-state gate, which is why `ENG-POOL-BEST-FIT`'s gate reads `DevicePool::stats()` rather than process bytes. Listed under `## Owed` O2 in [`pool-best-fit-retention.md`](../../specs/pool-best-fit-retention.md)
Row: ENG-POOL-BEST-FIT
State: UNKNOWN
Kind: gap
GitHub: 1926
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:723`

### Frozen archive evidence

> | [#1926](https://github.com/mudler/vllm.cpp/issues/1926) | `ENG-POOL-BEST-FIT` | The other shape-keyed process-lifetime caches on the forward path, each keyed by an EXACT token count and each holding memory for every distinct count ever seen: `row_idx_by_t` (`dense_attn_block.h`, host, retained deliberately because a captured graph bakes its address), `DenseAlignFor` and its second copy in `qwen3_5.cpp` (four device buffers per distinct `M`, never freed), `MoeFusedResident`/`MoeBf16Resident` `tok_map`, and `cuda_matmul.cu`'s `heurs`/`plans` (whose `plans` values hold cuBLASLt descriptors that are never destroyed). Found while measuring #1922: with the `ENG-POOL-BEST-FIT` fix in — and equally with the pool switched off entirely under `VT_POOL_BYPASS=1`, the clean control — the same twelve requests still grow the heap by ~390 KB on a model whose hidden size is 32. An order of magnitude under the pool, and each needs its own decision (size class instead of exact count, a bound with eviction, or preallocation at `max_num_batched_tokens`); `row_idx_by_t` additionally needs the capture lifetime in the answer, so none of them is a one-line change. It is the floor under any memory-steady-state gate, which is why `ENG-POOL-BEST-FIT`'s gate reads `DevicePool::stats()` rather than process bytes. Listed under `## Owed` O2 in [`pool-best-fit-retention.md`](../../specs/pool-best-fit-retention.md) | gap |

## Resolution

-

ID: ISSUE-GH-1165
Title: `--gpu-memory-utilization` is parsed (`src/vllm/entrypoints/openai/server_main.cpp:440-441`), threaded to both engines (`:952`, `:1039`), carried on the C ABI (`include/vllm.h:486`, `src/capi/vllm_c.cpp:577-580`), spelled by `examples/cli/main.cpp:118-119`, defaulted to 0.92 (`include/vllm/entrypoints/model_loader.h:90`) and then read by NOTHING: `LoadedEngine::ResolveNumBlocks` falls through knob 1 (`num_blocks`) and knob 2 (`kv_cache_memory_bytes`) to a bare `return 256` under a `TODO(ROAD-V1-MEM M3)` (`src/vllm/entrypoints/model_loader.cpp:954-959`), so a user who passes `--gpu-memory-utilization 0.85` believes they sized the KV pool and sized nothing. DISTINCT from [#83](https://github.com/mudler/vllm.cpp/issues/83), which owns IMPLEMENTING the utilization path (`ROAD-V1-MEM` M3, dgx-gated on a profile run and an oracle-matched pool). This row owns not lying about it: accept the flag, keeping vLLM's exact name and fraction semantics per `.agents/roadmap_v1.md:71`, and emit one notice per engine load when the caller set it explicitly AND the utilization path is the one that resolved the pool. Fixed in flow. Spec [`gpu-mem-util-inert.md`](../../specs/gpu-mem-util-inert.md)
Row: FIX-GPU-MEM-UTIL-INERT
State: UNKNOWN
Kind: bug
GitHub: 1165
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:359`

### Frozen archive evidence

> | [#1165](https://github.com/mudler/vllm.cpp/issues/1165) | `FIX-GPU-MEM-UTIL-INERT` | `--gpu-memory-utilization` is parsed (`src/vllm/entrypoints/openai/server_main.cpp:440-441`), threaded to both engines (`:952`, `:1039`), carried on the C ABI (`include/vllm.h:486`, `src/capi/vllm_c.cpp:577-580`), spelled by `examples/cli/main.cpp:118-119`, defaulted to 0.92 (`include/vllm/entrypoints/model_loader.h:90`) and then read by NOTHING: `LoadedEngine::ResolveNumBlocks` falls through knob 1 (`num_blocks`) and knob 2 (`kv_cache_memory_bytes`) to a bare `return 256` under a `TODO(ROAD-V1-MEM M3)` (`src/vllm/entrypoints/model_loader.cpp:954-959`), so a user who passes `--gpu-memory-utilization 0.85` believes they sized the KV pool and sized nothing. DISTINCT from [#83](https://github.com/mudler/vllm.cpp/issues/83), which owns IMPLEMENTING the utilization path (`ROAD-V1-MEM` M3, dgx-gated on a profile run and an oracle-matched pool). This row owns not lying about it: accept the flag, keeping vLLM's exact name and fraction semantics per `.agents/roadmap_v1.md:71`, and emit one notice per engine load when the caller set it explicitly AND the utilization path is the one that resolved the pool. Fixed in flow. Spec [`gpu-mem-util-inert.md`](../../specs/gpu-mem-util-inert.md) | bug |

## Resolution

-

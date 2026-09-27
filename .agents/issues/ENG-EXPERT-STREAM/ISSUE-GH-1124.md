ID: ISSUE-GH-1124
Title: `--device cuda` still cannot SERVE a larger-than-pool GGUF after [#1123](https://github.com/mudler/vllm.cpp/issues/1123); it refuses by name instead of dying mid-stream. The missing capability is a DEVICE expert slot store, and it is four pieces: `HostExpertSlotStore` is the only production `ExpertSlotStore` (`include/vllm/model_executor/host_expert_slot_store.h:28`, the only other subclass being a test double) while `include/vllm/model_executor/expert_streamer.h:8-9,30-31` claims "the production destination is a contiguous device-side slot array" and is FALSE today; the interface has no device-capable read, because `KqExpertSlice` reads back through `HostExpertSlotStore::Slot()`, the CONCRETE class (`qwen3_5.cpp:5258,5314`); the filler is `pread`-into-host, since `SlotForWrite` is handed straight to `::pread` (`expert_streamer.cpp:76-94`); and the consumer is device-gated by `is_cpu()` at `qwen3_5.cpp:5578`. Sized: 2790 slices per token at 2,490,368 B is 6.95 GB per token against a 119.631 GiB pool already holding the dense remainder. Deferred because W7 owns the pluggable backing store in the row's work breakdown, and the CPU arm's own decode bandwidth is still VOID (#912 F1 measured it with the step clock dead from token 3), so a device lane would be optimised against a number nobody has. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md)
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: gap
GitHub: 1124
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:343`

### Frozen archive evidence

> | [#1124](https://github.com/mudler/vllm.cpp/issues/1124) | `ENG-EXPERT-STREAM` | `--device cuda` still cannot SERVE a larger-than-pool GGUF after [#1123](https://github.com/mudler/vllm.cpp/issues/1123); it refuses by name instead of dying mid-stream. The missing capability is a DEVICE expert slot store, and it is four pieces: `HostExpertSlotStore` is the only production `ExpertSlotStore` (`include/vllm/model_executor/host_expert_slot_store.h:28`, the only other subclass being a test double) while `include/vllm/model_executor/expert_streamer.h:8-9,30-31` claims "the production destination is a contiguous device-side slot array" and is FALSE today; the interface has no device-capable read, because `KqExpertSlice` reads back through `HostExpertSlotStore::Slot()`, the CONCRETE class (`qwen3_5.cpp:5258,5314`); the filler is `pread`-into-host, since `SlotForWrite` is handed straight to `::pread` (`expert_streamer.cpp:76-94`); and the consumer is device-gated by `is_cpu()` at `qwen3_5.cpp:5578`. Sized: 2790 slices per token at 2,490,368 B is 6.95 GB per token against a 119.631 GiB pool already holding the dense remainder. Deferred because W7 owns the pluggable backing store in the row's work breakdown, and the CPU arm's own decode bandwidth is still VOID (#912 F1 measured it with the step clock dead from token 3), so a device lane would be optimised against a number nobody has. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md) | gap |

## Resolution

-

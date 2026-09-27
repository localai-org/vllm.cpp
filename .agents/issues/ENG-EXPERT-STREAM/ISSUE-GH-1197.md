ID: ISSUE-GH-1197
Title: `Gemma4MoE`'s device-expert LRU tests its slot cap BEFORE its eviction loop, so the eviction opt-in goes inert once the cap is reached. `DevExpertLru::MakeRoom` runs `if (slots.size() >= kMaxSlots) return false;` at `src/vllm/model_executor/models/gemma4_moe.cpp:498` @ `fd64c76ee`, two lines ahead of the `if (allow_evict) { while (used + need > bud && !slots.empty()) EvictOne(d); }` at `:499-500`, and `EvictOne` (`:457`, the DEVICE LRU's — the file carries a host-cache namesake at `:275`) is the only thing that shrinks `slots`. So after 24 admissions every later `MakeRoom` returns false at that first line, the eviction loop is never reached again, and `VT_GEMMA4_EXPERT_EVICT=1` becomes a no-op for the life of the process — the cache degrades permanently to the fill-only mode the opt-in exists to leave. It binds only when the slot cap is reached before the byte budget, i.e. when `24 * expert_bytes < BudgetBytes()` (below ~85.3 MiB per expert at the 2048 MiB default from `BudgetBytes`, `:416-436`, again the device one and not the host cache's at `:262`); above that the byte budget binds first and eviction behaves. Nothing reports which one happened. FILED, NOT FIXED, and not for effort: the one-line repair (move the cap test after the eviction loop, so it caps RESIDENT slots instead of stopping admission forever) wakes more `hipFree` under load, which the code's own comments call a permanent `kfd_wait` hang with the GPU idle, prefill done and no decode tokens (`:459-461` and `:486-488`), so the current ordering may be deliberate. Deciding it needs the dual-RDNA4 lab box of [`gemma4-rocm-fp8-moe.md`](../../specs/gemma4-rocm-fp8-moe.md); the host that found it has neither a ROCm nor a CUDA device. Found while establishing the facts for [#1126](https://github.com/mudler/vllm.cpp/issues/1126), which required reading `MakeRoom` line by line. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md)
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: bug
GitHub: 1197
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:378`

### Frozen archive evidence

> | [#1197](https://github.com/mudler/vllm.cpp/issues/1197) | `ENG-EXPERT-STREAM` | `Gemma4MoE`'s device-expert LRU tests its slot cap BEFORE its eviction loop, so the eviction opt-in goes inert once the cap is reached. `DevExpertLru::MakeRoom` runs `if (slots.size() >= kMaxSlots) return false;` at `src/vllm/model_executor/models/gemma4_moe.cpp:498` @ `fd64c76ee`, two lines ahead of the `if (allow_evict) { while (used + need > bud && !slots.empty()) EvictOne(d); }` at `:499-500`, and `EvictOne` (`:457`, the DEVICE LRU's — the file carries a host-cache namesake at `:275`) is the only thing that shrinks `slots`. So after 24 admissions every later `MakeRoom` returns false at that first line, the eviction loop is never reached again, and `VT_GEMMA4_EXPERT_EVICT=1` becomes a no-op for the life of the process — the cache degrades permanently to the fill-only mode the opt-in exists to leave. It binds only when the slot cap is reached before the byte budget, i.e. when `24 * expert_bytes < BudgetBytes()` (below ~85.3 MiB per expert at the 2048 MiB default from `BudgetBytes`, `:416-436`, again the device one and not the host cache's at `:262`); above that the byte budget binds first and eviction behaves. Nothing reports which one happened. FILED, NOT FIXED, and not for effort: the one-line repair (move the cap test after the eviction loop, so it caps RESIDENT slots instead of stopping admission forever) wakes more `hipFree` under load, which the code's own comments call a permanent `kfd_wait` hang with the GPU idle, prefill done and no decode tokens (`:459-461` and `:486-488`), so the current ordering may be deliberate. Deciding it needs the dual-RDNA4 lab box of [`gemma4-rocm-fp8-moe.md`](../../specs/gemma4-rocm-fp8-moe.md); the host that found it has neither a ROCm nor a CUDA device. Found while establishing the facts for [#1126](https://github.com/mudler/vllm.cpp/issues/1126), which required reading `MakeRoom` line by line. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md) | bug |

## Resolution

-

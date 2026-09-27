ID: ISSUE-GH-1389
Title: Metal, Vulkan and Tenstorrent inherit FlashAttention-2's head-size rule from a backend class they share by NAME, so a head size their own kernels run has no backend. Each registers `FlashAttentionBackend` for `FLASH_ATTN` (`src/vllm/v1/attention/backend.cpp:395-417`) and each returns `{"FLASH_ATTN"}` as its entire non-MLA priority list; the documented precondition for those rows is a shared NHD KV layout, not a shared kernel. Since [#1332](https://github.com/mudler/vllm.cpp/issues/1332) M1 that class also carries `supports_head_size` = `head_size % 8 == 0 && head_size <= 256` (`include/vllm/v1/attention/backend.h:394-397`, `flash_attn.py:170-178` @ pin `5559679229`), a true statement about a CUDA kernel none of those three devices runs. Same defect [#1371](https://github.com/mudler/vllm.cpp/issues/1371) reported for CPU, and the refusal names FlashAttention on boards that have never run FlashAttention. `kROCM` is NOT affected — `RocmAttentionBackend` declares no head-size constraint. CPU is fixable by mirroring upstream's own CPU answer; these three have no upstream equivalent, so the fix must AUTHOR each device's declared capabilities and needs its own spec. Found while fixing #1371 and not fixed in that flow: three platforms, no local runnable gate, and an authored capability surface is what `AGENTS.md` routes to the normal row and fresh-review path. Under `## Owed` in [`attn-validate-configuration.md`](../../specs/attn-validate-configuration.md)
Row: BACKEND-ATTN-REGISTRY
State: UNKNOWN
Kind: bug
GitHub: 1389
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:470`

### Frozen archive evidence

> | [#1389](https://github.com/mudler/vllm.cpp/issues/1389) | `BACKEND-ATTN-REGISTRY` | Metal, Vulkan and Tenstorrent inherit FlashAttention-2's head-size rule from a backend class they share by NAME, so a head size their own kernels run has no backend. Each registers `FlashAttentionBackend` for `FLASH_ATTN` (`src/vllm/v1/attention/backend.cpp:395-417`) and each returns `{"FLASH_ATTN"}` as its entire non-MLA priority list; the documented precondition for those rows is a shared NHD KV layout, not a shared kernel. Since [#1332](https://github.com/mudler/vllm.cpp/issues/1332) M1 that class also carries `supports_head_size` = `head_size % 8 == 0 && head_size <= 256` (`include/vllm/v1/attention/backend.h:394-397`, `flash_attn.py:170-178` @ pin `5559679229`), a true statement about a CUDA kernel none of those three devices runs. Same defect [#1371](https://github.com/mudler/vllm.cpp/issues/1371) reported for CPU, and the refusal names FlashAttention on boards that have never run FlashAttention. `kROCM` is NOT affected — `RocmAttentionBackend` declares no head-size constraint. CPU is fixable by mirroring upstream's own CPU answer; these three have no upstream equivalent, so the fix must AUTHOR each device's declared capabilities and needs its own spec. Found while fixing #1371 and not fixed in that flow: three platforms, no local runnable gate, and an authored capability surface is what `AGENTS.md` routes to the normal row and fresh-review path. Under `## Owed` in [`attn-validate-configuration.md`](../../specs/attn-validate-configuration.md) | bug |

## Resolution

-

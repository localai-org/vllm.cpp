ID: ISSUE-GH-1333
Title: The non-MLA sm_100 attention priority row misses the `use_non_causal` guard the pin added. `include/vllm/platforms/cuda_attn_priority.h::AttnPriorityTable` keys rows on `(use_mla, major)` and cites `vllm/platforms/cuda.py:143-166` `@ pin e24d1b24`; at the tree pin `5559679229` that arm reads `if device_capability.major == 10 and not use_non_causal` (`cuda.py:148`) with `use_non_causal` added as a fifth `_get_backend_priorities` parameter (`cuda.py:88`), because SM100f's non-causal cutlass path — the one DFlash attention uses — is known-bad (`cuda.py:145-147`). No selected name changes in this tree today, since `FLASHINFER` is registered for no device and both orderings fall through to `FLASH_ATTN`; it becomes a real divergence the moment a FlashInfer backend registers, and it is already a divergence in the record because the header claims a faithful, complete port of that function. Found while reconciling the four `@ pin e24d1b24` header anchors under [#1332](https://github.com/mudler/vllm.cpp/issues/1332) M0 and fixed in the same flow, since `use_non_causal` is a field M1 adds to `AttnSelectorConfig` regardless. Spec [`attn-validate-configuration.md`](../../specs/attn-validate-configuration.md)
Row: BACKEND-ATTN-REGISTRY
State: UNKNOWN
Kind: bug
GitHub: 1333
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:451`

### Frozen archive evidence

> | [#1333](https://github.com/mudler/vllm.cpp/issues/1333) | `BACKEND-ATTN-REGISTRY` | The non-MLA sm_100 attention priority row misses the `use_non_causal` guard the pin added. `include/vllm/platforms/cuda_attn_priority.h::AttnPriorityTable` keys rows on `(use_mla, major)` and cites `vllm/platforms/cuda.py:143-166` `@ pin e24d1b24`; at the tree pin `5559679229` that arm reads `if device_capability.major == 10 and not use_non_causal` (`cuda.py:148`) with `use_non_causal` added as a fifth `_get_backend_priorities` parameter (`cuda.py:88`), because SM100f's non-causal cutlass path — the one DFlash attention uses — is known-bad (`cuda.py:145-147`). No selected name changes in this tree today, since `FLASHINFER` is registered for no device and both orderings fall through to `FLASH_ATTN`; it becomes a real divergence the moment a FlashInfer backend registers, and it is already a divergence in the record because the header claims a faithful, complete port of that function. Found while reconciling the four `@ pin e24d1b24` header anchors under [#1332](https://github.com/mudler/vllm.cpp/issues/1332) M0 and fixed in the same flow, since `use_non_causal` is a field M1 adds to `AttnSelectorConfig` regardless. Spec [`attn-validate-configuration.md`](../../specs/attn-validate-configuration.md) | bug |

## Resolution

-

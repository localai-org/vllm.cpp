ID: ISSUE-GH-932
Title: Two arms of upstream `ICLoraPipeline` that [#923](https://github.com/mudler/vllm.cpp/issues/923) deliberately did not build. (1) `conditioning_attention_strength < 1.0` and `conditioning_attention_mask`: only the DEFAULT third branch of `iclora_utils.py:151-160` is served, which is why the row could ship at all — at strength 1.0 with no mask upstream computes no attention mask anywhere. The other two need `Ltx2LatentState` to carry a mask and `build_attention_mask`'s block structure (`mask_utils.py:170-243`); the DiT side already accepts one. NOT blocked behind [#930](https://github.com/mudler/vllm.cpp/issues/930), which closed in `c7cb59fbb`; the remaining reference-arm blockers are [#975](https://github.com/mudler/vllm.cpp/issues/975). (2) N-adapter fusion, which needs upstream's SECOND rounding pattern — `addmm_(B, A, alpha=strength)` at `fuse_loras.py:115`, which rounds differently from the first product and which #923 REFUSES rather than guesses. Listed under `## Owed` in [`ltx25-ic-lora.md`](../../specs/ltx25-ic-lora.md)
Row: -
State: UNKNOWN
Kind: feature
GitHub: 932
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:245`

### Frozen archive evidence

> | [#932](https://github.com/mudler/vllm.cpp/issues/932) | — | Two arms of upstream `ICLoraPipeline` that [#923](https://github.com/mudler/vllm.cpp/issues/923) deliberately did not build. (1) `conditioning_attention_strength < 1.0` and `conditioning_attention_mask`: only the DEFAULT third branch of `iclora_utils.py:151-160` is served, which is why the row could ship at all — at strength 1.0 with no mask upstream computes no attention mask anywhere. The other two need `Ltx2LatentState` to carry a mask and `build_attention_mask`'s block structure (`mask_utils.py:170-243`); the DiT side already accepts one. NOT blocked behind [#930](https://github.com/mudler/vllm.cpp/issues/930), which closed in `c7cb59fbb`; the remaining reference-arm blockers are [#975](https://github.com/mudler/vllm.cpp/issues/975). (2) N-adapter fusion, which needs upstream's SECOND rounding pattern — `addmm_(B, A, alpha=strength)` at `fuse_loras.py:115`, which rounds differently from the first product and which #923 REFUSES rather than guesses. Listed under `## Owed` in [`ltx25-ic-lora.md`](../../specs/ltx25-ic-lora.md) | feature |

## Resolution

-

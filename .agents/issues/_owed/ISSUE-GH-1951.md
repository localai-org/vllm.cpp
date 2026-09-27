ID: ISSUE-GH-1951
Title: **The DSpark draft takes the SAME second device copy of the target's embedding table that [#1946](https://github.com/mudler/vllm.cpp/issues/1946) removed from the DFlash lane, whenever its checkpoint omits one.** `LoadDsparkDraft` moves the target's table into `draft->dspark->backbone.embed_tokens`, which is a second `OwnedTensor`, and `ResidentWeight` caches its device upload on the `OwnedTensor` itself (`include/vllm/model_executor/models/dense_attn_block.h::ResidentWeight`) — so it is a second device allocation of identical bytes, the exact defect #1946 measured at 2,542,796,800 B on the 27B. NOT fixed in flow, and the reason is structural rather than scheduling: `BindDflashDraftSharedEmbed` works because `Qwen3DFlashWeights` can carry a BORROWED `const OwnedTensor*` beside its own table, while the DSpark backbone owns its table BY VALUE inside `Qwen3DSparkWeights`, so rebinding `draft.weights.embed_tokens` there would touch a field the DSpark forward never reads and leave the copy that costs the memory in place. The skip is by name (`if (draft.dspark != nullptr) return false;`) and `tests/vllm/v1/spec_decode/test_dflash2_embed_dedup.cpp` pins it, so the gap cannot become silent. Both published DSpark drafts SHIP their own table, so nothing on the default published path duplicates today. Owed under `## Owed` O2 of [the embed device dedup spec](../../specs/dflash2-embed-device-dedup.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1951
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:739`

### Frozen archive evidence

> | [#1951](https://github.com/mudler/vllm.cpp/issues/1951) | — | **The DSpark draft takes the SAME second device copy of the target's embedding table that [#1946](https://github.com/mudler/vllm.cpp/issues/1946) removed from the DFlash lane, whenever its checkpoint omits one.** `LoadDsparkDraft` moves the target's table into `draft->dspark->backbone.embed_tokens`, which is a second `OwnedTensor`, and `ResidentWeight` caches its device upload on the `OwnedTensor` itself (`include/vllm/model_executor/models/dense_attn_block.h::ResidentWeight`) — so it is a second device allocation of identical bytes, the exact defect #1946 measured at 2,542,796,800 B on the 27B. NOT fixed in flow, and the reason is structural rather than scheduling: `BindDflashDraftSharedEmbed` works because `Qwen3DFlashWeights` can carry a BORROWED `const OwnedTensor*` beside its own table, while the DSpark backbone owns its table BY VALUE inside `Qwen3DSparkWeights`, so rebinding `draft.weights.embed_tokens` there would touch a field the DSpark forward never reads and leave the copy that costs the memory in place. The skip is by name (`if (draft.dspark != nullptr) return false;`) and `tests/vllm/v1/spec_decode/test_dflash2_embed_dedup.cpp` pins it, so the gap cannot become silent. Both published DSpark drafts SHIP their own table, so nothing on the default published path duplicates today. Owed under `## Owed` O2 of [the embed device dedup spec](../../specs/dflash2-embed-device-dedup.md) | bug |

## Resolution

-

ID: ISSUE-GH-1066
Title: `Qwen35ExpertStream` (`src/vllm/model_executor/models/qwen3_5.cpp`) is a **process-lifetime singleton** and keyed its slot cache on `(TowerId(base), expert)`, where `base` is the expert tower's host buffer **ADDRESS**. Its own comment stated the premise and drew the wrong conclusion: "A tower's identity is its base pointer, which is stable for the model's life". The premise is true; the conclusion does not follow, because the CACHE is not scoped to one model's life. Free a model, load another, and the allocator hands the new towers addresses the old ones held, so the new model's expert resolves to an entry filled from a DIFFERENT checkpoint — returned as a HIT, which by contract moves no bytes, so no counter moves and nothing downstream has anything to observe. MEASURED on two synthetic 4-layer/4-expert MoE models in one process, instrumenting `KqExpertSlice` to `memcmp` each returned slot against the tower slice it claims to be: **24 towers occupied 21 distinct addresses, and 20 of 222 slices returned another tower's bytes**; end to end the two arms disagreed on all 160 logits while each arm was internally deterministic (0 differing values on a repeat), which rules out nondeterminism. Invisible to every existing test of this row by construction, because all of them build the cache, store and streamer by hand and none runs two models through the production seam. Reachable by any process that loads a model, releases it, and loads another. Fixed by `OwnedTensor::TowerUid()`, a lazily assigned process-unique counter stamped on the tensor and re-stamped when `bytes` moves (so a copy cannot inherit an identity along with a different buffer); a counter cannot collide because it never goes backwards. Found and fixed while repairing the F1-F11 wiring review for [#912](https://github.com/mudler/vllm.cpp/issues/912). Spec [`expert-streaming.md`](../../specs/expert-streaming.md)
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: bug
GitHub: 1066
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:310`

### Frozen archive evidence

> | [#1066](https://github.com/mudler/vllm.cpp/issues/1066) | `ENG-EXPERT-STREAM` | `Qwen35ExpertStream` (`src/vllm/model_executor/models/qwen3_5.cpp`) is a **process-lifetime singleton** and keyed its slot cache on `(TowerId(base), expert)`, where `base` is the expert tower's host buffer **ADDRESS**. Its own comment stated the premise and drew the wrong conclusion: "A tower's identity is its base pointer, which is stable for the model's life". The premise is true; the conclusion does not follow, because the CACHE is not scoped to one model's life. Free a model, load another, and the allocator hands the new towers addresses the old ones held, so the new model's expert resolves to an entry filled from a DIFFERENT checkpoint — returned as a HIT, which by contract moves no bytes, so no counter moves and nothing downstream has anything to observe. MEASURED on two synthetic 4-layer/4-expert MoE models in one process, instrumenting `KqExpertSlice` to `memcmp` each returned slot against the tower slice it claims to be: **24 towers occupied 21 distinct addresses, and 20 of 222 slices returned another tower's bytes**; end to end the two arms disagreed on all 160 logits while each arm was internally deterministic (0 differing values on a repeat), which rules out nondeterminism. Invisible to every existing test of this row by construction, because all of them build the cache, store and streamer by hand and none runs two models through the production seam. Reachable by any process that loads a model, releases it, and loads another. Fixed by `OwnedTensor::TowerUid()`, a lazily assigned process-unique counter stamped on the tensor and re-stamped when `bytes` moves (so a copy cannot inherit an identity along with a different buffer); a counter cannot collide because it never goes backwards. Found and fixed while repairing the F1-F11 wiring review for [#912](https://github.com/mudler/vllm.cpp/issues/912). Spec [`expert-streaming.md`](../../specs/expert-streaming.md) | bug |

## Resolution

-

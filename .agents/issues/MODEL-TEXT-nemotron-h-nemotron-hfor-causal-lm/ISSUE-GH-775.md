ID: ISSUE-GH-775
Title: `ForwardNemotronHForCausalLM` opened its type-erased handle with an unconditional `static_cast<NemotronHLoadedModel&>(model)`, so on any model that is not really one, every `nh.` member call was type confusion — UBSan's vptr check named `nemotron_h_registry.cpp:112:30` and `-fno-sanitize-recover=all` aborted `test_nemotron_h_scaffold`. Distinct from the TEST repair [#730](https://github.com/mudler/vllm.cpp/issues/730)/PR #784 made: that removed the stub being downcast, which cleared the symptom on the lane while leaving the cast unchecked, so the defect would have stayed invisible after the weight loader lands. FIXED by the checked `vllm::ModelAs<Model>` seam (`model_registry.h`) with the refusal authored once in `RaiseModelTypeMismatch`; spec [`nemotron-h-model.md`](../../specs/nemotron-h-model.md) §6d
Row: MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 775
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:207`

### Frozen archive evidence

> | [#775](https://github.com/mudler/vllm.cpp/issues/775) | `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` | `ForwardNemotronHForCausalLM` opened its type-erased handle with an unconditional `static_cast<NemotronHLoadedModel&>(model)`, so on any model that is not really one, every `nh.` member call was type confusion — UBSan's vptr check named `nemotron_h_registry.cpp:112:30` and `-fno-sanitize-recover=all` aborted `test_nemotron_h_scaffold`. Distinct from the TEST repair [#730](https://github.com/mudler/vllm.cpp/issues/730)/PR #784 made: that removed the stub being downcast, which cleared the symptom on the lane while leaving the cast unchecked, so the defect would have stayed invisible after the weight loader lands. FIXED by the checked `vllm::ModelAs<Model>` seam (`model_registry.h`) with the refusal authored once in `RaiseModelTypeMismatch`; spec [`nemotron-h-model.md`](../../specs/nemotron-h-model.md) §6d | bug |

## Resolution

-

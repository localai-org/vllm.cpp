ID: ISSUE-GH-2173
Title: **The Gemma-4 SigLIP2 vision tower has no production caller, so every measurement quoted about it measures a class rather than a capability.** Measured at `a1dcc74f4`: `grep -rn 'gemma4_vision.h' src/ include/` returns exactly one hit outside the header, `gemma4_vision.cpp:16` including its own header, and the only other includers are `tests/vllm/multimodal/test_gemma4_vision_tower.cpp:25` and `test_gemma4_registry_e2e.cpp:42`; every `Gemma4VisionForward` / `Gemma4VisionWeights` call site is inside those two files. The tower is unreached TWICE OVER, which is what distinguishes it from #1358: the engine driver `Gemma4GenerateGreedyViaRegistry` (`gemma4_mm.cpp:165`) takes `mm_projected` as a CALLER-SUPPLIED `const std::vector<float>&` and masked-scatters it at `:250-252` without ever calling the tower, and that driver's own only caller is `test_gemma4_registry_e2e.cpp:244`. Per [`.agents/reachability.md`](../../reachability.md) this is the test-only-driver shape, and a change with no production call site to delete has already answered the question — there is nothing to mutate. The 2026-07-29 `MM-IMAGE-E2E` fold recorded on the owning row IS real at the `ModelRegistry::Forward` layer (`gemma4_registry.cpp:151` routes `ModelForwardInput.mm` into `Gemma4Model::ForwardMm`); what is missing is everything above it that would build an `mm` field for Gemma-4 from an image. Consequence already observed: #2169's body claimed "Gemma-4 ran that pass on every image" about a per-weight `F32ToBF16` upload pass that no image reaches, and the operator repeated it. Third instance of a class whose other two are filed — #1358 (Qwen3-VL loads its tower and never reads it back) and #1566 (Muse Glimmer's encoder has no production caller) — and the one that was undisclosed. Filed from the fresh review of #2169; listed under `## Owed` in [`vision-tower-dtype-polarity.md`](../../specs/vision-tower-dtype-polarity.md)
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 2173
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:844`

### Frozen archive evidence

> | [#2173](https://github.com/mudler/vllm.cpp/issues/2173) | `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` | **The Gemma-4 SigLIP2 vision tower has no production caller, so every measurement quoted about it measures a class rather than a capability.** Measured at `a1dcc74f4`: `grep -rn 'gemma4_vision.h' src/ include/` returns exactly one hit outside the header, `gemma4_vision.cpp:16` including its own header, and the only other includers are `tests/vllm/multimodal/test_gemma4_vision_tower.cpp:25` and `test_gemma4_registry_e2e.cpp:42`; every `Gemma4VisionForward` / `Gemma4VisionWeights` call site is inside those two files. The tower is unreached TWICE OVER, which is what distinguishes it from #1358: the engine driver `Gemma4GenerateGreedyViaRegistry` (`gemma4_mm.cpp:165`) takes `mm_projected` as a CALLER-SUPPLIED `const std::vector<float>&` and masked-scatters it at `:250-252` without ever calling the tower, and that driver's own only caller is `test_gemma4_registry_e2e.cpp:244`. Per [`.agents/reachability.md`](../../reachability.md) this is the test-only-driver shape, and a change with no production call site to delete has already answered the question — there is nothing to mutate. The 2026-07-29 `MM-IMAGE-E2E` fold recorded on the owning row IS real at the `ModelRegistry::Forward` layer (`gemma4_registry.cpp:151` routes `ModelForwardInput.mm` into `Gemma4Model::ForwardMm`); what is missing is everything above it that would build an `mm` field for Gemma-4 from an image. Consequence already observed: #2169's body claimed "Gemma-4 ran that pass on every image" about a per-weight `F32ToBF16` upload pass that no image reaches, and the operator repeated it. Third instance of a class whose other two are filed — #1358 (Qwen3-VL loads its tower and never reads it back) and #1566 (Muse Glimmer's encoder has no production caller) — and the one that was undisclosed. Filed from the fresh review of #2169; listed under `## Owed` in [`vision-tower-dtype-polarity.md`](../../specs/vision-tower-dtype-polarity.md) | bug |

## Resolution

-

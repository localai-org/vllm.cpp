ID: ISSUE-GH-1566
Title: **The Muse Glimmer perception encoder has no production caller, so every change inside the tower lands unreached.** `MuseGlimmerVisionForward` is called only from `MuseGlimmerEncodePixelGroups` and `MuseGlimmerGenerateGreedyViaRegistry`, and neither has a caller in `src/`, `examples/` or `include/vllm.h` -- only `tests/`. `src/vllm/model_executor/models/muse_glimmer_registry.cpp:13-14` says so: "The perception encoder is still W3, so an image or video prompt is a pending brick." NOT fixed in this flow: wiring an image prompt through `ModelRegistry::Forward` is W4/W5 in `.agents/specs/muse-glimmer.md` §3 and is a model port, not a one-line repair. Filed while landing [#1545](https://github.com/mudler/vllm.cpp/issues/1545), which needs an OPEN issue to name for its `.agents/reachability.md` staged-slice exception and found the model's umbrella issue [#268](https://github.com/mudler/vllm.cpp/issues/268) closed. Owed under `## Owed` in [muse-glimmer.md](../../specs/muse-glimmer.md)
Row: MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation
State: UNKNOWN
Kind: gap
GitHub: 1566
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:549`

### Frozen archive evidence

> | [#1566](https://github.com/mudler/vllm.cpp/issues/1566) | `MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation` | **The Muse Glimmer perception encoder has no production caller, so every change inside the tower lands unreached.** `MuseGlimmerVisionForward` is called only from `MuseGlimmerEncodePixelGroups` and `MuseGlimmerGenerateGreedyViaRegistry`, and neither has a caller in `src/`, `examples/` or `include/vllm.h` -- only `tests/`. `src/vllm/model_executor/models/muse_glimmer_registry.cpp:13-14` says so: "The perception encoder is still W3, so an image or video prompt is a pending brick." NOT fixed in this flow: wiring an image prompt through `ModelRegistry::Forward` is W4/W5 in `.agents/specs/muse-glimmer.md` §3 and is a model port, not a one-line repair. Filed while landing [#1545](https://github.com/mudler/vllm.cpp/issues/1545), which needs an OPEN issue to name for its `.agents/reachability.md` staged-slice exception and found the model's umbrella issue [#268](https://github.com/mudler/vllm.cpp/issues/268) closed. Owed under `## Owed` in [muse-glimmer.md](../../specs/muse-glimmer.md) | gap |

## Resolution

-

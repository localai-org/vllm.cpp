ID: ISSUE-GH-1607
Title: **The DFlash2 startup notice prints TWICE on every draft load, and `docs/USAGE.md` called it one-time.** `CheckDflash2DraftArm` (`src/vllm/entrypoints/model_loader.cpp:502`) ends in an unconditional `std::cerr <<` of the whole notice paragraph with no once-flag, and the loader reaches it TWICE on one load of one `EngineParams`: directly from `FromModelDir` at `:1929`, deliberately placed ahead of every path, config, tokenizer and weight operation so a misclassified draft is caught before a 51.75 GiB target is mapped; and again from `ResolveSpecConfig` at `:1206`, which the `LoadedEngine` constructor runs in its member initializer at `:1538` on all three `new LoadedEngine(...)` returns (`:2172`, `:2341`, `:2359`). The server, the C ABI and the bench client therefore each emit the paragraph twice, on the safetensors arm and the GGUF arm alike. The tree already states that the resolution re-runs (`model_loader.cpp:2313`, `:871-874`); what nothing stated is that the notice re-runs with it. ESTABLISHED STATICALLY, by reading the call graph rather than by executing it -- a runtime confirmation needs a DFlash2 checkpoint and a rebuild of the whole 464-object library. Cosmetic rather than behavioural: nothing is loaded twice and nothing is refused twice, and `CheckDflash2DraftArm` returns early for every non-DFlash2 draft. Found by the THIRD fresh review of `SPEC-DFLASH2` W6 (#1314) and NOT fixed in flow, deliberately: the fix changes the production loader and needs its own red-first test and its own fresh review, while that wave was scoped to prose. `docs/USAGE.md` is corrected in the same change, so the shipped documentation is no longer wrong about the behaviour while this is open. Owed under `## Owed` O25 of [the DFlash2 spec](../../specs/dflash2-spec-decode.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1607
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:591`

### Frozen archive evidence

> | [#1607](https://github.com/mudler/vllm.cpp/issues/1607) | `SPEC-DFLASH2` | **The DFlash2 startup notice prints TWICE on every draft load, and `docs/USAGE.md` called it one-time.** `CheckDflash2DraftArm` (`src/vllm/entrypoints/model_loader.cpp:502`) ends in an unconditional `std::cerr <<` of the whole notice paragraph with no once-flag, and the loader reaches it TWICE on one load of one `EngineParams`: directly from `FromModelDir` at `:1929`, deliberately placed ahead of every path, config, tokenizer and weight operation so a misclassified draft is caught before a 51.75 GiB target is mapped; and again from `ResolveSpecConfig` at `:1206`, which the `LoadedEngine` constructor runs in its member initializer at `:1538` on all three `new LoadedEngine(...)` returns (`:2172`, `:2341`, `:2359`). The server, the C ABI and the bench client therefore each emit the paragraph twice, on the safetensors arm and the GGUF arm alike. The tree already states that the resolution re-runs (`model_loader.cpp:2313`, `:871-874`); what nothing stated is that the notice re-runs with it. ESTABLISHED STATICALLY, by reading the call graph rather than by executing it -- a runtime confirmation needs a DFlash2 checkpoint and a rebuild of the whole 464-object library. Cosmetic rather than behavioural: nothing is loaded twice and nothing is refused twice, and `CheckDflash2DraftArm` returns early for every non-DFlash2 draft. Found by the THIRD fresh review of `SPEC-DFLASH2` W6 (#1314) and NOT fixed in flow, deliberately: the fix changes the production loader and needs its own red-first test and its own fresh review, while that wave was scoped to prose. `docs/USAGE.md` is corrected in the same change, so the shipped documentation is no longer wrong about the behaviour while this is open. Owed under `## Owed` O25 of [the DFlash2 spec](../../specs/dflash2-spec-decode.md) | bug |

## Resolution

-

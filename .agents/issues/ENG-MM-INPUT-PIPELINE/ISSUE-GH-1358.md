ID: ISSUE-GH-1358
Title: `LoadQwen3VLWeights` reads the whole vision tower into `Qwen3VLWeights::vision` (`src/vllm/model_executor/models/qwen3_vl.cpp:418`, on the production path via `qwen3_vl_registry.cpp:97` -> `ModelRegistry::Load` -> `LoadedEngine::FromModelDir`) and NOTHING in `src/` ever reads it back. The only consumers of `Qwen3VLWeights` anywhere are three hardware e2e tests (`test_qwen3vl_e2e.cpp:110`, `test_qwen3vl_video_e2e.cpp:152`, `test_qwen3vl_registry_e2e.cpp:120`); the registered forward consumes ALREADY-MERGED embeddings off `ModelForwardInput.mm` and never touches `vl.weights().vision`. So on the server path the tower is paid for at load — widened bf16 -> host f32 — and never used. This is `.agents/reachability.md`'s unpassed-parameter shape wearing a loader's clothes, and unlike the usual case it costs memory rather than only being dead. FOUND while enumerating every production tower-load site for #607 L3, and it is why the L3 RSS measurement uses Muse Glimmer rather than this site. NOT fixed in flow: wiring the tower into the server's mm forward is the MM-SERVE-E2E residual `server_main.cpp:1315-1321` already names, a feature with its own spec and gate rather than a repair. What L3 does add is the flag that stops paying for it — `--language-model-only` now skips this exact load. Owned by `ENG-MM-INPUT-PIPELINE`; listed under `## Owed` in [`multimodal-track.md`](../../specs/multimodal-track.md) §1.5 L3
Row: ENG-MM-INPUT-PIPELINE
State: UNKNOWN
Kind: bug
GitHub: 1358
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:648`

### Frozen archive evidence

> | [#1358](https://github.com/mudler/vllm.cpp/issues/1358) | `ENG-MM-INPUT-PIPELINE` | `LoadQwen3VLWeights` reads the whole vision tower into `Qwen3VLWeights::vision` (`src/vllm/model_executor/models/qwen3_vl.cpp:418`, on the production path via `qwen3_vl_registry.cpp:97` -> `ModelRegistry::Load` -> `LoadedEngine::FromModelDir`) and NOTHING in `src/` ever reads it back. The only consumers of `Qwen3VLWeights` anywhere are three hardware e2e tests (`test_qwen3vl_e2e.cpp:110`, `test_qwen3vl_video_e2e.cpp:152`, `test_qwen3vl_registry_e2e.cpp:120`); the registered forward consumes ALREADY-MERGED embeddings off `ModelForwardInput.mm` and never touches `vl.weights().vision`. So on the server path the tower is paid for at load — widened bf16 -> host f32 — and never used. This is `.agents/reachability.md`'s unpassed-parameter shape wearing a loader's clothes, and unlike the usual case it costs memory rather than only being dead. FOUND while enumerating every production tower-load site for #607 L3, and it is why the L3 RSS measurement uses Muse Glimmer rather than this site. NOT fixed in flow: wiring the tower into the server's mm forward is the MM-SERVE-E2E residual `server_main.cpp:1315-1321` already names, a feature with its own spec and gate rather than a repair. What L3 does add is the flag that stops paying for it — `--language-model-only` now skips this exact load. Owned by `ENG-MM-INPUT-PIPELINE`; listed under `## Owed` in [`multimodal-track.md`](../../specs/multimodal-track.md) §1.5 L3 | bug |

## Resolution

-

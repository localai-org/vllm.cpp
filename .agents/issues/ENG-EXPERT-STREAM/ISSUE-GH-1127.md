ID: ISSUE-GH-1127
Title: `VT_DEVICE_WEIGHT_BUDGET_BYTES`, added by [#1123](https://github.com/mudler/vllm.cpp/issues/1123) to override the probed device memory pool for the load-time fit refusal, should be a weight-residency CONFIG key rather than an environment variable, for the reason `ENG-RESIDENCY-CONFIG` gives for the five `VT_GGUF_*` / `VT_MOE_EXPERT_STREAM*` knobs it is converting. It was left as an environment variable ON PURPOSE: [#1110](https://github.com/mudler/vllm.cpp/issues/1110) / PR #1119 is in flight, adds exactly the `vllm_cpp` namespace inside `--offload-config` this key belongs in plus `include/vllm/config/weight_residency.h`, and touches the same `src/vllm/entrypoints/model_loader.cpp`, so landing a competing config surface first would create the conflict both changes then resolve. Closing it means a `device_weight_budget_bytes` key under that object, the loader reading `EngineParams::weight_residency` instead of `std::getenv`, and `docs/ENVIRONMENT.md` plus `docs/USAGE.md` following. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md)
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: gap
GitHub: 1127
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:345`

### Frozen archive evidence

> | [#1127](https://github.com/mudler/vllm.cpp/issues/1127) | `ENG-EXPERT-STREAM` | `VT_DEVICE_WEIGHT_BUDGET_BYTES`, added by [#1123](https://github.com/mudler/vllm.cpp/issues/1123) to override the probed device memory pool for the load-time fit refusal, should be a weight-residency CONFIG key rather than an environment variable, for the reason `ENG-RESIDENCY-CONFIG` gives for the five `VT_GGUF_*` / `VT_MOE_EXPERT_STREAM*` knobs it is converting. It was left as an environment variable ON PURPOSE: [#1110](https://github.com/mudler/vllm.cpp/issues/1110) / PR #1119 is in flight, adds exactly the `vllm_cpp` namespace inside `--offload-config` this key belongs in plus `include/vllm/config/weight_residency.h`, and touches the same `src/vllm/entrypoints/model_loader.cpp`, so landing a competing config surface first would create the conflict both changes then resolve. Closing it means a `device_weight_budget_bytes` key under that object, the loader reading `EngineParams::weight_residency` instead of `std::getenv`, and `docs/ENVIRONMENT.md` plus `docs/USAGE.md` following. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md) | gap |

## Resolution

-

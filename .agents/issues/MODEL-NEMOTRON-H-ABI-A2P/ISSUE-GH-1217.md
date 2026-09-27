ID: ISSUE-GH-1217
Title: `ModelForwardInput::device_token_ids` carries the async runner's device-combined ids and its contract is that `token_ids` is STALE for decode rows whenever the pointer is non-null (`model_registry.h:314-324`). A registered forward that embeds the host vector then embeds the same placeholder id on EVERY decode step. The field's own comment says a model that ignores it "is simply never given one", but `runner.cpp:1408` sets the pointer for whatever model the step routes to, with no per-model opt-in and no check | two models have now been cut from the identical divergence: Kimi-Linear (`kimi_linear_device.cpp:2270-2280`, the GB10 9/128 case) and NemotronH's paged forward under [#1157](https://github.com/mudler/vllm.cpp/issues/1157), whose A3 gate read 4/24 on GB10 against 96/96 for the same binary on CPU where the pointer is always null | invisible because the runner sets it only under `VLLM_CPP_CUDA` with a live device mirror, so no CPU gate reaches the branch, and the failure is fluent wrong tokens rather than an error | two closes: give `ModelFactory` an explicit `honors_device_token_ids` and have the runner fall back to the synchronous host path for a forward that has not declared it, or add a checker over the registered `.forward` entry points (a file-level grep flags ~25 false positives because several models delegate through `detail::DeviceTokenIdsScope` or the shared dense block) | NOT fixed in the #1157 flow because one close changes a shared seam and every model factory and the other changes checker semantics, which is the "needs its own spec" case rather than the in-flow case. Listed under `## Owed` in [`nemotron-h-a2p-paged-forward.md`](../../specs/nemotron-h-a2p-paged-forward.md)
Row: MODEL-NEMOTRON-H-ABI-A2P
State: UNKNOWN
Kind: bug
GitHub: 1217
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:404`

### Frozen archive evidence

> | [#1217](https://github.com/mudler/vllm.cpp/issues/1217) | `MODEL-NEMOTRON-H-ABI-A2P` | `ModelForwardInput::device_token_ids` carries the async runner's device-combined ids and its contract is that `token_ids` is STALE for decode rows whenever the pointer is non-null (`model_registry.h:314-324`). A registered forward that embeds the host vector then embeds the same placeholder id on EVERY decode step. The field's own comment says a model that ignores it "is simply never given one", but `runner.cpp:1408` sets the pointer for whatever model the step routes to, with no per-model opt-in and no check \| two models have now been cut from the identical divergence: Kimi-Linear (`kimi_linear_device.cpp:2270-2280`, the GB10 9/128 case) and NemotronH's paged forward under [#1157](https://github.com/mudler/vllm.cpp/issues/1157), whose A3 gate read 4/24 on GB10 against 96/96 for the same binary on CPU where the pointer is always null \| invisible because the runner sets it only under `VLLM_CPP_CUDA` with a live device mirror, so no CPU gate reaches the branch, and the failure is fluent wrong tokens rather than an error \| two closes: give `ModelFactory` an explicit `honors_device_token_ids` and have the runner fall back to the synchronous host path for a forward that has not declared it, or add a checker over the registered `.forward` entry points (a file-level grep flags ~25 false positives because several models delegate through `detail::DeviceTokenIdsScope` or the shared dense block) \| NOT fixed in the #1157 flow because one close changes a shared seam and every model factory and the other changes checker semantics, which is the "needs its own spec" case rather than the in-flow case. Listed under `## Owed` in [`nemotron-h-a2p-paged-forward.md`](../../specs/nemotron-h-a2p-paged-forward.md) | bug |

## Resolution

-

ID: ISSUE-GH-1049
Title: `Ltx2Guidance` (`src/vllm/model_executor/models/ltx2_pipeline.cpp:526 @ c1fe35592`) is dead in production: `git grep -n 'Ltx2Guidance(' -- src include examples tests` returns the declaration (`ltx2_pipeline.h:330`), the definition, and ONE call, at `tests/vllm/models/test_ltx2_pipeline.cpp:710`. It is also the only path to two ported guiders — `Ltx2CfgDelta` (`ltx2_pipeline.cpp:532`, plus `test_ltx2_pipeline.cpp:615`) and `Ltx2StgDelta` (`:534`, plus `:630`) — so both are reachable from no product entry point either. `Ltx2BatchedPerturbationConfig` (`ltx2_pipeline.h:380`) is the same shape from a different direction: constructed only at `test_ltx2_pipeline.cpp:832-859`, while the LTX-2.5 text-to-audio path perturbs through `Ltx2DitPerturbation`, a different type, and no other path perturbs at all. This is the test-only-driver shape [`reachability.md`](../../reachability.md) names. All four landed with #641. `Ltx2MultiModalGuidance` was the fourth member of the set and is no longer one: [#1005](https://github.com/mudler/vllm.cpp/issues/1005) gave it a production call site in `ltx2_t2a.cpp`. Closing this means either routing a product path through `Ltx2Guidance` with a configured `Ltx2GuiderKind` — `Ltx2GuiderSigmaBin` and `Ltx2GuiderParamsForSigma` are already ported beside it — or retiring the unreached arms into `.agents/completed/` with their provenance. Found repairing the fresh review of [#1039](https://github.com/mudler/vllm.cpp/issues/1039) on PR [#1032](https://github.com/mudler/vllm.cpp/pull/1032), where an earlier draft of that row's spec §6b claimed the row ended all four test-only drivers; §6b now carries the measured table. Listed under `## Owed` in [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1049
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:305`

### Frozen archive evidence

> | [#1049](https://github.com/mudler/vllm.cpp/issues/1049) | — | `Ltx2Guidance` (`src/vllm/model_executor/models/ltx2_pipeline.cpp:526 @ c1fe35592`) is dead in production: `git grep -n 'Ltx2Guidance(' -- src include examples tests` returns the declaration (`ltx2_pipeline.h:330`), the definition, and ONE call, at `tests/vllm/models/test_ltx2_pipeline.cpp:710`. It is also the only path to two ported guiders — `Ltx2CfgDelta` (`ltx2_pipeline.cpp:532`, plus `test_ltx2_pipeline.cpp:615`) and `Ltx2StgDelta` (`:534`, plus `:630`) — so both are reachable from no product entry point either. `Ltx2BatchedPerturbationConfig` (`ltx2_pipeline.h:380`) is the same shape from a different direction: constructed only at `test_ltx2_pipeline.cpp:832-859`, while the LTX-2.5 text-to-audio path perturbs through `Ltx2DitPerturbation`, a different type, and no other path perturbs at all. This is the test-only-driver shape [`reachability.md`](../../reachability.md) names. All four landed with #641. `Ltx2MultiModalGuidance` was the fourth member of the set and is no longer one: [#1005](https://github.com/mudler/vllm.cpp/issues/1005) gave it a production call site in `ltx2_t2a.cpp`. Closing this means either routing a product path through `Ltx2Guidance` with a configured `Ltx2GuiderKind` — `Ltx2GuiderSigmaBin` and `Ltx2GuiderParamsForSigma` are already ported beside it — or retiring the unreached arms into `.agents/completed/` with their provenance. Found repairing the fresh review of [#1039](https://github.com/mudler/vllm.cpp/issues/1039) on PR [#1032](https://github.com/mudler/vllm.cpp/pull/1032), where an earlier draft of that row's spec §6b claimed the row ended all four test-only drivers; §6b now carries the measured table. Listed under `## Owed` in [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md) | bug |

## Resolution

-

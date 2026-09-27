ID: ISSUE-GH-847
Title: Residue of the registry type-confusion class after #775: 34 `prepare`/`forward` entry points across 32 model TUs still downcast a type-erased `LoadedModel&` with an unchecked `static_cast`. The `ModelAs` seam they need already exists, so the sweep is mechanical EXCEPT for one decision it must make first — `llama_registry.cpp`, `qwen3_5_dense.cpp` and `gemma4_registry.cpp` each register THREE architectures against ONE forward, so those sites have no single architecture name to refuse under. Owed by [`nemotron-h-model.md`](../../specs/nemotron-h-model.md) `## Owed` until a row claims it
Row: -
State: UNKNOWN
Kind: bug
GitHub: 847
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:208`

### Frozen archive evidence

> | [#847](https://github.com/mudler/vllm.cpp/issues/847) | — | Residue of the registry type-confusion class after #775: 34 `prepare`/`forward` entry points across 32 model TUs still downcast a type-erased `LoadedModel&` with an unchecked `static_cast`. The `ModelAs` seam they need already exists, so the sweep is mechanical EXCEPT for one decision it must make first — `llama_registry.cpp`, `qwen3_5_dense.cpp` and `gemma4_registry.cpp` each register THREE architectures against ONE forward, so those sites have no single architecture name to refuse under. Owed by [`nemotron-h-model.md`](../../specs/nemotron-h-model.md) `## Owed` until a row claims it | bug |

## Resolution

-

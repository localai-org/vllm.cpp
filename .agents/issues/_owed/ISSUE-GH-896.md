ID: ISSUE-GH-896
Title: No gate refuses a new unchecked `static_cast<...LoadedModel&>` in a registry entry point. The #847 sweep decided the checker is warranted — the class is exactly grep-able, unlike the unaligned-read class in #627, and it regrows the moment a new model port copies its neighbour — and deliberately did not bundle a repository-wide refusal gate into a 30-file mechanical sweep. Listed under `## Owed` in [`registry-downcast-sweep.md`](../../specs/registry-downcast-sweep.md) §6
Row: -
State: UNKNOWN
Kind: feature
GitHub: 896
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:237`

### Frozen archive evidence

> | [#896](https://github.com/mudler/vllm.cpp/issues/896) | — | No gate refuses a new unchecked `static_cast<...LoadedModel&>` in a registry entry point. The #847 sweep decided the checker is warranted — the class is exactly grep-able, unlike the unaligned-read class in #627, and it regrows the moment a new model port copies its neighbour — and deliberately did not bundle a repository-wide refusal gate into a 30-file mechanical sweep. Listed under `## Owed` in [`registry-downcast-sweep.md`](../../specs/registry-downcast-sweep.md) §6 | feature |

## Resolution

-

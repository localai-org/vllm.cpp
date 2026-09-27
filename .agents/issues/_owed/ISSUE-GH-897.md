ID: ISSUE-GH-897
Title: `ModelAs`'s `const LoadedModel&` overload has no caller anywhere in the tree: every registered `prepare`/`forward` takes a NON-const `LoadedModel&`, so all 35 call sites select the non-const overload and its `Model&` result merely binds to the `const auto&`. Proven by deleting the overload and compiling all 30 swept TUs `-fsyntax-only`. It is dead in the way that reads as covered — #847 cited its existence as the reason the 14 `const` sites needed no thought. Listed under `## Owed` in [`registry-downcast-sweep.md`](../../specs/registry-downcast-sweep.md) §3.3
Row: -
State: UNKNOWN
Kind: bug
GitHub: 897
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:238`

### Frozen archive evidence

> | [#897](https://github.com/mudler/vllm.cpp/issues/897) | — | `ModelAs`'s `const LoadedModel&` overload has no caller anywhere in the tree: every registered `prepare`/`forward` takes a NON-const `LoadedModel&`, so all 35 call sites select the non-const overload and its `Model&` result merely binds to the `const auto&`. Proven by deleting the overload and compiling all 30 swept TUs `-fsyntax-only`. It is dead in the way that reads as covered — #847 cited its existence as the reason the 14 `const` sites needed no thought. Listed under `## Owed` in [`registry-downcast-sweep.md`](../../specs/registry-downcast-sweep.md) §3.3 | bug |

## Resolution

-

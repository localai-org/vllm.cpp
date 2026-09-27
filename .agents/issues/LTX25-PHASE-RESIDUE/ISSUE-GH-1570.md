ID: ISSUE-GH-1570
Title: nothing bounds the instrument's own share of a leaf, so an `uncovered <= 2 * leaf_instrument` bound can widen SILENTLY while printing a small number -- moving the DiT `Tick` out of `Evaluate` would charge ~110 flushed writes to `denoise` and buy a budget larger than the floor it replaces. Found as F5 by that row's fresh review. This is also what would close [#1439](https://github.com/mudler/vllm.cpp/issues/1439), whose filed complaint is that the budget is a SHARE OF `wall` and therefore decides by box load: what is needed is a bound on a quantity the scheduler cannot move, and `## Design` 3 of the spec is the measured evidence that the obvious candidate is not it. Listed under `## Owed` in [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md)
Row: LTX25-PHASE-RESIDUE
State: UNKNOWN
Kind: bug
GitHub: 1570
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:606`

### Frozen archive evidence

> | [#1570](https://github.com/mudler/vllm.cpp/issues/1570) | `LTX25-PHASE-RESIDUE` | nothing bounds the instrument's own share of a leaf, so an `uncovered <= 2 * leaf_instrument` bound can widen SILENTLY while printing a small number -- moving the DiT `Tick` out of `Evaluate` would charge ~110 flushed writes to `denoise` and buy a budget larger than the floor it replaces. Found as F5 by that row's fresh review. This is also what would close [#1439](https://github.com/mudler/vllm.cpp/issues/1439), whose filed complaint is that the budget is a SHARE OF `wall` and therefore decides by box load: what is needed is a bound on a quantity the scheduler cannot move, and `## Design` 3 of the spec is the measured evidence that the obvious candidate is not it. Listed under `## Owed` in [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md) | bug |

## Resolution

-

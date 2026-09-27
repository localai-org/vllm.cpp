ID: ISSUE-GH-1572
Title: assertion (1c)'s span slack reds intermittently on `main` -- `decode.video` at `0.00256913` against a `0.00075` bound, 3.4x. Pre-existing from `6b48edb2c` and NOT `LTX25-PHASE-RESIDUE`'s, which does not touch (1c) and keeps its constants; observed by that row's fresh review and filed rather than repaired. Untouched by the record landing, since [#1556](https://github.com/mudler/vllm.cpp/issues/1556) is closed and no assertion moves. Listed under `## Owed` in [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md)
Row: LTX25-PHASE-RESIDUE
State: UNKNOWN
Kind: bug
GitHub: 1572
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:608`

### Frozen archive evidence

> | [#1572](https://github.com/mudler/vllm.cpp/issues/1572) | `LTX25-PHASE-RESIDUE` | assertion (1c)'s span slack reds intermittently on `main` -- `decode.video` at `0.00256913` against a `0.00075` bound, 3.4x. Pre-existing from `6b48edb2c` and NOT `LTX25-PHASE-RESIDUE`'s, which does not touch (1c) and keeps its constants; observed by that row's fresh review and filed rather than repaired. Untouched by the record landing, since [#1556](https://github.com/mudler/vllm.cpp/issues/1556) is closed and no assertion moves. Listed under `## Owed` in [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md) | bug |

## Resolution

-

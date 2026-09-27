ID: ISSUE-GH-1569
Title: `PhaseLog::WriteJson`'s clock ordering is ungated and its own mutation stays GREEN 10 of 10 -- `wall 0.0608987s, unaccounted 0.000534223s, table charge 0.000301655s` -- because the copy and sort of a three-record table are nanoseconds. The claim was WITHDRAWN from the source by `LTX25-PHASE-RESIDUE` rather than defended, which is the reason this is a filed gap and not a passing test. Gating it needs a table with enough records for the sort to be measurable and a `WriteJson` with nothing between it and the last `Close`. Listed under `## Owed` in [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md)
Row: LTX25-PHASE-RESIDUE
State: UNKNOWN
Kind: bug
GitHub: 1569
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:605`

### Frozen archive evidence

> | [#1569](https://github.com/mudler/vllm.cpp/issues/1569) | `LTX25-PHASE-RESIDUE` | `PhaseLog::WriteJson`'s clock ordering is ungated and its own mutation stays GREEN 10 of 10 -- `wall 0.0608987s, unaccounted 0.000534223s, table charge 0.000301655s` -- because the copy and sort of a three-record table are nanoseconds. The claim was WITHDRAWN from the source by `LTX25-PHASE-RESIDUE` rather than defended, which is the reason this is a filed gap and not a passing test. Gating it needs a table with enough records for the sort to be measurable and a `WriteJson` with nothing between it and the last `Close`. Listed under `## Owed` in [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md) | bug |

## Resolution

-

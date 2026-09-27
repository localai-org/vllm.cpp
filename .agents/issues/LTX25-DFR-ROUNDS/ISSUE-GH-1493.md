ID: ISSUE-GH-1493
Title: **DFR's unclamped `2**round_idx` tile count is mirrored and gated by nothing, because every fixture canvas in this tree has ONE keyframe segment.** `tile_ranges` clamps to `min(num_tiles, n_segments)` (`dfr_layout.py:171`), and the 9-frame fixture pads to a 25-frame canvas with a single segment, so round 1 asks for 2 windows and gets 1 and round 2 asks for 4 and gets 2. A port computing `round_idx + 1`, or `2 * round_idx`, or capping at 2 returns the SAME tile counts on every test here, with every downstream shape, frame count and exit status identical. NOT FIXED IN FLOW and the judgement is recorded rather than reversed: reaching 4 segments needs a materially longer canvas, so round 2 would denoise 4 tiles on a canvas already doubled twice - a new fixture and a substantially longer CPU run in a file that already carries 102 cases, not an assertion added to the existing render. The bound is stated in the test body and in `docs/USAGE.md`, so it was disclosed before it was owned; this row is the ownership. Closing it needs one render whose canvas carries at least 4 segments plus an assertion that `round_tile_counts` reads the unclamped `2**round_idx` for at least one round, which is the only shape that separates `2**round` from every expression agreeing with it at 1 and 2. Listed under `## Owed` in [`ltx25-dfr-rounds.md`](../../specs/ltx25-dfr-rounds.md)
Row: LTX25-DFR-ROUNDS
State: UNKNOWN
Kind: bug
GitHub: 1493
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:520`

### Frozen archive evidence

> | [#1493](https://github.com/mudler/vllm.cpp/issues/1493) | `LTX25-DFR-ROUNDS` | **DFR's unclamped `2**round_idx` tile count is mirrored and gated by nothing, because every fixture canvas in this tree has ONE keyframe segment.** `tile_ranges` clamps to `min(num_tiles, n_segments)` (`dfr_layout.py:171`), and the 9-frame fixture pads to a 25-frame canvas with a single segment, so round 1 asks for 2 windows and gets 1 and round 2 asks for 4 and gets 2. A port computing `round_idx + 1`, or `2 * round_idx`, or capping at 2 returns the SAME tile counts on every test here, with every downstream shape, frame count and exit status identical. NOT FIXED IN FLOW and the judgement is recorded rather than reversed: reaching 4 segments needs a materially longer canvas, so round 2 would denoise 4 tiles on a canvas already doubled twice - a new fixture and a substantially longer CPU run in a file that already carries 102 cases, not an assertion added to the existing render. The bound is stated in the test body and in `docs/USAGE.md`, so it was disclosed before it was owned; this row is the ownership. Closing it needs one render whose canvas carries at least 4 segments plus an assertion that `round_tile_counts` reads the unclamped `2**round_idx` for at least one round, which is the only shape that separates `2**round` from every expression agreeing with it at 1 and 2. Listed under `## Owed` in [`ltx25-dfr-rounds.md`](../../specs/ltx25-dfr-rounds.md) | bug |

## Resolution

-

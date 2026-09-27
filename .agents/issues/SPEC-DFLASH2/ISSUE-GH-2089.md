ID: ISSUE-GH-2089
Title: **The W11 draft-block route counters are blind to the production `P > 1` lane.** Both `NoteDflashBlockRoute` increments sit inside the `P == 1` branch (`qwen3_dflash.cpp:1487`, `:1502`); the materialized fallback at `:1888-1930` increments neither, so at every concurrency above one a route gate reads zero for both lanes while production runs a third route nothing names. #1890 put the counter inside the branch precisely so it would measure a capability rather than a class, and this is the hole that argument left. Listed under `## Owed` in [`specs/dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: test-gap
GitHub: 2089
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:803`

### Frozen archive evidence

> | [#2089](https://github.com/mudler/vllm.cpp/issues/2089) | `SPEC-DFLASH2` | **The W11 draft-block route counters are blind to the production `P > 1` lane.** Both `NoteDflashBlockRoute` increments sit inside the `P == 1` branch (`qwen3_dflash.cpp:1487`, `:1502`); the materialized fallback at `:1888-1930` increments neither, so at every concurrency above one a route gate reads zero for both lanes while production runs a third route nothing names. #1890 put the counter inside the branch precisely so it would measure a capability rather than a class, and this is the hole that argument left. Listed under `## Owed` in [`specs/dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md) | test-gap |

## Resolution

-

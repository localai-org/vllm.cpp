ID: ISSUE-GH-632
Title: The record's own `path:line` citations were range-checked and never reported. `check-agent-record.py` parsed BOTH forms -- markdown links, and bare `file.cpp:123` through `RAW_LOCAL_ANCHOR_RE` since `ee511ca8a` -- but it dropped a failing anchor with `continue` and then answered with `any()`, so one good sibling covered the rest. There was no symbol test and no report. 832 of the 867 in-scope citations (96.0%) were already parsed, so what this row adds is the symbol test and the report rather than the parser. Every stale anchor found by hand during the 2026-08-13/14 campaign was in range. Closed by a device-leakage-shaped ratchet over `scripts/record-anchor-baseline.json`, not a bulk rewrite, spec [`record-anchor-ratchet.md`](../../specs/record-anchor-ratchet.md)
Row: ENG-RECORD-ANCHOR-RATCHET
State: UNKNOWN
Kind: bug
GitHub: 632
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:205`

### Frozen archive evidence

> | [#632](https://github.com/mudler/vllm.cpp/issues/632) | `ENG-RECORD-ANCHOR-RATCHET` | The record's own `path:line` citations were range-checked and never reported. `check-agent-record.py` parsed BOTH forms -- markdown links, and bare `file.cpp:123` through `RAW_LOCAL_ANCHOR_RE` since `ee511ca8a` -- but it dropped a failing anchor with `continue` and then answered with `any()`, so one good sibling covered the rest. There was no symbol test and no report. 832 of the 867 in-scope citations (96.0%) were already parsed, so what this row adds is the symbol test and the report rather than the parser. Every stale anchor found by hand during the 2026-08-13/14 campaign was in range. Closed by a device-leakage-shaped ratchet over `scripts/record-anchor-baseline.json`, not a bulk rewrite, spec [`record-anchor-ratchet.md`](../../specs/record-anchor-ratchet.md) | bug |

## Resolution

-

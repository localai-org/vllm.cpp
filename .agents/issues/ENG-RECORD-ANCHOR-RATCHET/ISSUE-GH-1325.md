ID: ISSUE-GH-1325
Title: `scripts/record-anchor-baseline.json` stores a `"total"` that no code reads. `load_record_anchor_baseline` in `scripts/check-agent-record.py` returns `{bucket: int(data["buckets"][bucket]) for bucket in RECORD_ANCHOR_BUCKETS}` with `RECORD_ANCHOR_BUCKETS = ("stale", "broken")`, and it is the file's only reader; `check_record_anchors` iterates those two buckets, and `write_record_anchor_baseline`'s refusal compares `result.total > sum(previous.values())` — the buckets, not the stored `total`. Measured at `af87251c5`: mutating `"total": 38` to `39` (a file whose total disagrees with `32 + 6`) leaves `check-agent-record.py` exit 0, tree restored byte-for-byte by sha256. So this row's own budget file carries the exact shape the row exists to name: a recorded figure no gate reads, sitting beside the figures that are read and presenting as if it were checked. `--write-baseline` compounds it by printing `-> 38`, which reads as the value it stored and is the one no later run consults. TWO candidate resolutions, deliberately not chosen here because choosing belongs to the fixing row: read it and assert `total == stale + broken` on load, or drop the field and derive it at read time, which is the shape AGENTS.md §Records prefers. Either is a semantic change to `check-agent-record.py` owing a spec and a red-before case in `tests/scripts/test_agent_record.py` `RecordAnchorRatchet`. Distinct from [#1287](https://github.com/mudler/vllm.cpp/issues/1287) and [#1270](https://github.com/mudler/vllm.cpp/issues/1270), neither of which reaches the unread field. Also under `## Owed` in [`record-anchor-ratchet.md`](../../specs/record-anchor-ratchet.md)
Row: ENG-RECORD-ANCHOR-RATCHET
State: UNKNOWN
Kind: bug
GitHub: 1325
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:434`

### Frozen archive evidence

> | [#1325](https://github.com/mudler/vllm.cpp/issues/1325) | `ENG-RECORD-ANCHOR-RATCHET` | `scripts/record-anchor-baseline.json` stores a `"total"` that no code reads. `load_record_anchor_baseline` in `scripts/check-agent-record.py` returns `{bucket: int(data["buckets"][bucket]) for bucket in RECORD_ANCHOR_BUCKETS}` with `RECORD_ANCHOR_BUCKETS = ("stale", "broken")`, and it is the file's only reader; `check_record_anchors` iterates those two buckets, and `write_record_anchor_baseline`'s refusal compares `result.total > sum(previous.values())` — the buckets, not the stored `total`. Measured at `af87251c5`: mutating `"total": 38` to `39` (a file whose total disagrees with `32 + 6`) leaves `check-agent-record.py` exit 0, tree restored byte-for-byte by sha256. So this row's own budget file carries the exact shape the row exists to name: a recorded figure no gate reads, sitting beside the figures that are read and presenting as if it were checked. `--write-baseline` compounds it by printing `-> 38`, which reads as the value it stored and is the one no later run consults. TWO candidate resolutions, deliberately not chosen here because choosing belongs to the fixing row: read it and assert `total == stale + broken` on load, or drop the field and derive it at read time, which is the shape AGENTS.md §Records prefers. Either is a semantic change to `check-agent-record.py` owing a spec and a red-before case in `tests/scripts/test_agent_record.py` `RecordAnchorRatchet`. Distinct from [#1287](https://github.com/mudler/vllm.cpp/issues/1287) and [#1270](https://github.com/mudler/vllm.cpp/issues/1270), neither of which reaches the unread field. Also under `## Owed` in [`record-anchor-ratchet.md`](../../specs/record-anchor-ratchet.md) | bug |

## Resolution

-

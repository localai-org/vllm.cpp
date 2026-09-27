ID: ISSUE-GH-1324
Title: This row's own figures in `docs/BENCHMARKS.md` and `docs/STATUS.md` are read by NO gate. Measured at `af87251c5`: restoring the wrong pre-[#851](https://github.com/mudler/vllm.cpp/pull/851) values (`rot 39`, `7 broken`, `828 OK`) into the BENCHMARKS cell leaves `check-public-doc-tables.py` exit 0 AND `check-agent-record.py` exit 0, while the same run derived `stale=32, broken=6` beside them; tree restored byte-for-byte by sha256. The ratchet reads only the `code` and `tests` cells of matrix rows, so `docs/` prose is outside `RECORD_ANCHOR_FIELDS` by construction, and `check-public-doc-tables.py` measures shape and never a value. THREE instances, none caught by a gate: rounds four and five of the [#851](https://github.com/mudler/vllm.cpp/pull/851) review, each caught by a human reading, and the landing itself, which published `over 844 OK` where its own merge commit `678fc672c` reads `ok=847` (three good citations landed on `main` between verification and merge). NOT fixed in flow: a repair is a semantic checker change owing a spec and red-before evidence, and it has a design decision first — assert the derived figures against the page, which makes `docs/` a new lock of the shape [#364](https://github.com/mudler/vllm.cpp/issues/364) argues against, or stop publishing derived values and leave the derivation in `--report`, which is what `record-anchor-baseline.json` already chose for itself in its `_comment`. What IS fixed in flow is the drift: the live `OK` count is dropped from all four surfaces that carried it. Closest sibling [#667](https://github.com/mudler/vllm.cpp/issues/667) (same class in `.agents/model-matrix.md`), distinct from [#911](https://github.com/mudler/vllm.cpp/issues/911) (spec bodies) and [#1287](https://github.com/mudler/vllm.cpp/issues/1287) (the ratchet's own false negatives). Also under `## Owed` in [`record-anchor-ratchet.md`](../../specs/record-anchor-ratchet.md)
Row: ENG-RECORD-ANCHOR-RATCHET
State: UNKNOWN
Kind: bug
GitHub: 1324
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:433`

### Frozen archive evidence

> | [#1324](https://github.com/mudler/vllm.cpp/issues/1324) | `ENG-RECORD-ANCHOR-RATCHET` | This row's own figures in `docs/BENCHMARKS.md` and `docs/STATUS.md` are read by NO gate. Measured at `af87251c5`: restoring the wrong pre-[#851](https://github.com/mudler/vllm.cpp/pull/851) values (`rot 39`, `7 broken`, `828 OK`) into the BENCHMARKS cell leaves `check-public-doc-tables.py` exit 0 AND `check-agent-record.py` exit 0, while the same run derived `stale=32, broken=6` beside them; tree restored byte-for-byte by sha256. The ratchet reads only the `code` and `tests` cells of matrix rows, so `docs/` prose is outside `RECORD_ANCHOR_FIELDS` by construction, and `check-public-doc-tables.py` measures shape and never a value. THREE instances, none caught by a gate: rounds four and five of the [#851](https://github.com/mudler/vllm.cpp/pull/851) review, each caught by a human reading, and the landing itself, which published `over 844 OK` where its own merge commit `678fc672c` reads `ok=847` (three good citations landed on `main` between verification and merge). NOT fixed in flow: a repair is a semantic checker change owing a spec and red-before evidence, and it has a design decision first — assert the derived figures against the page, which makes `docs/` a new lock of the shape [#364](https://github.com/mudler/vllm.cpp/issues/364) argues against, or stop publishing derived values and leave the derivation in `--report`, which is what `record-anchor-baseline.json` already chose for itself in its `_comment`. What IS fixed in flow is the drift: the live `OK` count is dropped from all four surfaces that carried it. Closest sibling [#667](https://github.com/mudler/vllm.cpp/issues/667) (same class in `.agents/model-matrix.md`), distinct from [#911](https://github.com/mudler/vllm.cpp/issues/911) (spec bodies) and [#1287](https://github.com/mudler/vllm.cpp/issues/1287) (the ratchet's own false negatives). Also under `## Owed` in [`record-anchor-ratchet.md`](../../specs/record-anchor-ratchet.md) | bug |

## Resolution

-

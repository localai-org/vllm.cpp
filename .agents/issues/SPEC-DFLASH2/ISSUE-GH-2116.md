ID: ISSUE-GH-2116
Title: **A speculator vetoes the async input and sampler path at `runner.cpp:470` (`:425` at the base tree the measurement below names), so every spec step drains the queue in step, while vLLM keeps async scheduling ON for dflash because `DFlashModelTypes` is inside `EagleModelTypes`.** Scoped by [`specs/dflash2-async-spec-sampler.md`](../../specs/dflash2-async-spec-sampler.md), which discharges `## Owed` A2 of [`specs/spec-decode-async-scheduling.md`](../../specs/spec-decode-async-scheduling.md). The veto was MEASURED load-bearing on the CPU tier rather than argued: deleting `!spec_config_.has_value()` at both construction sites reds `test_mtp_depth`'s W7 identity case (10 cases / 123 assertions / exit 0 becomes 9 passed / 1 failed / exit 1) through a production refusal at `runner.cpp:1833`, because `sample_tokens_async` carries no verify arm — no rejection sampler and no propose — a reason the veto's own comment did not name. The comment's stated reason holds too, and holds invisibly: under the same mutation the non-draft-aware combine overwrites the LAST DRAFT of every verify block with the previous step's committed token (`draft=[6 18]` becomes `draft=[6 5]` where the previous step emitted `5`, at every position), and the emitted tokens never move, so every identity assertion still passes. That is #1366's acceptance-only shape a second time. The row therefore stays vetoed and the fix is staged A2-1 through A2-5, with the draft-equality gate G2 owed by the first wave
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 2116
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:825`

### Frozen archive evidence

> | [#2116](https://github.com/mudler/vllm.cpp/issues/2116) | `SPEC-DFLASH2` | **A speculator vetoes the async input and sampler path at `runner.cpp:470` (`:425` at the base tree the measurement below names), so every spec step drains the queue in step, while vLLM keeps async scheduling ON for dflash because `DFlashModelTypes` is inside `EagleModelTypes`.** Scoped by [`specs/dflash2-async-spec-sampler.md`](../../specs/dflash2-async-spec-sampler.md), which discharges `## Owed` A2 of [`specs/spec-decode-async-scheduling.md`](../../specs/spec-decode-async-scheduling.md). The veto was MEASURED load-bearing on the CPU tier rather than argued: deleting `!spec_config_.has_value()` at both construction sites reds `test_mtp_depth`'s W7 identity case (10 cases / 123 assertions / exit 0 becomes 9 passed / 1 failed / exit 1) through a production refusal at `runner.cpp:1833`, because `sample_tokens_async` carries no verify arm — no rejection sampler and no propose — a reason the veto's own comment did not name. The comment's stated reason holds too, and holds invisibly: under the same mutation the non-draft-aware combine overwrites the LAST DRAFT of every verify block with the previous step's committed token (`draft=[6 18]` becomes `draft=[6 5]` where the previous step emitted `5`, at every position), and the emitted tokens never move, so every identity assertion still passes. That is #1366's acceptance-only shape a second time. The row therefore stays vetoed and the fix is staged A2-1 through A2-5, with the draft-equality gate G2 owed by the first wave | bug |

## Resolution

-

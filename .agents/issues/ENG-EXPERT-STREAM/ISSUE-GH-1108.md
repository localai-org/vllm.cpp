ID: ISSUE-GH-1108
Title: Three of the four `Qwen35ExpertStreamStep` guards [#1091](https://github.com/mudler/vllm.cpp/issues/1091) finding 3 added land UNREACHED. Only `Qwen3_5MTPModel::ForwardPaged` is reachable from a production entry point (`src/vllm/v1/worker/gpu/runner.cpp:2183` -> `src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp:107,262`). `Qwen3_5MTPModel::Forward` is reached only through `ForwardLogitsHost`, which `qwen3_5_mtp.h:135` documents as "standalone parity convenience" and which itself has no caller outside `tests/`; `Qwen3_5Model::ForwardDense` is the parity reference by `qwen3_5.h:234` (callers `tests/parity/test_op_parity.cpp:1107`, `tests/vllm/v1/worker/test_runner.cpp:1278`, `tests/vllm/models/test_qwen35_paged_forward.cpp:293,320,385,403`); `Qwen3_5ReplayLayer` is per-layer parity replay by `qwen3_5.h:322` (only caller `tests/parity/test_op_parity.cpp:1050`). Per `.agents/reachability.md` a call site inside a test is not reach, so "every added path is reached from a production entry point at this commit" was true of one guard in four. NOTHING IS DELETED: the guards are correct where they sit, cost nothing, and become live the moment any of those entry points gains a production caller — and adding the guard later WITH the caller is precisely how this row lost its step boundary in the first place. What is owed is the record, so this is named as a staged slice that lands unreached rather than claimed as reached. Closes when one of those entry points gains a production caller, or when they are retired as parity references; neither is scheduled. Split out of [#1106](https://github.com/mudler/vllm.cpp/issues/1106) finding 2 so the debt stays open after that repair closes. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md)
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: bug
GitHub: 1108
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:319`

### Frozen archive evidence

> | [#1108](https://github.com/mudler/vllm.cpp/issues/1108) | `ENG-EXPERT-STREAM` | Three of the four `Qwen35ExpertStreamStep` guards [#1091](https://github.com/mudler/vllm.cpp/issues/1091) finding 3 added land UNREACHED. Only `Qwen3_5MTPModel::ForwardPaged` is reachable from a production entry point (`src/vllm/v1/worker/gpu/runner.cpp:2183` -> `src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp:107,262`). `Qwen3_5MTPModel::Forward` is reached only through `ForwardLogitsHost`, which `qwen3_5_mtp.h:135` documents as "standalone parity convenience" and which itself has no caller outside `tests/`; `Qwen3_5Model::ForwardDense` is the parity reference by `qwen3_5.h:234` (callers `tests/parity/test_op_parity.cpp:1107`, `tests/vllm/v1/worker/test_runner.cpp:1278`, `tests/vllm/models/test_qwen35_paged_forward.cpp:293,320,385,403`); `Qwen3_5ReplayLayer` is per-layer parity replay by `qwen3_5.h:322` (only caller `tests/parity/test_op_parity.cpp:1050`). Per `.agents/reachability.md` a call site inside a test is not reach, so "every added path is reached from a production entry point at this commit" was true of one guard in four. NOTHING IS DELETED: the guards are correct where they sit, cost nothing, and become live the moment any of those entry points gains a production caller — and adding the guard later WITH the caller is precisely how this row lost its step boundary in the first place. What is owed is the record, so this is named as a staged slice that lands unreached rather than claimed as reached. Closes when one of those entry points gains a production caller, or when they are retired as parity references; neither is scheduled. Split out of [#1106](https://github.com/mudler/vllm.cpp/issues/1106) finding 2 so the debt stays open after that repair closes. Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md) | bug |

## Resolution

-

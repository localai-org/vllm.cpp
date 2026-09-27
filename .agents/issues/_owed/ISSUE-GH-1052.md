ID: ISSUE-GH-1052
Title: `tests/vllm/v1/test_engine_core_proc.cpp:481` ("EngineCoreProc: immediate shutdown aborts in-flight requests") searches for the abort frame over a FIXED budget of 1000 dequeues while a `max_tokens=100000` request keeps the busy loop producing token deltas, so nothing bounds how many frames precede the abort and the budget is a bet on scheduling. MEASURED at `37e680cab`, same binary throughout, CPU-only Release on 20 cores: **2 failures in 3 `ctest -j4` runs** of the full 492-test suite (`CHECK( abort_seen ) is NOT correct!`), **0 in 25 solo runs** on an idle box at load 3.34, **0 in 25 solo runs against 20 spinning processes**, and 0 in two `ctest -R '^test_engine_core_proc$'` runs (`Passed 0.03 sec`). So CPU pressure alone does not reproduce it; it needs the `-j4` harness. The third `-j4` run failed `test_cpu_threadpool` INSTEAD, which is on the same load-dependent list, so the IDENTITY of the failing test rotates between runs of an unchanged binary and both pass alone with exit 0. NO ISSUE NAMED THIS TEST: PR [#1032](https://github.com/mudler/vllm.cpp/pull/1032)'s body attributed its flake to [#294](https://github.com/mudler/vllm.cpp/issues/294), which is a different defect in a different test (`test_async_llm` reusing an aborted request id), and a misattributed flake is worse than an untracked one because the next reader checks the citation, finds an open issue about something else, and stops looking. The assertion guards a real guarantee (an in-flight request gets a `kAbort` finish on immediate shutdown); the 1000-frame budget is the part that is a guess. Found repairing the fresh review of [#1039](https://github.com/mudler/vllm.cpp/issues/1039), on a branch that touches no file under `tests/vllm/v1/` or `src/vllm/v1/`. Listed under `## Owed` in [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1052
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:307`

### Frozen archive evidence

> | [#1052](https://github.com/mudler/vllm.cpp/issues/1052) | — | `tests/vllm/v1/test_engine_core_proc.cpp:481` ("EngineCoreProc: immediate shutdown aborts in-flight requests") searches for the abort frame over a FIXED budget of 1000 dequeues while a `max_tokens=100000` request keeps the busy loop producing token deltas, so nothing bounds how many frames precede the abort and the budget is a bet on scheduling. MEASURED at `37e680cab`, same binary throughout, CPU-only Release on 20 cores: **2 failures in 3 `ctest -j4` runs** of the full 492-test suite (`CHECK( abort_seen ) is NOT correct!`), **0 in 25 solo runs** on an idle box at load 3.34, **0 in 25 solo runs against 20 spinning processes**, and 0 in two `ctest -R '^test_engine_core_proc$'` runs (`Passed 0.03 sec`). So CPU pressure alone does not reproduce it; it needs the `-j4` harness. The third `-j4` run failed `test_cpu_threadpool` INSTEAD, which is on the same load-dependent list, so the IDENTITY of the failing test rotates between runs of an unchanged binary and both pass alone with exit 0. NO ISSUE NAMED THIS TEST: PR [#1032](https://github.com/mudler/vllm.cpp/pull/1032)'s body attributed its flake to [#294](https://github.com/mudler/vllm.cpp/issues/294), which is a different defect in a different test (`test_async_llm` reusing an aborted request id), and a misattributed flake is worse than an untracked one because the next reader checks the citation, finds an open issue about something else, and stops looking. The assertion guards a real guarantee (an in-flight request gets a `kAbort` finish on immediate shutdown); the 1000-frame budget is the part that is a guess. Found repairing the fresh review of [#1039](https://github.com/mudler/vllm.cpp/issues/1039), on a branch that touches no file under `tests/vllm/v1/` or `src/vllm/v1/`. Listed under `## Owed` in [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md) | bug |

## Resolution

-

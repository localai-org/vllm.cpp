ID: ISSUE-GH-1390
Title: `test_qwen3_5_decode_graph_seam` SIGSEGVs on `main` and every assertion passes. Measured at `5f68e60df`, which is `origin/main` exactly, CPU Release x86_64: 8 cases, 7 passed, 1 failed, an assertion line reading 135 of 135 passed and 0 failed, exit 139, with `W6: two spec shapes of EQUAL S and different q get two graphs` reporting `CRASHED: SIGSEGV`. The number a reader greps says 135/135, so only the exit status and the `CRASHED` line carry the verdict. ORDER-DEPENDENT: `-tc="W6*"` alone passes 9/9 exit 0, so the crash needs state an earlier case in the same process left behind — a doctest binary runs every case in one process, and a leaked pool block, a leaked backend or platform registration, or a driver slot captured under one shape and re-entered under another are all live candidates. `gdb -batch -ex run -ex bt` puts the fault inside `vt::cpu::PagedAttentionKernel` on a `vt::cpu::Threadpool` worker, which is what a block table, slot mapping or sequence length that does not describe the handed KV cache looks like. The crashing case is the one [#1374](https://github.com/mudler/vllm.cpp/issues/1374) added for the `(S, q, spec)` ring key. Found while landing [#1305](https://github.com/mudler/vllm.cpp/issues/1305) and NOT caused by it: reverse-applying that branch's whole source change and rebuilding leaves the same exit 139, and that change touches `qwen3_moe.cpp`, `deepseek_v2.cpp` and three registry translation units, none of which this binary's crashing case executes. NOT fixed in flow, because `AGENTS.md` routes a surprising fix to the normal row, spec and fresh-review path: it is a segmentation fault in another stage's newly landed code, its mechanism is an unlocated cross-case state leak, and `src/vllm/model_executor/models/qwen3_5.cpp` is under concurrent edit for [#1380](https://github.com/mudler/vllm.cpp/issues/1380). Owner: row `ENG-CUDAGRAPH-BREAK`, under `## Owed` in [eng-cudagraph-break.md](../../specs/eng-cudagraph-break.md)
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: bug
GitHub: 1390
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:471`

### Frozen archive evidence

> | [#1390](https://github.com/mudler/vllm.cpp/issues/1390) | `ENG-CUDAGRAPH-BREAK` | `test_qwen3_5_decode_graph_seam` SIGSEGVs on `main` and every assertion passes. Measured at `5f68e60df`, which is `origin/main` exactly, CPU Release x86_64: 8 cases, 7 passed, 1 failed, an assertion line reading 135 of 135 passed and 0 failed, exit 139, with `W6: two spec shapes of EQUAL S and different q get two graphs` reporting `CRASHED: SIGSEGV`. The number a reader greps says 135/135, so only the exit status and the `CRASHED` line carry the verdict. ORDER-DEPENDENT: `-tc="W6*"` alone passes 9/9 exit 0, so the crash needs state an earlier case in the same process left behind — a doctest binary runs every case in one process, and a leaked pool block, a leaked backend or platform registration, or a driver slot captured under one shape and re-entered under another are all live candidates. `gdb -batch -ex run -ex bt` puts the fault inside `vt::cpu::PagedAttentionKernel` on a `vt::cpu::Threadpool` worker, which is what a block table, slot mapping or sequence length that does not describe the handed KV cache looks like. The crashing case is the one [#1374](https://github.com/mudler/vllm.cpp/issues/1374) added for the `(S, q, spec)` ring key. Found while landing [#1305](https://github.com/mudler/vllm.cpp/issues/1305) and NOT caused by it: reverse-applying that branch's whole source change and rebuilding leaves the same exit 139, and that change touches `qwen3_moe.cpp`, `deepseek_v2.cpp` and three registry translation units, none of which this binary's crashing case executes. NOT fixed in flow, because `AGENTS.md` routes a surprising fix to the normal row, spec and fresh-review path: it is a segmentation fault in another stage's newly landed code, its mechanism is an unlocated cross-case state leak, and `src/vllm/model_executor/models/qwen3_5.cpp` is under concurrent edit for [#1380](https://github.com/mudler/vllm.cpp/issues/1380). Owner: row `ENG-CUDAGRAPH-BREAK`, under `## Owed` in [eng-cudagraph-break.md](../../specs/eng-cudagraph-break.md) | bug |

## Resolution

-

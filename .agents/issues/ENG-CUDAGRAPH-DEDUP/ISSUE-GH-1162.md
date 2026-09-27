ID: ISSUE-GH-1162
Title: We instantiate **one `cudaGraphExec` per padded decode bucket, per model**, and `grep -rn "cudaGraphExecUpdate" src include` returns nothing. `src/vt/cuda/cuda_backend.cu:222-232` instantiates a fresh exec per capture; `include/vllm/model_executor/models/decode_graph_sizes.h:32-41` yields 7 buckets at `max_num_seqs=32` and 11 at 64; eight drivers each build their own set. SGLang folds compatible captures onto one executable by hashing graph topology and calling `cudaGraphExecUpdate` on a signature hit (`cuda_graph_dedup_mixin.py:219-242`, logging "captured %d CUDA graphs, deduped to %d execs" at `:358`). Portable to us unchanged, because it is driver-level rather than PyTorch-level. It is a **memory and capture-time** change, NOT a throughput change — a deduped replay launches the same nodes — and it matters because on GB10 unified memory an OOM reboots the box, capture time is startup latency (a recorded gate axis), and bucket count is exactly what widening graph coverage would raise. Owed: the dedup registry behind the `vt` seam, a capture-count/exec-count log line, and a same-binary A/B proving a deduped replay is byte-identical rather than asserting it. Hazard already recorded: capture bakes host source addresses and a clean `compute-sanitizer` run is NOT evidence a capture path is safe (`specs/decode-graph-scratch-uaf-2026-07-18.md`). Spec [`sglang-breakable-cuda-graph.md`](../../specs/sglang-breakable-cuda-graph.md) `## Owed`. Analysis: [#1161](https://github.com/mudler/vllm.cpp/issues/1161)
Row: ENG-CUDAGRAPH-DEDUP
State: UNKNOWN
Kind: perf
GitHub: 1162
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:362`

### Frozen archive evidence

> | [#1162](https://github.com/mudler/vllm.cpp/issues/1162) | `ENG-CUDAGRAPH-DEDUP` | We instantiate **one `cudaGraphExec` per padded decode bucket, per model**, and `grep -rn "cudaGraphExecUpdate" src include` returns nothing. `src/vt/cuda/cuda_backend.cu:222-232` instantiates a fresh exec per capture; `include/vllm/model_executor/models/decode_graph_sizes.h:32-41` yields 7 buckets at `max_num_seqs=32` and 11 at 64; eight drivers each build their own set. SGLang folds compatible captures onto one executable by hashing graph topology and calling `cudaGraphExecUpdate` on a signature hit (`cuda_graph_dedup_mixin.py:219-242`, logging "captured %d CUDA graphs, deduped to %d execs" at `:358`). Portable to us unchanged, because it is driver-level rather than PyTorch-level. It is a **memory and capture-time** change, NOT a throughput change — a deduped replay launches the same nodes — and it matters because on GB10 unified memory an OOM reboots the box, capture time is startup latency (a recorded gate axis), and bucket count is exactly what widening graph coverage would raise. Owed: the dedup registry behind the `vt` seam, a capture-count/exec-count log line, and a same-binary A/B proving a deduped replay is byte-identical rather than asserting it. Hazard already recorded: capture bakes host source addresses and a clean `compute-sanitizer` run is NOT evidence a capture path is safe (`specs/decode-graph-scratch-uaf-2026-07-18.md`). Spec [`sglang-breakable-cuda-graph.md`](../../specs/sglang-breakable-cuda-graph.md) `## Owed`. Analysis: [#1161](https://github.com/mudler/vllm.cpp/issues/1161) | perf |

## Resolution

-

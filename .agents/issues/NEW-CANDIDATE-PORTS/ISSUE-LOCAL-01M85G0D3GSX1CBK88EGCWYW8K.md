ID: ISSUE-LOCAL-01M85G0D3GSX1CBK88EGCWYW8K
Title: olala CPU decode: bounded-thread-pool row-parallel host GEMVs (5.4x gap lever 1+2)
Row: NEW-CANDIDATE-PORTS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-24
Updated: 2026-11-24
Closed: 2026-11-24

## Problem

The published p150-olala-bf16 CPU baseline decodes at 0.3990 tok/s vs the llama.cpp fork's 2.158 (5.41x gap); the olala host ops are documented single-threaded scalar code on a 128-core host. Profile (perf, decode): LayerTailHost 30% (MoE 19.7%), MambaStepHost 29%, VAttentionStepHost 6.9%, LmHeadHost 5%. Every hot op is a set of whole-row serial double dots, so output-row splitting across a bounded blocking worker pool is bit-identical. Fix: add HostRowPool (blocking condvar workers, no busy-wait, OLALA_HOST_THREADS override, default 96 measured) and split output rows of the profiled GEMV sites; measured decode 0.3990->1.8053 tok/s, prefill 0.4579->1.9247 tok/s, greedy stream bit-identical, test_olala_paged_engine 18/18 EXACT, test_olala_ops 305/305.

## Resolution

2026-11-24: landed in two commits on row/NEW-CANDIDATE-PORTS-olala-bench —
`d02eff903` (bounded 96-thread blocking worker pool + output-row-parallel
GEMVs at the profiled sites; decode 0.3990 -> 1.8053 tok/s, prefill 0.4579 ->
1.9247) and `9a720e64e` (profiled waste removal: dead prefill placeholder
dot, hoisted B/C rms and mimo coefficient tables, row-parallel prefill
attention; decode -> 1.8908, prefill -> 2.0534; cumulative 4.74x/4.49x vs the
published baseline). The 64-id greedy stream is byte-identical at every
stage; test_olala_paged_engine 18/18 EXACT and test_olala_ops 305/305 in
Release AND bounds-checked Debug. A barrier-fusion experiment measured 0.95
tok/s (locality loss) and was reverted. Evidence:
docs/bench-evidence/olala-cpu-opt-20261123/; published page
docs/benchmarks/p150-olala-bf16.md moved to the new measured values. The
remaining 1.14x decode / 1.27x prefill gap to the fork stays open as a
traceable hypothesis (serial double-FADD row chains), not a ceiling.


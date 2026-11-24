ID: ISSUE-LOCAL-01M85G0D3GSX1CBK88EGCWYW8K
Title: olala CPU decode: bounded-thread-pool row-parallel host GEMVs (5.4x gap lever 1+2)
Row: NEW-CANDIDATE-PORTS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-24
Updated: 2026-11-24
Closed: -

## Problem

The published p150-olala-bf16 CPU baseline decodes at 0.3990 tok/s vs the llama.cpp fork's 2.158 (5.41x gap); the olala host ops are documented single-threaded scalar code on a 128-core host. Profile (perf, decode): LayerTailHost 30% (MoE 19.7%), MambaStepHost 29%, VAttentionStepHost 6.9%, LmHeadHost 5%. Every hot op is a set of whole-row serial double dots, so output-row splitting across a bounded blocking worker pool is bit-identical. Fix: add HostRowPool (blocking condvar workers, no busy-wait, OLALA_HOST_THREADS override, default 96 measured) and split output rows of the profiled GEMV sites; measured decode 0.3990->1.8053 tok/s, prefill 0.4579->1.9247 tok/s, greedy stream bit-identical, test_olala_paged_engine 18/18 EXACT, test_olala_ops 305/305.

## Resolution

-

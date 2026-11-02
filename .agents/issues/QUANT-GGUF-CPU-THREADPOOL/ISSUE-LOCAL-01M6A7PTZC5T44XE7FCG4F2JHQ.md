ID: ISSUE-LOCAL-01M6A7PTZC5T44XE7FCG4F2JHQ
Title: CPU decode: paged attention runs its per-head loop inline at T=1
Row: QUANT-GGUF-CPU-THREADPOOL
State: CLOSED
Kind: performance
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: 2026-11-01

## Problem

At decode the paged-attention kernel splits work over query tokens only, so T=1 hands ParallelForRows one row and the caller runs every q head of every key single-threaded while the pool workers spin. The W5 SIGPROF and VT_POOL_GAP profile measured 1,596 calls totalling 0.303 s of a 6.7-7.0 s 32-token greedy request (190 us per call) in the inline body, which is 5.5 percent of the current 5.495 s wall; the per-head output elements are independent, so the same output-range split the W5 kernels use applies once the per-token scratch is rescoped per head.

## Resolution

2026-11-01. `PagedAttentionKernel` now hands `ParallelForRows` the flat
`total_q * hq` range and maps a flat index `i` to `t = i / hq` and
`h = i % hq`, so a decode call divides its eight q heads across the pool
instead of running them in the inline caller. The softmax scratch and the
widened query row moved into the pair body; `qrow` is widened once per token
and cached on the last token a chunk visited, and `diff -w` shows the pair body
is the previous head-loop iteration. Same-binary interleaved A/B, seven pairs,
two runs per arm, both arms linked from the same deployment archive and
differing only in `cpu_paged_attn.cpp.o`: 5.078 s to 4.830 s wall (1.051x) and
91.28 s to 86.90 s CPU, six of seven pairs favouring the after arm and all 14
greedy runs byte-identical; 402-token prefill 0.306/0.304 s to 0.246/0.243 s,
so the finer chunking does not regress the other shape. A fresh reviewer
mutated the head: the extent mutant fails 69 / 171 / 6, the shifted store 1586
/ 171 with the threadpool battery green because the battery cannot see a
deterministic mis-map, the `qrow` cache 98 / 16 / 0 at 1 / 3 / 20 threads, a
deleted `probs` sizing dies in SIGSEGV, and a sized-but-unwritten `probs` is
green because pass 1 writes every element pass 2 reads. Three prose findings
from that review were corrected in `0903be579` and a second fresh reviewer
returned PASS. Gate: `test_ops_paged_attn` 15 / 1838,
`test_ops_paged_attn_dtype` 3 / 172, `test_cpu_threadpool` 9 / 19650 (19651
with the pool size set) at unset, 1, 3 and 20 threads. The deployed binary
(sha256 `9661f2cc`, W5 backup `0f8d6cff`) answers `/health` with 200 at 6.54 GB
RSS, and two `verify.py --expect-cache on` captures match the W5 `verify-final`
capture field for field outside `id` and `created`; the 32-of-115 and 96-of-115
hit deltas are the same pair W5 recorded in the opposite order, so they are
cache warmth. Evidence: `evidence/gsq-lumen-w6-20261101/` in the serving
deployment. `kRmsNorm` is now
the largest inline site at 0.081 s per 32-token request and stays a separate
gap, and the served artifact still cannot be built from this branch
(ISSUE-LOCAL-01M6A8SZSN03ZNFY8Y7HB6M7TZ).

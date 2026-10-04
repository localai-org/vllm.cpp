ID: ISSUE-LOCAL-01M3918K0WTSCZ2X1S2WZA5NHW
Title: 27B decode exhausts tt-metal DRAM after ~N requests per process (bank_manager OOM in the keepquant decode path)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-30
Closed: -

## Problem

UPSTREAM FILED 2026-09-25: tenstorrent/tt-metal#57970 (ttnn deferred-reader retention, the falsification-proven holder). The package at docs/bench-evidence/tt-ttm-retention-repro-20260925/ is the evidence; the weekly-refresh task tracks the upstream state and this issue closes when the upstream fix lands and our pin advances past it.

Root-cause row: [tenstorrent-ttm-retention-rootcause](../../specs/tenstorrent-ttm-retention-rootcause.md) (2026-09-25).

Spec: [tenstorrent-27b-dram-growth](../../specs/tenstorrent-27b-dram-growth.md) (attribution-first, 2026-09-24).
UPDATE 2026-09-25 (root-cause row): the TT-LEDGER allocator instrument (temporary, TT_LEDGER-gated, on the pinned tree's build_instr) NAMES the structure: one (16,777,216 B + 3,932,160 B) pair of `ttnn::to_layout` tilize outputs leaked per GDN layer (48 layers) per request, on each request's FIRST decode step, in the GDN state-scatter web (GdnStateScatterKernel -> ScatterRowsExact/ScatterRowsDevice, tenstorrent_gdn.cpp:1004-1095/:1501-1540) — 48x(16+3.75) MiB = 948 MiB/request, the ledger's -949.3 MB/request exactly. The buffers are the web's unregistered transients (the not-resident rows upload, :1039-1041/:1054-1064, and the re-tilized scatter returns); the program cache is EXONERATED (request 3 leaks its full step-0 set at ZERO PC-INSERTs; the leg is fully eager — no capture cycles exist). The our-side workaround (force-reclaiming the scatter's dead transients after the commit) measured -964.8 → -16.5 MiB/request on the settled ledger but was FALSIFIED BY GATE 2 (the 4-prompt anchor leg shifted request 2's tokens — token 7: 16→17 vs the before-anchor) and REVERTED: the reference that blocks the scope-exit free is a LIVE DEFERRED READER of the transients, so the retention's mechanism is ttnn-internal (the Tensor::from_vector -> to_device -> to_layout(TILE) chain must drain it or own the tensors) and the fix is upstream's. Evidence: docs/bench-evidence/tt-ttm-retention-rootcause-20260925.md + the upstream package docs/bench-evidence/tt-ttm-retention-repro-20260925/ (the minimal reproducer — a negative control: the pure web does NOT leak outside vllm.cpp — plus the instrument patch and the boundary analysis).
UPDATE 2026-09-25: the first fix attempt (request-invariant repair planes, spec tenstorrent-keepquant-retention.md) produced an IDENTICAL staircase and a decisive control — the same P=64 prefill ran twice with equal retention both times (-0.195 GiB/bank each). Retention is per-PREFILL-EVENT, not per-new-shape: the program-cache pins-per-new-shape attribution is REFUTED. The owner is tt-metal-side per-execution retention (allocator/program-cache workspace under VT_TT_PROGRAM_CACHE=1), escalated per the spec stop condition; the fix attempt was reverted clean.

The APEX-I-Nano 27B decode path OOMs on tt-metal at bank_manager.cpp:495 (~268 MB DRAM allocation against ~63 MB largest free block) in Qwen3_5DenseDecodeGraph::Step -> DecodeKeepQuantWordsF32 -> ttnn::where. One 64-request process dies after ~60 requests; 16-request after ~14; 8-request after ~7; single-request processes complete. The growth-with-requests pattern indicates a leak or unbounded allocation accumulation on the decode path. This blocks the INT8DOT wide band sweep (gate 1), the 27B Q4KM sibling gate, and by extension the 27B trace-capture row. Evidence: docs/bench-evidence/tt-int8dot-flip-gates-20260923.md on row/TT-KEEPQUANT-INT8DOT-DEFAULT (2026-09-23/24, P150).

## Resolution

-

UPDATE 2026-09-30 (engine-side re-check on the NEW pin vllm-cpp-pin/20260923-adv @ 6449cf13f7b, eager arm VT_TT_DECODE_CAPTURE=0, 27B Q4_K_M, 8 prompts in 32, c=4, VT_TT_ALLOC_TRACE ledger): the 948 MiB/request staircase is REFUTED on this pin. Settled per-forward free DRAM (block/0/pre boundary) is EXACTLY flat at 301,930,048 B/bank for 26 consecutive forwards across wave 1 and 300,485,696 B/bank for 23 consecutive forwards across wave 2; the per-new-request step is ~2.9 MiB/request, not 948 MiB. No bank_manager OOM, no request died; both request joins fully observed (the 9000 s wrapper timeout cut only the final decode steps of requests 7/8). One of the ~576 commits between the old pin and this one drained the deferred-reader retention. Evidence: docs/bench-evidence/tt-retention-newpin-20260930.md. Upstream reply on tt-metal#57970 held pending this result; nothing posted.

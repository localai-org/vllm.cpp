ID: ISSUE-LOCAL-01M433M0TNT8FWC6SMT4R3700W
Title: TT GDN decode: state-slot churn across c2 sequence re-admission kills the batched decode graph (engine-fatal tenstorrent_gdn.cpp:558)
Row: BACKEND-TENSTORRENT-KEEPQUANT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-06
Closed: 2026-10-06

## Problem

During the INT8DOT flip 64-prompt band sweep (c2, --num-prompts 64, Qwen3.8-27B Q4_K_M, row/int8dot-default-flip @ 6fbe576dc), the INT8DOT=1 leg died mid-run with an engine-fatal: 'vt: tenstorrent gdn_decode: state-slot indices changed during trace capture — decode slots must be stable across a sequence's steps; the recapture cadence owns a slot change (reset the graph, then re-warm on the eager boundary step)' at src/vt/tenstorrent/tenstorrent_gdn.cpp:558 (log /tmp/int8dot-flip/dot1-c2-64p.log, BENCH_EXIT=1, 2026-10-04). Every earlier c2 gate leg ran only --num-prompts 2 (one wave, no re-admission), so the slot-reassignment path across retirements was never exercised. Any multi-wave c2 workload hits it. This blocks multi-prompt batched TT decode gates at c2.

## Resolution

Fixed 2026-10-06 (row/gdn-slot-churn @ 76aef007c): the captured decode graph bakes the slot bindings through the content-keyed GdnSsmIdxEntry/GdnConvIdxEntry device tensors, so a replay after a membership change re-reads the capture-time slots (the c2 run-to-run churn) and a capture after a change aborts at :558. Both Qwen3.5 decode-graph drivers (dense + MoE) now record the GDN non-spec state-index content per SizeSlot and route a content change through the existing reset lane: destroy the graph, run the boundary step eagerly (re-warming the idx caches with the new content, capture inactive), re-capture on the next step. Red: the checkpoint-gated 0.8B multi-wave churn gate (tests/parity/test_qwen35_paged_engine.cpp, 549f652d0) aborts on the pre-fix driver; green: all requests finish and two independent runs produce byte-identical streams. 27B c-concurrency-4 8-prompt money leg: docs/bench-evidence/tt-gdn-slot-churn-27b-20261006.md.

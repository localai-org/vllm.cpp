ID: ISSUE-LOCAL-01M433M0TNT8FWC6SMT4R3700W
Title: TT GDN decode: state-slot churn across c2 sequence re-admission kills the batched decode graph (engine-fatal tenstorrent_gdn.cpp:558)
Row: BACKEND-TENSTORRENT-KEEPQUANT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

During the INT8DOT flip 64-prompt band sweep (c2, --num-prompts 64, Qwen3.8-27B Q4_K_M, row/int8dot-default-flip @ 6fbe576dc), the INT8DOT=1 leg died mid-run with an engine-fatal: 'vt: tenstorrent gdn_decode: state-slot indices changed during trace capture — decode slots must be stable across a sequence's steps; the recapture cadence owns a slot change (reset the graph, then re-warm on the eager boundary step)' at src/vt/tenstorrent/tenstorrent_gdn.cpp:558 (log /tmp/int8dot-flip/dot1-c2-64p.log, BENCH_EXIT=1, 2026-10-04). Every earlier c2 gate leg ran only --num-prompts 2 (one wave, no re-admission), so the slot-reassignment path across retirements was never exercised. Any multi-wave c2 workload hits it. This blocks multi-prompt batched TT decode gates at c2.

## Resolution

-

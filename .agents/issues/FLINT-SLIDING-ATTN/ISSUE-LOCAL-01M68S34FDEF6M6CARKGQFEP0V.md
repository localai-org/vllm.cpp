ID: ISSUE-LOCAL-01M68S34FDEF6M6CARKGQFEP0V
Title: Sliding attention is 37% of the Flint serving step and grows 1.94x with sequence length despite the bounded window
Row: FLINT-SLIDING-ATTN
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: -

## Problem

TT-NONROUTED-ATTRIBUTION (landed lineage) measured attention-sliding at 37% of the keep-quant serving step, growing 46.7 -> 90.8 ms/step (1.94x) from seq ~16 to seq ~207 while the routed phase stays flat (0.994) -- confirmed independently by implementer and reviewer. The kernel's window math is bounded (cpu_paged_attn.cpp:337-343), so the growth is not the window. Hypotheses: H1 KV-append/table maintenance inside the timer scope; H2 arena-scatter cache misses on the windowed gather; H3 batch-1 serial attention (ParallelForRows splits by token rows; concurrency 1 = one row = one thread for 32 heads x 513 keys while the pool idles). Diagnose stage attributes the growth with per-phase + thread-occupancy evidence BEFORE any fix; fixes adopt through the shared kernel with all token gates green (test_flint_paged 208/208, dtype bit-identity, keepquant serving gate; MiMoV2 sink gate mandatory on kernel-region changes). Spec: .agents/specs/flint-sliding-attn.md

## Resolution

-

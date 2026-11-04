ID: ISSUE-LOCAL-01M6KSG9891SWMZYV0V3WE3AXS
Title: The CPU descent continues: the attribution table's remaining levers post-NEON-dot
Row: FLINT-CPU-DESCENT-2
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-04
Updated: 2026-11-04
Closed: -

## Problem

The keep-quant paged arm is at 7.01-7.38 tok/s @16t (from 1.20; pool match + sliding decomposition + NEON dot landed). The attribution table names the remaining levers in value order: (1) router host top-k (2.6% @16t, doubling at 32t -- serial partial_sort; parallel/radix-select/device candidate); (2) MoE shared expert + combine (7.4% -- same weights every token, a batching/caching candidate); (3) lm_head (3.1% -- check the weight class and what kernel serves it; the NEON dot may not apply directly). The routed phase keeps the grouped-GEMM follow-up as a separate row if still dominant post-NEON. Spec: .agents/specs/flint-cpu-descent-2.md. Measurement-driven: re-run the attribution table with the NEON dot landed (shares moved), adopt the top lever through the shared seam, repeat until the levers exhaust or the residual is attributed elsewhere. Serving gate + dtype bit-identity + MiMoV2 sinks on every change.

## Resolution

-

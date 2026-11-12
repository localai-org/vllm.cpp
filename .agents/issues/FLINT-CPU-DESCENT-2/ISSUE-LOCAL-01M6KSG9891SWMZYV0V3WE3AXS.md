ID: ISSUE-LOCAL-01M6KSG9891SWMZYV0V3WE3AXS
Title: The CPU descent continues: the attribution table's remaining levers post-NEON-dot
Row: FLINT-CPU-DESCENT-2
State: CLOSED
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-04
Updated: 2026-11-04
Closed: 2026-11-04

## Problem

The keep-quant paged arm is at 7.01-7.38 tok/s @16t (from 1.20; pool match + sliding decomposition + NEON dot landed). The attribution table names the remaining levers in value order: (1) router host top-k (2.6% @16t, doubling at 32t -- serial partial_sort; parallel/radix-select/device candidate); (2) MoE shared expert + combine (7.4% -- same weights every token, a batching/caching candidate); (3) lm_head (3.1% -- check the weight class and what kernel serves it; the NEON dot may not apply directly). The routed phase keeps the grouped-GEMM follow-up as a separate row if still dominant post-NEON. Spec: .agents/specs/flint-cpu-descent-2.md. Measurement-driven: re-run the attribution table with the NEON dot landed (shares moved), adopt the top lever through the shared seam, repeat until the levers exhaust or the residual is attributed elsewhere. Serving gate + dtype bit-identity + MiMoV2 sinks on every change.

## Resolution

2026-11-04: landed on row/FLINT-CPU-DESCENT-2 c5e147b9f. Stage 1 re-attribution table recorded (docs/bench-evidence/flint-cpu-descent-2-20261101/README.md): the shared+combine bucket is 97% shared-expert bf16 GEMVs (combine 0.2%), a bf16-dot tier owed to a separate row. Lever LANDED: NEON vec_dot_q6_K_q8_K for the Q6_K lm_head (VT_CPU_Q6K_DOT=auto), bit-identity 6370/6370, keep-quant gate UNMOVED 53/60 worst 2.48648, in-situ lm_head -21% @16t, serving A/B +6.3% @16t +1.7% @32t on a contended box. Lever STOPPED: router top-k selection — partial_sort tie order unspecified, cannot prove bit-identity (test_flint_router_topk_equiv records the falsification); only the workspace hoist landed. Residual routed-expert loop 32-37% keeps its grouped-GEMM row.

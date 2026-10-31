ID: ISSUE-LOCAL-01M686KWK9PXFK67593AZ7ZP0Z
Title: Attribute the ~88% of Flint keep-quant serving decode time outside the routed-expert phase
Row: TT-NONROUTED-ATTRIBUTION
State: CLOSED
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: 2026-10-31

## Problem

FLINT-KQ-CPU-PERF (ea530a5c7 lineage) matched the threadpool and took the keep-quant paged arm to 5.51 tok/s @16t -- but its evidence records only ~12% of serving decode is the routed-expert phase. The other ~88% (attention, dense, norms, lm_head, sampling, orchestration) is unattributed, so no lever can be ranked. Spec: .agents/specs/tt-nonrouted-attribution.md. Deliverable: one wall-clock-verifiable phase table at 16t and 32t summing to ~100%; adoption of any discovered fix files its own row; only instrumentation defects are in-row fixes.

## Resolution

- 2026-10-31 CLOSED with the landing measurement: docs/bench-evidence/tt-nonrouted-attribution-20261031/README.md. The premise falsified twice over: the routed-expert phase is 33.1% of the 181 ms step in situ (60.3 ms), 2.7x the micro projection that produced the "12%" derivation, so the unattributed remainder was ~56%, not 88%. The table closes to 99.3% (16t) / 101.9% (32t) of the step wall with no unexplained gap bucket. The named levers: sliding attention 37.0% whose cost still grows with sequence length despite the window (next hypothesis), the routed loop 2.7x above its matched-pool floor (NEON vec_dot), shared expert 7.4%, lm_head 3.1% near its measured floor.

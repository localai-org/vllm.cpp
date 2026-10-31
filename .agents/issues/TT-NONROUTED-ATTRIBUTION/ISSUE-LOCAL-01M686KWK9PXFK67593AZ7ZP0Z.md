ID: ISSUE-LOCAL-01M686KWK9PXFK67593AZ7ZP0Z
Title: Attribute the ~88% of Flint keep-quant serving decode time outside the routed-expert phase
Row: TT-NONROUTED-ATTRIBUTION
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: -

## Problem

FLINT-KQ-CPU-PERF (ea530a5c7 lineage) matched the threadpool and took the keep-quant paged arm to 5.51 tok/s @16t -- but its evidence records only ~12% of serving decode is the routed-expert phase. The other ~88% (attention, dense, norms, lm_head, sampling, orchestration) is unattributed, so no lever can be ranked. Spec: .agents/specs/tt-nonrouted-attribution.md. Deliverable: one wall-clock-verifiable phase table at 16t and 32t summing to ~100%; adoption of any discovered fix files its own row; only instrumentation defects are in-row fixes.

## Resolution

-

ID: ISSUE-LOCAL-01M6VZZE9W9Q1JB8VX4GH4ADET
Title: Stage A: the trace contract and the both-ways single-token measurement -- the dispatch-vs-compute bound that decides DFlash2's GO/NO-GO
Row: QWEN38-DFLASH2-STAGE-A
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-07
Updated: 2026-11-07
Closed: -

## Problem

The stage-0 study (docs/design/qwen38-dflash2-integration-20261102.md) re-scoped honestly: the DFlash2 drafter is ~6% of the target's per-token work, but whether the pattern wins on our path depends on a bound the study could not settle -- dispatch-bound (~8s/token, DFlash2 loses) vs compute-bound (DFlash2 wins). Stage A measures it: the qwen3.8-27B TT serving decode traced both ways (region-captured vs eager, the sticky-decline cliff characterized), the per-step host-touch count profiled (the discriminator), the verdict recorded. Also: which arm (replay vs eager) the rebench legs served -- the rebench README does not record it. Spec: .agents/specs/qwen38-dflash2-stage-a.md; the study: docs/design/qwen38-dflash2-integration-20261102.md on mine/row/QWEN38-DFLASH2.

## Resolution

-

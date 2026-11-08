ID: ISSUE-LOCAL-01M6VZZE9W9Q1JB8VX4GH4ADET
Title: Stage A: the trace contract and the both-ways single-token measurement -- the dispatch-vs-compute bound that decides DFlash2's GO/NO-GO
Row: QWEN38-DFLASH2-STAGE-A
State: CLOSED
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-07
Updated: 2026-11-07
Closed: 2026-11-07

## Problem

The stage-0 study (docs/design/qwen38-dflash2-integration-20261102.md) re-scoped honestly: the DFlash2 drafter is ~6% of the target's per-token work, but whether the pattern wins on our path depends on a bound the study could not settle -- dispatch-bound (~8s/token, DFlash2 loses) vs compute-bound (DFlash2 wins). Stage A measures it: the qwen3.8-27B TT serving decode traced both ways (region-captured vs eager, the sticky-decline cliff characterized), the per-step host-touch count profiled (the discriminator), the verdict recorded. Also: which arm (replay vs eager) the rebench legs served -- the rebench README does not record it. Spec: .agents/specs/qwen38-dflash2-stage-a.md; the study: docs/design/qwen38-dflash2-integration-20261102.md on mine/row/QWEN38-DFLASH2.

## Resolution

- 2026-11-07 (thalia P150, 9 legs, `docs/bench-evidence/qwen38-dflash2-stage-a-20261103/`): the bound is COMPUTE-BOUND. Captured whole-graph arm TPOT median 6637.35 ms with 3.7 ms host wall and 0 op dispatches per step (device tail 6623 ms = 99.9%); region-replay arm +0.14% (65 enqueues/step); eager arm +0.65% with 6.29 s/step of host op dispatch (~98 ms/layer x 64 layers). The rebench served the CAPTURED whole-graph arm (S=1, 119.7 MiB staging, kind=cold→capture→replay). The sticky region-decline cannot fire: `region_declined` (qwen3_5.cpp) has no writer and `WholeGraphTraceFits` has no caller — declared but unimplemented; capture succeeds in both scopes (max region 2.20 MiB against the 50 MiB budget). GO for stages B-D per the spec's rule; the stage-D acceptance gate carries the row-scaling break-even arithmetic (verify block ~9 rows of device work / ~6.5 accepted).


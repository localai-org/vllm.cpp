# SPEC — `QWEN38-DFLASH2-STAGE-A` — the trace contract and the both-ways single-token measurement

- Owning row: `QWEN38-DFLASH2-STAGE-A`
- Parent row: `QWEN38-DFLASH2` (stage 0's study: docs/design/qwen38-dflash2-integration-20261102.md)
- Issue: created in the same flow (row-scoped local issue)

## What stage A decides

The stage-0 study re-scoped honestly: the drafter is ~6% of the target's
per-token work, but whether DFlash2 wins on our path depends on a bound
the study could not settle — **dispatch-bound (~8 s/token, DFlash2
loses outright) vs compute-bound (DFlash2 wins)**. Which one holds is
stage A's measurement. Stage A also establishes which arm (replay vs
eager) the rebench legs served — the rebench README does not record it.

## Deliverables

1. **The trace contract**: our qwen3.8-27B TT serving decode traced —
   the region-scoped capture arm (qwen3_5.cpp:12143, 13008-13018, 50
   MiB/region, sticky eager decline) measured BOTH WAYS: (a) the
   region-captured arm's per-step cost and (b) the eager arm's, on the
   same host/governor/artifact — the study's open question. The region-
   decline cliff (the sticky decline at :12250) characterized: what
   triggers it, whether it fires at our shape.
2. **The both-ways single-token measurement**: TPOT (and ms/step) on
   the production path, current head, with the per-step host-touch
   count profiled (how many dispatch round-trips per decode step — the
   dispatch-bound vs compute-bound discriminator: dispatch-bound means
   the host round-trips dominate; compute-bound means the device work
   does). The measurement targets: batch 1, input 32, 200 generated
   tokens, INT8DOT=1, governor performance, 3-rep medians.
3. **The verdict**: dispatch-bound or compute-bound, with the numbers
   that decide it — recorded in docs/bench-evidence/
   qwen38-dflash2-stage-a-20261103/ and driving the GO/NO-GO for
   stages B–D (compute-bound → stages proceed; dispatch-bound → the
   row records the loss and the trace-contract work becomes the
   prerequisite lever — the on-card loop the bundle has and we lack).

## Non-goals

- No drafter work (stages C/D). No bucket system. No product-code
  behavior change (the trace contract is measurement instrumentation;
  if it touches the forward, env-gated default-off).

## Gates

check-agent-record, style, trailers, preflight (named host dotnet
failure), claims file RUNNING→DONE, governor logged per leg, NO
timeout guards. The standing device recipe (mutex, reset, envs,
substrate unmodified).

## Git integration

One pull request, `row/QWEN38-DFLASH2-STAGE-A`, spec committed first.

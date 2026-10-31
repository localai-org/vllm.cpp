# SPEC — `TT-NONROUTED-ATTRIBUTION` — attribute the ~88% of Flint keep-quant serving decode time outside the routed-expert phase

- Owning row: `TT-NONROUTED-ATTRIBUTION`
- Kind: performance (attribution/measurement; adoption of fixes files its own rows)
- Issue: created in the same flow (row-scoped local issue)

## Problem

FLINT-KQ-CPU-PERF (landed lineage ea530a5c7) matched the threadpool to the
cpus and took the keep-quant paged arm from 1.20 to 5.51 tok/s @16t — and
its evidence records that only ~12% of serving decode time is the
routed-expert phase the row optimized. **~88% of the step is somewhere
else**, unattributed: attention (paged, hybrid sliding/full), the dense
layers, norm kernels, lm_head/sampling, the threadpool barriers outside the
routed loop, or pure host orchestration. No lever can be ranked until the
88% has names and numbers.

## Deliverable

One attribution table over the serving decode step, phase-named, summing to
~100% of the step: attention / dense MLP / norms / lm_head / sampling /
routed-expert phase (the known 12%) / orchestration-and-gaps (whatever the
phases do not cover). Method: instrument or reuse existing timers around
the production forward (the flint_kq_moe_class_ab harness already loads via
`ModelRegistry::Load` — extend the same pattern), verify the phases sum
against wall-clock, and record per-phase ms at 16t and 32t.

## The acceptance rule (adjudication)

The table is the deliverable. Adoption of any fix found along the way files
its own row. The one in-row fix allowed: a timer/phase-name bug in
flint.cpp or the harness (an instrumentation defect is in scope; a
performance change is not).

## Red-first

There is no number today — the "88%" itself is derived (100 minus the
routed 12%). The red is the empty table; the green is the completed,
wall-clock-verifiable table.

## Gates

check-agent-record, style, trailers, preflight (named host dotnet failure).
Serving legs: bench zone, idle-gated, claims file, governor logged, 3-rep
medians.

## Git integration

One pull request, `row/TT-NONROUTED-ATTRIBUTION`, spec committed first.

# SPEC — `QWEN38-DFLASH2-STAGE-A` — the trace contract and the both-ways single-token measurement

- Owning row: `QWEN38-DFLASH2-STAGE-A` (record ID `BACKEND-TENSTORRENT-QWEN38-DFLASH2-STAGE-A`)
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

## Scope

The stage-0 study re-scoped honestly: whether DFlash2 wins on our path
depends on a bound the study could not settle — dispatch-bound (~8
s/token, DFlash2 loses outright) vs compute-bound (DFlash2 wins). Which
one holds is stage A's measurement. IN: env-gated default-off per-step
instrumentation (the dense decode driver + the op-provider seam), the
both-ways device measurement, the trace contract written from
measurement, the evidence directory, the arm the rebench served. OUT:
drafter work (stages C/D), the bucket system, any product behavior
change.

## Upstream chain

vLLM is the mirror for the serving path behavior; the decode capture
arm mirrors upstream's `FULL_AND_PIECEWISE` decode capture
(`vllm/config/compilation.py:63,65-66,630-632` via
`include/vt/breakable_graph.h`). DFlash2's on-card verify-loop
structure is the dflash2-vision bundle
(`mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4` @
`7855cf0d2982064350991bdb8e1ec1db82f914b3`), quoted as provenance,
never a mirror source (oracle registry `vllm-tt-plugin` /
`TT-PLUGIN-QWEN38-ORACLE`). Stage A measures; it ports nothing.

## Our baseline

TPOT 25.23 s, TTFT 1203.6 s, 0.03 tok/s output, 100% warm JIT,
governor-swept 3-rep medians (`docs/bench-evidence/qwen38-p150-rebench-20261102/README.md`),
served through the production AsyncLLM path with
`VT_TT_KEEPQUANT_INT8DOT=1`. The arm (captured vs eager) that baseline
ran on is UNKNOWN — stage A establishes it.

## Port map

No port. Stage A is measurement + read-only instrumentation:
`src/vt/op_provider.cpp::GetOp` (the dispatch counter),
`src/vllm/model_executor/models/qwen3_5.cpp::Qwen3_5DenseDecodeGraph::Step`
(the per-step phase/capture-state/segment emit, extending the existing
`VT_TT_STEP_PHASES` clock), both env-gated default-off.

## Tests to port

None. The upstream reference has no test for our instrumentation; the
gate is the measurement itself plus the standing record/style gates.

## Dependencies

Device: the thalia P150 (local board, `flock $HOME/gpu.lock` mutex,
`luwen` reset, `/tmp/umdtrial-install` substrate, `TT_METAL_HOME`
`~/Sources/tt/tt-metal-pin`), build `/tmp/build-stagea`, artifact
`/mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf`,
`VT_TT_KEEPQUANT_INT8DOT=1` mandatory. Stages B-D depend on this row's
verdict (GO/NO-GO).

## Work breakdown

W1: the instrumentation (dispatch counter, capture-state marker,
segment count) — env-gated, default-off, committed separately. W2: the
measurement series (defaults x3, region-replay x3, eager x3, medians,
idle-gated, governor logged) + the sticky-decline characterization. W3:
the trace contract + evidence directory + verdict + the rebench-arm
finding.

## Risks/decisions

D1: the sticky region-decline is, as of the static read, DECLARED but
never SET (`qwen3_5.cpp` `region_declined` has no writer); stage A
measures what the code actually does and records the gap rather than
repairing it. D2: the whole-graph arm's 3.15 GB staging demand vs ~298
MB free means the defaults leg characterizes the real cliff. D3: the
instrumentation is a getenv-gated single stderr line per step; the
unset path pays one static bool compare.

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

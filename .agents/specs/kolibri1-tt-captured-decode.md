# SPEC — `KOLIBRI1-TT-CAPTURED-DECODE` — the kolibri1 P150 captured-decode campaign: remove the ~100 blocking host round-trips per step

- Owning row: `KOLIBRI1-TT-CAPTURED-DECODE`
- Parent rows: `BACKEND-TENSTORRENT` (the gap analysis: docs/gap-kolibri1-tt-captured-decode.md), `KOLIBRI1-DRAFTER-STUDY` (the 2026-11-07 study that sharpened this campaign's premise and rejected the drafter alternative)
- Issue: `ISSUE-LOCAL-01M4N8Y7XPDC74SED0WEDN0YFR` (reassigned to this row; open since 2026-10-11)

## What this campaign is

kolibri1 on the P150 decodes at ~1.82 tok/s (548.5 ms/step median,
`the 2026-10-21 P150 numbers refresh (row `P150-NUMBERS-REFRESH`, evidence `kolibri1-tt-p150-refresh-20261021`)`) through a
per-LAYER host-touch loop: the router downloads its full `[t,384]` f32
logits to host every layer (kolibri1_tt_forward.cpp:563-564), runs the
sigmoid-logit-add top-6 on host (`:582`), and the routed experts stream
from host (dispatch plan `:673-677`, fetch list `:678-680`, per-expert H2D
stage with a full-slot D2H readback pivot `:685-702`, bf16 dequant memo
`:669`). Counted plainly that is two-plus blocking host round-trips per
layer, ~100+ per step. Every step also allocates fresh `DBuf`s — no
persistent per-step device buffers, no `Replay()` (gap doc, missing item 1).

The qwen3.5 captured-decode template exists and is proven on this card
(~12 ms wall / 0.3 ms replay per served step). kolibri1 implements none of
its hooks. This campaign closes that gap in stages, each stage gated
greedy-token-exact against the current serving arm.

## Why a campaign, not a task

The 2026-11-07 drafter study (the 2026-11-07 drafter study (PR #3551, branch `row/KOLIBRI1-DRAFTER-STUDY`))
priced the ordinary capture enablement at 3-5 device-days with ~zero win
WHILE items 3/4 still serialize every layer — the per-layer D2H and the
host-built data-dependent expert fetch are exactly what forbids trace
capture. The lever only pays after the loop is capture-shaped. The study's
verified-but-slower device-router composite (kolibri1_tt_forward.cpp:540-548,
env-gated off) shows the routing half alone is not a win: the stages below
must land as a set before the capture stage can be judged. DFlash2 stage A's
compute-bound verdict on qwen3.8 (6623 ms device tail = 99.9% of TPOT;
captured host wall 3.7 ms/step vs eager 6.29 s) is the template for this
row's own stage-0 bound question on kolibri1's shape.

## Stages

- **Stage 0 (bound question)**: measure kolibri1's P150 decode both ways —
  the current arm vs a host-free prototype with the router tail kept on
  device — to decide the same dispatch-bound-vs-compute-bound question
  stage A answered for qwen3.8. GO/NO-GO for the rest.
- **Stage 1 (capture-shaped loop)**: persistent per-step device buffers
  (no fresh DBufs), the routing decision resident on device (a vt::MoeRouterTopK
  TT port or the gated composite — measured, not assumed), and the expert
  fetch made trace-compatible (the slot-pool plan + fetch list resolved
  without mid-loop D2H). Gate: greedy token identity vs the current arm on
  the identical workload; MiMoV2 sink gates + dtype bit-identity on every
  kernel-region change.
- **Stage 2 (the capture)**: wire the qwen3.5 decode-graph template's hooks
  (driver, Warm* hooks, Replay) for kolibri1's graph. Gate: capture/replay
  stability across 197+ steps, greedy identity, then the TPOT comparison
  that justifies the campaign.

## Scope

IN: kolibri1 TT serving-path changes routed through the shared seams
(ModelRegistry::Forward, dense_attn::AttnBlock, VT shared decode machinery),
the per-stage evidence dirs (flat per-run layout), greedy-exact gates.
OUT: the drafter (rejected NO-GO, kolibri1-drafter-study), any quant-arm
change, CPU-path changes.

## Weights

The pinned kolibri1 GGUF artifacts and the fp8 checkpoint per docs/USAGE.md
(this row serves existing arms only; any weight pin change is its own
record edit).

## Risks/decisions

- The device-router composite is verified but measured SLOWER; stage 1 must
  not simply enable it — the composite's cost structure changes once the
  expert fetch is also resident, so re-measure inside stage 1 before adopting.
- The fp8 expert-slot pool and the dequant memo interact with residency
  capture the same way the qwen3.5 state slots did (baked content-keyed
  addresses); the commit-or-restore pattern stage B built for qwen3.8's GDN
  state is the precedent if kolibri1's slots need rollback.
- The stream-bound guard's loud refusal past the B1 per-token ceiling
  (kolibri1_tt_stream.cpp:209-217) is a product contract — capture must not
  silently widen it.

## Gates

Every stage: greedy token identity vs the incumbent arm on the identical
workload (divergence = wrong implementation, STOP), MiMoV2 sink gates +
dtype bit-identity on kernel-region changes, preflight green except known
host gaps, evidence in `docs/bench-evidence/kolibri1-tt-captured-decode-<stage>-<date>/`,
governor + idle gate + reset per leg. Stage 2 additionally: capture/replay
stability 197+ steps and the TPOT verdict that either adopts or records
the loss.

## Git integration

One pull request per stage (each stage is its own unit of work); the spec
and the issue reassignment land first as this records-only change. Push to
`mine`.

## Stop conditions

- Stage 0 returns dispatch-bound at kolibri1's shape with no capture-shaped
  path to compute-bound: record the loss on the issue, close the campaign
  honestly, keep the gap doc.
- Greedy divergence at any stage: STOP, the accept rule is the incumbent's
  own argmax.

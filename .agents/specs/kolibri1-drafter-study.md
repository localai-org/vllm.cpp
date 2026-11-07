# SPEC — `KOLIBRI1-DRAFTER-STUDY` — the DFlash2 adoption sketch for kolibri1 on the P150: what a drafter needs, what we have, what the lift is

- Owning row: `KOLIBRI1-DRAFTER-STUDY`
- Kind: study/design (a written adoption sketch; no product code)
- Issue: created in the same flow (row-scoped local issue)
- Parent: TT-DRAFTER-STUDY (the DFlash2 study, `docs/design/tt-drafter-study-20261103.md` on mine/row/TT-DRAFTER-STUDY)

## Why kolibri1 first

The developer picked kolibri1 as the first adoption target. Three reasons
from the DFlash2 study:

1. **Its TT decode is already on-card**: kolibri1's P150 decode runs
   captured/replayed device work with one host touch per step — the
   closest match to the bundle's loop structure. The drafter pattern's
   marginal integration cost is smallest here.
2. **Its numerics are ratified**: the W0 band gate + the chain gate are
   the standing correctness contracts; a drafter changes the DECODE
   LOOP's acceptance semantics (greedy verify = the target's own argmax,
   which the chain gate already measures) — the contracts to satisfy are
   already in the tree.
3. **The measured baseline exists**: the P150 anchoring session's
   983.5 ms/step (1.02 tok/s) + the kolibri1 TT attribution table
   (attention 50.3%, routed 33.1%) are the baseline a drafter would be
   measured against.

## What a DFlash2-style drafter needs from the target model (from the study)

1. **Residual taps** the drafter consumes (the target's post-layer
   residual — kolibri1's TT forward has them).
2. **A verify pass over the draft block** (the target's own forward —
   kolibri1's TT forward, captured/replayed).
3. **A TRAINED DRAFTER** — the real lift. The DFlash2 drafter is a
   5-layer transformer consuming the target's residual taps with its own
   paged context KV. For kolibri1: a drafter checkpoint must be TRAINED
   (out of scope for this row — the sketch names the training plan) or
   an existing model repurposed. The incoai DFlash2 drafter is
   qwen3.8-specific and does not transfer.
4. **Bucket geometries with hysteresis** (per-user verify shapes with
   hold semantics).

## Deliverable

`docs/design/kolibri1-drafter-study-20261107.md` (following
docs/design/tt-drafter-study-20261103.md's shape): the adoption sketch
answering —

1. **kolibri1's TT decode structure today**: the captured/replayed
   traces, the host-touch boundary, the residual stream shape — with
   file:line citations into src/vllm/model_executor/models/
   kolibri1_tt*.cpp.
2. **The drafter fit**: where a DFlash2-style drafter plugs in (the
   residual taps, the verify trace, the acceptance math), what kolibri1
   provides already vs what must be built.
3. **The drafter checkpoint plan**: size (5 layers x ~1.8B-class? a
   smaller kolibri1-derived draft?), training data + compute estimate,
   the acceptance-rate target (the study's ~1.5 tokens/iteration
   break-even floor), and the honest risk register.
4. **The staged adoption plan**: the five stages from the DFlash2 study
   adapted to kolibri1, each with its gate.
5. **The decision**: GO/NO-GO with the basis — the measured baseline
   (1.02 tok/s), the projected drafter speedup at the modeled acceptance
   rate, and the training cost. NO GO = the sketch records why (e.g.
   the model is too small for a drafter to pay for itself at kolibri1's
   scale — the DFlash2 target is 27B; kolibri1 is smaller, and the
   drafter's overhead share is correspondingly larger).

## Method

- Read kolibri1's TT decode path (kolibri1_tt*.cpp, the serving gate's
  recipe) with file:line citations.
- Read the DFlash2 study (docs/design/tt-drafter-study-20261103.md, on
  mine/row/TT-DRAFTER-STUDY) for the mechanism + the loop + the bucket
  system.
- Every kolibri1 claim cites the tree; every DFlash2 claim cites the
  study (which cites the bundle sources).
- No device run: the sketch is arithmetic + structure (the baseline is
  measured; the acceptance-rate model is arithmetic on the study's
  break-even floor). A device measurement of the draft-forward cost is
  the adoption row's first stage, not this study's.

## Gates

check-agent-record, style, trailers, preflight (named host dotnet
failure). Device-free row.

## Git integration

One pull request, `row/KOLIBRI1-DRAFTER-STUDY`, spec committed first.

# SPEC — `QWEN38-DFLASH2-STAGE-C` — the drafter: DFlash2 block-diffusion drafting on the captured verify arm

- Owning row: `QWEN38-DFLASH2-STAGE-C` (record ID `BACKEND-TENSTORRENT-QWEN38-DFLASH2-STAGE-C`)
- Parent row: `QWEN38-DFLASH2`; prerequisite DONE: stage B (`34fbb6154` on origin/main — the recorded-geometry fix, the GDN commit-or-restore, the `(S, 9, spec=true)` captured verify arm)
- Issue: created in the same flow (row-scoped local issue)

## What stage C delivers

Stage A proved compute-bound; stage B made the captured verify trace
correct and speculative-ready. Stage C adds the DRAFTER: the bundle's
DFlash2 block-diffusion drafter (one 5-layer forward over the target's
residual-stream taps, producing 8 draft rows `[anchor, MASK×7]` at FIXED
shapes) ported to our stack, wired so the captured verify arm replays the
9-row speculative verify, with host-side greedy accept by token-id
compare. The modeled win at the bundle's acceptance is ~8.5x; the row's
gate is what we actually measure.

Provenance (never a mirror source; oracle registry `vllm-tt-plugin`,
gateable = no): the dflash2-vision bundle
(`mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv` @
`28add9a1fbd0f259cba61f955023a65a10748064`, local read
`/tmp/df2-bundle/`, code sha256 `ce4a1982…`). Anchors quoted: the drafter
forward `extra_code/qwen38/dflash2.py` (`DFlash2Draft` component-validated
form, `DFlash2Drafter` production form; 8-row fixed-shape per-iteration
forward; `_hctx` = rms(fc(taps)) host taps [1,S,25600] -> [1,n,H];
`_ctx_kv` context K/V), the serving loop `dflash2_serving.py:898-1053`
(three captured traces replaying from fixed-address buffers; host runs
only accept math), greedy accept = token-id compare, sampled = exact
rejection sampling (`spec_sampling.py:280` — OUT of scope here, greedy
only).

## Waves

- **W1 — weights: obtain, convert, pin.** The drafter's weights must ship
  through OUR loader. Source of truth: the bundle checkpoint's drafter
  tensors (fc, hnorm, per-layer projections + norms; the target model's
  own weights are already served). Deliverable: a converter script (the
  glm5-next converter precedent) producing the loader's expected format,
  the exact files + repo + revision + sha256 pinned in `docs/USAGE.md`
  grouped under the qwen3.8 arm (the say-which-weights rule — a repo id
  alone is not a pin), and a refusal-by-name message if the drafter
  weights are absent at load time. Downloading the checkpoint requires
  the developer's go (large asset rule); ask at W1 start if not already
  given.
- **W2 — the drafter forward, component-validated.** Port
  `DFlash2Draft`'s per-iteration forward (8 fixed-shape rows over the
  taps) with a component gate vs the bundle's reference numbers (PCC
  band per the bring-up precedent) before any loop wiring. Runs on the
  P150 under lease + mutex; residual taps come through the aux seam
  stage B fixed (per-tap `[1,H]` into the persistent `[S,H*taps]` slot,
  recorded geometry preserved).
- **W3 — the loop.** Wire draft -> captured verify replay ->
  host accept through `Qwen3_5DenseDecodeGraph` (the DENSE driver only;
  the MoE driver untouched, the stage-B seam call sites are the anchor).
  Greedy only: accept by token-id compare with the bundle's cascade rule;
  on reject, the stage-B GDN commit-or-restore rolls the state back. The
  accept/commit math stays on HOST for this stage (the bundle's deeper
  on-card move is a later row).
- **W4 — the gate.** Identical-workload measurement, 3-rep medians,
  governor + idle gate + reset per leg, evidence FLAT dir
  `docs/bench-evidence/qwen38-dflash2-stage-c-<date>/`: (a) greedy token
  identity — the spec-decode arm's FINAL tokens must equal plain decode's
  (divergence = wrong implementation, STOP); (b) acceptance rate +
  TPOT vs the stage-B arm's 6606.84 ms — the row's verdict is the
  measured multiple, recorded honestly either way.

## Scope

IN: drafter weights conversion + pin, the drafter forward on TT, the
speculative loop on the dense driver, host accept. OUT: sampled-mode
rejection sampling, the on-card accept/commit move, the bucket system
(stage D), quant-arm changes, the MoE driver.

## Our baseline

Stage B's arm: TPOT 6606.84 ms warm (eager 6648.14), greedy-identical,
greedy tokens pinned (sha256 `d49c53d2…` over 200 tokens, batch 1 /
input 32 / output 200, seed 0). Stage C's W4 gate compares against this
exact configuration.

## Tests to port

The bundle's component validation (PCC bands per
`tests/dflash2_ttnn_bringup.py` precedent quoted in the drafter header)
as our W2 gate; host-side accept-rule unit tests (cascade on the 8-row
draft, reject path exercising the stage-B rollback — extend the
stage-B rollback unit); the token-identity device gate.

## Dependencies

Device: local P150, `${HOME}/gpu.lock` mutex, reset recipe, governor
logged; build recipe as stage B's (durable-pin prefix + topk isystem
flag); runtime env durable-libs-first, `TT_METAL_RUNTIME_ROOT` at the
pin. Weights: the bundle checkpoint (W1) — DOWNLOAD NEEDS DEVELOPER GO.
Stages D (buckets + hysteresis) depend on this row's measured
acceptance.

## Risks/decisions

- The drafter weights' storage format in the bundle checkpoint is
  UNVERIFIED at spec time — W1's first task is to enumerate them and
  name the conversion; if the checkpoint withholds the drafter tensors,
  STOP and report (the modeled 8.5x is then unbuyable from this source).
- The taps host->device staging per draft iteration is host traffic the
  bundle avoids by living on-card; stage C accepts host staging cost and
  MEASURES it (the modeled win assumed it away — the W4 verdict must
  attribute the gap honestly).
- Divergence in greedy final tokens is STOP, not a tuning gap: the accept
  rule is the target's argmax.
- If acceptance < ~1.5x breakeven at this shape, the row records the loss
  honestly (the stage-A arithmetic: 9-row verify ≈ 1 row's weight
  streaming was the premise — W4 measures it).

## Gates

1. W2 component PCC band green before loop wiring.
2. Focused: accept-rule + rollback units red-before where applicable,
   green-after.
3. Full: preflight `--staged` green except known host gaps (dotnet;
   glm4 deliberately-red lanes; NEON now fixed on main).
4. W4 device gate: greedy token identity (STOP on divergence) +
   acceptance/TPOT table vs stage-B baseline.
5. Reachability: the reviewer deletes the draft call site and the W4
   gate must go red (the loop, not a class, must be measured).

## Git integration

One pull request for the whole stage (the stage-A/B pattern); branch
`row/QWEN38-DFLASH2-STAGE-C`, worktree `/tmp/q38-stage-c`, push to
`mine`. Commit per wave; trailer block byte-exact last.

## Stop conditions

- W1 finds no drafter tensors in the reachable checkpoint: STOP,
  NEEDS_DECISION (source unbuyable).
- Greedy divergence at W4: STOP.
- W4 acceptance below breakeven: record honestly, close the row DONE
  with the loss, keep the arm behind its knob.

# Spec: the capture-economics curve — TPOT vs output length on the serving arm

Row: `BACKEND-TENSTORRENT`. State: DRAFT (2026-09-27).
Issue: files against the perf-campaign row; the curve is the evidence
the next optimization targets hang from.
Follow-up to: #3327 (the replay arm serves — 12 ms/replay, TPOT
cold-dominated at output_len 4) and the decompose (#3322: the fixed
per-request costs — cold body, capture pass, trace-wait).
Git integration: one pull request (spec + the curve + the model), branch
`row/TT-CAPTURE-ECONOMICS`.

## Problem

The serving arm's TPOT at output_len 4 is ~34 s/token — cold-dominated:
the per-REQUEST fixed costs (the eager cold step ~33 s, the capture pass
~5.7 s, the post-capture trace-wait ~31 s in the next step's gap) dwarf
the per-TOKEN served-replay cost (12 ms wall, 0.3 ms replay). Whether
this matters is a function of generation length: at output_len 256, the
fixed costs amortize over 251 served steps and TPOT should collapse
toward ~12 ms/token-class behavior — IF nothing else grows with length
(the GDN state windows, the KV pages, the per-step keepquant repair
transients all could).

## The measurement

1. **The curve**: TPOT (mean/median/P99) at output_len in {4, 16, 64,
   256} on the anchor (4 prompts, c=1, the committed fixture, seed 0,
   temperature 0, ignore-eos), one fresh process per leg (the
   per-request OOM ceiling stands — the ttnn deferred-reader issue is
   still open upstream; the curve reports what fits in one process).
2. **The phase mix per leg**: `VT_TT_STEP_PHASES` +
   `VT_DECODE_GRAPH_STATS` — how the cold/capture/served split shifts
   with length; confirm the served steps stay ~12 ms or reveal what
   grows.
3. **The model**: fit TPOT(len) = fixed/len + per_token — the two
   coefficients are the deliverable. The fixed term is the capture
   economics target; the per-token term is the dispatch row's
   denominator.

## Deliverable

A dated benchmark-record section: the curve table, the fitted model,
the verdict (which term dominates at real generation lengths), and the
named next targets. Interpretation binds: if per-token stays ~12 ms
class, the dispatch row is the target and the capture pass is second;
if per-token grows with length, the growth mechanism is the target.

## Gates

- Tokens at each leg recorded (byte-comparison across lengths is not a
  gate — different lengths produce different streams; the served-replay
  test's green is the correctness state).
- Reproducible: the 4-token leg re-runs within noise of #3327's
  recorded 34,165 ms.
- Standard gates; instruments read-only.

## Risks

- A leg may hit the per-request retention ceiling before its length
  completes (the ~5-request process cap): the curve reports the leg's
  completion honestly; if output_len 256 cannot complete in one
  process, the curve's top point is the truncation and the model notes
  it.
- The 256-token legs are long (256 x 34 s if cold-dominated = hours):
  the legs run detached; if the curve bends early (the model fits from
  4/16/64), the 256 leg is the confirmation, not the discovery.

## Non-goals

- No optimization; no changes to the capture path; the dispatch row is
  next (spec'd separately).

## Stop conditions

- A leg cannot complete (retention ceiling or device fault after 3
  retries): report the partial curve with its model; do not extrapolate
  silently.

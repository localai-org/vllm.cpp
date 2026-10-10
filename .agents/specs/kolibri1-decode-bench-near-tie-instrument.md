# Spec: the decode-bench near-tie instrument (NEON lane adjudication)

Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
Issue: ISSUE-LOCAL-01M221WAVSK2STJ49Y3WTDSDS3 (GitHub #3438, open for
Ettore's B-vs-C preference). This spec is the B pilot: it changes ONLY the
bench gate; production numerics are untouched.

## Problem

`test_kolibri1_decode_bench` pins the greedy chain byte-exactly (anchor
109726, alternating 101807/109726). The aarch64 NEON lane of
`vt::PagedAttention` agrees with the scalar body to 1 bf16 ulp, which flips
the first decoded token (33382 vs 101807) at a position whose scalar
top1–top2 gap is 0.197 nats — inside the row's ratified 2.5-nat near-tie
band. A byte-exact gate cannot express "correct up to the ratified band",
so a numerically sound lane fails.

## Design

The bench gains the W3 methodology, cross-arm instead of cross-implementation:

1. TEACHER-FORCED GAP: for each compared position, the scalar chain is the
   reference; the lane's token is scored teacher-forced on the scalar prefix
   (the same procedure W3's `neartie_gap_mnats` uses, and the tooling
   `scripts/glm4-moe-lite-neartie-gap.py` established). The gap is the nats
   distance from the scalar argmax to the lane's token under the scalar
   distribution.
2. THE GATE: with the lane ON, a position passes when EITHER the token is
   identical OR (the gap is ≤ 2.5 nats AND the lane's token is inside the
   scalar top-K at that position). A gap beyond the band, or a token outside
   top-K, is a HARD FLIP and fails — the same rule W3 enforces.
3. THE CAPTURE: the per-position gaps are computed at run time by the bench
   itself (both arms run in-process; no oracle capture is needed — the
   scalar arm is the reference). The worst gap and the flip count are
   printed, and the numbers are recorded in the row's evidence doc.
4. STRICT MODE: `VT_CPU_PAGED_ATTN_NEON=0` keeps the byte-exact anchor
   exactly as it is today; the scalar path is untouched.

## Tests

- The bench test case gains the instrument; with the lane OFF it must
  produce byte-identical chains and zero flips (unchanged pass).
- RED-FIRST: with the lane ON on the current code, the byte-exact check
  fails today (33382) — the instrumented gate must pass it (0.197 nats,
  in-band) while a synthetic beyond-band mutation (e.g. compare against a
  deliberately perturbed arm) must still fail.
- W3 stays byte-exact-where-it-is and near-tie where it already is; this
  spec changes no W3 assertion.

## Gates

- The instrumented bench, lane OFF: golden chain, exit 0.
- The instrumented bench, lane ON: 0 hard flips, all gaps ≤ 2.5 nats,
  worst gap recorded (expected ≈ 0.2 from the adjudication).
- The unit sweep `test_ops_paged_attn_neon`: unchanged, green.

## Risks

- BAND LAUNDERING: a wide band could mask a real defect. Mitigations: the
  band is the row's already-ratified 2.5 nats (not chosen here); hard flips
  (outside top-K) always fail; the worst gap is printed and recorded, so a
  change from ~0.2 to ~2.0 is visible in the record.
- The instrument measures scalar-vs-NEON, not model-vs-upstream. W3 remains
  the upstream-facing gate and is unaffected.

## Owed / stop conditions

- If Ettore picks C, this instrument is still correct for the transition
  (it measures the arms against each other during the re-capture) but the
  default-flip decision moves to the C plan; the pilot stops there.
- Idle-host A/B re-measurement (the disposition's other precondition) lands
  with the flip decision, not in this spec.

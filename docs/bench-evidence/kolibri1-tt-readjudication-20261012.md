# kolibri1 TT P150: teacher-forced readjudication on the corrected instrument (2026-10-12)

Issue: `ISSUE-LOCAL-01M4ER0E9HHM95YZYJB7T5FECN`. Branch: `row/tt-kolibri-readj`
(off `row/tt-b2ii-index` @ `2c4efce88`, which fixed the teacher-forced walk's
shifted chain — the walk now feeds `generated_ids[d-1]`, not
`generated_ids[step-1]`).

## Why this run exists

The prior hard-flip classification ("router-gain amplification of upstream
bf16 drift", recorded 2026-10-10 on the `:76`-era instrument) was computed
against a SHIFTED prefix: the walk fed the golden token one position early, so
every teacher-forced logit row answered a different question than the gate's
decode. All teacher-forced flip counts from that instrument era are INVALID as
per-step evidence. This run re-derives them on the corrected walk.
Determinism on this base is verified (3 identical chains per arm), so
cross-run comparison is valid.

## Runs

- Host: P150 (thor), `tt-smi -r 0` + clean `tt-metal-cache` / shm before each
  arm; substrate = `/tmp/umdtrial-install` (carries the rms_norm fp32-accum
  patch and the port fixes); model `/mnt/models/Aleph-Alpha/Kolibri-1`.
- Binary: `test_kolibri1_tt_b2ii --test-case='*device token gate*'`,
  commit `2c4efce88` + the readjudication RouteDump probe (see below).
- `SKIPPED`: 0 in every leg; every device leg ran.
- Instrumentation: `VT_KOLIBRI1_TT_STAGE_DUMP=1` (existing per-stage bf16
  bit-hash probes, both arms, plus the CPU-vs-TT `SCRATCH dbg TF prompt`
  teacher-forced leg) and a NEW scratch probe in the TT `RouteDump`
  (`kolibri1_tt_forward.cpp`) printing the sorted top-8 BIASED router
  selection scores per layer per token (`route_scores`), so the per-layer
  top1-top2 / top6-top7 gaps are measurable against the incoming drift.

## Adjudication table (corrected instrument, this run)

| Arm | free divergences | tf flips | near-ties | HARD | worst tf gap | chain (matched/compared) |
|---|---|---|---|---|---|---|
| host-free ON | 170 | 69 | 29 | **40** | 10.617 nats | 3/11 (8 flips: 3 near-tie, 5 hard) |
| host-free OFF | 158 | 44 | 27 | **17** | 8.647 nats | 12/20 (8 flips: 6 near-tie, 2 hard) |

Both arms reproduce the recorded #3453-era corrected-instrument numbers
exactly (ON 69/29/40/10.6, OFF 44/27/17/8.6). The residual hard-flip
population is REAL, not an instrument artifact (the zero-stop condition did
not fire).

## Q1: where does the drift originate (corrected walk)

From `SCRATCH dbg TF prompt` (CPU arm and TT arm in one process,
teacher-forced on the golden chain, per-stage bf16 bit hashes; prompts 1
'der Mond…' 25 steps, 5 'Translation to German…' 23 steps, 4 'x1 = 3…' 10
steps):

- The first hash-bearing stage of every decode forward is `L0 dh2` (the L0
  MoE input), and it diverges at EVERY decode step from step 0 on all three
  prompts — before it, the L0 embed/attention stages match the CPU arm.
- L0 `dh2` relative magnitude drift vs CPU: typically 0.03%–2.5% per step
  (outlier steps to ~10x when the sum passes near zero), consistent with the
  1-bf16-ULP-class L0 input-norm origin the 2026-10-10 record established
  element-level; the corrected walk does NOT move the origin. The drift
  origin remains the L0 input norm / device attention chain at 1 bf16 ULP,
  present from decode step 0.

## Q2: does the router-selection-flip mechanism hold? YES.

Per-layer CPU-vs-TT top-6-of-384 selected-expert overlap under the corrected
teacher-forced walk (1600 layer-steps per prompt):

| Prompt | mean overlap (of 6) | full-agreement layer-steps | median router logit_sum rel diff |
|---|---|---|---|
| 'der Mond…' | 3.66 | 6.0% | 6.65% |
| 'Translation to German…' | 3.71 | 7.1% | 7.05% |
| 'x1 = 3…' | 3.05 | 6.4% | 9.65% |

The selection disagreement is present AT EVERY LAYER including L0 (per-layer
mean overlap 1.4–5.7 of 6; e.g. L0 step 0: CPU {47,284,19,228,359,141} vs TT
{359,47,284,246,141,144}). The mechanism holds: a ~0.1–2.5% continuous input
drift through the unit-norm-router GEMM (K=2560) injects percent-level
absolute noise into O(1)-scale biased selection scores, and the top-6
boundary flips 2–3 of 6 experts at nearly every layer-step.

The gap structure pins WHERE the selection is fragile (route_scores probe,
OFF arm, teacher-forced pass at each hard-flip step; 50 layers):

- top1–top2 gap: median across layers 0.12–0.48 (robust — rank-1 is almost
  never the flip site);
- top6–top7 gap: median across layers 0.017–0.050, min 2.6e-5–1.8e-3; at
  every hard-flip step, 26–41 of 50 layers (OFF arm) and 27–44 of 50 layers
  (ON arm) hold a top6–top7 gap BELOW 0.05 —
  the incoming per-layer noise (percent-level absolute on O(1) scores)
  exceeds the rank-6/7 margin at most layers, so a boundary flip is
  expected at essentially every layer, exactly as observed in the overlap
  table. Top-1 selection is not the defect; the top-6-of-384 boundary is.

## Q3: hard-flip distribution (band-2.5 standard input)

Per-prompt / per-step-position distribution of the HARD teacher-forced flips:

OFF arm (17 hard): 'der Mond…' steps 0,1 (opening burst); 'Australia' step 18
(late isolated); 'Wissen…' steps 1,2,3 (burst), 14, 17 (late isolated);
'Translation…' steps 5,10,22 (spread); 'Ein gutes…' step 7; 'counting…'
steps 0–4 (contiguous burst).

ON arm (40 hard): 'der Mond…' 0,1; 'Australia' 0,11,14,15,16,18,21,26
(spread across the whole chain); 'Wissen…' 1,3,4,6,7,11,14,16,17,21 (spread);
'x1 = 3…' 2,4,5,9; 'Translation…' 4,5,10; 'Ein gutes…'
0,2,4,7,9,11,13,22,24 (spread); 'counting…' 0,1,2,3.

First-divergent-layer for every hard flip: on the three instrumented
diverging prompts the corrected walk's first divergent stage is `L0 dh2` at
every recorded flip step (steps 0–24, 0–22, 0–9 respectively) — there is no
late-switching localized defect layer; the drift is resident from step 0 and
the flip step is decided by the LM-head margin at that step, not by where the
drift begins.

Concentration verdict: the flips do NOT concentrate at a single repairable
site. They appear as opening bursts (steps 0–4) on some prompts and as
isolated mid/late-chain flips (steps 14–26) on others, on 7 of 8 prompts, in
BOTH arms. Under the row's band-gate question this is the STRUCTURAL class:
the residual is the continuous bf16 device-arm drift (L0 input-norm origin)
amplified by a high-gain router onto a top-6-of-384 boundary that is
sub-0.05-nat at most layers — no single-layer or single-op repair clears it,
and the arm delta (17 vs 40) shows the host-free shadow reconciliation
remains the largest single lever.

## Recommended gate-standard input

Keep the 2.5-nat / top-5 adjudication. Report the P150 band as STRUCTURAL
(spread), not fixable-by-a-patch: teacher-forced hard flips 17 (OFF) / 40
(ON) spread over prompts and step positions, driven by a step-0-resident
1-ULP-class norm/attention-chain drift x router gain. The next levers, in
order: (1) host-free shadow reconciliation (halves the hard count), (2)
f32-enabling the device attention/norm chain (removes the origin), (3) any
future router-margin widening is a model change, not a backend repair — out
of scope for this row.

## Raw evidence

`/tmp/readj/`: `arm-off.log` (OFF, full stage dump), `arm-on3.log` (ON, full
stage dump, reproduces the clean ON counts exactly), `arm-on2.log` (ON, clean
counts), `tf-p{1,5,4}.log` (CPU-vs-TT teacher-forced drift origin), parsed
tables `off.json`, `on.json`, `tf_origin.json`.

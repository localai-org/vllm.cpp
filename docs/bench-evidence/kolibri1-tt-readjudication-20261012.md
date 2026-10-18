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

## Step-label mapping (SCRATCH dbg TF walk vs the gate's decode steps)

The `SCRATCH dbg TF prompt` legs print a raw `TFstep` counter that is NOT
the gate's decode position. Walk iteration `i` feeds `input_ids[i]` for
`i < input_len` and `generated_ids[i - input_len]` afterwards; its logits
sit at context position `i` and therefore predict
`generated_ids[i - input_len + 1]`. The print emitted `d = i - input_len`
BEFORE that +1, so:

    raw `TFstep N`  ==  DECODE position N + 1

The last prompt token (`TFstep -1`) produces decode-position-0 logits, and
the first teacher-forced feed (`TFstep 0`) produces decode-position-1
logits. The gate's own `HARD flip ... step d` lines use
`d = walk_step - input_len`, which IS the decode position (that forward
predicts `generated_ids[d]`) — no offset there. The test now prints both
(`TFstep N (decode position N+1)`; verified on device with a short
prompt-4 leg), and every step number below that comes from the SCRATCH
walk is given as a decode position. The committed parser
`kolibri1-readj/parse_tf_origin.py` derives `tf_origin.json` under the
corrected labels.

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
  MoE input), and it diverges at EVERY decode position from 0 on all three
  prompts (decode positions 0–25, 0–23, 0–10 respectively, including
  position 0 — the last-prompt-token forward the pre-fix logs labelled
  `TFstep -1`; see `kolibri1-readj/tf_origin.json`, where every row's
  `l0_dh2_hash_cpu != l0_dh2_hash_tt`) — before it, the L0 embed/attention
  stages match the CPU arm.
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
mean overlap 1.4–5.7 of 6; e.g. the walk's first forward, prompt token 1,
at L0: CPU {47,284,19,228,359,141} vs TT {359,47,284,246,141,144}). The mechanism holds: a ~0.1–2.5% continuous input
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
every decode position (0–25, 0–23, 0–10 respectively; corrected labels, see
the mapping section and `kolibri1-readj/tf_origin.json`) — there is no
late-switching localized defect layer; the drift is resident from step 0 and
the flip step is decided by the LM-head margin at that step, not by where the
drift begins.

Concentration verdict: the flips SPREAD across prompts and decode positions
— opening bursts (decode positions 0–4) on some prompts and isolated
mid/late-chain flips (positions 14–26) on others, on 7 of 8 prompts, in
BOTH arms. What the evidence supports is that this spread is CONSISTENT
WITH the shared L0-origin drift amplification: the drift is resident at
every decode position from 0 (Q1), and the top-6-of-384 selection boundary
is sub-0.05-nat at most layers at every hard flip (Q2), so a boundary flip
is expected wherever the LM-head margin is thin, not at a localised site.
This does NOT prove that no single chain-wide change clears the residual:
a chain-wide f32 fix of the device attention/norm chain (lever 2 below)
addresses the shared origin itself, and the spread does not rule out that
such a fix clears most of it. What the spread DOES rule out is a
single-layer or single-position patch. Under the row's band-gate question
this is the STRUCTURAL class (drift x router gain), and the arm delta
(17 vs 40) shows the host-free shadow reconciliation remains the largest
single lever.

## Recommended gate-standard input

Keep the 2.5-nat / top-5 adjudication. Report the P150 band as STRUCTURAL
(drift x router gain): teacher-forced hard flips 17 (OFF) / 40 (ON) spread
over prompts and decode positions, consistent with a position-0-resident
1-ULP-class norm/attention-chain drift amplified by the router. The spread
is evidence against single-site patches, not against a chain-wide repair —
lever 2 targets the origin itself and remains live. The next levers, in
order: (1) host-free shadow reconciliation (halves the hard count), (2)
f32-enabling the device attention/norm chain (removes the origin), (3) any
future router-margin widening is a model change, not a backend repair — out
of scope for this row.

## Raw evidence

Committed, in `docs/bench-evidence/kolibri1-readj/` (a sibling directory of
this file):

- `OFF.json`, `ON.json` — the parsed adjudication tables (summary, argmax
  chain, and per-hard-flip route_scores gap records).
- `tf_origin.json` — the L0 drift-origin table under CORRECTED decode
  position labels (see the mapping section; includes decode position 0).
- `tf-p{1,5,4}.log.gz` — the raw CPU-vs-TT teacher-forced stage-dump legs
  (9.3 MB each uncompressed, 2.2 MB gzipped).
- The parsers that turn the logs into every table above:
  `parse_adjudication.py` (`OFF`/`ON.json`), `parse_tf_origin.py`
  (`tf_origin.json`), `parse_overlap.py` (the Q2 overlap table). Rerun
  from the repo root:

  ```sh
  python3 docs/bench-evidence/kolibri1-readj/parse_tf_origin.py
  python3 docs/bench-evidence/kolibri1-readj/parse_overlap.py
  python3 docs/bench-evidence/kolibri1-readj/parse_adjudication.py \
    <arm-log> OFF
  ```

Not committed — the full-gate arm logs are too large: `arm-off.log`
95.7 MB, `arm-on3.log` 97.7 MB (the ON leg that reproduces the clean ON
counts exactly), `arm-on.log` 22.0 MB, `arm-on2.log` 3.4 MB. They remain
under `/tmp/readj/` on the measurement host and are regenerable with the
run recipe in `## Runs` plus the `parse_adjudication.py` invocations in
its docstring.

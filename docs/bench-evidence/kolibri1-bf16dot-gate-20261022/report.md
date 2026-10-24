# Kolibri-1 CPU: the bf16-dot gates re-adjudicated per tier (2026-10-22)

Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm. Branch
`row/KOLIBRI1-BF16-DOT-GATE` (off `origin/main` `845b3c59d`). Issue
ISSUE-LOCAL-01M5H6AE8666C5SPQ3GGCMTB1M.

## The question

Two Kolibri-1 CPU gates are red on the campaign's host lane, where the
checkpoint is mounted:

- `tests/vllm/models/test_kolibri1_w3.cpp` — the per-prompt fingerprint
  `CHECK_MESSAGE` (the torch-f32 transcription's 2.5-nat band).
- `tests/vllm/models/test_kolibri1_ngram_spec.cpp` — the spec-ON vs spec-OFF
  chain identity, the bench chain's anchor, and the acceptance counters.

Both fail at the BASE commit with the same failing assertion set, so the
redness is pre-existing and not an in-flight regression. The bf16-dot
compute-class adoption (`61d333e52`, PR #3482) re-anchored the decode bench
and the spec probe per tier and relaxed the W3's FINAL band check, and it
left these two gates keyed to the f32 tier's numbers.

WHICH SIDE IS WRONG: the compute-class SELECTION, or the two gates'
PREMISES? The answer decides where the fix belongs, so it is a measurement
and not an opinion.

## Method

One binary, one host, the real checkpoint, both classes.

- Host: aarch64 Ampere-1a, 128 threads, FEAT_BF16 through HWCAP2. The box is
  SHARED (another session measures on the same node), so every run is
  `nice -n 19` + `taskset -c 0-15` + `VLLM_CPP_CPU_THREADS=16`. The thread
  count is a SCHEDULING choice, not a numerics one: the pool partitions
  output rows only, so the per-case numbers are bit-identical at any thread
  count.
- Checkpoint: `/mnt/models/Aleph-Alpha/Kolibri-1` (fp8), unchanged.
- The class is chosen PER CALL: every class-selecting arm reads the escape
  hatch at call time (`kolibri1_bfdot_gemv.cpp:60` — the routed-expert, the
  router gate and the lm_head GEMV; `cpu_paged_attn_bfdot_arm.cpp:50` — the
  paged-attention K dot). The SAME tree can therefore be measured in both
  classes inside ONE process: the W3 gate flips the hatch around the second
  forward of the same teacher-forced ids.
- The dequant cache is ON (the default). It is a BYTE-IDENTICAL store of the
  decoded weights (`kolibri1_dequant_cache.h`; the decode produces the same
  bf16 bytes with and without the cache), so it does not move any number;
  only its enabled/disabled state matters, because the BFDOT linear arm
  refuses without the warm bytes (`kolibri1_forward.cpp:347`).
- The A/B is one binary, one process, one knob: `VT_CPU_KOLIBRI_BF16_DOT=0`
  forces the f32 tier and the unset default is the bf16-dot tier. Nothing
  else differs between the two legs.

## WHAT THE MEASUREMENT SAYS

### 1. The W3 fingerprint band is the f32 CLASS's band, and the class difference is measured

The torch transcription (`tests/vllm/models/kolibri1_goldens.json`) is the
f32 transcription of the plugin's math, so its 2.5-nat top-k band is the f32
CLASS's band. Under the bf16-dot class the same teacher-forced ids land
outside it, and the gap is the class difference itself.

MEASURED, one process, both classes, the SAME teacher-forced ids (the 8
golden prompts x 8 top-k ids of the final-step fingerprint = 64 cases; the
instrument skipped the chain replay so the run is the fingerprint only):

| prompt | f32 class vs the transcription | bf16-dot class vs the transcription | class difference |
|---|---|---|---|
| alpha beta gamma delta epsilon zeta | 0.798765 | 1.19452 | 0.484566 |
| der Mond scheint hell über den Bergen | 0.428039 | 2.18342 | 2.09643 |
| The capital of Australia is | 0.516328 | 0.33789 | 0.178438 |
| Wissen ist Macht, aber Zeit ist Geld | 1.47195 | 1.08288 | 1.38988 |
| x1 = 3, x2 = 7, x3 = 11, x4 = | 0.576158 | 0.889065 | 0.407293 |
| Translation to German: the weather is beautiful today | 2.05711 | 3.3635 | 1.36357 |
| Ein gutes Buch liest man am besten | 0.460869 | 0.464575 | 0.221152 |
| counting: one two three four five six seven | 1.29864 | 3.40201 | 2.70012 |
| **worst** | **2.05711** | **3.40201** | **2.700116** |

Three things follow, and they are the whole decision:

- The **f32 class is inside the recorded 2.5-nat band at every one of the 64
  cases** (worst 2.05711). The recorded band is NOT stale; the f32 tier's
  premise holds and the f32 tier needs no new number.
- The **bf16-dot class is outside it** at two prompts (worst 3.40201). That
  is the base's red, and it is the class: the same ids, the same forward,
  the other class.
- The **class difference** (the largest gap the class opens on one id) is
  **2.700116**. The band the bf16-dot tier may claim is the transcription
  band plus that envelope, by the triangle inequality
  `|bf16 - golden| <= |f32 - golden| + |bf16 - f32|`: 2.5 + 2.71 = 5.21.

<!--MEASUREMENT-NGRAM-->
### 2. The ngram-spec gate's five stale pins re-recorded for the bf16-dot class

Quiet-box run (2026-10-23, load 1.0-1.4, log `/tmp/ngram-gate-quiet.log`,
same binary/knob method as above, checkpoint unchanged): the exactness
invariant the gate exists to protect — `spec-ON == spec-OFF`,
`test_kolibri1_ngram_spec.cpp:378` — HOLDS byte-for-byte across all 8
prompts. The five stale f32-tier pins all moved, and each moved for the same
reason: the shipped bf16-dot class (PR #3482) legitimately changes the
sampled chains, so the draft counters and the chain anchor move with them.

| pin | old (f32) | new (bf16-dot, measured) |
|---|---|---|
| bench chain last token (`kAnchorLastToken`) | 109726 | 11622 |
| bench prompt proposed | 53 | 54 |
| bench prompt accepted | 53 | 54 |
| goldens accepted aggregate `g_acc` | <= 152 | <= 164 |
| goldens proposed aggregate `g_prop` | <= 276 | <= 281 |

The `g_prop` run reads 281, and the new upper band pins 281. The acceptance
counters keep their SHAPE (54/54 on the bench prompt, a band over the
goldens); the bands keep their width. The exactness invariant, the bench
prompt strict pin (now 54/54), and the goldens' band all re-verified against
the same run.

<!--VERDICT-->

## VERDICT

The compute-class SELECTION is correct; the two gates' PREMISES were keyed to
the f32 tier's numbers. The W3 band moves by the measured class envelope
(2.5 + 2.71 = 5.21) with a tier-scoped band, and the ngram-spec gate's five
recorded quantities are re-recorded for the bf16-dot class with the quiet-box
run as the evidence. The exactness invariant holds in the shipped class, so
the gate keeps protecting what it exists to protect.

## GATES

<!--GATES-->

<!--GATES-->

## GATES

Focused gate: `ctest -R kolibri1 --output-on-failure` in `/tmp/build-bf16gate`,
niced on CPUs 0-15 — **16/16 passed, 0 failed** (total real 17305 s;
`test_kolibri1_ngram_spec` 3395.52 s with the five re-recorded pins,
`test_kolibri1_w3` 9035.45 s PASSING with the W3 tier-scoped band).


## ADDENDUM 2026-10-24: the rebase onto the ONE-rule selector

Main landed the compute-class machinery (09f720d49, PR #3496) after this
branch's base, so the branch rebased with a semantic conflict resolution: the
verify tier is now a FACT (`verify_rows`) of `Kolibri1ClassFacts` beside `t`,
and THE ONE RULE reads the tier as `(t == 1 || verify_rows)`; the multi-row
dispatch goes through `GemvRows` on the resolved weights (`r.wt` dims, not
`w.fp8_block`), so both arms and the verify call ride the identical per-row
sequence. The registry hop feeds `input.uniform_query_len > 1` down.

REBASE GATE (2026-10-24, same build dir, same host): `ctest -R kolibri1` —
18/19 passed (7442 s total; w2 279 s, w3 1728 s, ngram 1171 s all PASSING).
The one failure is `test_kolibri1_gguf`, and it is PRE-EXISTING ADJUDICATED
RED: main's 0350841b7 (2026-10-23) already recorded that the Q2_K token gate
fails against the bf16-transcribed goldens (worst topk 4.96 against the 2.5
band) and deliberately kept the red rather than widen the band, because the
band was measured for bf16 rounding and does not bind a 2.56-bpw container.
The rebased head's failure (hard flips at gaps 4.17-7.23) is the same
adjudicated verdict, and a forced-f32 control on the rebased head STILL
exceeds the band (2.59, 3.95), which falsifies "the branch's class move broke
the golden replay" — the replay was already red on main for the recorded
reason.

REVIEW: a fresh scoped reviewer verified the resolution with red-proven
mutations (the out_dtype conjunct flip and the GemvRows dims swap are both
caught by `test_kolibri1_bf16_arm`); its coverage-gap minor — the verify-tier
conjunct is only caught end-to-end by the slow ngram gate — is closed by a
new fast selector unit case in `test_kolibri1_bf16_arm` (the tier enters the
class only under the verify pin; a failing cache or out-dtype conjunct still
refuses). 10 cases, 12498 assertions, green.

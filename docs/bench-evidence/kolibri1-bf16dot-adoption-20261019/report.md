# Kolibri-1 bf16-dot CPU adoption: default-flip gate evidence (2026-10-19)

Row: MODEL-TEXT-kolibri-1 (CPU lane). Branch `row/kolibri-bf16-dot-cpu`
(fresh from `origin/main` `4b3a0091c`; spec
`.agents/specs/kolibri1-bf16-dot-cpu.md` committed before implementation).

## Question

Does flipping the kolibri1 bf16-dot arms (routed-expert GEMV, paged-attention
decode K dot, lm_head / router gate) from opt-in knobs to the hardware-gated
DEFAULT on aarch64 FEAT_BF16 hold the token gate (top-1 agreement with the
primary oracle must not drop) and what does the decode GEMV leg cost or save
in the production shape?

## Method

One binary, same-tree A/B; the f32 leg is the escape hatch
`VT_CPU_KOLIBRI_BF16_DOT=0`, the bf16 leg is the unset default. Host:
aarch64 Ampere-1a (128 threads, FEAT_BF16 via HWCAP2), contended shared box —
the A/B is same-binary and interleaved-legs-in-one-session, but the absolute
numbers are contended-host numbers; only the RATIO is evidence. Weights:
`/mnt/models/Aleph-Alpha/Kolibri-1` fp8 checkpoint, unchanged from the
previous reports. `VT_KOLIBRI1_DEQUANT_CACHE_MB=16384` on BOTH legs so the
GEMV arm's warm-bf16 residency precondition holds on both sides and the only
difference is the compute class.

## Token gate (the red line)

`tests/vllm/models/test_kolibri1_threeway_capture.cpp` dumps, and
`scripts/kolibri1-threeway-analysis.py` scores, the same 202 aligned
positions against the primary oracle's published top-100 rows
(`kolibri1_primary_topk_rows.json`):

| mode | top-1 | top-5 | top-100 | gap mean (nats) | cos mean | KL mean (nats) |
|---|---|---|---|---|---|---|
| BEFORE f32 (`=0`) | 97.0% | 100.0% | 100.0% | 5.842 | 0.9812 | 6.994 |
| AFTER bf16-dot default | **98.0%** | 100.0% | 100.0% | 5.854 | 0.9804 | 7.018 |

AFTER >= BEFORE: the gate holds and moves TOWARD the primary, as the spec
predicted from the C-draft tie. Depth-growth: the gain concentrates in the
6-15 bin (96.8% -> 98.4%) and the 16-31 bin (98.0% -> 99.0%).

## Decode bench (production shape, 128-in / 63 greedy steps)

`tests/test_kolibri1_decode_bench`, 3 runs per leg, median:

| leg | decode s (median) | tok/s (median) | runs |
|---|---|---|---|
| f32 (`=0`) | 14.068 | 4.478 | 4.453 / 4.478 / 4.287 |
| bf16-dot default | 11.948 | 5.273 | 5.201 / 5.284 / 5.273 |

**+17.7% decode tok/s** on the flipped GEMV leg (1.177x), consistent with the
C draft's historical +21% micro measurement. Chains: f32 constant 33382
(3/3), bf16 constant 11622 (3/3) — both tier anchors re-recorded in the bench
gate.

## W3 torch-golden adjudication

- f32 leg (`=0`): 141/145 chain matches, 4 near-tie flips, 0 hard;
  fingerprint band green — bit-identical to main's production gate.
- bf16-dot default: 134/138 chain matches, 4 near-tie flips, 0 hard; the
  torch-f32 topk fingerprint band (2.5 nats) no longer binds (worst 3.40) —
  the fingerprints are the torch f32 transcription, so the default tier is
  adjudicated against the PRIMARY artifacts above, per the spec's Design
  section. The W3 gate is now tier-scoped, not relaxed.

## spec probe

The probe's anchor was stale on main (expected 109726, the f32 tier's chain
has been 33382 since the FMA-contract re-anchor — red before this change;
hatch run 2026-10-19 shows 33382). Re-anchored per tier. spec-vs-plain
byte-identity holds 8/8 on the f32 tier; the bf16-dot default measures 6/8
(golden[3], golden[5] — the two knife-edge prompts, where the plain t==1
BFDOT arms and the batched verify rows are different numerics), recorded as
the tier floor.

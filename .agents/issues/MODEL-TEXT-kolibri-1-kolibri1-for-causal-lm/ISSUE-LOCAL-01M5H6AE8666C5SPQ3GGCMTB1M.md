ID: ISSUE-LOCAL-01M5H6AE8666C5SPQ3GGCMTB1M
Title: kolibri1 CPU: the bf16-dot adoption left test_kolibri1_w3 and test_kolibri1_ngram_spec red on main
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-22
Updated: 2026-10-22
Closed: -

## Problem

The bf16-dot compute-class adoption (61d333e52, PR #3482) relaxed only the
FINAL band check and left two per-case assertions binding:

- `tests/vllm/models/test_kolibri1_w3.cpp:371` — the per-prompt fingerprint
  `CHECK_MESSAGE(diff <= kLogitBand)`. The 2.5-nat band is the torch f32
  transcription's band, which is the f32 CLASS's band; under the bf16-dot
  DEFAULT the class moves our logits past it. The rest of the gate
  reproduces its recorded numbers exactly (134/138 positions match, 4
  near-ties, 0 hard flips, worst topk diff 3.40201).
- `tests/vllm/models/test_kolibri1_ngram_spec.cpp:378` — the spec-ON vs
  spec-OFF chain identity, plus the bench chain's anchor (109726, STALE for
  BOTH tiers since the FMA-contract paged-attention re-anchor of 2026-10-10:
  the f32 tier's chain has ended at 33382 since, which is what the decode
  bench and the spec probe pin) and the acceptance counters (the f32 tier's
  53/53 on the bench prompt and its 146/270 over the goldens).

Both fail at the base commit 845b3c59d with an IDENTICAL error set, verified
by building the base in a scratch tree and diffing the failing assertion
sets, so the redness is pre-existing and not an in-flight regression. The
compute class is selected by the FEAT_BF16 probe and by the single escape
hatch `VT_CPU_KOLIBRI_BF16_DOT=0`, which forces the f32 tier.

The question the record must answer is which side is wrong: either the class
selection must be pinned so the ON and OFF arms select the same class, or
the identity assertion must be adjudicated against the measured class
difference rather than deleted. It is a measurement, and the measurement
decides where the fix belongs.

## Resolution

- RESOLVED 2026-10-24, quiet-box run 2026-10-23 (load 1.0-1.4, log
  `/tmp/ngram-gate-quiet.log`, evidence
  `docs/bench-evidence/kolibri1-bf16dot-gate-20261022/report.md`).
- The measurement adjudicates: the compute-class SELECTION is correct and the
  two gates' f32-tier premises were stale. The W3 gate takes a tier-scoped
  band (the measured class difference 2.700116 gives the 2.71 envelope;
  worst bf16-dot case 3.40201 inside 2.5 + 2.71 = 5.21). The ngram-spec gate's
  five recorded quantities re-record to the measured bf16-dot values
  (anchor 11622, bench 54/54, goldens 164/281); the exactness invariant
  `spec-ON == spec-OFF` HOLDS byte-for-byte, so the gate keeps protecting
  what it exists to protect. Nothing deleted; every pin keeps its assertion
  shape.


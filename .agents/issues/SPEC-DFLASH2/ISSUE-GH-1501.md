ID: ISSUE-GH-1501
Title: **The row recorded DFlash2's T>0 walk as "inverse CDF" from its opening brief; upstream draws GUMBEL-MAX at BOTH pull-request heads it has cited.** `.agents/specs/dflash2-spec-decode.md` `## Upstream chain` mechanism 2 said "At T>0 the walk is by inverse CDF and returns q over the K candidates", and the `SPEC-DFLASH2` engine-matrix row said the same. At `19c9351904df4c63042671bc67a866ca48dc7d6f` the non-greedy branch of `_selector_walk_kernel` drew `uniform = tl_rand32(gumbel_seed, candidates, includes_zero=False)`, `noise = -tl.log(-tldevice.log1p(-uniform))` and took the argmax of `scores / temperature + noise` — Gumbel-max; at `66e5414c6d75a8529473d977f7458c140bbab8a0` that branch is replaced by one call to `gumbel_noised_argmax`, the same draw. No inverse-CDF walk exists upstream. BLAST RADIUS is a mis-scoped wave rather than a shipped defect: W4 ships the GREEDY arm, which is byte-for-byte upstream's `SAMPLE_PROBABILISTIC=False` arm, and the noised arm is not ported (`ParseSpeculativeConfigJson` refuses `draft_sample_method: "probabilistic"` by name against an accept-iff-equal verify) — but an implementer scoped to write an inverse-CDF walk would have written the wrong algorithm with no oracle to catch it, because the acceptance gate that would notice is itself owed. FIXED IN THE SAME FLOW by W4: both records corrected in place rather than annotated, `## Risks/decisions` D13 records why the noised arm is unreachable here, and `## Owed` O12 records its layout and the Triton Philox stream (`tl.randint4x`, Philox 4x32-10, keyed by the candidate token ids) a bit-parity port would need. Also corrected there: the six-item enumeration of the speculator head move was verified item by item against the two blobs and is correct, but it was missing a SEVENTH change in the base class (`DraftModelSpeculator.__init__` now calls a virtual `draft_logits_spec`) and the `+24 / -6` file it lives in was absent from the delta table. Listed under `## Risks/decisions` D13 in [dflash2-spec-decode.md](../../specs/dflash2-spec-decode.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1501
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:531`

### Frozen archive evidence

> | [#1501](https://github.com/mudler/vllm.cpp/issues/1501) | `SPEC-DFLASH2` | **The row recorded DFlash2's T>0 walk as "inverse CDF" from its opening brief; upstream draws GUMBEL-MAX at BOTH pull-request heads it has cited.** `.agents/specs/dflash2-spec-decode.md` `## Upstream chain` mechanism 2 said "At T>0 the walk is by inverse CDF and returns q over the K candidates", and the `SPEC-DFLASH2` engine-matrix row said the same. At `19c9351904df4c63042671bc67a866ca48dc7d6f` the non-greedy branch of `_selector_walk_kernel` drew `uniform = tl_rand32(gumbel_seed, candidates, includes_zero=False)`, `noise = -tl.log(-tldevice.log1p(-uniform))` and took the argmax of `scores / temperature + noise` — Gumbel-max; at `66e5414c6d75a8529473d977f7458c140bbab8a0` that branch is replaced by one call to `gumbel_noised_argmax`, the same draw. No inverse-CDF walk exists upstream. BLAST RADIUS is a mis-scoped wave rather than a shipped defect: W4 ships the GREEDY arm, which is byte-for-byte upstream's `SAMPLE_PROBABILISTIC=False` arm, and the noised arm is not ported (`ParseSpeculativeConfigJson` refuses `draft_sample_method: "probabilistic"` by name against an accept-iff-equal verify) — but an implementer scoped to write an inverse-CDF walk would have written the wrong algorithm with no oracle to catch it, because the acceptance gate that would notice is itself owed. FIXED IN THE SAME FLOW by W4: both records corrected in place rather than annotated, `## Risks/decisions` D13 records why the noised arm is unreachable here, and `## Owed` O12 records its layout and the Triton Philox stream (`tl.randint4x`, Philox 4x32-10, keyed by the candidate token ids) a bit-parity port would need. Also corrected there: the six-item enumeration of the speculator head move was verified item by item against the two blobs and is correct, but it was missing a SEVENTH change in the base class (`DraftModelSpeculator.__init__` now calls a virtual `draft_logits_spec`) and the `+24 / -6` file it lives in was absent from the delta table. Listed under `## Risks/decisions` D13 in [dflash2-spec-decode.md](../../specs/dflash2-spec-decode.md) | bug |

## Resolution

-

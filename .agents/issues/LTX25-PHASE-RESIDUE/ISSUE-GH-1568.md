ID: ISSUE-GH-1568
Title: `denoise.step` and `denoise.update` are open to a seconds transfer that no assertion in `test_ltx2_video.cpp` can see: (1b') compares `start_seconds` only, and no per-part floor separates the honest share (0.45% to 11.15% across four boxes) from a transfer (~0%). Leaving `denoise.step` open across the post-process and emitting `denoise.update` empty after it preserves the alternation, both counters, containment, non-overlap, exclusivity, (1c) and (2), while moving 100% of the decomposed seconds onto one name. Found by the fresh review of [#1536](https://github.com/mudler/vllm.cpp/issues/1536); `LTX25-PHASE-RESIDUE` claimed it closed and WITHDREW the claim when a reviewer measured it. Closing it needs an anchor INSIDE the callee. Listed under `## Owed` in [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md)
Row: LTX25-PHASE-RESIDUE
State: UNKNOWN
Kind: bug
GitHub: 1568
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:604`

### Frozen archive evidence

> | [#1568](https://github.com/mudler/vllm.cpp/issues/1568) | `LTX25-PHASE-RESIDUE` | `denoise.step` and `denoise.update` are open to a seconds transfer that no assertion in `test_ltx2_video.cpp` can see: (1b') compares `start_seconds` only, and no per-part floor separates the honest share (0.45% to 11.15% across four boxes) from a transfer (~0%). Leaving `denoise.step` open across the post-process and emitting `denoise.update` empty after it preserves the alternation, both counters, containment, non-overlap, exclusivity, (1c) and (2), while moving 100% of the decomposed seconds onto one name. Found by the fresh review of [#1536](https://github.com/mudler/vllm.cpp/issues/1536); `LTX25-PHASE-RESIDUE` claimed it closed and WITHDREW the claim when a reviewer measured it. Closing it needs an anchor INSIDE the callee. Listed under `## Owed` in [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md) | bug |

## Resolution

-

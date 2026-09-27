ID: ISSUE-GH-1470
Title: **`test_ltx2_video` false-redded once on `main` under load and the failing case's identity was never captured.** The prediction it records (F12 of #1441's fifth review) was that a coverage floor whose per-boundary sampling cost is fixed while the leaf shrinks is a false-RED risk and never a false pass. Measured by `LTX25-PHASE-RESIDUE`: the prediction is right about the polarity and WRONG about the term. The `denoise` coverage miss is not sampling cost, it is the sampler's post-process and Euler step, which no anchor wrapped and which scale with the latent -- 49 us per step at nine frames against 343 us at 81, in ONE run of one binary, and instrument cost does not move 7x with the latent. **STILL OPEN, and the correction above is the whole of what this row establishes.** The row's branch anchored that work as `denoise.update` and [#1556](https://github.com/mudler/vllm.cpp/issues/1556) is CLOSED rather than merged, so nothing anchors it on `main` today; the anchor is owed under [#1668](https://github.com/mudler/vllm.cpp/issues/1668). An earlier draft of this row said the issue was closed by anchoring, which was true of the branch and never of the tree. Spec [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md)
Row: LTX25-PHASE-RESIDUE
State: UNKNOWN
Kind: bug
GitHub: 1470
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:601`

### Frozen archive evidence

> | [#1470](https://github.com/mudler/vllm.cpp/issues/1470) | `LTX25-PHASE-RESIDUE` | **`test_ltx2_video` false-redded once on `main` under load and the failing case's identity was never captured.** The prediction it records (F12 of #1441's fifth review) was that a coverage floor whose per-boundary sampling cost is fixed while the leaf shrinks is a false-RED risk and never a false pass. Measured by `LTX25-PHASE-RESIDUE`: the prediction is right about the polarity and WRONG about the term. The `denoise` coverage miss is not sampling cost, it is the sampler's post-process and Euler step, which no anchor wrapped and which scale with the latent -- 49 us per step at nine frames against 343 us at 81, in ONE run of one binary, and instrument cost does not move 7x with the latent. **STILL OPEN, and the correction above is the whole of what this row establishes.** The row's branch anchored that work as `denoise.update` and [#1556](https://github.com/mudler/vllm.cpp/issues/1556) is CLOSED rather than merged, so nothing anchors it on `main` today; the anchor is owed under [#1668](https://github.com/mudler/vllm.cpp/issues/1668). An earlier draft of this row said the issue was closed by anchoring, which was true of the branch and never of the tree. Spec [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md) | bug |

## Resolution

-

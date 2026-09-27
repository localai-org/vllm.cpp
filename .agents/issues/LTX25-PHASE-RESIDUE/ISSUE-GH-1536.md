ID: ISSUE-GH-1536
Title: **`test_ltx2_video` was persistently red rather than load-flaky, and it was the only failing test on `main`.** The issue asks for the residue to be DECOMPOSED rather than re-argued, and names `d995c52f0` (the temporal x2 upsampler) as the first hypothesis to test. The decomposition REFUTES that hypothesis and settles the cause: **92% of the un-named time is one region**, `Ltx2VideoEngine::Load` from the timeline's origin to `Open("load.dit")` -- 17.661 ms of a 19.178 ms residue -- while the upsampler's own work sits inside `phase.upsample_latent`, a named leaf that does not appear in the residue at all. The remaining four gaps are 4.95%, 1.30%, 1.09% and 0.56%, and the sixteen gaps between adjacent named phases hold 6.8 us each, which is the instrument and nothing else. **The RED is gone and the issue is NOT.** `519303d15` ([#1622](https://github.com/mudler/vllm.cpp/pull/1622), row `LTX25-DEVICE-RESIDENCY`) names that same 92% region `load.open` -- same open point, same close point, same `Scope::Close` shape as this row's `load.setup` -- so the sum floor is repaired on `main` by another row, and the coverage floor is repaired by `6b48edb2c` moving it to 0.75, which that change's own comment describes as a holding action in substance, though not in those words. What is owed is the naming that would make a tight floor honest again, under [#1668](https://github.com/mudler/vllm.cpp/issues/1668). Spec [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md)
Row: LTX25-PHASE-RESIDUE
State: UNKNOWN
Kind: bug
GitHub: 1536
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:602`

### Frozen archive evidence

> | [#1536](https://github.com/mudler/vllm.cpp/issues/1536) | `LTX25-PHASE-RESIDUE` | **`test_ltx2_video` was persistently red rather than load-flaky, and it was the only failing test on `main`.** The issue asks for the residue to be DECOMPOSED rather than re-argued, and names `d995c52f0` (the temporal x2 upsampler) as the first hypothesis to test. The decomposition REFUTES that hypothesis and settles the cause: **92% of the un-named time is one region**, `Ltx2VideoEngine::Load` from the timeline's origin to `Open("load.dit")` -- 17.661 ms of a 19.178 ms residue -- while the upsampler's own work sits inside `phase.upsample_latent`, a named leaf that does not appear in the residue at all. The remaining four gaps are 4.95%, 1.30%, 1.09% and 0.56%, and the sixteen gaps between adjacent named phases hold 6.8 us each, which is the instrument and nothing else. **The RED is gone and the issue is NOT.** `519303d15` ([#1622](https://github.com/mudler/vllm.cpp/pull/1622), row `LTX25-DEVICE-RESIDENCY`) names that same 92% region `load.open` -- same open point, same close point, same `Scope::Close` shape as this row's `load.setup` -- so the sum floor is repaired on `main` by another row, and the coverage floor is repaired by `6b48edb2c` moving it to 0.75, which that change's own comment describes as a holding action in substance, though not in those words. What is owed is the naming that would make a tight floor honest again, under [#1668](https://github.com/mudler/vllm.cpp/issues/1668). Spec [`ltx25-phase-residue.md`](../../specs/ltx25-phase-residue.md) | bug |

## Resolution

-

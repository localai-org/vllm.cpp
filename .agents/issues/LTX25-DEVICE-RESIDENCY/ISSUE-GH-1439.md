ID: ISSUE-GH-1439
Title: `ltx2 video: a render through the ABI emits a phase table that SUMS to wall` asserts `CHECK(leaves >= 0.95 * wall)` (`tests/vllm/multimodal/test_ltx2_video.cpp:2854`) and is **RED on `origin/main`**, not on the branch that found it. MEASURED on one x86 box, one build directory, `CMAKE_BUILD_TYPE` empty as CI has it, with this lane's four files reverted so the binary IS main at `89261c955`: six in-suite runs read 94.32%, 95.20%, 93.74%, 94.20%, 94.69% and one `-tc` run 94.19% - **five of six red**, 93 cases / 3321 assertions / 1 failed, exit 1. The W0-live merge reads 93.82%, 93.68%, 94.34%, 94.62%, 94.39% in-suite (96 cases / 3555 assertions / 1 failed) and 94.12% with `VLLM_RENDER_PROGRESS=0`, so neither the new emitter nor its call site is the term; one `-tc` run passed at 95.40%. It is NOT box contention, and the run that disproves that is main's single green: it had `wall=0.579684s`, more than double every other run, because the box was LOADED - a slower render passes, since the un-named residue grows more slowly than the wall it is divided by. The residue is 4.80% to 6.32% of `wall` across all twelve runs (0.0128 s to 0.0278 s against a `wall` of 0.220 s to 0.580 s), so a 95% floor sits INSIDE the measurement's own range at the 64x64 / 9-frame FIXTURE scale and the case decides by coin flip, mostly red. The tolerance was argued for the 21.004 B render, where the same residue would be a far smaller fraction. NOT FIXED IN FLOW, deliberately: naming the un-named time, or bounding `unaccounted_seconds` beside the ratio so the assertion says the same thing at both scales, is a change to a gate's semantics and needs its own row, spec and red-first evidence per `AGENTS.md` "Changing the rules or a checker". Found while merging `origin/main` into `row/LTX25-RESIDENCY-W0-LIVE` after [#1419](https://github.com/mudler/vllm.cpp/pull/1419) was auto-closed by its base branch being deleted. Listed under `## Owed` in [`ltx25-device-residency.md`](../../specs/ltx25-device-residency.md)
Row: LTX25-DEVICE-RESIDENCY
State: UNKNOWN
Kind: bug
GitHub: 1439
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:492`

### Frozen archive evidence

> | [#1439](https://github.com/mudler/vllm.cpp/issues/1439) | `LTX25-DEVICE-RESIDENCY` | `ltx2 video: a render through the ABI emits a phase table that SUMS to wall` asserts `CHECK(leaves >= 0.95 * wall)` (`tests/vllm/multimodal/test_ltx2_video.cpp:2854`) and is **RED on `origin/main`**, not on the branch that found it. MEASURED on one x86 box, one build directory, `CMAKE_BUILD_TYPE` empty as CI has it, with this lane's four files reverted so the binary IS main at `89261c955`: six in-suite runs read 94.32%, 95.20%, 93.74%, 94.20%, 94.69% and one `-tc` run 94.19% - **five of six red**, 93 cases / 3321 assertions / 1 failed, exit 1. The W0-live merge reads 93.82%, 93.68%, 94.34%, 94.62%, 94.39% in-suite (96 cases / 3555 assertions / 1 failed) and 94.12% with `VLLM_RENDER_PROGRESS=0`, so neither the new emitter nor its call site is the term; one `-tc` run passed at 95.40%. It is NOT box contention, and the run that disproves that is main's single green: it had `wall=0.579684s`, more than double every other run, because the box was LOADED - a slower render passes, since the un-named residue grows more slowly than the wall it is divided by. The residue is 4.80% to 6.32% of `wall` across all twelve runs (0.0128 s to 0.0278 s against a `wall` of 0.220 s to 0.580 s), so a 95% floor sits INSIDE the measurement's own range at the 64x64 / 9-frame FIXTURE scale and the case decides by coin flip, mostly red. The tolerance was argued for the 21.004 B render, where the same residue would be a far smaller fraction. NOT FIXED IN FLOW, deliberately: naming the un-named time, or bounding `unaccounted_seconds` beside the ratio so the assertion says the same thing at both scales, is a change to a gate's semantics and needs its own row, spec and red-first evidence per `AGENTS.md` "Changing the rules or a checker". Found while merging `origin/main` into `row/LTX25-RESIDENCY-W0-LIVE` after [#1419](https://github.com/mudler/vllm.cpp/pull/1419) was auto-closed by its base branch being deleted. Listed under `## Owed` in [`ltx25-device-residency.md`](../../specs/ltx25-device-residency.md) | bug |

## Resolution

-

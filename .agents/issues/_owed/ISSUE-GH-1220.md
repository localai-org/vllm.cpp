ID: ISSUE-GH-1220
Title: The two LTX schedule-anchor cases in `tests/vllm/multimodal/test_ltx2_video.cpp` return the REQUEST step count from their render lambda under a comment saying the count is read back out of the render, so the four `REQUIRE(x.steps == rendered_steps)` lines compare four copies of one request field rather than four renders. WEAKENED, not vacuous, and that is why it is filed rather than repaired in flow: the lambda pins the step count at 3 with a stated reason -- a 2-step schedule is `{1, 0.1, 0}` for every token count, so the 4096 anchor cannot reach the trajectory -- and the assertions below recompute `Ltx2SigmaSchedule` at it, so lowering that literal still reds them by name. What it cannot see is an engine that ignored the request and ran a different number of steps. `Ltx2ConditioningTrace::dit_evaluations` (`include/vllm/multimodal/ltx2_video.h:757-774`) already carries the observation, incremented at one site inside the shared `Evaluate` lambda, and is `steps` per phase on the Euler and ancestral arms against `2 * steps + 1` on res_2s. Not repaired during the review repair of [#1096](https://github.com/mudler/vllm.cpp/issues/1096) because it changes what a LANDED case measures on both pipelines and on the res_2s control, so it owes its own red-first evidence that the derived count reproduces the current literal on every arm rather than a quiet re-derivation of a passing assertion. Listed under `## Owed` in [`ltx25-keyframe-interp.md`](../../specs/ltx25-keyframe-interp.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1220
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:391`

### Frozen archive evidence

> | [#1220](https://github.com/mudler/vllm.cpp/issues/1220) | — | The two LTX schedule-anchor cases in `tests/vllm/multimodal/test_ltx2_video.cpp` return the REQUEST step count from their render lambda under a comment saying the count is read back out of the render, so the four `REQUIRE(x.steps == rendered_steps)` lines compare four copies of one request field rather than four renders. WEAKENED, not vacuous, and that is why it is filed rather than repaired in flow: the lambda pins the step count at 3 with a stated reason -- a 2-step schedule is `{1, 0.1, 0}` for every token count, so the 4096 anchor cannot reach the trajectory -- and the assertions below recompute `Ltx2SigmaSchedule` at it, so lowering that literal still reds them by name. What it cannot see is an engine that ignored the request and ran a different number of steps. `Ltx2ConditioningTrace::dit_evaluations` (`include/vllm/multimodal/ltx2_video.h:757-774`) already carries the observation, incremented at one site inside the shared `Evaluate` lambda, and is `steps` per phase on the Euler and ancestral arms against `2 * steps + 1` on res_2s. Not repaired during the review repair of [#1096](https://github.com/mudler/vllm.cpp/issues/1096) because it changes what a LANDED case measures on both pipelines and on the res_2s control, so it owes its own red-first evidence that the derived count reproduces the current literal on every arm rather than a quiet re-derivation of a passing assertion. Listed under `## Owed` in [`ltx25-keyframe-interp.md`](../../specs/ltx25-keyframe-interp.md) | bug |

## Resolution

-

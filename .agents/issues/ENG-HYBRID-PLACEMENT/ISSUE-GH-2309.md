ID: ISSUE-GH-2309
Title: **The fp4-resident MoE refusal was lost when W3c moved every architecture onto the shared placement seam.** `RunMoeBlockPlaced` refused the arm; the refactor left that helper dead and the live `RunMoePlaced` path accepted it. Placing an fp4-resident arm uploads every expert at load and then computes on the host across the bus, so it is SLOWER than not placing — and **a token gate cannot see it**, because the tokens stay correct and only the placement is wrong. Refusal restored as a `placeable` / `unplaceable_reason` contract on the seam itself rather than in each caller, so a newly wired architecture inherits it; callers pass `layer.moe.expert_gate_fp4.empty()`. It fires only when a placement is in force (`placed_on != engine_device`), leaving an ordinary unplaced load untouched, since a guard that fired there would break every load. Proved by a COMPILING mutation: with the guard rewritten never to fire, `test_device_placement` goes red at 1 case / 2 assertions. The first mutation attempt failed to compile under `-Werror` on the unused parameters and the stale binary reported 19/19 SUCCESS, which is a passing mutant proving nothing — the mutant build's rc=0 is part of the evidence. Found while gating W3c, fixed in the same flow. Spec [`specs/hybrid-placement.md`](../../specs/hybrid-placement.md) §W3d
Row: ENG-HYBRID-PLACEMENT
State: UNKNOWN
Kind: bug
GitHub: 2309
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:893`

### Frozen archive evidence

> | [#2309](https://github.com/mudler/vllm.cpp/issues/2309) | `ENGINE-HYBRID-PLACEMENT` | **The fp4-resident MoE refusal was lost when W3c moved every architecture onto the shared placement seam.** `RunMoeBlockPlaced` refused the arm; the refactor left that helper dead and the live `RunMoePlaced` path accepted it. Placing an fp4-resident arm uploads every expert at load and then computes on the host across the bus, so it is SLOWER than not placing — and **a token gate cannot see it**, because the tokens stay correct and only the placement is wrong. Refusal restored as a `placeable` / `unplaceable_reason` contract on the seam itself rather than in each caller, so a newly wired architecture inherits it; callers pass `layer.moe.expert_gate_fp4.empty()`. It fires only when a placement is in force (`placed_on != engine_device`), leaving an ordinary unplaced load untouched, since a guard that fired there would break every load. Proved by a COMPILING mutation: with the guard rewritten never to fire, `test_device_placement` goes red at 1 case / 2 assertions. The first mutation attempt failed to compile under `-Werror` on the unused parameters and the stale binary reported 19/19 SUCCESS, which is a passing mutant proving nothing — the mutant build's rc=0 is part of the evidence. Found while gating W3c, fixed in the same flow. Spec [`specs/hybrid-placement.md`](../../specs/hybrid-placement.md) §W3d | bug |

## Resolution

-

ID: ISSUE-GH-1667
Title: **Both DFlash2 arms judge the LEGS before the arm record reaches disk, so one 63-token prompt discards ~2h of leased evidence.** `require_no_reasons(hook_reasons(...) + leg_reasons(...))` sits after the clock window and before the write: in `capture()` on the oracle arm, whose `main()` writes only after `capture()` returns, and in `main()` on ours. A non-clock refusal therefore takes `records`, `blocks`, `output_token_ids` and the O26 provenance with it. Proven by execution through the existing `OurArmRunEntryPointTest` fixture at `completion = 63`: `arm JSON on disk: ABSENT` against the control run's `PRESENT`. SEPARATE FROM #1657 and not a regression of it -- #1657 moved the CLOCK judgement after the write on exactly this reasoning and both arms carry the comment saying so, but that guarantee is scoped to the clock and says nothing about the leg and hook checks, while a leg refusal is the likelier one. `git log -S` places both call sites at `208559a79` (#1653), predating #1657 and predating the branch that found it. NOT fixed in flow: it changes the refusal ordering of both arms and wants its own red-first test per arm. Owned by [`dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md) `## Owed` O31
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1667
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:614`

### Frozen archive evidence

> | [#1667](https://github.com/mudler/vllm.cpp/issues/1667) | `SPEC-DFLASH2` | **Both DFlash2 arms judge the LEGS before the arm record reaches disk, so one 63-token prompt discards ~2h of leased evidence.** `require_no_reasons(hook_reasons(...) + leg_reasons(...))` sits after the clock window and before the write: in `capture()` on the oracle arm, whose `main()` writes only after `capture()` returns, and in `main()` on ours. A non-clock refusal therefore takes `records`, `blocks`, `output_token_ids` and the O26 provenance with it. Proven by execution through the existing `OurArmRunEntryPointTest` fixture at `completion = 63`: `arm JSON on disk: ABSENT` against the control run's `PRESENT`. SEPARATE FROM #1657 and not a regression of it -- #1657 moved the CLOCK judgement after the write on exactly this reasoning and both arms carry the comment saying so, but that guarantee is scoped to the clock and says nothing about the leg and hook checks, while a leg refusal is the likelier one. `git log -S` places both call sites at `208559a79` (#1653), predating #1657 and predating the branch that found it. NOT fixed in flow: it changes the refusal ordering of both arms and wants its own red-first test per arm. Owned by [`dflash2-spec-decode.md`](../../specs/dflash2-spec-decode.md) `## Owed` O31 | bug |

## Resolution

-

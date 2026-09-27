ID: ISSUE-GH-1298
Title: `scripts/agent-integration.py` CANNOT RUN. `AGENTS.md` §Commands names it as the pre-merge gate, but `main()` calls `cutover_oid()` before it reads anything and that function loads `.agents/policy-cutover`, DELETED by `0f3e44eee` ("policy: the code is the state, git is the history", 2026-08-09) and never restored -- so every invocation exits `INTEGRATION FAILED: missing policy cutover` whatever the tree, the pull request or the trailers hold. Measured at `origin/main` `489a9a4c0`. `scripts/agent-integration.py:107` is the tree's ONLY `--cutover` caller, and it sits behind that raise; `.agents/specs/fix-trailer-lane-cutover.md` already recorded the file as absent when it rejected `--cutover` as an instrument, but not that the absence bricks the command. NO SUITE IS RED, because `tests/scripts/test_agent_gates.py:205` exercises `cutover_oid` against a SYNTHETIC repository it writes the anchor into: the function is tested, the command is not. Found while `GATE-PR-BODY-TRAILERS` ([#1263](https://github.com/mudler/vllm.cpp/issues/1263)) evaluated it as the home for the pre-merge body check and rejected it on this ground -- wiring a new check there would land it behind a permanent refusal. NOT fixed in flow: the repair is a decision (drop `--cutover` from the command, or restore an anchor and say what value it should hold and why), so it owes its own spec, red-before and reviewer. Owed under `## Owed` in [`gate-pr-body-trailers.md`](../../specs/gate-pr-body-trailers.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1298
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:423`

### Frozen archive evidence

> | [#1298](https://github.com/mudler/vllm.cpp/issues/1298) | — | `scripts/agent-integration.py` CANNOT RUN. `AGENTS.md` §Commands names it as the pre-merge gate, but `main()` calls `cutover_oid()` before it reads anything and that function loads `.agents/policy-cutover`, DELETED by `0f3e44eee` ("policy: the code is the state, git is the history", 2026-08-09) and never restored -- so every invocation exits `INTEGRATION FAILED: missing policy cutover` whatever the tree, the pull request or the trailers hold. Measured at `origin/main` `489a9a4c0`. `scripts/agent-integration.py:107` is the tree's ONLY `--cutover` caller, and it sits behind that raise; `.agents/specs/fix-trailer-lane-cutover.md` already recorded the file as absent when it rejected `--cutover` as an instrument, but not that the absence bricks the command. NO SUITE IS RED, because `tests/scripts/test_agent_gates.py:205` exercises `cutover_oid` against a SYNTHETIC repository it writes the anchor into: the function is tested, the command is not. Found while `GATE-PR-BODY-TRAILERS` ([#1263](https://github.com/mudler/vllm.cpp/issues/1263)) evaluated it as the home for the pre-merge body check and rejected it on this ground -- wiring a new check there would land it behind a permanent refusal. NOT fixed in flow: the repair is a decision (drop `--cutover` from the command, or restore an anchor and say what value it should hold and why), so it owes its own spec, red-before and reviewer. Owed under `## Owed` in [`gate-pr-body-trailers.md`](../../specs/gate-pr-body-trailers.md) | bug |

## Resolution

-

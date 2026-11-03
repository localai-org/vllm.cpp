# Spec: ENG-ROLE-DISCIPLINE-988482882

Row: `ENG-ROLE-DISCIPLINE-988482882` · issue
[#3008](https://github.com/mudler/vllm.cpp/issues/3008)

## Scope

Record a permanent `check-role-discipline.py` failure on `main` at `988482882`,
and the rule that prevents another. This row repairs nothing in the tree,
because the defect is a commit message on `main` and `main` is never
force-pushed. It exists so the red is attributable instead of mysterious.

## What is red

```
ERROR: 988482882: repository change (include/vllm/v1/worker/gpu/prepare_inputs.h,
include/vllm/v1/worker/gpu/runner.h, src/vllm/v1/worker/gpu/prepare_inputs.cpp,
src/vllm/v1/worker/gpu/runner.cpp, ... (+2)) reached main without arriving on a
task branch.
```

## Cause

Landing [#2991](https://github.com/mudler/vllm.cpp/pull/2991) the operator
brought `main` into an integration branch with
`git merge --no-ff --no-edit origin/main` and did not amend the result, so it
carries git's default `Merge remote-tracking branch 'origin/main' into land3`.

`arrives_via_row_pr` accepts a merge whose own message matches
`row/[A-Za-z0-9_.-]+` or a PR reference, or whose merged-in commits do. `land3`
is neither, and the second parent `4269469da` is itself
`merge: origin/main into the A2-4 optimistic correction`, which also names
neither. So the merge fails on its message.

**The flagged files did not arrive through this commit.** They came from
`4269469da`, already on `main` before the push. They appear new only because
merging `main` INTO the integration branch makes the integration branch the
first parent, so `main`'s own content shows up on the first-parent walk as
though this commit introduced it.

## Why it is not repaired

The fix is a commit message, and `main` is never force-pushed. Advancing
`scripts/ci-enforcement-floor.txt` past it would make the gate green by widening
its scope to cover the operator's own error, which AGENTS.md forbids and which
this row declines to do. The red stands as visible debt.

## The rule

When merging `main` into an integration branch before pushing, the merge commit
is AUTHORED and names the row branch or PR it lands. `--no-edit` is what fails.
Four merges in the same session were amended and passed; the fifth was not and
did not.

## Blast radius

Stated rather than assumed. A pull request's CI walks `BASE..HEAD` from its own
base, so a branch cut from current `main` does not include this commit. The push
lane on `main` walks from the last gated commit and would include it, and CI does
not run on direct pushes to `main`. Whether any other contributor's run is
reddened by it is NOT established here; if one is, that is the harm and #3008
owns it.

## Owed

- Confirm whether any CI lane other than a direct `main` push actually walks
  across `988482882`, and if one does, decide with the developer whether the
  floor should advance for that reason -- which is a different argument from
  hiding the error, and must be made on its own merits (#3008).

- Triaging `ISSUE-GH-250`: `a5b52047` reached `main` without a task branch and
  `check-role-discipline` could not be waived for that landing. The record
  predates the cutover and carries `Row: -`; the discipline floor it measures is
  this row's own, so the identity is owned here rather than in `_intake`.

## Now

`RECORDED`. No tree change. The gate is red at `988482882` and stays red.

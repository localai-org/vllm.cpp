ID: ISSUE-GH-1130
Title: A merged pull request is necessary before removing a worktree and is NOT sufficient. The worktree for PR [#1035](https://github.com/mudler/vllm.cpp/pull/1035) was reaped while its branch carried two commits that were on neither `main` nor its own remote branch, preserved only by `rescue/es-cuda-grouped-unpushed` (tip `3ca7c23d8`). On a squash-only `main` ancestry carries no information: `git merge-base --is-ancestor` returns false for work that landed perfectly and `git cherry origin/main` marks landed commits `+`. Measured on that case, four instruments gave three different answers, and only `git log @{u}..HEAD` plus a CONTENT check agreed with the truth, which is that all three commits' content DID reach `main` in squash `b493f4981`. So the rule is two-part: verify `@{u}..HEAD` is empty AND that the content reached `main` by `git diff`/`git log -S`, because step 1 alone blocks a safe reap and step 2 alone allows an unsafe one. Fixed in flow in [`workflow.md`](../../workflow.md) `## Isolation`. Two sibling rescue refs remain unadjudicated, `rescue/cuda-breadth-sm75-audit` and `rescue/fp8-native`
Row: ENV-GPU-LEASE-METHODOLOGY
State: UNKNOWN
Kind: bug
GitHub: 1130
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:332`

### Frozen archive evidence

> | [#1130](https://github.com/mudler/vllm.cpp/issues/1130) | `ENV-GPU-LEASE-METHODOLOGY` | A merged pull request is necessary before removing a worktree and is NOT sufficient. The worktree for PR [#1035](https://github.com/mudler/vllm.cpp/pull/1035) was reaped while its branch carried two commits that were on neither `main` nor its own remote branch, preserved only by `rescue/es-cuda-grouped-unpushed` (tip `3ca7c23d8`). On a squash-only `main` ancestry carries no information: `git merge-base --is-ancestor` returns false for work that landed perfectly and `git cherry origin/main` marks landed commits `+`. Measured on that case, four instruments gave three different answers, and only `git log @{u}..HEAD` plus a CONTENT check agreed with the truth, which is that all three commits' content DID reach `main` in squash `b493f4981`. So the rule is two-part: verify `@{u}..HEAD` is empty AND that the content reached `main` by `git diff`/`git log -S`, because step 1 alone blocks a safe reap and step 2 alone allows an unsafe one. Fixed in flow in [`workflow.md`](../../workflow.md) `## Isolation`. Two sibling rescue refs remain unadjudicated, `rescue/cuda-breadth-sm75-audit` and `rescue/fp8-native` | bug |

## Resolution

-

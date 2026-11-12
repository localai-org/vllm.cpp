ID: ISSUE-LOCAL-01M67JAVTJ1RT2ZFRZKYXNTAXW
Title: agent-pr-body.py cannot read a pull request from a fork checkout
Row: GATE-PR-BODY-TRAILERS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-30
Closed: -

## Problem

scripts/agent-pr-body.py derives its repository from the 'origin' remote alone (scripts/agent-pr-body.py:92). On a fork checkout origin is the fork, which carries none of the upstream pull request numbers, so every invocation returns exit 3 REMOTE_UNVERIFIED and the pre-merge body read is unusable for the workflow it was written for. Measured on the phantomic12/vllm.cpp checkout on 2026-10-30: 'agent-pr-body.py --pr 3474' returned rc=3 with 'Could not resolve to a PullRequest with the number of 3474', while the same command with --repo localai-org/vllm.cpp returned rc=0 'OK: commit trailer contract'. gh pr view 3474 --repo phantomic12/vllm.cpp exits 1 and the fork has no pull requests at all, so the failure is repository selection and not a reachability problem. A pull request number is unique only inside one repository, so the tool must ask the repository the pull request actually lands in.

## Resolution

-

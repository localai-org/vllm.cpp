ID: ISSUE-LOCAL-01M5ZEVF7HQ8SFK9F8A9WSREEP
Title: agent-record flaky: strix disk-headroom exhausts on shared runner
Row: CI-AGENT-RECORD-DISK
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-27
Updated: 2026-10-27
Closed: -

## Problem

The agent-record CI job fails intermittently on main (runs 40521291883, 40452744901) and reproduced on the #3513 rerun (job 122404172875): 49 tests in tests.tools.test_strix_vllm_oracle fail with ValueError('disk headroom exhausted') from tools/bench/strix_vllm_oracle/worker.py:257, plus CacheDropError (cache-drop proof retained 6291456 resident bytes). Environmental shared-runner pressure, not owned by a product row. Spec: .agents/specs/ci-agent-record-disk.md. Fix: free runner disk before the tools suite in .github/workflows/ci.yml.

## Resolution

-

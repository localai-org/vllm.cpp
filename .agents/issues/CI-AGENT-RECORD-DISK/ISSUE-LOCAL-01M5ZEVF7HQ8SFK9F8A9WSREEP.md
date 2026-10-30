ID: ISSUE-LOCAL-01M5ZEVF7HQ8SFK9F8A9WSREEP
Title: agent-record flaky: strix disk-headroom exhausts on shared runner
Row: CI-AGENT-RECORD-DISK
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-27
Updated: 2026-10-28
Closed: 2026-10-28

## Problem

The agent-record CI job fails intermittently on main (runs 40521291883, 40452744901) and reproduced on the #3513 rerun (job 122404172875): 49 tests in tests.tools.test_strix_vllm_oracle fail with ValueError('disk headroom exhausted') from tools/bench/strix_vllm_oracle/worker.py:257, plus CacheDropError (cache-drop proof retained 6291456 resident bytes). Environmental shared-runner pressure, not owned by a product row. Spec: .agents/specs/ci-agent-record-disk.md. Fix: free runner disk before the tools suite in .github/workflows/ci.yml.

## Resolution

Landed 2026-10-28 as f484c76ad (PR #3515). The wall was not runner disk pressure on / but the /tmp tmpfs of the ubuntu-26.04 image (actions/runner-images#14777): the strix worker's headroom() measures disk_usage of self.local and self.output, both under /tmp (worker.py:226,256), so the 20 GiB floor was unmeetable on 26.04 runners regardless of how empty / was. Proof the first attempt was insufficient: run 40586147690 job 122477980562 showed 113 GiB free on /dev/root and still failed 49+2 strix tests; build-test-cpu (122477933406) and sanitize-cpu (122477933291) failed identically in the same run. Fix: mask tmp.mount and lazy-unmount /tmp as the first step of the six ubuntu-latest compile/ctest lanes. Verified green: run 40637909804 -- agent-record and build-test-vulkan passed with df -h / /tmp showing /tmp on /dev/root at 88 GiB free.

# SPEC — `CI-AGENT-RECORD-DISK` — free runner disk before the agent-record tools suite

- Owning row: `CI-AGENT-RECORD-DISK`
- Kind: bug (CI infrastructure)
- Issue: created in the same flow (row-scoped local issue)

## Scope

Add a runner disk-reclamation step to the `agent-record` job in
`.github/workflows/ci.yml`, before the step that runs the tools suite. No
product, checker, or test change.

## Problem

The agent-record job fails intermittently on `main` (2 of its last 5 runs:
40521291883, 40452744901) and reproduced on the #3513 rerun (job
122404172875, 2026-10-27T19:16Z). Two environmental failure modes, both
caused by the shared runner's free disk dropping under what the tools suite
demands:

1. 49 tests in `tests.tools.test_strix_vllm_oracle` fail with
   `ValueError("disk headroom exhausted")` from
   `tools/bench/strix_vllm_oracle/worker.py:257`: the worker's
   `DEFAULT_LIMITS` demand 20 GiB `min_disk_bytes` and the check runs against
   the real runner disk.
2. `tools.bench.drop_file_cache.CacheDropError: cache-drop proof retained
   6291456 resident bytes`: the cache-drop test cannot prove eviction under
   memory/disk pressure.

The paged branch does not touch the strix worker; this is shared-runner
pressure, and `main` suffers from it too.

## Design

One workflow step before the tools-suite step in the `agent-record` job:

```yaml
- name: Free runner disk
  run: |
    sudo rm -rf /usr/local/lib/android /usr/share/dotnet /opt/ghc \
      /usr/local/.ghcup /usr/lib/jvm || true
    sudo docker image prune -af || true
    df -h /
```

Standard GitHub-hosted-runner reclamation: preinstalled Android SDK, .NET,
GHC, and JVM trees are the largest removable tenants, and stale docker
layers accumulate across image updates. The step is best-effort (`|| true`)
and reports the resulting free space so the log carries the evidence.

The thresholds in `worker.py` are unchanged: the worker's 20 GiB floor is a
correct safety property of the oracle runner; the fix removes the pressure,
not the check.

## Risks

- Removing `/usr/share/dotnet` breaks a job that needs dotnet. The repo has
  no dotnet job on GitHub-hosted runners (`windows-msvc` uses MSVC directly);
  the local `check-windows-portability.py` dotnet dependency is host-side,
  not runner-side. Verified by `grep -r dotnet .github/workflows/`.
- An `apt` re-fetch cost after removal: none of the reclaimed trees are
  apt-managed.

## Tests

Red-before: the failure exists on `main` without the step (runs 40521291883,
40452744901; #3513 rerun job 122404172875). Green-after: the agent-record
job on this branch's pull request passes with the step present and logs the
post-reclamation `df -h /` line. There is no local unit to run: the change
is workflow-only, and the existing checkers
(`check-agent-record.py`, `check-commit-style.py`,
`check-commit-trailers.py`, `agent-preflight.sh --staged`) gate the change
locally.

## Gates

- `python3 scripts/check-agent-record.py`
- `python3 scripts/check-commit-style.py` and
  `scripts/check-commit-trailers.py` on the commit range
- `scripts/agent-preflight.sh --staged`
- Remote: the `agent-record` job green on the pull request

## Evidence

- Failed main runs: 40521291883 (2026-10-27T14:27Z), 40452744901
  (2026-10-27T04:59Z)
- Reproduction on #3513 rerun: job 122404172875, log line
  `FAILED (failures=49, errors=2, skipped=59)` with
  `ValueError: disk headroom exhausted` at `worker.py:257` and
  `CacheDropError: cache-drop proof retained 6291456 resident bytes`
- Passing main run on the same day (14:27Z, run 40521291883's successor
  per check-run listing) shows the failure is intermittent, i.e.
  environmental.

## Stop conditions

- If GitHub's runner image changes so that the reclaimed paths no longer
  exist and the failure recurs, the next hypothesis is a per-step
  `df`-gated reclamation or lowering the test-side limits through a spec'd
  checker change.

## Git integration

One pull request, `row/CI-AGENT-RECORD-DISK`, spec committed before the fix.

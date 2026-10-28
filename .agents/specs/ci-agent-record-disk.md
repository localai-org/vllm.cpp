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

**Correction 2026-10-28.** The first attempt (reclamation only) was
insufficient: run 40586147690 job 122477980562's `df -h /` showed 113 GiB
free on `/dev/root` while the same job still failed 49+2 strix tests with
`ValueError("disk headroom exhausted")` from `worker.py:257`. The reason:
`headroom()` does not measure `/` — it measures
`shutil.disk_usage(self.local).free` and `shutil.disk_usage(self.output).free`
(`worker.py:256`), and both paths live under `/tmp` (the tests create their
scratch tree with `tempfile.TemporaryDirectory(prefix="strix3043-test-")`,
`test_strix_vllm_oracle.py:35-37`, and the worker creates `self.local` with
`mkdtemp(prefix="strix-vllm3043-", dir="/tmp")`, `worker.py:226`).

`ubuntu-latest` is rolling from 24.04 to 26.04 (actions/runner-images#14777),
and on 26.04 `/tmp` is a RAM-backed tmpfs mounted at 50% of RAM (~7.8 GiB)
with a per-user quota (~6.3 GiB) — while `/` has ~90+ GiB free. A job whose
runner lands on 26.04 fails the worker's floor no matter how empty `/` is;
one on 24.04 passes. The intermittency is runner-image assignment, not
workload. The same wall hits every lane that compiles or runs the tools
suites on `ubuntu-latest`: `agent-record`, `build-test-cpu`, `sanitize-cpu`
(all three proven failing in run 40586147690), plus `build-test-vulkan`,
`build-newest-gcc`, and `cuda-fat-build` by the same exposure class. The
`ubuntu-24.04-arm` lanes are pinned to an image whose `/tmp` stays on the
root disk and are out of scope.

Fix, the runner-images maintainer-endorsed workaround (#14777): each
affected job masks `tmp.mount` and lazy-unmounts `/tmp` as its FIRST step
(files already in `/tmp` are not visible after the unmount, so nothing may
have written there yet), then reports `df -h / /tmp`. On a 24.04 runner the
step is a harmless no-op (the unit mask succeeds or is absent, the unmount
finds no mount). The reclamation step stays in `agent-record`: with `/tmp`
back on the root disk, freeing the Android/.NET/GHC/JVM trees and stale
docker layers buys the headroom the suites and compilers actually consume.

The thresholds in `worker.py` are unchanged: the worker's 20 GiB floor is a
correct safety property of the oracle runner; the fix removes the pressure,
not the check.

```yaml
- name: Put /tmp back on the root disk
  run: |
    sudo systemctl mask tmp.mount || true
    sudo umount -l /tmp || true
    df -h / /tmp
```

First step of each affected job, before anything writes to `/tmp`. On a
24.04 runner both commands are harmless no-ops (the unmount finds no
mount), and the `|| true` keeps the step from reding on image drift.

```yaml
- name: Free runner disk
  run: |
    sudo rm -rf /usr/local/lib/android /usr/share/dotnet /opt/ghc \
      /usr/local/.ghcup /usr/lib/jvm || true
    sudo docker image prune -af || true
    df -h /
```

Standard GitHub-hosted-runner reclamation, in the `agent-record` job:
preinstalled Android SDK, .NET, GHC, and JVM trees are the largest
removable tenants, and stale docker layers accumulate across image
updates. The step is best-effort (`|| true`) and reports the resulting
free space so the log carries the evidence.

## Risks

- Removing `/usr/share/dotnet` breaks a job that needs dotnet. The repo has
  no dotnet job on GitHub-hosted runners (`windows-msvc` uses MSVC directly);
  the local `check-windows-portability.py` dotnet dependency is host-side,
  not runner-side. Verified by `grep -r dotnet .github/workflows/`.
- An `apt` re-fetch cost after removal: none of the reclaimed trees are
  apt-managed.

## Tests

Red-before: the failure exists on `main` without the step (runs 40521291883,
40452744901; #3513 rerun job 122404172875). Red-also-before the corrected
design: PR #3515's first commit (reclamation only) left the job red with
113 GiB free on `/` -- run 40586147690 job 122477980562. Green-after: the
`agent-record`, `build-test-cpu`, and `sanitize-cpu` jobs on this branch's
pull request pass with the mask step first in their logs and the `df -h /
/tmp` line showing `/tmp` on `/dev/root`. There is no local unit to run: the
change is workflow-only, and the existing checkers
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
- PR #3515's first commit (reclamation only) red on the corrected
  mechanism: run 40586147690 (2026-10-27T21:21Z), agent-record job
  122477980562 -- `df -h /` printed 113 GiB free on `/dev/root`, then the
  tools suite still failed `FAILED (failures=49, errors=2, skipped=59)`.
  The same run's `build-test-cpu` (job 122477933406) and `sanitize-cpu`
  (job 122477933291) failed on the identical `worker.py:257` raise,
  proving the wall is not agent-record-specific.
- Upstream root cause: actions/runner-images#14777 -- on `ubuntu-26.04`
  `/tmp` is a 50%-of-RAM tmpfs with a per-user quota; the endorsed
  workaround (mask `tmp.mount`, lazy-unmount `/tmp` as the first step) is
  the one this change lands, and runner-images#14814 is the image-side fix.

## Stop conditions

- If GitHub's runner image changes so that the reclaimed paths no longer
  exist and the failure recurs, the next hypothesis is a per-step
  `df`-gated reclamation or lowering the test-side limits through a spec'd
  checker change.

## Git integration

One pull request, `row/CI-AGENT-RECORD-DISK`, spec committed before the fix.

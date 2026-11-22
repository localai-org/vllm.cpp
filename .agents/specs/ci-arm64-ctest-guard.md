# CI-MAIN-CTEST-RESIDUE: make the arm64-full ctest guard execute and count honestly

Row: `CI-MAIN-CTEST-RESIDUE`.
Issue: `ISSUE-LOCAL-01M80VZXFK7HRAC9WSQ5P5J51W`.
Kind: bug fix in CI semantics; CPU-only, no oracle pin change.

## Scope

One job: `build-test-cpu-arm64-full` in `.github/workflows/ci.yml`. The job is
schedule-only and `continue-on-error`, so this change does not gate a pull
request. It changes how the job reports what it executed, not which tests are
registered or which assertions any test makes.

Out of scope: the vLLM pin, the test binaries and their assertions, the other
CI lanes, and any GPU measurement.

## Anchors

- `.github/workflows/ci.yml`, job `build-test-cpu-arm64-full`, step `Test`:
  the guard whose semantics this spec changes.
- `tests/CMakeLists.txt:18` `vllm_cpp_add_test`, which sets
  `SKIP_RETURN_CODE 77` on every registered test.
- Raw job logs archived under `docs/bench-evidence/44904678515/` and
  `docs/bench-evidence/44952527245/`:
  - `build-test-cpu-arm64-full`, jobs `136847599665` and `136993376238`,
    head `cb0da7d66adadb1ed56bba963efe69efdd777d89`.
  - `baseline-summary`, job `136896966427`, which records the same RED.

## Baseline

Two consecutive scheduled runs of the same head end the `Test` step with
`Process completed with exit code 8`. The step's own shell script is echoed in
the log, but neither its `cat ctest-run.txt` output nor the guard's `ctest
EXECUTED ...` line appears. The enumeration step in the same job does print its
`cat ctest-enumeration.txt` output (`Total Tests: 823`).

The difference is ctest's exit status. The enumeration step's last command
succeeds; the `Test` step's first command fails. GitHub runs `run:` blocks with
`bash -e`, so the failing `ctest` aborts the step before `cat` and before the
guard. The guard therefore never executes on a failing run, and no failing run
can carry its own triage evidence.

A second defect is visible without a run. The guard extracts
`ran` with `sed -n 's/^.*tests passed, .* out of \([0-9][0-9]*\)$/\1/p'`.
ctest's summary line is `N% tests passed, M tests failed out of T`; the
captured group is `T`, the enumeration total. `ran` therefore equals `count`
for every complete run, whatever the pass, fail or skip counts are. The
comparison cannot detect a selection mismatch or a skip.

A controlled ctest experiment (three tests: one pass, one `SKIP_RETURN_CODE
1`, one fail) confirms the format: per-test lines carry `***Passed`,
`***Skipped` or `***Failed`; the summary carries `out of 3` regardless of the
skip; and ctest exits `8` as a generic error code, not as a failed-test count.

## Design

1. Add `scripts/check-ctest-accounting.py`. It reads the captured ctest output
   and the enumerated count. It counts terminal per-test status lines by kind:
   passed, failed, skipped, not-run and timeout. It prints each count. It exits
   0 only when every enumerated test produced exactly one terminal status. The
   executed count is passed plus failed plus timeout; skipped and not-run
   entries are reported separately and never added to it. A skip is therefore
   not a pass, but it is not a missing test either. ctest's own exit status
   still decides whether any failure is acceptable.
2. In the `Test` step, keep ctest's exit status without letting `bash -e` abort
   the step: initialise `status=0` and use
   `ctest ... > ctest-run.txt 2>&1 || status=$?`. Then `cat ctest-run.txt`
   always runs. Then call the checker and propagate its refusal before
   propagating ctest's status. No assertion is deleted and no failure is
   accepted: both the accounting and ctest must pass for the step to pass.

## RED-before evidence

Captured against the unmodified `Test` step before any implementation edit:
the behavioural test in `tests/scripts/test_check_arm64_ctest_guard.py` runs
the step's own `run:` block under `bash -e` with a fabricated failing ctest.
Results are recorded in this spec's `## Outcome` when the work lands.

## Tests

`tests/scripts/test_check_arm64_ctest_guard.py` executes the real `run:` block
from the workflow under `bash -e` with a fake `ctest` on `PATH` and fabricated
output, so both directions are executable:

- a failing ctest still prints its output and reaches the guard (RED before,
  GREEN after);
- a fully accounted all-pass run exits 0 (accept);
- a skipped test is reported as not executed and never enters the executed
  count, even when ctest itself exits 0 (reject a skip counted as a pass);
- an accounted count below the enumeration is refused (reject a short run);
- a genuine ctest failure still fails the step when the accounting is clean.

`scripts/check-ctest-accounting.py` is exercised through its function API in
the same file for the parsing edge cases: no terminal lines, duplicate status
lines, and the "did not run" section.

## Gates

- `scripts/agent-preflight.sh` (all record gates and mutation suites) before
  each commit and before push.
- `python3 scripts/check-agent-record.py --report` reports rot 0 before push.
- `python3 -m unittest discover -s tests/scripts -p 'test_check_arm64_ctest_guard.py'`
  stays green after the fix; the same command is RED before it.
- `python3 scripts/check-commit-style.py` on the pushed range and
  `python3 scripts/check-commit-trailers.py --message-file` on each message.
- Local CPU-only aarch64 reproduction of the job's configure, build,
  enumerate and test sequence; its ctest output is archived under
  `docs/bench-evidence/` and the failing test names are recorded in the issue
  resolution. The lane itself is not required to turn green by this change:
  genuine test failures remain genuine failures.

## Risks

- Parsing ctest's human-readable output is version-sensitive. The parser keys
  on the `***<Status>` marker that every terminal per-test line carries and
  fails closed when no terminal line is found.
- A `bash -e` exemption via `||` can hide a real ctest status. The step stores
  `$?` immediately and re-exits it after the guard, so it cannot hide one.
- Counting a skip as a missing test can keep the lane RED when the skip is
  legitimate. The local aarch64 run shows `test_modelopt_mixed_precision_checkpoint`
  skipping by design, so skips are accounted for and reported, not refused.
  They are still excluded from the executed count, and ctest's own status
  remains the pass/fail authority.

## Stop conditions

- The guard is not executable from the workflow, or a test cannot be driven
  with fabricated ctest output: report `NEEDS_CONTEXT`.
- The ctest failures need a device, a checkpoint or a developer decision:
  record the exact failing names and report `NEEDS_DECISION` rather than
  changing a test assertion.
- The vLLM pin would move, or a checker would be widened to accept a skip:
  stop; both are outside this row.

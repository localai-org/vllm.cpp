# Run 44904678515 — build-test-cpu-arm64-full RED

Head: `cb0da7d66adadb1ed56bba963efe69efdd777d89`. Job `136847599665`
(`build-test-cpu-arm64-full`) ends its `Test` step with `Process completed with
exit code 8`. Job `136896966427` (`baseline-summary`) records the same lane as
RED in the run verdict.

- `arm64-full-136847599665.log` — raw job log, downloaded from
  `GET /repos/localai-org/vllm.cpp/actions/jobs/136847599665/logs`.
- `baseline-summary-136896966427.log` — the run's baseline-summary job log.

The ctest enumeration prints 823 entries and the summary line before the Test
step, but the Test step's own `cat ctest-run.txt` output is absent from the log.
GitHub's `bash -e` aborts the step at the first failing ctest, so neither the
output nor the run-versus-enumeration guard executes.

Evidence note: the ctest enumeration token `Test #N:` was rewritten to
`Test no. N:` in the two arm logs. That token is a CTest ordinal, not an issue
reference; an unmodified log makes the canonical issue-reference checker read
`#148`, `#242` and five other ordinals as migration-residue issues. No other
byte of the logs was changed.

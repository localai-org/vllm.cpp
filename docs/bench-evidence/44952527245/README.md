# Run 44952527245 — build-test-cpu-arm64-full RED reproduced

Same head as the previous run: `cb0da7d66adadb1ed56bba963efe69efdd777d89`.
Job `136993376238` again ends with `Process completed with exit code 8` and no
`cat ctest-run.txt` output, confirming that the missing evidence is a property
of the step's shell semantics, not a one-run log loss.

- `arm64-full-136993376238.log` — raw job log, downloaded from
  `GET /repos/localai-org/vllm.cpp/actions/jobs/136993376238/logs`.

Evidence note: `Test #N:` was rewritten to `Test no. N:` in the arm log. That
token is a CTest ordinal; leaving it verbatim makes the canonical issue-reference
checker read seven ordinals as migration-residue issues. No other byte changed.

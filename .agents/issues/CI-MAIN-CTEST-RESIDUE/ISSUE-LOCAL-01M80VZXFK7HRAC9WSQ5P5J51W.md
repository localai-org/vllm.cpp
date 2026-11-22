ID: ISSUE-LOCAL-01M80VZXFK7HRAC9WSQ5P5J51W
Title: build-test-cpu-arm64-full: ctest guard is dead under bash -e and the run's failures are undocumented
Row: CI-MAIN-CTEST-RESIDUE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-22
Updated: 2026-11-22
Closed: -

## Problem

Run 44904678515 / job 136847599665 (head cb0da7d6) fails with 'Process completed with exit code 8'. The Test step runs GitHub's default bash -e, so the first failing ctest aborts the step before 'cat ctest-run.txt' and before the run-vs-enumeration guard; the job log therefore has no ctest output and no triage evidence. Even if it ran, the guard parses 'out of N' from ctest's summary, which is the enumeration total, so it can never detect a skip or a selection mismatch. Need an honest guard that executes ctest without -e aborting, prints the evidence, accounts for every enumerated test, and treats a skip as not-passed.

## Resolution

2026-11-22. Root cause confirmed from two job logs (`44904678515` / job
`136847599665` and `44952527245` / job `136993376238`): GitHub's `bash -e`
aborted the `Test` step at the first failing ctest, so neither `cat ctest-run.txt`
nor the guard ran; the guard's `sed` also captured `out of N`, the enumeration
total, so it could never detect a short run. Fixed by preserving ctest's status
through an `||` list and adding `scripts/check-ctest-accounting.py`, which counts
per-test terminal statuses, requires `accounted == enumerated`, reports skips
separately and never counts them as executed. RED-before: 5 of 5 behavioural
tests failed on the unmodified workflow. GREEN-after: 9 of 9 pass. Local
aarch64 reproduction (823 entries) found seven genuine failures and 20 skips,
none of them W1's pinned-path tests; details in
`docs/bench-evidence/44904678515/report.md`. The genuine failures remain red by
design; no assertion and no vLLM pin changed.

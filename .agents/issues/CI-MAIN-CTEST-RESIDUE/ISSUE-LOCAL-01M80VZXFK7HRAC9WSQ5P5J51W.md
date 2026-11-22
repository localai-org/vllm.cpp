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

-

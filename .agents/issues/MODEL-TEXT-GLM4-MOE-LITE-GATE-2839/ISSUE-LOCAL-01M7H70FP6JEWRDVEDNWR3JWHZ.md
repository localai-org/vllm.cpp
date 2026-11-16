ID: ISSUE-LOCAL-01M7H70FP6JEWRDVEDNWR3JWHZ
Title: K1 and K2 have no runner, so the two measurements that decide the campaign are manual and can clobber the tracked goldens
Row: MODEL-TEXT-GLM4-MOE-LITE-GATE-2839
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-15
Closed: 2026-11-15

## Problem

scripts/glm4-moe-lite-neartie-gap.py writes our_ids.npy and neartie_gap_mnats.npy into --golden-dir, which is the tracked tests/parity/goldens/glm4_moe_lite_greedy directory. Running it there, as the header's own example implies, overwrites committed artifacts in a working tree that the next commit can sweep up. Neither it nor scripts/glm4-moe-lite-path-determinism.py has a runner, so K1 and K2 are two remembered command lines with hand-copied arguments, and nothing compares the regenerated artifacts against the committed ones so a reader can see whether the oracle still reproduces its own capture. The runner copies the goldens to a scratch directory, invokes both tools through one python interpreter, diffs every artifact against the committed copy, and writes summary.json with the K1 sub-millinat count and the K2 verdict, so the lease command is one line and the comparison is automatic.

## Resolution

Runner landed on row/MODEL-TEXT-GLM4-MOE-LITE-GATE-2839-K1K2-RUN based at 13542b40a. scripts/glm4-moe-lite-k1k2-run.py stages the committed goldens into a scratch directory, refuses a scratch path inside the tracked directory, runs K1 then K2 through one interpreter, diffs our_ids.npy, neartie_gap_mnats.npy, teacher_forced_gap_nats.npy and teacher_forced_argmax.npy cell by cell against the committed capture, and writes summary.json with the differing counts, the K1 sub-millinat count and the K2 verdict. Red before: tests/scripts/test_glm4_k1k2_runner.py failed 7 of 7 with the runner absent. Green after: 9 of 9. Registered in scripts/agent-preflight.sh and .github/workflows/ci.yml behind the numpy guard; check-test-registration.py green. The runtime half is unexecuted on this host: no GPU and no 31.2B snapshot, and the runner needs K1 (#3565) and K2 (#3566) merged before a real run adds a measurement.

# CHECKER-HARDENING — the red-first captures of the four checker changes

Row: `CHECKER-HARDENING`
Spec: `.agents/specs/checker-hardening.md`
Date: 2026-10-30. Host: the shared dev checkout; every capture is offline,
standard library only, no GPU, no network.

Each section records the observed failure of the PRE-change world (or, for a
new checker, the failure the checker produces on the mutated tree), verbatim,
before the fix that the same commit carries.

## 1. Evidence report vs oracle pin sha256 (`check-evidence-pin-sha`)

Mutation: the artifact sha256 in
`docs/bench-evidence/flint-cpu-parity-20261029/report.md` replaced with 64
zeroes, checker run against the mutated tree, then the file restored
byte-for-byte:

```text
$ python3 scripts/check-evidence-pin-sha.py; echo rc=$?
check-evidence-pin-sha: evidence reports disagree with their pins:
  .agents/oracles/llama-cpp-flint.md vs docs/bench-evidence/flint-cpu-parity-20261029/report.md: evidence report and oracle pin disagree on the artifact sha256: the report records 0000000000000000000000000000000000000000000000000000000000000000, the pin file records 20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e
Re-measure the artifact and record the SAME sha256 in both the
evidence report and the oracle pin, or re-quantize whichever file
recorded the stale hash.
rc=1
```

Green after restore: `check-evidence-pin-sha: ok` at rc=0, self-test
`6 evidence-pin-sha fixtures` ok, unit suite `tests/scripts/test_check_evidence_pin_sha.py` OK.

## 2. Registry multiplicity (`check-oracle-pins`)

Red of the PRE-change world: the `llama-cpp-flint` row duplicated in the
AGENTS.md oracle-registry table (scratch copy, restored byte-for-byte after
the capture) — the set comparison passed it:

```text
$ python3 scripts/check-oracle-pins.py; echo red-rc=$?
oracle-pins ok (23 oracles pinned)
red-rc=0
```

Green after the multiset change on the clean tree, then red on the same
mutation:

```text
$ python3 scripts/check-oracle-pins.py; echo restored=$?
oracle-pins ok (23 oracles pinned)
restored=0

$ python3 scripts/check-oracle-pins.py; echo green-check-rc=$?   # AGENTS.md row duplicated again
oracle-pins: llama-cpp-flint: admitted 2x by the AGENTS.md table but pinned 1x — a duplicated registry row
oracle-pins FAILED (1 error(s)) — see AGENTS.md "When vLLM has no implementation"
green-check-rc=1
```

Self-test extended with a 4-case registry corpus (agree, duplicated row,
absent row both directions); unit tests
`test_duplicated_agents_registry_row_is_reported` and
`test_a_singly_admitted_row_stays_green` added.

## 3. CI wiring coverage of the agent-record job (`check-test-registration`)

Red of the PRE-change world: the `check-evidence-refs` step deleted from the
agent-record job (scratch copy, restored byte-for-byte after the capture) —
the checker passed it:

```text
$ python3 scripts/check-test-registration.py >/dev/null; echo red-rc=$?
red-rc=0
```

After the pin, the same deletion names the missing step (the span also
carried the item-1 step, which is named beside it):

```text
ERROR: check-evidence-pin-sha.py is missing from the agent-record job's checker steps
ERROR: check-evidence-refs.py is missing from the agent-record job's checker steps
rc=1
```

Two further reds the pin catches: an unpinned checker step added to the job
(`check-new-unpinned.py runs in the agent-record job but is not in
AGENT_RECORD_CHECKER_STEPS`), and a job-level `if: always()` on the job
(`the agent-record job carries a job-level \`if:\` — a step behind a
condition is not a registration`). Green restored byte-for-byte afterwards
(`restored-green`). Unit tests added in
`tests/scripts/test_check_test_registration.py` (87 tests, OK).

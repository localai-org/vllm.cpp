# Spec: re-point tests/scripts/test_agent_record.py at the restructured checker's modules

Row: `BACKEND-TENSTORRENT` (the record-infrastructure row). State: DRAFT
(2026-09-27). Issue: the CI-rot repair train's newly-exposed layer.
Git integration: one pull request (spec + the test-infrastructure repair
+ gates), branch `row/FIX-TEST-AGENT-RECORD-IMPORTS`.

## Problem

`tests/scripts/test_agent_record.py` (the record gate's 124-test suite)
references `agent_record.check_issue_records` and
`agent_record.issue_records` — functions from the PRE-restructure
checker. The current `scripts/check-agent-record.py` (1,253 lines, the
`a9f6186c2`-era restructure) has ZERO `issue_records` references and NO
`check_issue_records` function: the issue-record validation was either
inlined into `check_issue_table` or split into
`scripts/issue_records.py` (which carries `parse_issue_file`,
`render_issue_record`, `IssueRecord` — but NOT `check_issue_records`).

The suite errors at import/attribute level on pristine main (93 tests,
37 failures, 57 errors), masked all week because the rot failures fired
earlier in the CI job. The rot repair (#3329) unmasked it.

## The repair

1. **Load `scripts/issue_records.py` as its own module** in the test's
   header (the importlib pattern, 3 lines) and attach it as
   `agent_record.issue_records` for the tests' backward-compatible
   references (11 sites — minimal diff).
2. **Re-point the `check_issue_records` test calls** at the current
   validation path: the checker's `check_issue_table` plus
   `issue_records`'s validation primitives, composed per the current
   code's semantics (read `check_issue_table` first to understand what
   the restructured checker actually validates).
3. **The mutation evidence**: each re-pointed test's red-first state is
   the CURRENT error (the AttributeError) — captured; the green state
   is the test passing against the re-pointed path.
4. **The suite verdict**: 124 tests, errors reduced to the
   host-environmental set (the known rocprof/tools failures), the
   record-relevant tests green.

## Gates

- `python3 -m unittest tests.scripts.test_agent_record` — errors
  eliminated, the record-relevant failures resolved.
- `python3 scripts/check-agent-record.py` stays green (the repair is
  test-side only).
- Standard gates (style, trailers, agent-record).

## Risks

- The re-pointed tests may exercise a validation path whose semantics
  differ from the old `check_issue_records` — each re-pointed test's
  assertions must match the CURRENT code's behavior, not the old
  function's (read `check_issue_table` + `issue_records` first).
- If `check_issue_table` proves insufficient for a test's needs, the
  test exercises a DELETED capability — stop, record, and the
  capability's restoration becomes its own unit (do not reimplement
  inside the test).

## Non-goals

- No checker-script changes; no issue-file changes; no product code.
- No new validation logic — the tests exercise what exists.

## Stop conditions

- A test's needs cannot be met by the current code's semantics → stop,
  record, the capability gap becomes its own unit.

## Owed

- `ISSUE-LOCAL-01M3JMSD9P7REKNQVEG9HCEWTD` owns this spec's remaining stop condition: the tree must satisfy the restored restructured checker (claim annotations, roadmap issue-row refusal, canonical issue paths, anchor ratchet) and the record suite must pass against it.


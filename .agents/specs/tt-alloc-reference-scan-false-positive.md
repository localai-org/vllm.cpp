# Spec: reference-scan false positive on evidence-log allocator IDs

Issue: ISSUE-LOCAL-01M7EJAFM401RJRNXD49S6QPWB
Row: TT-EMBED-DUAL-RESIDENCY (in-flow checker repair, blocks the row's gate)

## Scope

`scripts/check-agent-record.py` reference discovery only
(`ISSUE_REFERENCE_RE` / `issue_references_in_text`). No record, evidence log,
or enforced property changes.

## Problem

`python3 scripts/check-agent-record.py` on `row/TT-EMBED-DUAL-RESIDENCY`
reports 18 errors: the GitHub-issue-number tokens 652/834/835 and
148/242/250/296/362/469 resolve to `_intake`, plus the `ISSUE-GH` twins. Both
forms are false positives from committed evidence logs:

1. `[TT-ALLOC] 652 label=EnsureKeepQuantRowWords/pre banks=8 ...` device
   allocator lines in `docs/bench-evidence/tt-embed-dual-residency-20261114/{q6kl-cap-1,q6kl-cap16-1}.stderr.log`.
   The `#N` is an allocator event ID.
2. `preflight-staged-repair.log` quotes the checker's own diagnostics
   verbatim (lines beginning `ERROR: issue reference` that name a token and
   its `_intake` record), so the scanner matches both forms in the checker's
   own output text.

## Design

The enforced property is unchanged: an issue reference in a record, commit
body, or changed file must resolve to a canonical record. The fix narrows
token classification inside `issue_references_in_text`:

- Skip a line when it is a device allocator-log line (matches
  `\[[A-Z][A-Z0-9-]*-ALLOC\]`). `#N` there is an allocator event ID.
- Skip a line when it is a quoted checker diagnostic (matches
  `^\s*(?:ERROR|WARNING): issue reference\b`). Both `#N` and `ISSUE-GH-N`
  there are the checker describing itself, not the author claiming a
  reference.

Line-scoped, not file-scoped: a real reference elsewhere in an evidence log
is still detected. A real reference sharing a line with an allocator tag is
not detected; accepted, because allocator lines are machine-generated.

## Tests (tests/scripts/test_agent_record.py)

Red on the unfixed checker, green after:

- a changed file containing `[TT-ALLOC] #652 label=...` yields no reference
  and passes `check_canonical_issue_references` against an intake record for 652.
- a changed file quoting an `ERROR: issue reference` diagnostic line (token
  plus `_intake` record ID) yields no reference.
- a changed file containing a prose bare-token reference still resolves to the
  intake record and still fails canonical ownership (property survives).
  record and still fails canonical ownership (property survives).

## Evidence

- Red before: `docs/bench-evidence/tt-embed-dual-residency-20261114/reference-scan-false-positive-red-before.log`
  (checker, 18 errors) and `...-red-before-tests.log` (new tests red on the
  unfixed checker).
- Green after: the branch runs `check-agent-record.py` clean with the
  evidence logs intact.

## Stop conditions

If a guard class silently hides a genuine intake reference in the future,
revert to line-scoped masks and require the evidence file to cite references
in a fenced, non-diagnostic form instead.

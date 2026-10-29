# SPEC — `FIX-EVIDENCE-LOG-GATE` — evidence dirs must not silently lose their raw logs

- Owning row: `FIX-EVIDENCE-LOG-GATE`
- Kind: bug (measurement-integrity infrastructure: a checker-shaped hole)
- Issue: created in the same flow (row-scoped local issue)

## Problem

`.gitignore:34` ignores `*.log` globally. Every bench-evidence directory that
"captured raw logs" therefore committed only its README/report: `git add` on
a directory silently skipped the log files, no gate checks evidence-dir
contents, and nobody noticed until the FLINT-CPU-PARITY fresh reviewer found
its report listing 9 raw logs of which zero were in the tree.

Confirmed on `main` (2026-10-29): the TT-KOLIBRI1-READBACK,
FIX-PAGED-REGRESSION, and FIX-TT-NUMA-DEGRADE evidence dirs contain no raw
logs. Their worktrees were removed after merge, which took the untracked
logs with them — **the raw measurement records for those three landed rows
are unrecoverable**. The reviews stood at review time (the reviewers saw the
real files), but the landed record is summary assertions only. That is the
exact shape the protocol's evidence discipline exists to prevent: an
assertion whose backing bytes no longer exist.

FLINT-CPU-PARITY's logs survived only because its worktree was still open
when the reviewer flagged the gap; they are rescued onto its branch
separately.

## Design

Two changes, both small:

1. **Whitelist evidence logs from the ignore rule.** Negate the global rule
   for committed evidence:
   ```gitignore
   # Logs and local capture artifacts (demo recordings, screencasts)
   *.log
   !docs/bench-evidence/**/*.log
   ```
   Evidence logs are precisely the artifacts the protocol requires to be
   kept; demo recordings and scratch logs stay ignored.
2. **A checker: every file an evidence README references must exist in the
   tree and be committed.** `scripts/check-evidence-refs.py`: scan
   `docs/bench-evidence/**/report.md` and `README.md` for filenames they
   assert as captured (a simple filename-reference scan over the directory's
   own listing is enough — do not parse prose semantics); for each, require
   the file to exist on disk AND be tracked. Exit 1 naming the missing or
   untracked file. Wire it into the agent-record lane (ci.yml agent-record
   job) and preflight.

This converts the failure mode — "the README says a log exists and the log
is gone" — into a red checker at commit time, on every push.

## Red-first

The checker, run against `origin/main` today, must FAIL: the readback,
paged-regression, and numa evidence dirs each reference logs that are not in
the tree. That is the red-before on the checker itself. (Those three rows'
missing logs cannot be resurrected; their dirs carry a dated loss note added
in this row — what is missing, why, and that the reviews verified the logs
at review time. The loss is recorded, not papered over.)

## Risks

- The whitelist may resurface large logs (the readback lever logs were
  ~9 MB). Acceptable: measurement evidence is the reason this repository
  keeps gateable oracles and pinned artifacts; a 9 MB committed log is a
  small price. If a future log is huge, compress it — do not drop it.
- The checker's filename scan may under-capture prose references. It is a
  floor, not a ceiling: it must catch the observed failure (report lists
  logs, logs absent), not parse intent.

## Tests

- Checker unit tests (tests/scripts/test_check_evidence_refs.py) following
  the house pattern: a fixture evidence dir with a referenced-present file
  (pass), a referenced-missing file (fail), a referenced-untracked file
  (fail), an unreferenced extra file (pass — the floor does not police the
  other direction).
- Red-before: `check-evidence-refs.py --self-test` plus the real `main` run
  failing on the three landed dirs (recorded in the evidence README).
- Green-after: with the whitelist + rescue notes, the checker passes on
  `main`; on a scratch mutation deleting a committed log, it fails.

## Gates

check-agent-record, commit style, trailers, preflight (named host dotnet
failure), and the checker wired into both ci.yml (agent-record job) and
preflight — a checker wired nowhere is a decoration.

## Stop conditions

- If the whitelist negation pattern does not work for nested dirs on any
  supported git version (verify `git check-ignore -v` on a nested evidence
  log after the change), fall back to renaming the whitelist to a dedicated
  extension (`.evlog`) — record the choice.

## Outcome

Landed 2026-10-29 as 007c67b6a (PR #3522). The whitelist !docs/bench-evidence/**/*.log is in .gitignore (verified on git 2.54: nested evidence logs tracked; root and scratch logs still ignored) and scripts/check-evidence-refs.py is wired into the agent-record CI job (ci.yml:704) and preflight CHECKERS/SUITES -- no job-level if touched. Red-before: 46 failure entries / 41 distinct filenames across NINE evidence dirs (the spec's three plus six more: flint-gate-harden, flint-keepquant-dispatch, kolibri1-dequant-cache, tt-rmsnorm-w1, rocm-quant-gather, rocm-rdna3-quant-wmma) -- the hole was 3x the spec's estimate. Each dir carries a dated EVIDENCE-LOST note; the raw logs for the pre-gate rows are unrecoverable and recorded as lost, not papered over. Review MERGEABLE (6 claims; M1/M2/M4/M6 RED prove the checker, whitelist, and tracked-check load-bearing). Named known gaps left open: the CI wiring of the checker is itself unguarded (deleting the ci.yml step reds nothing; check-test-registration pins only its own step and three suites) -- a wiring-coverage checker is the candidate; the fence exemption is a floor by design; check-env-doc still enforces presence, not semantics drift.

## Git integration

One pull request, `row/FIX-EVIDENCE-LOG-GATE`, spec committed first. The
checker, the whitelist, the loss notes for the three affected dirs, and the
unit tests land together.

# FIX-EVIDENCE-LOG-GATE — evidence-log loss gate: red-before, green-after, mutation

Row: `FIX-EVIDENCE-LOG-GATE`. Date: 2026-10-30. Host: local workstation,
git 2.54.0. Standard library only, offline throughout.

## What this row changes

1. `.gitignore`: whitelist `!docs/bench-evidence/**/*.log` beside the global
   `*.log` rule. Verified with `git check-ignore -v` on this git: the nested
   evidence path matches the negation (last match wins, so it is tracked),
   while a root-level log and a `scratch/` log still match the `*.log` rule. No `.evlog` fallback needed.
2. `scripts/check-evidence-refs.py`: every `.log` / `.log.txt` filename a
   `docs/bench-evidence/*/README.md` or `report.md` references must exist on
   disk AND be tracked. A dated `EVIDENCE-LOST` note in the same file records
   a loss instead of a live reference. Unreferenced extra files pass — the
   checker is a floor, not a completeness gate. Wired into the `agent-record`
   CI job and `scripts/agent-preflight.sh` (`CHECKERS` + `SUITES`).

## Red-before (checker against origin/main, 3c69df47a4, 2026-10-30)

`python3 scripts/check-evidence-refs.py` in a detached worktree of
`origin/main` exits 1 with 47 references missing or untracked across **nine**
evidence dirs — the three the spec names, plus six more the checker exposed
with the identical failure mode:

> EVIDENCE-LOST (2026-10-30): every filename below is a recorded loss from the
> red-before run; the per-dir notes carry the reasons.
- `tt-kolibri1-readback-2026-10-28` — `lever12.log`, `mutation-red-drain-verify.log`
- `fix-paged-regression-20261028` — `final-spec_probe-prompt-raw.log`, `final-decode_bench-anchor-raw.log`
- `fix-tt-numa-degrade-2026-10-29` — `01-patched-four-suites.log`, `02-patched-strict-arm.log`
- `flint-gate-harden-20261025`, `flint-keepquant-dispatch-20261027`,
  `kolibri1-cpu-dequant-cache-default-20261021`, `tt-rmsnorm-w1-20261025`,
  `rocm-quant-gather`, `rocm-rdna3-quant-wmma` — same global-ignore loss.

Excerpt (first lines of the run):

```
check-evidence-refs: evidence references point at files that are missing or untracked:
  docs/bench-evidence/fix-paged-regression-20261028/README.md:3: final-spec_probe-prompt-raw.log
  docs/bench-evidence/fix-paged-regression-20261028/README.md:7: final-decode_bench-anchor-raw.log
  docs/bench-evidence/fix-tt-numa-degrade-2026-10-29/README.md:60: 01-patched-four-suites.log
  docs/bench-evidence/fix-tt-numa-degrade-2026-10-29/README.md:91: 02-patched-strict-arm.log
  docs/bench-evidence/tt-kolibri1-readback-2026-10-28/README.md:32: lever12.log
```

Scope note: the checker resolves references against the CURRENT tree only.
The FLINT-CPU-PARITY row's raw logs are rescued on its own branch, not on
`main`, so `origin/main` carries only its report; this row records that
cross-reference rather than failing it. The six extra dirs above are
recorded-loss cases like the spec's three: their logs died with their
worktrees, and each now carries the same dated `EVIDENCE-LOST` note.

## Green-after (this branch, whitelist + loss notes)

`python3 scripts/check-evidence-refs.py` → exit 0:
`check-evidence-refs: ok — every referenced evidence log is tracked and on disk`.
`python3 tests/scripts/test_check_evidence_refs.py` → 12 tests OK, including
the shipped-tree test that reruns the full scan.

## Mutation (scratch copy of this tree, one whitelisted log deleted)

Delete the tracked `w0-after` log of `flint-tt-w2-20261025` (a log the
report references) in a scratch clone:

```
check-evidence-refs: evidence references point at files that are missing or untracked:
  docs/bench-evidence/flint-tt-w2-20261025/report.md:121: w0-after.log
Either restore the file under docs/bench-evidence/, or record the loss with an `EVIDENCE-LOST` note naming it.
(exit 1)
```

The gate detects the exact defect it exists to catch.

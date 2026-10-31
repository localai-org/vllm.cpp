# TT-SMALL-FIXES evidence — 2026-10-31

Row `TT-SMALL-FIXES`, spec `.agents/specs/tt-small-fixes.md`, issue
`ISSUE-LOCAL-01M686P391579S3SRN675V2XT7`. Two fixes, one capture each.

## Fix 1 — B2BI/B2II env silent-skip

- RED: `red-b2ii-silent-skip.log` — at HEAD `762098300`, build
  `/tmp/build-ttsmallfix` (`-DVLLM_CPP_TENSTORRENT=ON`, Release), the blackhole
  card present, standing recipe (mutex `~/gpu.lock`, `luwen reset`, tt-metalium
  env), and ONLY `VT_KOLIBRI1_TT_B2II_MODEL=/mnt/models/Aleph-Alpha/Kolibri-1`
  set (the documented spelling, real checkpoint): the device leg prints
  `SKIPPED: VT_KOLIBRI1_TT_B2BI_MODEL not set` and the suite reports
  `10 | 10 passed | 0 failed` / `SUCCESS!` — the phantom green.
- GREEN: `green-b2ii-device-leg.log` — same build and recipe at the fix, same
  single env var: the leg now RUNS (`[kolibri1-tt-b2bi] device leg start:
  model=...`); 203 assertions pass, RC=0.
- GREEN (refusal arm): `green-b2ii-bogus-path-refusal.log` — bogus
  `VT_KOLIBRI1_TT_B2II_MODEL=/nonexistent/bogus-checkpoint`: the leg refuses BY
  NAME (`SKIPPED: /nonexistent/bogus-checkpoint is not a Kolibri-1 checkpoint`)
  instead of silently skipping, which also proves the variable is read.
## Fix 2 — preflight `--staged` SyntaxError

- RED: `red-preflight-syntaxerror.txt` — at HEAD `762098300`,
  `python3 scripts/agent-preflight.sh --staged` dies with
  `SyntaxError: unmatched ')'` at `scripts/agent-preflight.sh:86`
  (`--staged) STAGED=1 ;;`).
- Diagnosis: the report's "inline python in the bash script" theory is FALSE —
  `agent-preflight.sh` carries no inline python. The crash is Python parsing
  the BASH file after a `python3` invocation; the parser stops at the first
  bash-only token, the line-86 case label, which is why the report names that
  line. The documented invocation (`bash scripts/agent-preflight.sh --staged`,
  AGENTS.md "Commands") runs clean — verified at HEAD with a full `--staged`
  preflight in this worktree and in a scratch repo (only the named host
  failure `check-windows-portability.py` / dotnet). A bash file cannot defend
  itself against a python3 parse, so the repair is the regression test
  `StagedInvocationTests` in `tests/scripts/test_agent_preflight_skip_report.py`:
  it executes the documented `--staged` invocation in the harness scratch repo
  and asserts no `SyntaxError`/`Traceback` in either stream, the
  `Staged change:` heading reached, and an exit status from the documented set.
  Mutation: mangling the line-86 case label turns the test red; restored, green.

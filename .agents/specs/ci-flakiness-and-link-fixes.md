# SPEC — `CI-FLAKINESS-AND-LINK-FIXES`: sanitize-cpu thread resilience and stale pin prose

Row: `CI-FLAKINESS-AND-LINK-FIXES`
Base: `92ea7d3a8306ec70e82512a25ff6668519e706b7` (current canonical main, 2026-11-24)
Issue: `ISSUE-LOCAL-01M864Q3642P4D1Z7Z3JH6QYSN`

## Trigger / problem

The coordinator relayed on 2026-11-23 that `sanitize-cpu (thread)` flips fail/pass
on `main` within the same day. The failure is a runner death at roughly 80
minutes, not a sanitizer finding: the two long cases under TSan are
`test_ltx2_video` (~20 min) and `test_dots3_note_scaffold` (~14.5 min), and a
dead runner discards the whole serial ctest run without a terminal ctest
summary. The lane is `continue-on-error` on pull requests and is counted by
`baseline-summary` on schedule/dispatch only, so a lost run is invisible to
merge review.

Two smaller record items ride in the same unit:
- `.agents/upstream-sync.md:7` prose still describes the 2026-09-03
  `e126687a9a` advance, while the authoritative `parity-pin` block in the same
  file (line 101) holds `a7c23ac96d7806e7c7e7d862eadbce5a33529b94`
  (`0.3.0.dev267`, pinned 2026-09-26, advanced by commit `4f11dfc10`).
- The relayed "unguarded `DeviceAvailable()` in qwen3_5.cpp `GdnBlockPaged`"
  item is **falsified by the tree**: commit `3abacff58` (2026-11-18) wrapped all
  three unguarded `vt::tenstorrent::DeviceAvailable()` call sites — two inside
  `GdnBlockPaged` — in `#ifdef VLLM_CPP_TENSTORRENT`. It is closed with that
  evidence here and is NOT re-fixed.

## Scope

1. `.github/workflows/ci.yml`, `sanitize-cpu` job, `thread` lane only: run the
   suite in three shards — all tests except the two heavy ones, then
   `test_ltx2_video`, then `test_dots3_note_scaffold` — each through a
   runner-death retry wrapper that retries exactly once.
2. `scripts/ctest-run-complete.py`, a small detector that exits 0 only when a
   ctest log contains a terminal run summary, plus its focused unit test.
3. `.agents/upstream-sync.md:7` prose correction. No pin block, oracle file,
   checker or assertion is changed.

Explicitly out of scope: changing the ASan+UBSan lane, changing sanitizer
flags or suppressions, removing or skipping any test, making a failing test
pass by retry, changing `continue-on-error` or `baseline-summary`, touching
`qwen3_5.cpp`, adding a matrix row (recorded under `## Owed`), and any public
document.

## Design

A retry is permitted only for an **incomplete run**, never for a failed test.
`ctest` always prints a terminal summary (`N% tests passed` or
`Total Test time =`) when it finished; a runner death before that point leaves
no such line. The wrapper runs a shard, streams its log, calls the detector,
and:
- returns the ctest status unchanged when the run is complete (including a
  real finding — never retried into green);
- retries once when the run is incomplete, then passes through whatever status
  the second attempt produced.

Sharding isolates the two long tests so a death in one shard costs that shard,
not the 80-minute suite. Each shard is a separate ctest process with the same
environment and serial execution; the address/undefined lane keeps its single
unchanged Test step.

## Tests / evidence

- `tests/scripts/test_ctest_run_complete.py`: green after a passing summary,
  green after a failing summary, red on empty or truncated logs, red on a log
  containing only `Start N:` lines. Mutation: weaken the detector to accept any
  non-empty log and the truncated cases go green — the test must catch it.
- Focused CI YAML inspection proves the `thread` step uses three shards and the
  detector, and that the address/undefined step is unchanged.
- Local `scripts/agent-preflight.sh --staged` and `check-ctest-accounting.py`
  on the workflow run (when a gate run is available). The decisive evidence is
  a `sanitize-cpu (thread)` CI run on main after this lands; until one exists,
  the reliability claim stays OPEN, not PASSED.

## Risks

- A genuinely flaky sanitizer finding could look like a runner death if ctest
  is killed before its summary. Mitigation: the retry is one attempt only and
  the final status is always ctest's own; a real finding is never deleted,
  skipped or allowlisted.
- Sharding changes process boundaries but not the test bodies, env or serial
  ordering; the two heavy tests still run under the same TSan configuration.
- `upstream-sync.md` is a shared record; the edit is a scoped one-paragraph
  replacement with all other lines byte-identical.

## Gates / stop conditions

Report `PASS` only after the focused tests are green, the detector mutation is
red, preflight is clean apart from the pre-existing environment-side
`check-windows-portability.py`, and a fresh reviewer passes the immutable head.
Stop and return `NEEDS_DECISION` if a CI rerun shows a real TSan finding rather
than a runner death. Never declare the flakiness cured from a single green run.

## Git integration

One PR, spec and implementation together, branch `row/CI-FLAKINESS-AND-LINK-FIXES`,
push only to `mine`; the user merges. No merge or force on main.

## Now

ACTIVE — spec-first, implementation pending.

## Owed

- A `*-matrix.md` row for `CI-FLAKINESS-AND-LINK-FIXES`; the row is made
  canonical by this spec's `Row:` declaration, as on the Kolibri rows.
- A post-landing `sanitize-cpu (thread)` run on main as reliability evidence.

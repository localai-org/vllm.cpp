# Raw measurement logs (added 2026-10-29)

- `final-spec_probe-prompt-raw.log` — raw output of
  `test_kolibri1_spec_probe` (DEFAULT environment, taskset -c 32-39, this
  host), with per-prompt output: the knife-edge flip at position 5 and the
  5/8 byte-identity floor.
- `final-decode_bench-anchor-raw.log` — raw output of
  `test_kolibri1_decode_bench` (DEFAULT environment, taskset -c 32-39,
  this host), capturing the tier-scoped anchors {101807, 109726} (the
  101807 anchor measurement, alternate-near-tie chain).

Both files back the numbers asserted in `DECISIONS.md` ("Consequences
re-measured under the restored default") and the final gate table.
Recorded in response to the fresh reviewer's finding, 2026-10-29.

> EVIDENCE-LOST (2026-10-30): `final-spec_probe-prompt-raw.log`, `final-decode_bench-anchor-raw.log`
> The raw logs named above are unrecoverable: the pre-2026-10-30 global `*.log` rule
> dropped them at `git add` and the worktree removal took the untracked files; the reviews
> verified them at review time. Recorded 2026-10-30 by FIX-EVIDENCE-LOG-GATE.

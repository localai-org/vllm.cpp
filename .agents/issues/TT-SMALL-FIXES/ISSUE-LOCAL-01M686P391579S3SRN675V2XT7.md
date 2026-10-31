ID: ISSUE-LOCAL-01M686P391579S3SRN675V2XT7
Title: Two named small defects: the B2BI/B2II env silent-skip, and the preflight --staged SyntaxError
Row: TT-SMALL-FIXES
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: 2026-10-31

## Problem

(1) tests/vllm/models/test_kolibri1_tt_b2bi.cpp:1021 reads VT_KOLIBRI1_TT_B2BI_MODEL while every doc/instruction sets VT_KOLIBRI1_TT_B2II_MODEL -- the device leg silently skips while the suite reads 10/10 SUCCESS (TT-ANCHORING evidence; the 0.29s phantom-green ctest runs). Fix: read B2II first, fall back to B2BI, comment citing TT-ANCHORING; red-first = with only B2II set HEAD silently skips, post-fix it runs or refuses by name. (2) agent-preflight.sh --staged crashes with a Python SyntaxError around line 86 (kolibri1-GGUF implementer report, 2026-10-30); non-staged run clean. Fix + a regression case running --staged in a scratch repo asserting no SyntaxError. Spec: .agents/specs/tt-small-fixes.md. One branch, two commits.

## Resolution

- Fix 1 (2026-10-31, `aa1f79f6`): the leg reads `VT_KOLIBRI1_TT_B2II_MODEL`
  first and falls back to `VT_KOLIBRI1_TT_B2BI_MODEL`. RED on the card at
  `762098300` (documented env only → silent skip + green banner), GREEN leg run
  and bogus-path refusal-by-name, in
  `docs/bench-evidence/tt-small-fixes-20261031/`.
- Fix 2 (2026-10-31): the reported "inline python SyntaxError" diagnosis is
  FALSIFIED by the tree — `agent-preflight.sh` carries no inline python. The
  crash reproduces exactly as reported (`SyntaxError: unmatched ')'` at line 86,
  the `--staged)` case label) but only under a `python3` invocation of the bash
  script (`python3 scripts/agent-preflight.sh --staged`); the documented
  `bash scripts/agent-preflight.sh --staged` runs clean, verified at HEAD by a
  full `--staged` preflight in this worktree and in a scratch repo (only the
  named host failure `check-windows-portability.py` / dotnet). A bash file
  cannot defend itself against a python3 parse, so the repair is the regression
  test `StagedInvocationTests` in `tests/scripts/test_agent_preflight_skip_report.py`
  (executes the documented `--staged` invocation in the scratch repo; asserts no
  `SyntaxError`/`Traceback`, the `Staged change:` gate reached, documented exit
  status; mutation-verified red on a mangled case label).

ID: ISSUE-LOCAL-01M864Q3642P4D1Z7Z3JH6QYSN
Title: CI: sanitize-cpu(thread) runner-death retry and shard the heavy tests; stale pin prose
Row: CI-FLAKINESS-AND-LINK-FIXES
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-24
Updated: 2026-11-24
Closed: 2026-11-24

## Problem

sanitize-cpu (thread) flips fail/pass on main within the same day: a runner death around 80 minutes discards the whole serial ctest run. The two long cases are test_ltx2_video (~20 min) and test_dots3_note_scaffold (~14.5 min) under TSan. The lane is continue-on-error on PRs and counted by baseline-summary on schedule/dispatch only, so a lost run is invisible to merge review. Separately, .agents/upstream-sync.md:7 prose still names the 2026-09-03 e126687a9a advance while the authoritative parity-pin block at line 101 holds a7c23ac96d (0.3.0.dev267, pinned 2026-09-26). A third relayed item — unguarded DeviceAvailable() in qwen3_5.cpp GdnBlockPaged — is already falsified by the tree (commit 3abacff58, 2026-11-18, wrapped all three call sites in #ifdef VLLM_CPP_TENSTORRENT) and is recorded as closed by evidence in the spec, not re-fixed.

## Resolution

- 2026-11-24: `sanitize-cpu (thread)` now runs the suite in three shards in
  `.github/workflows/ci.yml` — everything except the two heavy TSan cases, then
  `test_ltx2_video`, then `test_dots3_note_scaffold`. Each shard goes through a
  one-attempt runner-death retry gated by
  `scripts/ctest-run-complete.py`: a completed ctest run (pass OR fail) keeps
  its own status; only a run with no terminal summary is retried, once. The
  address/undefined lane keeps its original single Test step, and no test,
  assertion, sanitizer flag or `continue-on-error` setting was changed.
  `tests/scripts/test_ctest_run_complete.py` executes the workflow's own run
  block with a fake ctest: 7/7 green; mutating the detector to accept any
  non-empty log and mutating the retry to never fire each red two cases.
- 2026-11-24: `.agents/upstream-sync.md:7` prose now points at the
  authoritative `parity-pin` block (line 101, `a7c23ac96d`) instead of naming
  the superseded `e126687a9a` advance as current.
- 2026-11-24: the relayed `DeviceAvailable()` item is CLOSED BY EVIDENCE, not
  re-fixed: commit `3abacff58` (2026-11-18) wrapped the two then-unguarded
  `vt::tenstorrent::DeviceAvailable()` sites in qwen3_5.cpp — one inside
  `GdnBlockPaged` (GDN-spec debug dump) and one in `DenseForwardLayers`
  (`tt_lbl` pre label) — in `#ifdef VLLM_CPP_TENSTORRENT`; the third site at
  the prior `:10556` label was already guarded. That is the current tree.
  Post-landing CI evidence for the thread lane is recorded as owed in the
  spec; the reliability claim is not made from local tests alone.
- 2026-11-24 record correction after fresh review: the earlier wording in
  this resolution said "all three unguarded" and "two inside GdnBlockPaged";
  the commit diff shows two newly guarded call sites, one in GdnBlockPaged,
  and one site already guarded. The falsification conclusion is unchanged.

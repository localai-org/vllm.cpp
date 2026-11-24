ID: ISSUE-LOCAL-01M864Q3642P4D1Z7Z3JH6QYSN
Title: CI: sanitize-cpu(thread) runner-death retry and shard the heavy tests; stale pin prose
Row: CI-FLAKINESS-AND-LINK-FIXES
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-24
Updated: 2026-11-24
Closed: -

## Problem

sanitize-cpu (thread) flips fail/pass on main within the same day: a runner death around 80 minutes discards the whole serial ctest run. The two long cases are test_ltx2_video (~20 min) and test_dots3_note_scaffold (~14.5 min) under TSan. The lane is continue-on-error on PRs and counted by baseline-summary on schedule/dispatch only, so a lost run is invisible to merge review. Separately, .agents/upstream-sync.md:7 prose still names the 2026-09-03 e126687a9a advance while the authoritative parity-pin block at line 101 holds a7c23ac96d (0.3.0.dev267, pinned 2026-09-26). A third relayed item — unguarded DeviceAvailable() in qwen3_5.cpp GdnBlockPaged — is already falsified by the tree (commit 3abacff58, 2026-11-18, wrapped all three call sites in #ifdef VLLM_CPP_TENSTORRENT) and is recorded as closed by evidence in the spec, not re-fixed.

## Resolution

-

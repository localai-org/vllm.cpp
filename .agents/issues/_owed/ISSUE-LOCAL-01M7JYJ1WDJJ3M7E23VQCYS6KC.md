ID: ISSUE-LOCAL-01M7JYJ1WDJJ3M7E23VQCYS6KC
Title: TT paged-attention decode with the program cache DISABLED under rich suite state: value-level garbage at seq>=2 (bf16 re-read also wrong)
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-16
Updated: 2026-11-16
Closed: -

## Problem

The TT-PA-SMALL-WINDOW W1 characterization's program-cache-disabled arm (ProgramCacheToggleForTest(false), the falsified cache-hit-run-args hypothesis's instrument) exposed a SECOND PA decode defect distinct from the fixed dtype-width one: in a fresh process the arm is green (0 garbage / 8 seqs, w3-all-arms-GREEN.log), but inside the full suite (80+ cases of allocator/program history ahead of it) it fails deterministically at seq=2..8 with VALUE-level garbage — the bf16 re-read of the output buffer shows the same 1e33-magnitude wrong values as the f32 read (w22-consume-gdn-suite2.log: 'seq=2: max_abs=1.91669e+33, bf16-reread max_abs=1.9167e+33'), so the computed values themselves are wrong, not the write width. The program cache is enabled by default in production and the engine never disables it, so this is out of the production path's reach today; it is recorded because the characterization arm that found it ships in the suite and stays red there. Owed: root-cause the cache-disabled decode's value corruption under rich state (suspects: the per-call recompile's kernel-arg/scratch reuse), fix or gate the arm by the finding.

## Resolution

-

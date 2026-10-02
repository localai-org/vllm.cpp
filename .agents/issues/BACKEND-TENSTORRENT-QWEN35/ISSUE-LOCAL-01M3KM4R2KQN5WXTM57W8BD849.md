ID: ISSUE-LOCAL-01M3KM4R2KQN5WXTM57W8BD849
Title: RacIdxCache batched lane mishandles a page-table width change
Row: BACKEND-TENSTORRENT-QWEN35
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-28
Updated: 2026-09-28
Closed: -

## Problem

WarmRacIdx keys RacIdxCache by (num_slots, block_size) but not page-table width. The C=1 lane reallocates on block_table_cols != e.pt_width (retire + realloc, the #1105 discipline); the batched (num_slots>1) lane added in e39f2cf3f has no such guard: its refresh branch indexes batched_pt_host with the CALLER's block_table_cols against a vector sized at allocation width (OOB read) and copy_to_device's a [1, new_cols] host tensor into a [1, old_cols] device tensor — TT_FATAL 'Host tensor has different shape' (tensor_apis.cpp:161). Exposed by the new batched-PA capture doctest (cols=2) running after the batched-RAC doctest (cols=1) in the full suite.

## Resolution

- 2026-09-28 (worktree row/tt-27b-capture-write) FIXED. The batched lane now
  mirrors the C=1 lane's `pt_width` discipline: any `block_table_cols !=
  e.batched_pt_width` on a live entry retires the per-user page tables into
  `batched_retired_pts` (kept alive — never free a buffer a recorded trace
  addresses, #1105), reallocates them at the new width, and resets
  `batched_pt_host`; `batched_pt_width` records the allocation width.
  `batched_update_idxs` ([1] per user) and the sharded inputs are
  width-independent and untouched. Evidence: before the fix the new
  batched-PA capture doctest (page-table cols=2) threw TT_FATAL
  "Host tensor has different shape" (tensor_apis.cpp:161) when it ran after
  the batched-RAC doctest (cols=1) in the full suite — both share RacIdxCache
  key (num_slots=2, block_size=32); /tmp/suite-pa.log. After the fix the full
  96-case suite is 95/96 with the only failure the pre-existing owed RAC
  residual (/tmp/suite-pa3.log: 126/128, user-1 second head — the signature
  recorded at e39f2cf3f, unchanged); the PA case reads replay-vs-eagerB
  0/2048 mismatched elems in the same full-suite run.
-

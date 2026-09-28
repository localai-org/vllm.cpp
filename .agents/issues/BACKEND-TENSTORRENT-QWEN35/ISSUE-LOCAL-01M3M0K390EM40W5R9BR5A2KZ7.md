ID: ISSUE-LOCAL-01M3M0K390EM40W5R9BR5A2KZ7
Title: 27B c1 decode: RAC C=1 lane routes through unallocated batched tensors — segfault at the first cold decode step
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

At row/tt-27b-region-capture HEAD ec4e8a824, the Qwen3.8-27B-Q4_K_M c1 leg (--concurrency 1) segfaults in ttnn::copy inside ReshapeAndCacheKernel during the COLD eager decode step (capturing=0; /tmp/leg-control-c1.log, /tmp/leg-region-c1-diag.log, 2026-09-28, thalia). Control leg without VLLM_CPP_REGION_CAPTURE crashes identically, so this is pre-existing on the base, not the region arm. Root cause: e39f2cf3f rewrote TryReshapeAndCacheDeviceDecode as one per-user batched loop (rac_entry.batched_in[u], batched_update_idxs[u], batched_page_table[u]) but WarmRacIdx allocates those ONLY for num_slots>1 — for C=1 it allocates the shared sharded_in/sharded_in_v/update_idxs/page_table and its warm gate admits C=1 on `allocated` alone, so the loop indexes empty vectors (empty ttnn::Tensor -> null storage -> ttnn::Tensor::memory_config() segfault). The commit's claim 'The C=1 lane is untouched' is false; no c1 leg ran on this branch since e39f2cf3f (the doctrine legs were c2). Fix: restore the proven C=1 sequence verbatim (build_input over the whole shadow into the shared sharded tensors, one paged_fused_update_cache against the shared update_idxs/page_table) beside the batched loop.

## Resolution

-

## Resolution

- 2026-09-28 (worktree row/tt-27b-region-capture-spec): FIXED same-flow. The
  C=1 lane is restored verbatim from `git show e39f2cf3f~1`
  (build_input over the whole rope shadow into the shared sharded_in /
  sharded_in_v, one paged_fused_update_cache against the shared update_idxs /
  page_table) beside the batched loop, which now serves num_slots > 1 only.
  RED: /tmp/leg-control-c1.log and /tmp/leg-region-c1-diag.log (segfault at
  RAC native4 -> ttnn::copy, capturing=0, identical with and without
  VLLM_CPP_REGION_CAPTURE). GREEN: /tmp/leg-region-c1-fix.log — the cold step
  and the capture pass run; the leg proceeds to 8 segment captures before
  dying at the unrelated fit wall recorded in
  ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ and
  docs/bench-evidence/tt-region-capture-20260928.md. Focused device gate for
  the RAC lane itself rides the standing TT suite (see the row spec's ## Now).

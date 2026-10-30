ID: ISSUE-LOCAL-01M5Z079R7RFSCPB1F4P9C94S5
Title: Multi-group paged serving writes every group's cache with the TARGET group's slot mapping
Row: MODEL-TEXT-FLINT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-27
Updated: 2026-10-27
Closed: 2026-10-27

## Problem

MakeCommonAttentionMetadata (src/vllm/v1/attention/backend.cpp) selects ONE group's slot_mapping (step.slot_mapping[kv_cache_group_id]) into the single CommonAttentionMetadata the model forward receives, and GPUModelRunner gathers per-group block tables onto the multi_kv channel. The groups' pools allocate DIFFERENT block ids for the same positions (measured on flint-Q4_K_M: the full-attention group's first allocated block is 1, the sliding-window group's is 2), so any multi-group forward that writes its non-target groups' caches with the CAM's slot mapping puts every key in the wrong page; the pages the group's own block table names are still zero and the layer's attention output is exactly 0. Measured on the Flint paged engine (MODEL-TEXT-FLINT W-c): every sliding-window layer's attention output sum was exactly 0.0; after deriving slot = table[pos/bs]*bs + pos%bs from the layer's OWN published group table, the outputs came back. MiMoV2 (mimo_v2.cpp:364) reads attn_meta.slot_mapping for every layer the same way and is exposed identically whenever its groups' block ids diverge.

## Resolution

Fixed 2026-10-27 in the Flint paged arm in the same flow (worktree /tmp/flint-paged, branch row/FLINT-PAGED): when multi_kv is present the arm derives each layer's slot mapping from that layer's OWN published group block table (slot = table[pos/block_size]*block_size + pos%block_size) and writes the cache with it. Red-first evidence: every sliding layer's attention output sum exactly 0.0 with the CAM mapping on flint-Q4_K_M; nonzero after the fix. The engine-side single-CAM seam stays as found; MiMoV2 (mimo_v2.cpp:364) is exposed identically whenever its groups' ids diverge — the engine-side single-CAM seam belongs to the engine row that owns it and stays as found here.

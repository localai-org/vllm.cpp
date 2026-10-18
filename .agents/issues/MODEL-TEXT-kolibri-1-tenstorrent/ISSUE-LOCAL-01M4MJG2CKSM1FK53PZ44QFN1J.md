ID: ISSUE-LOCAL-01M4MJG2CKSM1FK53PZ44QFN1J
Title: kolibri1 TT streaming MoE: the slot-dequant memo grows without bound and exhausts device DRAM on free-running decode
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-11
Updated: 2026-10-11
Closed: 2026-10-11

## Problem

Kolibri1TTStreamingDeviceContext::memo (src/vllm/model_executor/models/kolibri1_tt_forward.h:160) caches one bf16 dequant per (layer, expert) ever routed to, 3*inter*h*2 = 7.86 MB per entry at the Kolibri-1 dims (h=2560, inter=512, 50 layers, 384 experts, top-6). Nothing bounds the map: a free-running decode selects ever-new experts and the entries' device buffers accumulate until the tt-metal allocator refuses (observed on the production-shape decode bench: TT_FATAL Out of Memory, 2.6 MB request with 300 KB free, deterministic at prefill step ~75 of the first run; the b2ii/b2bi gates never hit it because they run tens of forwards). The reset lane only clears on a slot-content change, which is not a growth bound.

## Resolution

2026-10-11: FIXED in the same flow (branch row/tt-kolibri-decode-bench,
src/vllm/model_executor/models/kolibri1_tt_forward.cpp, the memo insert):
the memo is capped at 512 entries (~4 GiB bf16, beside the 3.5 GiB resident
context and the slot pool). Eviction is correctness-free: a dropped entry is
re-dequanted bit-identically from the authoritative, readback-verified slot
bytes on the next touch. Evidence: without the cap the bench died with
TT_FATAL Out of Memory (2.6 MB request, 300 KB free) deterministically at
prefill step ~75; with the cap it completes a 192-forward free run in both
arms, and the b2ii/b2bi/tt gates stay green (13/13, 10/10, 16/16).


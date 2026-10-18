ID: ISSUE-LOCAL-01M4MHZM17MAAJJF1Z071YA7R0
Title: kolibri1 TT streaming MoE: a miss fetch's LRU eviction can evict an expert the same step selected
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

In MoeBlock's routed path (src/vllm/model_executor/models/kolibri1_tt_forward.cpp), PlanKolibri1TTMoEDispatch classifies residents/misses against the pre-fetch table, then the fetch loop Touch()es each miss. Touch on a full table evicts the global LRU slot, which can BE an expert this step's route selected as a resident; the compute loop then hits the 'expert N selected on layer L but no slot holds it' miss-handle refusal on a fully warm table. Observed twice on the production-shape decode bench (row/tt-kolibri-decode-bench), model /mnt/models/Aleph-Alpha/Kolibri-1, gfx under the GPU lease: layer 3, expert 258 resident per dispatch, evicted by the same step's expert-323 fetch; earlier run evicted 374. Resident slots' recency is only stamped at fill time, never on reuse, so a long-resident but-this-step-needed expert is the prime victim.

## Resolution

2026-10-11: FIXED in the same flow (branch row/tt-kolibri-decode-bench,
src/vllm/model_executor/models/kolibri1_tt_forward.cpp, the MoeBlock routed
path): the fetch loop now refreshes every SAME-STEP resident's recency
(Touch on a resident only restamps it) BEFORE staging the misses, so a
miss's LRU victim can no longer be an expert this step's route selected.
Evidence: the bench previously refused deterministically at layer 3
(experts 374 then 258 across runs); with the fix the bench completes a
192-forward free run on the card in BOTH arms (host-free on and off), and
the b2ii device token gate stays 13/13, b2bi 10/10, tt 16/16.


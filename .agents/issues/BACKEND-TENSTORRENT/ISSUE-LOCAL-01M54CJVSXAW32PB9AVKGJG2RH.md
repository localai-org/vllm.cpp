ID: ISSUE-LOCAL-01M54CJVSXAW32PB9AVKGJG2RH
Title: kolibri1 P150: the deterministic-decode memo/shadow fixes cost ~1.6x step time — the shadow drop re-stages evicted expert weights
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-17
Updated: 2026-10-17
Closed: -

## Problem

ad8d9b45b (memo eviction order + weight-view shadow drops at UnregisterHostBuffer/MarkScratchAcquired) fixed the run-to-run nondeterminism but read 1576 ms/step (median of 3, taskset -c 32-63, tt-smi reset recipe) vs 1000 ms/step on the immediate pre-fix parent under the identical recipe; the historical 610 ms/step eager-lever median was NOT reproducible today (host load, concurrent builds). Direct attribution (env-gated shadow-drop no-op measurement leg): prefill 214.6->166.7 s, decode 1556->1130 ms/step with the drop disarmed — the drop forces full re-staging of the 7.86 MB expert dequant views after every memo eviction; the residual gap to pre-fix is the eviction-order change plus host noise. PRICED FIX: make the memo entry own its staged device tensors (stage the three views at dequant time into the SlotDequant, serve from there) so eviction drops them exactly once and the pointer-keyed shadow map never caches transient-buffer views; UnregisterHostBuffer then needs no range scan on the pool re-hand path.

## Resolution

-

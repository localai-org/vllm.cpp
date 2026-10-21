ID: ISSUE-LOCAL-01M54CJVSXAW32PB9AVKGJG2RH
Title: kolibri1 P150: the deterministic-decode memo/shadow fixes cost ~1.6x step time — the shadow drop re-stages evicted expert weights
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-17
Updated: 2026-10-21
Closed: -

## Problem

ad8d9b45b (memo eviction order + weight-view shadow drops at UnregisterHostBuffer/MarkScratchAcquired) fixed the run-to-run nondeterminism but read 1576 ms/step (median of 3, taskset -c 32-63, tt-smi reset recipe) vs 1000 ms/step on the immediate pre-fix parent under the identical recipe; the historical 610 ms/step eager-lever median was NOT reproducible today (host load, concurrent builds). Direct attribution (env-gated shadow-drop no-op measurement leg): prefill 214.6->166.7 s, decode 1556->1130 ms/step with the drop disarmed — the drop forces full re-staging of the 7.86 MB expert dequant views after every memo eviction; the residual gap to pre-fix is the eviction-order change plus host noise. PRICED FIX: make the memo entry own its staged device tensors (stage the three views at dequant time into the SlotDequant, serve from there) so eviction drops them exactly once and the pointer-keyed shadow map never caches transient-buffer views; UnregisterHostBuffer then needs no range scan on the pool re-hand path.

## Resolution

2026-10-21: STILL TRUE, with the attribution changed. The PRICED FIX this issue
names landed as `93e37e703` (PR #3468): the memo entry now owns its staged
device tensors (`SlotDequant::staged`,
`src/vllm/model_executor/models/kolibri1_tt_forward.h:172-180`, staged at
`kolibri1_tt_forward.cpp:882-892`), the shadow sweeps are off the decode hot
path (172,800 owned serves, 142,740 owned stages, 0 sweep events), and the
pointer-keyed shadow map no longer caches views of the transient buffer. The
mechanism this issue names -- the shadow drop re-staging evicted expert weights
-- no longer exists.

The cost, however, persists and is re-attributed: step time is unchanged (1559
-> 1567 ms/step) because the memo thrashes. The cap of 512 entries against about
300 distinct experts per forward gives a hit rate near zero, so every touch
re-uploads 7.86 MB H2D. The pre-#3452 1000 ms/step was fast because the stale
pointer-keyed cache skipped that upload while serving the previous tenant's
weights, which is the bug #3452 closed.

What settles it: one of the three levers the landing commit names -- a
content-keyed expert staging cache, a device-side fp8->bf16 dequant, or raising
the cap -- followed by a step-time measurement. The residual has no open owner:
the memo-bound record `ISSUE-LOCAL-01M4MJG2CKSM1FK53PZ44QFN1J` is CLOSED.

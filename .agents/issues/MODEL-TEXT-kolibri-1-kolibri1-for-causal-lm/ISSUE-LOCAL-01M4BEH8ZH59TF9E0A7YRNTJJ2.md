ID: ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2
Title: fp8 dequant cache: cached blocks aliased while leased; W3 fails with an active cache
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-07
Updated: 2026-10-07
Closed: 2026-10-07

## Problem

The kolibri-1 CPU perf session (2026-10-07, worktree /tmp/vllm-kolibri-neon2, branch row/kolibri-neon2) implemented the recorded next lever, an LRU dequant cache (VT_KOLIBRI1_DEQUANT_CACHE_MB) over kolibri1_forward.cpp LinearBT's fp8 arm. Unit-level the cache is byte-faithful: tests assert a hit never re-invokes the decode callback and returns bit-identical bytes (red-first, green). In-production it is NOT output-faithful: with an active cache the W3 gate fails (7 hard flips, worst topk logit diff 7.15 vs band 2.5), while hits=0 runs reproduce the landed gate byte-for-byte. Diagnostics: (1) per-hit memcmp against a fresh decode matched on every hit in the bench build, yet W3 outputs changed materially (step-1 logit 5.86 vs 7.08, different token); (2) a decode-target tripwire fired POOL DOUBLE-HAND-OUT: DequantFp8Block's fresh DBuf landed on a pointer still owned by a live cache entry; (3) W3 profile with active cache at 10 forwards: dequant_fp8_block 1.65 s vs 16.5 s uncached, total_forward 18.26 s vs ~35 s — the performance mechanism works. Suspect: a DevicePool lease/Put interaction (blocks leased via DBuf::ReleaseShared while cached are re-handed to new allocations), or a second owner created outside the pool; a raw free-list tripwire over DevicePool::Put/Get is inconclusive because multiple pool instances share one static set. Full A/B tables, harness shapes and repro recipes: docs/bench-evidence/kolibri1-dequant-cache-negative-20261007.md. The lever was REVERTED from the tree per the row's bit-exactness contract; this issue owns the follow-up investigation (fresh reviewer + mutation on the pool lease path) before the cache lever can land.

## Resolution

- 2026-10-07 (row/kolibri-pool-lease): the pool half is localized, reproduced
  red at pool level, and fixed. Red first: two new `test_device_pool` cases
  drive the cache's lease pattern (`allocate -> DBuf::ReleaseShared carrier ->
  same-class traffic -> Get`) and the defect itself. The pattern case is GREEN
  on main — an innocent pool never double-hands — so the observed aliasing
  required a second return of a live-leased block; the refusal case
  (`DevicePool::Put` of a block whose carrier is alive) is RED on main with
  the exact observed symptom: the pool accepted the second return in silence,
  the block entered the free list, and the next `DBuf` of the class received
  the leased pointer with the carrier's bytes overwritten (the
  `POOL DOUBLE-HAND-OUT` / `CACHE MISMATCH` mechanism, at
  device_pool.h `Put` -> `TakeBlockClass`, which could not tell a second
  owner's return from a legal one). Fix: `DBuf::ReleaseShared` now registers
  the block as LEASED with its own pool, the carrier's deleter returns it
  through `DevicePool::PutLeased` (which discharges the lease), and any OTHER
  `Put` of a leased block throws by name instead of aliasing. `Put`/`Get`
  signatures and reuse behavior are unchanged; both debug lanes
  (`VT_POOL_BYPASS`, `VT_POOL_EXACT`) and the existing 14-case suite stay
  green. REMAINING OPEN: the cache's own side is NOT exonerated and cannot be
  verified here — the lever is reverted and its diff was never committed, so
  which of its paths performed the second return (a second carrier over the
  same block, or weight-identity keying handing one block to two entries) is
  unknown. Any re-attempt of the cache lever must be built against the
  hardened pool: a second return now aborts loudly at the `Put`, so the
  corruption mode this issue records cannot recur silently. kolibri-1 gates on
  the fix tree: test_kolibri1 27/186, test_kolibri1_dequant 3/10,
  test_kolibri1_w3 900/900 chain 141/145 (4 near-tie, 0 hard), fingerprints
  2.18646/2.6763 — no regression.

- 2026-10-07 (row/kolibri-cache2): CLOSED. The cache's second-return question
  is resolved BY CONSTRUCTION — the re-landed cache's entries OWN their bytes
  (independent `std::vector<uint16_t>` allocations sized to the block), never a
  pool block, a `DBuf`, or any memory the cache did not allocate, so a cache
  entry cannot be handed out from under by the pool, and there is no second
  return for the hardened pool to refuse. The hardened-pool throw never fired
  in any cached run. The passing hitting-budget gates close the behavior half:
  `test_kolibri1_w3` at `VT_KOLIBRI1_DEQUANT_CACHE_MB=16384` with
  hits=348418 is 900/900, ARGMAX chain 141/145 (4 near-tie, 0 hard),
  fingerprints 2.18646/26763 — byte-identical to the landed baseline;
  `test_kolibri1_decode_bench` anchor 109726 with a byte-identical CHAIN at
  off/8/16 GiB (16 GiB hits=71157); `test_kolibri1` 27/186;
  `test_kolibri1_dequant` 3/10; new `test_kolibri1_dequant_cache` 3/34
  (byte identity cold/hot, budget invariance in one binary, ownership under
  pool traffic), compile-red on base, two mutation reds. Decode bench 2.50x
  (49.40 -> 19.78 s) at 16 GiB. Evidence:
  docs/bench-evidence/kolibri1-dequant-cache-reland-20261007.md.

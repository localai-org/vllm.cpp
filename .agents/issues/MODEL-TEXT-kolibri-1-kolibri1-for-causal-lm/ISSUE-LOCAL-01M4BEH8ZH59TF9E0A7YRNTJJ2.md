ID: ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2
Title: fp8 dequant cache: cached blocks aliased while leased; W3 fails with an active cache
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-07
Updated: 2026-10-07
Closed: -

## Problem

The kolibri-1 CPU perf session (2026-10-07, worktree /tmp/vllm-kolibri-neon2, branch row/kolibri-neon2) implemented the recorded next lever, an LRU dequant cache (VT_KOLIBRI1_DEQUANT_CACHE_MB) over kolibri1_forward.cpp LinearBT's fp8 arm. Unit-level the cache is byte-faithful: tests assert a hit never re-invokes the decode callback and returns bit-identical bytes (red-first, green). In-production it is NOT output-faithful: with an active cache the W3 gate fails (7 hard flips, worst topk logit diff 7.15 vs band 2.5), while hits=0 runs reproduce the landed gate byte-for-byte. Diagnostics: (1) per-hit memcmp against a fresh decode matched on every hit in the bench build, yet W3 outputs changed materially (step-1 logit 5.86 vs 7.08, different token); (2) a decode-target tripwire fired POOL DOUBLE-HAND-OUT: DequantFp8Block's fresh DBuf landed on a pointer still owned by a live cache entry; (3) W3 profile with active cache at 10 forwards: dequant_fp8_block 1.65 s vs 16.5 s uncached, total_forward 18.26 s vs ~35 s — the performance mechanism works. Suspect: a DevicePool lease/Put interaction (blocks leased via DBuf::ReleaseShared while cached are re-handed to new allocations), or a second owner created outside the pool; a raw free-list tripwire over DevicePool::Put/Get is inconclusive because multiple pool instances share one static set. Full A/B tables, harness shapes and repro recipes: docs/bench-evidence/kolibri1-dequant-cache-negative-20261007.md. The lever was REVERTED from the tree per the row's bit-exactness contract; this issue owns the follow-up investigation (fresh reviewer + mutation on the pool lease path) before the cache lever can land.

## Resolution

-

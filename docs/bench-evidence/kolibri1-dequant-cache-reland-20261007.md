# Kolibri-1 CPU: dequant-cache lever RELANDED (MODEL-TEXT-kolibri-1)

Date: 2026-10-07. Host: aarch64, 128 cores, 255 GB RAM, Linux. Worktree
`/tmp/vllm-kolibri-cache2`, branch `row/kolibri-cache2`, base = `origin/main`
d780204dd merged with `mine/row/kolibri-pool-lease` b6b7eb790 (the pool-lease
hardening; merge commit 33c043a09). All runs `VLLM_CPP_CPU_THREADS=8`; Release,
`-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_TENSTORRENT=OFF`, built in `/tmp`.

VERDICT: the lever LANDS. The first attempt was reverted when the row's token
gate failed with an active cache (7 hard flips; post-mortem:
`kolibri1-dequant-cache-negative-20261007.md`, branch `row/kolibri-neon2`).
The re-land passes EVERY gate, with the cache active and hitting, byte-for-byte
on the landed fingerprints. Both halves of
ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2 are resolved: the pool half by the
hardened pool, the cache half by OWNED-BYTE entries — see "Ownership" below.

## What changed vs v1 (the reverted lever)

v1 stored each entry's decoded block in a POOL BLOCK held through
`DBuf::ReleaseShared`. The pool re-handed that live block to a fresh `DBuf`
and a live cache entry became another tensor's bytes — the `POOL DOUBLE-HAND-OUT`
/ `CACHE MISMATCH n=512 k=2560` mechanism.

The re-land (`src/vllm/model_executor/models/kolibri1_dequant_cache.h`):

- OWNERSHIP: an entry OWNS its bytes in an independent `std::vector<uint16_t>`
  sized to the block. No entry ever holds a pool block, a `DBuf`, or any memory
  the cache did not allocate. The `GetOrDequant` return value is a pointer into
  entry-owned storage, valid until the caller's NEXT call on the same `Cache`
  — strictly longer than the one `vt::MatmulBT` in `LinearBT` needs. `LinearBT`
  (`kolibri1_forward.cpp`) feeds it to the GEMM as a non-owning `MakeTensor`
  view and holds no further cache call in between; that IS the lifetime
  contract, documented in the header.
- KEYING: unchanged from v1 and unchanged in kind — the weight's OWN
  packed/scale `OwnedTensor` data pointers plus (n, k, block_n, block_k). Those
  buffers are `Kolibri1Weights`-owned with process lifetime; the key never
  names a pool block or any transient allocation.
- BUDGET: `VT_KOLIBRI1_DEQUANT_CACHE_MB` (default **0 = DISABLED** — v1
  defaulted 8192). Disabled keeps the ORIGINAL threaded pool decode in
  `LinearBT` byte-for-byte, so main-line behavior is unchanged unless the var
  is set. An entry larger than the whole budget is never cached. LRU eviction.
- COLD-MISS DECODE: threaded through `host_parallel::ForOutputRows` exactly as
  `DequantFp8Block` partitions (output rows only, so bit-identical to serial by
  construction). Measured WITHOUT this first: a serial cold decode tripled the
  bench prefill (44.3 s vs 12.8 s) and ate most of the win; the shipped form
  restores prefill to ~15-16 s.
- COUNTERS: hits/misses/evictions/decode_calls, printed by the
  `VT_KOLIBRI1_PROFILE` report.

## Red-first evidence (`tests/vllm/models/test_kolibri1_dequant_cache.cpp`)

- RED (compile, base copy): the new test target against base 33c043a09 fails —
  `test_kolibri1_dequant_cache.cpp:31:10: fatal error:
  vllm/model_executor/models/kolibri1_dequant_cache.h: No such file or
  directory`.
- GREEN: 3/3 cases, 34 assertions:
  1. BYTE IDENTITY — hits serve bit-identical raw-uint16 bytes to a fresh
     kernel decode, cold AND hot, over all 256 e4m3 bytes x 8 representative
     scales (incl. the 64-tie 1.015625 and full-mantissa 1.1), plus same-
     pointer stability across repeated hits.
  2. BUDGET INVARIANCE — one binary, one decode chain over 8 distinct weights
     with revisits: budgets 0 (off), 2-entry (forcing LRU eviction), and
     16-entry all produce the identical chain hash, equal to the no-cache
     kernel chain; the evicting budget shows evictions > 0 and hits > 0.
  3. OWNERSHIP — a warm hit entry's bytes (the 512x2560 shape the CACHE
     MISMATCH named) are unchanged, pointer-identical, after 8 rounds of
     same-size-class pool Get/Put traffic whose blocks are first filled with a
     junk pattern.
- MUTATION reds (each run, then restored byte-for-byte):
  - garbled decode tail → 3/3 cases FAIL (24 assertions fail) — tests 1-3 all
    detect a wrong byte.
  - v1-style aliasing (all entries view one shared scratch) → 1 case, 1
    assertion FAIL — the byte-identity/hot-hit path detects cross-entry
    aliasing, the exact v1 corruption shape.

## Gates (final binary; cache ACTIVE and hitting)

- `test_kolibri1_w3` at `VT_KOLIBRI1_DEQUANT_CACHE_MB=16384`,
  `VT_KOLIBRI1_PROFILE=1`, 8 threads: **900/900 assertions PASS; ARGMAX chain
  141/145 (4 near-tie, 0 hard); FINAL-STEP FINGERPRINT worst topk 2.18646,
  worst sum diff 26763 — identical to the landed baseline.** Counters:
  hits=348418, misses=519935, evictions=514482, decode_calls=868353. Whole-gate
  wall 8:43.87 (baseline 9:18.71), max RSS 96.5 GB (baseline 76.5 GB: the
  16 GiB cache on top of the same working set).
- `test_kolibri1_decode_bench`: anchor last token **109726** (chain
  101807/109726 alternating) at EVERY budget — off, 8 GiB, 16 GiB — with the
  CHAIN lines byte-identical (same md5) across all three runs. Runs and hits:
  off n/a; 8 GiB hits=69377, misses=20497, evictions=18321; 16 GiB
  hits=71157, misses=18717, evictions=13264.
- `test_kolibri1`: 27/27 cases, 186/186 assertions PASS (the 27/186-split, as
  on main).
- `test_kolibri1_dequant`: 3/3 cases, 10/10 assertions PASS (as on main).
- `test_kolibri1_dequant_cache`: 3/3, 34/34 (above).
- The hardened pool's `still leased` throw NEVER fired in any cached run: with
  owned bytes there is no second return to catch. The diagnostic is unneeded
  by construction, not suppressed.

## A/B: production-shape decode bench, 8 threads, this tree

Off vs on measured on THIS tree (base + pool fix + cache). The NEON linear_gemm
lever is landing in parallel on another branch; these numbers will be
re-measured on the post-NEON tree before any production claim. Machine had a
concurrent foreign `test_kolibri1_w3` during the off run (single-digit percent
contention at most on a 128-core host; recorded for honesty).

| axis (128-in, 63 greedy tok) | cache OFF | 8 GiB | 16 GiB |
|---|---|---|---|
| decode wall | 49.40 s | 20.98 s | 19.78 s |
| decode multiple vs OFF | 1.00x | 2.35x | **2.50x** |
| decode tok/s | 1.275 | 3.004 | 3.185 |
| prefill wall | 12.83 s | 16.21 s | 15.45 s |
| whole wall | 62.23 s | 37.18 s | 35.23 s |
| greedy chain | 109726 | 109726 | 109726 |

Higher than the post-mortem's 2.10x because that figure was measured with a
SERIAL cold decode on the ON side and because the budget sweep there was
measured on the scratch harness; here the miss path matches the uncached
threaded decode. Budget sweep at the gate shape: W3 (working set ≈ 10 GB,
fresh re-prefill per forward) at 16 GiB still evicts heavily
(evictions≈misses) — the hit rate is budget-bound and grows with budget, as
the post-mortem recorded. The default stays 0 (off): the cache is an explicit
opt-in until a budget policy is chosen for production.

## Records

- Fix + lever: `kolibri1_dequant_cache.h`, `kolibri1_forward.cpp` (LinearBT
  fp8 arm + profile report), tests + CMake target, allowlist line for
  `VT_KOLIBRI1_DEQUANT_CACHE_MB`.
- ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2: CLOSED by this change.

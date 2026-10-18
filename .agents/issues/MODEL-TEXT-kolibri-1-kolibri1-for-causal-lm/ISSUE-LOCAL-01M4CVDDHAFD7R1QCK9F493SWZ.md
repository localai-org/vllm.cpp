ID: ISSUE-LOCAL-01M4CVDDHAFD7R1QCK9F493SWZ
Title: maint-bot review of PR #3414 (row/kolibri-cache2): dequant-cache identity outlives weights, hit bytes unprotected across the GEMM, POSIX-only test
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: 2026-10-08

## Problem

The re-landed kolibri1 fp8 dequant cache (kolibri1_dequant_cache.h, landed via ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2 on row/kolibri-cache2, PR #3414) has three review findings. P1a: the process-wide ProcessCache() keys entries on the raw packed/scale buffer addresses, but retains entries past the owning model's lifetime — after unload+reload, reused addresses with the same geometry serve the previous model's decoded bytes. P1b: GetOrDequant returns a raw pointer after unlocking; a concurrent caller (distinct engines share the static cache) can evict/overwrite the bytes before the consuming GEMM finishes. P2: tests/vllm/models/test_kolibri1_dequant_cache.cpp unconditionally includes sys/wait.h/unistd.h and calls fork/execle/waitpid — MSVC cannot compile it, though CMake registers the target on all platforms.

## Resolution

- 2026-10-08 (row/kolibri-cache2, follow-up to HEAD 24c7d1ae9, PR #3414): all
  three findings repaired in one change.
  - P1a — MODEL-GENERATION IDENTITY IN THE KEY (chosen over teardown
    invalidation: no model-destructor coupling to the process-lifetime static
    cache, so no static-destruction-order hazard, and stale entries age out
    through ordinary LRU pressure with no new teardown path to get wrong).
    `detail::Key` gains the model generation; `Kolibri1LoadedModel`'s
    constructor bumps the process-wide atomic generation at every model
    load (kolibri1_registry.cpp), so a reload that reuses the previous
    model's weight-buffer addresses can never hit the previous model's
    entries. Regression (test case 6): two borrowed-arena weights — the
    mmap shape real safetensors weights have — occupy the SAME addresses by
    construction (deterministic under any allocator, ASan's quarantine
    included) with the same geometry and different bytes; the served bytes
    must equal a fresh decode of the second model. RED evidence: (i) scenario
    red on the pre-fix code with allocator-reused addresses
    (`CHECK_EQ(memcmp(got2, want2), 0)` -> -1, stale hit); (ii) mutation red
    on the fixed tree with the generation dropped from the key -> case 6
    fails at the same assertion. GREEN: 7/7 cases, 52/52 assertions.
  - P1b — OWNERSHIP LEASE (chosen over serializing lookup-and-consume:
    holding the cache mutex across the GEMM would serialize distinct
    engines' GEMMs — the expensive half — for no correctness gain). Entries
    hold their bytes through a `shared_ptr`; `GetOrDequant` returns a
    `Lease` that co-owns the bytes, so eviction frees the cache's reference
    but a live lease keeps the bytes alive and unchanged — the guarantee is
    by construction, the same ownership argument that fixed v1, and the
    smaller diff (no pinned flags, no deferred removal, no lock across the
    GEMM). `LinearBT` holds the lease across `vt::MatmulBT` and releases it
    at scope end. The disabled/oversized path now decodes into a fresh
    per-call lease-owned allocation (the old shared member scratch could be
    clobbered by the next call, which the lease contract forbids);
    `LinearBT` never calls `GetOrDequant` when disabled, so main-line
    behavior and its allocation profile are unchanged. Budget note: an
    evicted-but-leased entry is counted as evicted, so a live lease
    transiently overshoots the budget by at most one in-flight GEMM's
    weight. Regression (test case 7): barrier-synchronized two callers — A
    holds a hit live while B evicts A's entry and drives same-size scratch
    decodes; A's bytes must remain intact. RED evidence: (i) scenario red
    on the pre-fix code (`CHECK(a_bytes_intact)` -> false, B's eviction
    reused A's freed bytes); (ii) AddressSanitizer on the pre-fix TU:
    `heap-use-after-free`, READ of 65536 bytes by caller A
    (test_kolibri1_dequant_cache.cpp:370) on memory freed by caller B's
    eviction (lines 366/376); (iii) mutation red on the fixed tree with
    non-owning leases -> case 7 fails at `CHECK(a_bytes_intact)`. GREEN:
    7/7 cases, 52/52 assertions; ASan (CI-lane env `VT_POOL_BYPASS=1`,
    `detect_leaks=1`) clean, no use-after-free, no leaks; TSan clean, no
    warnings.
  - P2 — PORTABILITY GUARDS. Every POSIX-only construct in
    test_kolibri1_dequant_cache.cpp (the `<sys/wait.h>`/`<unistd.h>`
    includes and the whole fork/execle/waitpid probe body of case 5) is
    gated on `#if !defined(_WIN32)`; on Windows the probe is
    disabled-and-documented via a `WARN` in the case body. The portable
    cache cases (1-4, 6, 7) are ungated and compile in both branches; the
    CMake target stays registered on all platforms and the helper binary is
    portable (it only uses `getenv` and the cache header). Verified: the test
    TU and the helper TU both pass a `-D_WIN32 -fsyntax-only -Werror`
    compile (the `_WIN32` branch is what gets checked), and the `_WIN32`
    preprocessed output contains no `fork`/`execle`/`waitpid`/`sys/wait`/
    `unistd`/`_exit` from this file (remaining matches are libstdc++/glibc
    header internals and the WARN message string). No cross toolchain was
    available, so MSVC-cleanliness is by this syntax check plus inspection.
- Gates (Release, /tmp/build-kolibri-cache2, VLLM_CPP_CPU_THREADS=8,
  aarch64 128-core host): test_kolibri1_dequant_cache 7/7 cases 52/52
  assertions PASS; test_kolibri1 27/27 cases 186/186 assertions PASS;
  test_kolibri1_dequant 3/3 cases 10/10 assertions PASS;
  test_kolibri1_w3 at VT_KOLIBRI1_DEQUANT_CACHE_MB=16384,
  VT_KOLIBRI1_PROFILE=1: 900/900 assertions PASS, ARGMAX chain 141/145 (4
  near-tie, 0 hard), FINAL-STEP FINGERPRINT worst topk 2.18646, worst sum
  diff 26763 — identical to the landed baseline; cache counters identical
  to the landed run (hits=348418, misses=519935, evictions=514482,
  decode_calls=868353); wall 532 s (landed 8:43.87). decode_bench A/B:
  anchor last token 109726 at BOTH budgets with byte-identical CHAIN lines
  (md5 8768afca321c4c5a6a62ef96f48d61d3 off and on); decode wall 50.21 s off
  vs 19.93 s at 16 GiB (landed 19.78 s, +0.8% — within the ±10% band, so the
  lease design does not change cache performance materially); 16 GiB
  counters identical to the landed run (hits=71157, misses=18717,
  evictions=13264). check-agent-record rc=0; agent-preflight rc=0.

## Owed

- (none — all three findings repaired and gated in this change)

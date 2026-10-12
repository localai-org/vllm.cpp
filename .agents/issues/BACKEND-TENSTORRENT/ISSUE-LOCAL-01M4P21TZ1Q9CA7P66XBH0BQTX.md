ID: ISSUE-LOCAL-01M4P21TZ1Q9CA7P66XBH0BQTX
Title: kolibri1 TT greedy decode is run-to-run nondeterministic on P150 (thalia)
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-11
Updated: 2026-10-12
Closed: 2026-10-12

## Problem

On the P150 (host thalia, /tmp/umdtrial-install, /tmp/tt-metal-umdtrial), the kolibri1 Tenstorrent greedy decode chain is not reproducible run to run on identical code and input: three runs of the same one-token-per-step walk produced first tokens 346, 30686, and 10, and the b2ii gate's hard flips move between runs on unmodified origin/main alone. Chain identity is not adjudicable on this path; the b2ii gate is a band gate (141/145, 2.5-nat near-tie teacher-forced). This nondeterminism poisons every measurement. Investigate which decode arm(s) are nondeterministic (host-free ON vs OFF), bisect to the first run-to-run divergent stage via VT_KOLIBRI1_TT_STAGE_DUMP, classify (residency/stale read, async queue race, host thread-pool op ordering, or device FP nondeterminism), and fix if it is a bug in our path.

## Resolution

- ROOT CAUSE (two bugs, both in our path, both on the kolibri1 B2b-ii
  streaming-MoE memoized dequant cache; classification: residency/stale-read
  + use-after-free, NOT an async queue race and NOT device FP nondeterminism):

  1. DANGLING-ITERATOR USE-AFTER-FREE in the memo eviction
     (kolibri1_tt_forward.cpp MoeBlock). The memo is capped at 512 entries;
     the old order emplaced the new (layer, expert) dequant and THEN evicted
     the smallest key (`while (size > cap) erase(begin())`). Once the memo's
     smallest key caught up to the current (layer, expert) — first observed
     at the 11th forward of the decode bench — the victim WAS the
     just-emplaced entry: `it` dangled, and the ExpertMlp below read
     gate/up/down views plus a 7.86 MB dequant buffer that the erase had
     already be.Free'd. Measured symptom: routed-expert GEMM outputs of
     uninitialized-DRAM scale (~1e28-1e37) for the caught-up experts while
     the gathered inputs were bit-identical; the divergence enters the hidden
     chain at that expert's layer and propagates to the logits.

  2. STALE WeightViewShadow SERVE ON ADDRESS REUSE
     (tenstorrent_residency.cpp). The memo's up/down expert-weight views are
     INTERIOR pointers into the 7.86 MB dequant buffer, cached in
     WeightViewShadows() keyed by the view's host pointer. That map was never
     invalidated when the buffer was freed: malloc reuses the address for the
     next dequant (measured: 2357 reused addresses in one 192-step run), and
     the new expert's up/down GEMMs stale-served the OLD expert's device
     weights (EnsureWeightViewDevice's hit path returns the cached tensor
     without comparing bytes). Onset exactly when the memo cap is first
     crossed and evictions begin (the 3rd forward of the bench walk). The
     leak also OOM'd the near-full device on the longer b2ii gate (3/3
     pre-fix runs died at the same 2.5 MB bank allocation; the post-fix gate
     completes).

  Both arms (host-free ON and OFF) were nondeterministic — the defect is in
  the shared residency/memo layer, upstream of the arm-specific residual
  paths. CPU was deterministic (3/3 identical chains). Kolibri1 never
  captures (no capture call in the TT path), so the only decode modes are the
  two eager host-free arms.

- BISECTION EVIDENCE: per-step logits/dnorm/hidden/res hashes (scratch,
  since removed) localized the first divergence to the 3rd forward, in the
  mid layers, entering the hidden chain through the MoE; the memo trace
  (FETCH/HIT/MEMO/EVICT) showed the first behavioral difference as the
  router's expert selection diverging after a corrupted MoE output; the
  per-expert weight host-byte hashes pinned the first differing weight to a
  freshly dequanted expert whose memo entry the eviction had just erased
  (victim == the just-emplaced key). Control experiments: no-eviction runs
  (raised cap) and leak-the-evicted-entries runs were deterministic through
  the compared prefix; host-leak-only (device shadow still destroyed) was
  deterministic, isolating the host free + address reuse as the trigger.

- FIX (row/tt-kolibri-nondet, two changes):
  1. kolibri1_tt_forward.cpp: evict BEFORE emplacing
     (Kolibri1TTMemoEvictForEmplace, extracted as a testable helper), so the
     just-emplaced entry is never the victim and `it` stays valid.
  2. tenstorrent_residency.cpp: DropWeightViewShadowsInRange sweeps
     WeightViewShadows + Bfp8WeightShadows for keys inside a retiring
     allocation, called from UnregisterHostBuffer (be.Free; no drain — the
     memo eviction's queue is already drained by the preceding scatter
     readback) and from MarkScratchAcquired (pool re-hand; drains first).
     Retention was measured and rejected: any retained transient OOMs this
     near-full device (the prior session's retired-shadow-ring rejection).

- ACCEPTANCE (the deterministic check): 3 identical bench runs per arm,
  identical full 65-token greedy chains:
    host-free ON  (default): md5 75f2b5c4a4c12f40410e2a964c333498 x3
    host-free OFF (VT_TT_HOST_FREE_DECODE=0): md5 89e5b411c6e3790cd090874de36db68a x3
  Pre-fix the chains differed every run (first tokens 100/44/379 ON,
  2213/44/861 OFF).

- REGRESSION TESTS: tests/vllm/models/test_kolibri1_tt.cpp — the memo
  eviction order (the just-emplaced entry survives; the old order erased it
  — the deterministic red); tests/vt/test_tenstorrent_backend.cpp — freeing
  a buffer drops the weight-view shadows keyed inside it (the old code leaked
  the entry — the deterministic red). test_kolibri1_tt: 17/17 pass.

- GATES: TT unit suite (test_tenstorrent_backend): 104/107 — the 3
  failures are exactly the 2 known failures (kGdnDecode wide-range state,
  kMatmulBTQuantGrouped P=1 Q4_K) + the 1 known order-state flake (batched
  decode RAC num_slots=2); no regression, and the new sweep test passes.
  b2ii gate (both arms): completes deterministically now (pre-fix it OOM'd
  3/3 this session on the leaked WeightViewShadow tensors, and its flips
  moved between runs). The gate's CHECKs stay RED by design (the B2b-ii
  record owns the residual). Post-fix verdicts: ON — ARGMAX CHAIN 0/8
  (7 hard, 1 near), instrument 140 tf flips (128 HARD, worst 13.5 nats);
  OFF — 0/8 (7 hard, 1 near), instrument 142 tf flips (128 HARD, worst 16.2
  nats); identical across repeated runs. All flips are semantic (nat gaps
  2-16), not garbage.

- FOUND, NOT FIXED (out of scope, reported for the gate's owner): the b2ii
  ARGMAX CHAIN compares the d-th decoded token against
  generated_ids[step] instead of generated_ids[step - input_len] — an
  off-by-input_len index introduced with the one-token prompt walk
  (9124217b3). The recorded 26/33 baseline predates that rewrite; under the
  current indexing the compared/match counts are not comparable to it. With
  the correct index, the post-fix run matches the golden at the first
  decoded step for 3 of 8 prompts (the rest are semantic near-ties/hard
  flips, the known B2b-ii residual).

- PERF NOTE: decode ~104 s vs ~73.5 s pre-fix on the bench (+40%): the cost
  of destroying the ~136k stale weight-view tensors at each memo eviction
  (correctness requires the drop; retention OOMs the device). Prefill
  ~214 s vs ~181 s. A generation-counter scheme could cut the destruction
  count ~30x — follow-up if the decode regression matters.


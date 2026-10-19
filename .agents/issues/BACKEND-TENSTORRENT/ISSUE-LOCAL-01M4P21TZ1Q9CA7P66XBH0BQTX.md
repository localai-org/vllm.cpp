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

- FIXED (row/tt-b2ii-index, 2026-10-12): the b2ii ARGMAX CHAIN compared
  the d-th decoded token against generated_ids[step]. `step` is the GLOBAL
  walk step (the prompt walks one token per step, 9124217b3), while the
  goldens' generated_ids is laid out over DECODE positions only (32
  entries, no prompt prefix — kolibri1_goldens.json, max_new_tokens=32),
  so every scored position was off by input_len. The correct index is
  generated_ids[step - input_len] = generated_ids[decode position], the
  W3 methodology's indexing (test_kolibri1_w3.cpp:314). Three sites in
  tests/vllm/models/test_kolibri1_tt_b2ii.cpp carried the same class of
  bug; all three are fixed: (1) the ARGMAX CHAIN comparison — its flip
  records now name the DECODE position, comparable to the W3 records;
  (2) the near-tie instrument's teacher-forced walk fed
  generated_ids[step-1] = generated_ids[input_len + d - 1], a SHIFTED
  chain, so pass 1 never ran on the golden prefix and its teacher-forced
  flip counts were meaningless — it now feeds generated_ids[d-1]; (3) the
  SCRATCH dbg TF walk fed generated_ids[i - input_len - 1], off by one
  and an out-of-bounds read (SIZE_MAX index) at the first decode step — it
  now feeds the golden token at position i.
  RED (wrong index, this tree, both arms, 2026-10-12; reproduces this
  issue's recorded verdicts exactly): ON — ARGMAX CHAIN 0/8 (7 hard, 1
  near), instrument 140 teacher-forced flips (128 HARD, worst 13.5
  nats); OFF — 0/8 (7 hard, 1 near), instrument 142 teacher-forced flips
  (128 HARD, worst 16.2 nats). All 16 flips sit at each prompt's FIRST
  compared position, recorded at step = input_len with want =
  generated_ids[input_len] — the wrong positions: the gate scored 8
  positions in total, one per prompt, and never left decode position 0.
  GREEN (fixed index, both arms, 2026-10-12; two identical runs per arm,
  the acceptance above re-checked on the corrected indexing):
  ON — ARGMAX CHAIN 3/11 (3 near-ties, 5 hard): prompt 4 matches at
  decode 0, prompt 5 at decode 0 and 1, the other six prompts flip at
  decode 0 (genuine first-position drift, gaps -0.998 to -10.617 nats);
  instrument 170 free-running divergences, 69 teacher-forced flips (29
  near-ties, 40 HARD, worst 10.617 nats). Both runs identical, streaming
  counters byte-identical (slot fills 4718, staged 18551930880 B,
  readback-verified 101316034560 B, memo hits 4452).
  OFF — ARGMAX CHAIN 12/20 (6 near-ties, 2 hard): prompt 3 matches decode
  0-8 and flips at decode 9 (gap -0.474), prompt 5 matches decode 0-1 and
  flips at decode 2 (gap -0.970), prompt 7 matches decode 0 and flips at
  decode 1 (gap -1.727); the other five prompts flip at decode 0 (gaps
  -0.236 to -6.157). Instrument 158 free-running divergences, 44
  teacher-forced flips (27 near-ties, 17 HARD, worst 8.647 nats). Both
  runs identical.
  The counts moved from 0/8 to 3/11 (ON) and 12/20 (OFF): the gate now
  scores the real decode chain, and the flips are the known B2b-ii
  residual (semantic near-ties just past the band on the OFF arm, the
  signature recorded in the 2026-10-09 evidence). The CHECKs stay RED by
  design.
  RECORD CORRECTION: the note above ("the recorded 26/33 baseline
  predates that rewrite") was wrong. The 26/33 was recorded 2026-10-09/10
  (docs/bench-evidence/kolibri1-tt-b2ii-20261009.md sections 6-7) on trees
  that already carried the walk 9124217b3 — the pre-walk gate scored 0/8
  on the zero-KV step 0, so a 26/33 verdict cannot predate the walk.
  Under the wrong index even a PERFECT engine scores 48/54: the six
  varied-golden prompts flip at decode position 0 (generated_ids[0] !=
  generated_ids[input_len]) and only the two repetitive-golden prompts
  (121598-repeated, 12843-repeated) are unaffected, so 26/33 decomposes
  exactly as 6x1 plus 27 from those two — a wrong-index artifact, not a
  correct-index baseline. The comparable reference is the W3
  methodology's decode-position indexing (the 141/145 CPU baseline the
  gate's CHECK is calibrated to).

- PERF NOTE: decode ~104 s vs ~73.5 s pre-fix on the bench (+40%): the cost
  of destroying the ~136k stale weight-view tensors at each memo eviction
  (correctness requires the drop; retention OOMs the device). Prefill
  ~214 s vs ~181 s. A generation-counter scheme could cut the destruction
  count ~30x — follow-up if the decode regression matters.


- BAND-GATE RATIFICATION + RESCORE (row/TT-B2II-BAND-GATE, 2026-10-19):
  the developer RATIFIED the band gate as the P150 correctness standard
  for this device: a sequence is ON iff it has ZERO hard flips —
  divergences inside the established 2.5-nat near-tie band are not gate
  failures; the exact-token ON/OFF counting is no longer the bar; the
  primary oracle rows (the W3 141/145 CPU baseline) remain the
  denominator. The scorer now prints the band-gate verdict line
  (`BAND GATE (ratified 2026-10-19): X/Y sequences ON`, with the worst
  in-band divergence against the band edge) beside the unchanged
  exact-token reporting — a reporting change, not a tolerance change;
  the classifier (the per-position nat gap vs the golden chain,
  `test_kolibri1_tt_b2ii.cpp:1222` `-gap <= kNatBand`, band 2.5) is
  unchanged.
  THREE-RUN VERDICT on origin/main (61d333e52, P150 non-fleet, the
  device recipe; byte-identical flip sets across all three runs per
  arm, no classifier defect found — no flip beyond the band is recorded
  as a near-tie on either arm):
  - HOST-FREE ON (VT_TT_HOST_FREE_DECODE default): BAND GATE 3/8
    sequences ON, worst in-band divergence -1.747 nats (margin 0.753
    nats to the 2.5-nat edge); exact-token unchanged at 3/11 (3
    near-ties, 5 hard: Australia, Wissen, x1=, Ein gutes, counting).
  - HOST-FREE OFF (VT_TT_HOST_FREE_DECODE=0): BAND GATE 6/8 sequences
    ON, worst in-band divergence -2.157 nats (margin 0.343 nats);
    exact-token unchanged at 12/20 (6 near-ties, 2 hard: der Mond step
    0, counting step 0).
  Under the ratified standard the gate verdict is STABLE and REPORTABLE
  (3/8 ON, 6/8 ON): it still fails its "all sequences ON" ideal, so the
  gate remains open on the hard-flip sequences, but the ON/OFF state is
  now a measured, dated, stable quantity instead of an exact-token
  coincidence count. The three pre-existing assertion failures remain,
  documented: `:1263` (argmax_matches >= 141), `:1264`
  (hard_flips == 0), `:1419` (instr_hard_flips == 0).

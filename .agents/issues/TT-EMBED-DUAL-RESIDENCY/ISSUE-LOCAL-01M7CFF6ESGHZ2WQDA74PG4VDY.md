ID: ISSUE-LOCAL-01M7CFF6ESGHZ2WQDA74PG4VDY
Title: TT embed dual residency: the tied embed table lives twice on device (9.95 GiB row words + 2.44 GiB bf16 twin) — eliminate the duplicate and unlock the captured arm
Row: TT-EMBED-DUAL-RESIDENCY
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-14
Updated: 2026-11-15
Closed: -

## Problem

TT-STAGING-FOOTPRINT's W3 recorded the captured-arm serving gap (~80 MB trace-region short at KV 8, fragmentation-bound) and named the lever: the tied embed table is resident twice — 9.95 GiB keep-quant row words + 2.44 GiB bf16 twin. The dual copy comes from the tied keep-quant non-share fall-through in LoadEmbedAndHead (qwen3_5_gguf_weights.cpp:883 share gate -> :897 OwnMatmulWeight, deliberate non-share :869-880); device side: head row words tenstorrent_keepquant.cpp:107 (pre-pass tenstorrent_capture.cpp:473) beside the gather bf16 twin tenstorrent_residency.cpp:30 (call tenstorrent_ops.cpp:335). Removing the twin frees ~2.4 GB: the captured-arm gap clears with margin, KV headroom grows, and UD-Q8_K_XL viability gets re-checked on a board that is no longer full. Owed: W1 consumer census (which consumers read the bf16 twin vs row words, file:line; the share-gate reason from git history before overturning it), W2 elimination per the census's smallest correct design, W3 THE GATE: captured-arm greedy stream byte-identical to the recorded eager stream b36bc61a... on UD-Q6_K_XL (divergence = STOP) + TPOT comparison + KV headroom + UD-Q8_K_XL viability re-check. Evidence: docs/bench-evidence/tt-staging-footprint-20261113/ + docs/bench-evidence/qwen38-q6k-vs-q4km-20261113/. Spec: .agents/specs/tt-embed-dual-residency.md

## Resolution

- 2026-11-14, W1+W2 (commits 881752c9b, be416cc5a): census + elimination
  landed on `row/TT-EMBED-DUAL-RESIDENCY` — `kEmbeddingQuant` registered on
  kTENSTORRENT, tied keep-quant share arm on TT, the 2.44 GiB bf16 twin gone
  (host and device). Focused tests green; eager-arm byte-identity `b36bc61a…`
  preserved.
- 2026-11-14, W3 attempt: **the unlock gate FAILED — STOP hit.** Captured
  UD-Q6_K_XL KV 8 completes WITHOUT the ~80 MB trace-region OOM (allocator
  ledger: 13,638,720 B/bank free before -> ~329 MB/bank at the tightest
  snapshot, ~2.5 GB freed) and is deterministic 3/3 (`d54b29f9…`), but the
  captured stream DIVERGES from the recorded eager stream `b36bc61a…` (embed
  contribution collapses to the attractor). The W3 dtype-churn fix (slot-
  current dtype+layout before every in-place copy; F32 row shadow) is
  coherent and innocent: at W2 `be416cc5a` the Q4_K_M captured arm already
  yields the identical wrong `a0033dd8…` vs the recorded `d49c53d2…`
  (bisection scratch build, `bisect-q4km.log`). Regression owned by W2's
  captured quant-row branch; eager arm correct at every head. Evidence:
  `docs/bench-evidence/tt-embed-dual-residency-20261114/`. Owed: root-cause
  the W2 captured divergence; W3(b)-(d) re-measure after it lands. State
  stays OPEN.
- 2026-11-15, root cause + fix + STOP lifted: **the recorded
  `ttnn::copy` FROM the persistent row shadow replays CAPTURE-TIME
  bytes.** Poison probe (`VT_TT_EMB_POISON`, the bisection instrument): the
  per-step refresh provably writes fresh, correct rows into the SAME buffer
  address the trace pinned (`[TT-EMB-DBG]` prints: constant shadow_addr,
  per-step hash changes; the eager arm — which decodes through the same
  `RefreshQuantEmbedRows` — is byte-identical, so the decode is exact), yet
  zeroing the ENTIRE shadow before every replay left the captured stream
  bit-for-bit unchanged. The read side, not the refresh, is frozen: the
  capture bakes the staged buffer's content, not just its address (#2812
  class), for the high-level copy op — while `ttnn::embedding`'s recorded
  ids read (WarmDecodeIds) and the custom kernels' recorded reads
  (WarmRacIdx) are replay-live. The W3 conversion (`typecast`/`to_layout`
  in-region) and the pure-copy variant diverge identically because BOTH
  route the commit through that frozen recorded copy. Fix per the stage-B
  contract ("representation changes land OUTSIDE captured regions"): the
  quant arm commits the embedding OUTSIDE the region — new
  `EmbedQuantRowArm(table_host, n, h)` query, drivers `qwen3.cpp` and
  `qwen3_5.cpp` gate the in-region `EmbedDeviceIdsInto` on it and commit
  beside `WarmEmbedQuantRows` per replay step (no allocation; program
  warmed by the R4 mirror); a float table keeps the in-region embedding
  (proven replay-live). qwen3_5.cpp's capture step additionally gained the
  missing pre-mirror `WarmEmbedQuantRows` (qwen3.cpp:1022-1025 already had
  it). Gates all green (3 reps each): Q4_K_M captured 64 blocks
  `d49c53d2…` 3/3 AND eager `d49c53d2…` unchanged; UD-Q6_K_XL captured
  KV 8 `b36bc61a…` 3/3 AND eager `b36bc61a…`; TPOT (correct stream now a
  real result) captured 5329.15/5319.70/5318.76 ms vs eager 5341.31 ms;
  KV 16 fits and is deterministic 2/2 (`5b34331d…`); DRAM ledger kept
  (tightest `total_free=2,634,189,824 B`, ~2.5 GB/bank-set freed, no OOM);
  `test_gguf_keep_quant` 63/63 cases, 12205/12205 assertions. Row stays
  OPEN for the fresh-review cycle.

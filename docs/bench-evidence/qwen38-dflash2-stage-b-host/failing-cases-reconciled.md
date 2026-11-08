# Reconciled failing-case attribution, QWEN38-DFLASH2-STAGE-B (2026-11-08)

This supersedes the attribution in `full-suite.log` /
`baseline-failing-cases.log` (d4dff2eb0). That pair was irreconcilable as
presented: the baseline log names 13 failing cases, the head log's summary
said 15, only 14 names were recoverable from it, and it showed
`kPagedAttention pure-decode matches host within BF16 envelope` failing at
head where the baseline showed it passing.

## What was rerun

- HEAD (row/QWEN38-DFLASH2-STAGE-B @ 059d17be6, the repaired tree): FULL
  suite `build/tests/test_tenstorrent_backend` on the P150
  (/dev/tenstorrent/0), `TT_METAL_HOME=~/Sources/tt/tt-metal-pin`, under the
  `${HOME}/gpu.lock` mutex. TWO runs, logs:
  `full-suite-head-repaired.log` (rep 1) and
  `full-suite-head-repaired-rep2.log` (rep 2). Both report
  `112 cases | 96 passed | 16 failed` — the SAME 16 case names both times.
  Both binaries then segfault in teardown AFTER the doctest summary prints
  (invalid-permission fault in device teardown); the summary line is
  complete and the segfault is recorded here as a teardown defect, not a
  test-result defect.
- Baseline (9996ce94d): the committed `baseline-failing-cases.log` is
  REUSED unchanged (13 failing cases; the run recipe matches the one above).
  The failure-to-case mapping was redone against the BASELINE source's line
  numbers (the earlier attribution mapped baseline error lines against the
  head source, which shifted names).

## The true sets (exact counts)

- Failing at BOTH baseline and head: 13
- Failing ONLY at head: 3
- Failing ONLY at baseline: 0
- Baseline total: 13. Head total: 16 (both runs).

### The 13 that fail at both

1. kTENSTORRENT batched decode PagedAttention is capture-safe (num_reqs=2)
2. kTENSTORRENT kAdd matches a host F32 reference (elementwise + bias broadcast)
3. kTENSTORRENT kEmbedding matches a host F32 reference (row gather)
4. kTENSTORRENT kGdnDecode matches the CPU f32 oracle (rank-1 step, both state_idx forms, NULL slot)
5. kTENSTORRENT kGdnPostConv matches the CPU f32 oracle (fused split + l2norm + g/beta)
6. kTENSTORRENT kL2Norm matches the CPU f32 oracle (GDN q/k row shape)
7. kTENSTORRENT kLayerNorm matches a host F32 reference (affine + plain)
8. kTENSTORRENT kMatmulBTQuantGrouped matches the CPU grouped provider inside the ratified envelope
9. kTENSTORRENT kMoeSiluMul matches host F32 within BF16 envelope
10. kTENSTORRENT kQkvSplit device path matches host within BF16 envelope
11. kTENSTORRENT kRelu matches a host F32 reference
12. kTENSTORRENT kRmsNormGated matches the CPU f32 oracle (silu + sigmoid, padded gate stride)
13. kTENSTORRENT kSiluAndMul matches host F32 within BF16 envelope

### The 3 that fail only at head

- `kTENSTORRENT kPagedAttention pure-decode matches host within BF16
  envelope` (`CHECK(max_abs < 0.5f)`): fails in BOTH full-suite runs at
  head, passed at baseline. In ISOLATION (`--test-case`) it PASSES 3/3 at
  head. So the PagedAttention delta REPRODUCES under the full suite and is
  NOT a random flake — it is order/state-dependent within the suite (an
  earlier case leaves device state the isolated run does not have). Not
  explained by this row's three seam commits; needs a bisection against
  9996ce94d with the suite context, not the single case.
- `kTENSTORRENT batched decode RAC is capture-safe (num_slots=2)`
  (`CHECK(k_ok == total)`, `CHECK(v_ok == total)`): same shape — fails in
  both full-suite runs, passes 3/3 in isolation at head, passed at
  baseline. Order/state-dependent, not random.
- `kTENSTORRENT kRmsNormGated serves a [256,6144] committed gate shadow at
  the batched-prefill [12288,128] geometry` (`CHECK(d.within)`): fails
  deterministically at head — in both full-suite runs AND 3/3 in
  isolation. The case exists at 9996ce94d and PASSES there. This one is a
  REAL, reproducible regression introduced between 9996ce94d and this
  branch head (candidates: bedb24739's VT_TT_SPEC_STATE_SHADOW staging,
  49f04cb17's replay-tap probe). It is NOT explained by the D2D
  recorded-geometry fix under repair here; it is recorded as an OPEN item
  for the operator, not fixed in this repair series.

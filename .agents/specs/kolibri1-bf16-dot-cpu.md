# SPEC: kolibri1-bf16-dot-cpu — adopt the bf16-dot compute class as the default on aarch64

Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
Status: ACTIVE (spec committed before implementation; one PR carries spec + implementation — recorded developer preference)
Decision authority: developer ruling 2026-10-19 — "we can support bf16 in hardware both on aarch64 and p150 so makes no sense not doing it" (resolves the direction of #3438).

## Problem

Kolibri-1's CPU production path dequantizes fp8 to f32 and multiplies in f32.
The primary oracle (aleph-alpha-inference via vLLM 0.31.0+cpu) computes the
same matmuls in the bf16-dot class (BFMMLA, 11,328 instances, #3438 evidence).
The three-way analysis (#3469, landed a70efb8cb) shows production and the
C draft are a TIE against the primary (~98% top-1 agreement either way) — so
production f32-dequant is the outlier, not the reference. The bf16-dot arms
exist behind default-off knobs (#3464) but nothing reaches them in production.

## Scope

Flip to default-on (hardware-gated, not env-gated) the kolibri1 arms that
consume bf16 inputs directly on aarch64 when `DetectArmBf16()` proves
FEAT_BF16 at runtime:

1. Routed-expert GEMV — `kolibri1_bfdot_gemv.cpp` (was `VT_CPU_KOLIBRI_BFDOT_GEMV=1` opt-in).
2. Paged-attention decode dot — `cpu_paged_attn_bfdot_arm.cpp` (was `VT_CPU_PAGED_ATTN_BFDOT=1` opt-in).
3. The lm_head / router bf16 extensions from #3464 (same gating).

Out of scope: x86 (no bf16-dot tier exists; the f32 path is unchanged and
remains the only tier there), non-bf16 aarch64 (fallback tier unchanged),
the P150 (separate surface — the band-gate re-score is a follow-up row),
the silu LUT (already unconditional and bit-exact — untouched), W8A8
(falsified, stays off).

## Design

- Dispatch: feature-gated at runtime (`DetectArmBf16()` + the existing
  per-arm usability checks). No environment variable gates the DEFAULT path;
  vLLM dispatches numerics kernels on CPU feature probes, not env knobs.
  An escape hatch remains: `VT_CPU_KOLIBRI_BF16_DOT=0` forces the f32 tier
  for the arms above (single knob, replaces the three per-arm opt-ins; the
  old knobs keep working as explicit `=1` for one release, then go away).
- Numerics contract: bf16 inputs, f32 accumulate (BFDOT semantics) — the
  class the primary oracle uses. The f32 fallback tier's byte stream is
  unchanged, so a non-bf16 host never moves.
- Gate re-adjudication: the CPU goldens scored against
  `kolibri1_goldens.json` (the torch transcription) move to the primary
  artifacts (`kolibri1_goldens_primary.json` + the top-100 rows, landed via
  #3463). Expected: first-token agreement with the primary RISES (the C
  draft already matched it on the streams where production flipped). The
  greedy chain gates get re-recorded under the new default, with the old
  f32-tier streams preserved under the escape hatch as the A/B leg.

## Upstream anchors

- vLLM CPU bf16 GEMM dispatch: runtime feature probes, no env gates.
- Primary oracle numerics: BFMMLA class, #3438 evidence thread.
- `scripts/gen-kolibri1-primary-goldens.py` / `-topk-rows.py` — the
  denominators for re-adjudication.

## Tests

- Existing arm tests (from #3464) flip from knob-gated-red to
  default-path-green on aarch64 CI; add one test asserting the DEFAULT
  dispatch selects the bf16-dot arm when `DetectArmBf16()` is true, and the
  escape hatch returns the f32 stream.
- Full kolibri1 CPU suite + the three-way capture test
  (`test_kolibri1_threeway_capture.cpp`) green under the new default.
- Non-aarch64 CI: zero behavioral change (the source-level guarantee the
  #3464 arch-guard incident taught us to verify explicitly).

## Gates

- aarch64 gate (this host) and the x86 CI lanes green.
- Token gate: greedy decode vs the primary oracle rows — per-position
  agreement recorded BEFORE (f32 production) and AFTER (bf16-dot default);
  the AFTER number must be >= the BEFORE number for the row to land.
  This is the row's red line: adoption may not move AWAY from the primary.

## Evidence

- Before/after agreement numbers, the re-recorded goldens, and the bench
  number for the flipped arms (the C draft measured +21% on the GEMV leg
  historically; record the production-shape number, not the micro).

## Risks / stop conditions

- If the AFTER agreement drops below BEFORE, STOP: file the divergence,
  revert to opt-in, and take it to the developer — the ruling sanctions the
  class, not a regression.
- If the escape hatch cannot reproduce the old f32 stream bit-for-bit
  (dispatch accidentally altered the fallback), STOP: that is a dispatch
  bug, not a numerics trade.

## Owed

- P150 band-gate re-score (separate row; needs the developer's explicit
  band-gate confirmation).
- W8A8 stays falsified/off — no arm.

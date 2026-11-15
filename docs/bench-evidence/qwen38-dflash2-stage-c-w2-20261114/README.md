# W2 evidence — QWEN38-DFLASH2-STAGE-C: the drafter forward on TT

Date: 2026-11-14. Host: thalia (P150, blackhole), mutex
`${HOME}/gpu.lock`, `luwen reset` before every leg, runs from
`/home/lu_zero/Sources/tt/tt-metal-pin` (the tt-metal runtime-root cwd
requirement), `taskset -c 32-63` runs / `64-127` builds. Weights: W1's pin
(`/mnt/models/qwen38-dflash2-drafter/`, incoai/Qwen3.8-27B-DFlash2 @
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`). Build: `/tmp/build-stagec-w2`.

## What W2 ships

- `src/vt/tenstorrent/tenstorrent_ops.cpp`: `kDFlashGroupedConv` on TT (the
  DFlash2 draft's grouped dynamic conv — the op had no TT kernel, so the
  drafter forward could not run on the P150 at all), plus the `kIndexCopy`
  TT registration the context-KV append needs. The conv is a composition of
  the shared TT primitives (slice/matmul/add/multiply/concat) preserving the
  CPU kernel's one-round-per-step structure.
- `tests/vllm/models/test_qwen38_dflash2_drafter_tt.cpp`: the W2 component
  gate (5 cases) on the REAL pinned drafter weights.

## The substrate discovery (recorded honestly)

The pinned tt-metal's f32->bf16 rounding at exact half-ulp ties is NEITHER
RNE NOR a consistent direction: exact tie 170.5 ulp rounds up (i=8), exact
tie 207.5 ulp rounds down (the worst case). BIT-IDENTITY IS UNATTAINABLE for
the conv: a tie in `k = bf16(b + d)` lands one k-ulp away, and a k=b+d
cancellation flips a whole tap term (cpu 0 vs tt 0.03125). The conv gate is
therefore a documented operand-magnitude ENVELOPE, not bit equality.

## Gate results (final legs, this date)

1. `DFlashGroupedConv matches CPU within the tie-rounding envelope` — PASS.
   T=17 (odd, exercises the `&` arm), H=5120, groups=320, taps=2:
   35582/87040 elements bit-differ, rel_rms **0.00382381**,
   maxabs **0.0625** <= bound 0.05*xmax = 0.1. PASS.
2. `paged attention at the draft block shape` — PASS. rel_rms
   **1.7e-07** (TT vs CPU kernel, GQA 32q/8kv, bidirectional 8-over-16).
3. `the 8-row block forward over tap context matches the bundle reference
   (PCC band)` — stage gates: fc combine PCC 0.999679, ctxK/ctxV l0 PCC
   0.999763/0.999736 (each >= 0.98 floor); full chain vs the bundle
   reference — see "the remaining divergence" below.
4. `the CPU arm of the chain matches the bundle reference` — the isolation
   instrument.
5. `the PCC gate is RED on a corrupted fc row` — PASS (red proof): a
   sign-flipped fc output row collapses the chain to PCC 0.046, so the PCC
   gate detects a wrong drafter.

## The op-by-op replay (layer-0 isolation, CPU ops, fp64 reference)

`W2_REF_OP_REPLAY=1`. Every chain op, on the real layer-0 weights, stage by
stage against the fp64 reference:

| stage | PCC | rel_rms |
|---|---|---|
| after in_ln | 0.999999 | 0.00168 |
| after ac prepare (conv side 0) | 0.999999 | 0.00104 |
| q normed+roped | 0.999995 | 0.00306 |
| k normed+roped | 0.999995 | 0.00315 |
| attention out | 0.999995 | 0.00316 |
| after ac finish (conv side 1) | 0.999993 | 0.00366 |
| after post_ln (fused add+RMSNorm) | 0.99999 | 0.00451 |
| after mc prepare | 0.999978 | 0.00665 |
| after silu*up | 0.999954 | 0.00965 |
| after mc finish | 0.999996 | 0.00279 |

ALL TEN stages pass. The fp64 Gdc itself matches the CPU op EXACTLY
(rel_rms 0, the earlier probe's own OOB `b01[i]` -> `b01[i-kH]` fixed) —
"the CPU op was right all along".

## The remaining divergence — TWO bugs, both found and fixed

The op-by-op replay (context-free AND with-context, all ten stages PCC >=
0.9995) proved every chain op faithful, which moved the suspicion to the
REFERENCE. Two reference-side defects explained the entire gate gap:

1. **THE REFERENCE'S O_PROJ READ A STRIDED MATRIX AS CONTIGUOUS.**
   `RefLayerForward` allocated its attention output `o2d` as `[B, kH=5120]`
   (H-padded) and wrote `o2d[t*kH + h*kHD + d]`, then handed it to `MulWT`
   which reads `[T, K]` CONTIGUOUS at K = kNH*kHD = 4096. Rows 1..7 of every
   reference layer read shifted data. Signature: row 0 PCC ~0.70, rows 1..7
   ~0 — exactly what the gate reported. Fixed to `[B, kNH*kHD]`; the
   un-padded fp64 diff then shows attention output `o2d` rel_rms **0** vs the
   chain.

2. **`MatmulBT`'s TT kernel ran the ttnn DEFAULT compute config for bf16
   outputs** (LoFi fidelity, bf16 partial accumulation — the W3 noise map's
   2.3e-2 rel_rms per GEMM), while the bundle's own compute kernel config is
   HiFi4 + fp32_dest_acc_en (dflash2.py `CKC`). The fc combine alone carried
   74 sign-flip-class outliers. Fix: `F32AccKernelConfig()` for BOTH output
   dtypes (src/vt/tenstorrent/tenstorrent_ops.cpp, kMatmulBT) — parity with
   the bundle, not tuning. The fc outliers collapsed 74 -> 1.

## FINAL GATE (2026-11-14, this date, exactness arm + fixed reference)

`test_qwen38_dflash2_drafter_tt` — **451/451 assertions, Status: SUCCESS,
EXIT 0** (log: this directory's gate transcript; full log
`/tmp/w2-gate-final2.log`):

- TT 8-row block forward vs the corrected bundle reference, per row:
  **PCC 0.9917 / 0.9917 / 0.9945 / 0.9975 / 0.9980 / 0.9972 / 0.9974 /
  0.9976** (floor 0.98 — the bundle's own band, model.py:1132) — ALL rows
  PASS. ALL-rows PCC **0.9956**, rel_rms **0.0935** (envelope 0.10).
  Stage gates: fc combine PCC 0.999999 (outliers 74 -> 1), ctxK 0.999995,
  ctxV 0.999997.
- `DFlashGroupedConv` TT vs CPU: rel_rms 0.0038, maxabs 0.0625 <= 0.1 —
  the documented tie-rounding envelope (bit identity unattainable on the
  pinned tt-metal f32->bf16 tie behavior; see the kernel comment).
- `PagedAttention` TT vs CPU at the draft shape: rel_rms 1.7e-07.
- The CPU arm of the chain (device-free) vs the corrected reference: layer
  downs PCC 0.9999 / 0.9995 / 0.9996 / 0.9944 / 0.9876 — the chain
  semantics are exact to the bundle math on the host.
- RED proof: one sign-flipped fc row collapses the TT forward to PCC
  0.9648 < 0.98 — the gate detects a wrong drafter (the margin is thin
  because the exactness arm compresses the damage; recorded honestly).

Instrumentation kept in the test (all env-gated, default off):
`W2_REF_OP_REPLAY` (the ten-stage layer-0 replay), `W2_REF_CTX_REPLAY`
(the context-inclusive replay + the ckv/weight-identity probes),
`W2_REF_INT_DUMP`/`W2_REF_STAGE_DUMP`/`VT_DBG_DFLASH_DUMP` (stage dumps —
the tools that isolated both defects).

Bisect trail (all measured, not argued): op replay all-pass -> reference
o2d stride bug (fixed) -> fc outliers = bf16-partial accumulation
(fixed by the CKC-parity lever) -> gate green.

## Bundle-reference bugs found during bring-up (fixed in the harness)

- per-head RoPE (:327-342 of the draft harness lineage): the reference
  broadcast the tables over the wrong axis / applied once, not per head.
- o_proj K/N: the matmul used the transposed geometry.
- pre_residual_down out-param: the reference lacked the chain's
  pre-residual MLP output handle (the chain's per_layer_out is the
  conv-finished MLP output BEFORE the residual add, qwen3_dflash.cpp:704-711).
- bf16 rounding of the conv coefficients: the op consumes bf16 OPERANDS
  (k = round(bf16(b) + bf16(d))), not round(b+d) of f32 — a 100% error of k
  wherever b+d cancels (b=1, d=-1.0035 -> op k exactly 0).
- CombineAuxFeatures returns the PRE-norm fc output; the chain applies
  hidden_norm later (qwen3_dflash.cpp:367) — comparing pre vs post reported
  a fictitious rel_rms-33 scale mismatch.
- ckv.k OOB read fixed.
- THE o2d stride bug (this wave's root cause): see "The remaining
  divergence" above — the reference allocated [B, kH] where the bundle
  reshapes [B, NH*HD] contiguous.

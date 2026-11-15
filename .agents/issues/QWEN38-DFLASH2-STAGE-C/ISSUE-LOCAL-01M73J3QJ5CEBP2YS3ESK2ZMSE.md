ID: ISSUE-LOCAL-01M73J3QJ5CEBP2YS3ESK2ZMSE
Title: Stage C: port the DFlash2 block-diffusion drafter and wire the speculative loop through the captured verify arm
Row: QWEN38-DFLASH2-STAGE-C
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-10
Updated: 2026-11-15
Closed: -

## Problem

Stage B's captured verify arm is speculative-ready but nothing drafts. The bundle's DFlash2 drafter (extra_code/qwen38/dflash2.py, DFlash2Draft/DFlash2Drafter: one 5-layer forward over the target's residual taps producing 8 fixed-shape rows [anchor, MASK x7]) is the missing producer; the modeled win at the bundle's acceptance is ~8.5x on the 6606.84 ms stage-B arm. Owed: W1 obtain/convert/pin the drafter weights (download needs developer go; converter + docs/USAGE.md pin + refuse-by-name), W2 component-validated drafter forward on TT (PCC band vs the bundle reference), W3 the loop through Qwen3_5DenseDecodeGraph (draft -> captured verify replay -> host greedy accept by token-id compare; reject rolls back via the stage-B GDN commit-or-restore; dense driver only), W4 the gate (greedy token identity vs plain decode, STOP on divergence; acceptance/TPOT table vs the stage-B baseline; the host-staging cost the modeled win assumed away must be attributed honestly). Spec: .agents/specs/qwen38-dflash2-stage-c.md

## Resolution

W2 resolution (2026-11-14, the drafter forward on TT):

- SHIPPED: `kDFlashGroupedConv` on TT (src/vt/tenstorrent/tenstorrent_ops.cpp)
  — the DFlash2 draft's grouped dynamic conv had no TT kernel (the drafter
  forward could not run on the P150 at all) — plus the `kIndexCopy` TT
  registration the context-KV append needs. Both are compositions of the
  shared TT primitives; the conv preserves the CPU kernel's
  one-round-per-step structure.
- THE LEVER: `kMatmulBT` on TT passed `F32AccKernelConfig()` only for f32
  outputs; bf16 outputs (every drafter GEMM) ran the ttnn default config
  (LoFi + bf16 partials, the W3 noise map's 2.3e-2 rel_rms per GEMM), while
  the bundle's own CKC is HiFi4 + fp32_dest_acc_en (dflash2.py). Now the
  exactness arm for BOTH output dtypes — parity with the bundle, not tuning.
  Effect on the fc combine: 74 sign-flip-class outliers -> 1.
- THE REFERENCE BUG (the gate's whole gap): the test harness's
  `RefLayerForward` allocated its attention output `o2d` as [B, kH=5120]
  (H-padded) but the o_proj matmul reads it contiguous at [B, kNH*kHD=4096]
  — rows 1..7 of EVERY reference layer corrupted (row 0 matched, rows 1+
  collapsed; the exact signature the gate reported). Fixed to [B, kNH*kHD].
  The ten-stage op-by-op replay (context-free AND with-context, every stage
  PCC >= 0.9995 vs independent fp64) is what moved the suspicion onto the
  reference; the real chain was never the divergence.
- ALSO FIXED in the harness on the way: per-head RoPE, the o_proj K/N
  geometry, the pre_residual_down out-param, bf16 rounding of the conv
  coefficients (the op rounds bf16(b)+bf16(d), not b+d), CombineAuxFeatures'
  PRE-norm contract, a ckv.k OOB read, and a null-CPU-queue crash
  misattributed to "CPU MatmulBT raw-NK segfaults".
- SUBSTRATE (recorded honestly): the pinned tt-metal's f32->bf16 rounding at
  exact half-ulp ties is NEITHER RNE nor consistent-direction (tie 170.5 ulp
  rounds up, tie 207.5 ulp rounds down), so the conv's k = bf16(b + d) can
  land one k-ulp away and flip a cancellation tap. BIT IDENTITY IS
  UNATTAINABLE; the conv gate is the documented operand-magnitude envelope
  (rel_rms 0.0038, maxabs 0.0625 <= 0.1 bound).
- GATE GREEN: `tests/vllm/models/test_qwen38_dflash2_drafter_tt` 451/451
  assertions, EXIT 0, on the pinned drafter weights on the P150: TT 8-row
  block forward vs the corrected bundle reference — per-row PCC
  0.9917..0.9980 (floor 0.98, the bundle's own band, model.py:1132),
  ALL-rows 0.9956 / rel_rms 0.0935; conv envelope PASS; paged attention
  rel_rms 1.7e-07; the CPU arm (device-free) per-layer 0.9876..0.9999; RED
  proof (sign-flipped fc row -> PCC 0.9648 < 0.98) still detects a wrong
  drafter. Evidence: docs/bench-evidence/qwen38-dflash2-stage-c-w2-20261114/
  (FLAT: README.md, gate-final.log). W3 (the loop) is next.

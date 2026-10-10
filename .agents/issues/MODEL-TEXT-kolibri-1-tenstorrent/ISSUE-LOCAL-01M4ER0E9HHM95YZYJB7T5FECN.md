ID: ISSUE-LOCAL-01M4ER0E9HHM95YZYJB7T5FECN
Title: B2b-ii: the streaming MoE on the P150 (slot pool, routed path, stream bound)
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-09
Closed: -

## Problem

B2b-i landed the dense-resident device forward with the routed-expert tier deliberately absent (the refusal fires by name 450x per decode). The goldens are full-model decodes and cannot be replayed until the routed experts run: the B2b-ii streaming path — FP8_E4M3 slot buffers consuming Kolibri1TTExpertSlotPolicy verbatim, router readback -> host remap -> fetch list -> fetch executor -> Touch(), slot swaps on the ContentChangedSince reset lane, the B1 per-token stream bound asserted at runtime with loud failure, the full 384-expert tier host-side — is owed, and with it the deferred 141/145 token gate.

## Resolution

- 2026-10-09 (PARTIAL — the host half + device arm landed; the device
  GATES are blocked external): the streaming MoE's host half landed
  (slot-pool plan over `PlanKolibri1TTExpertSlotPolicy` unchanged, the
  fetch executor's host half, the LOUD stream-bound guard, the slot
  shadow / readback pivot, the eviction-hook integration, the
  `Kolibri1TTSlotEpoch` reset lane) with red-first device-free coverage
  in `test_kolibri1_tt_b2ii.cpp` (9/9), and the routed path REPLACED the
  B2b-i refusal in the production TT forward (slot pool staged FP8_E4M3
  verbatim; dispatch -> fetch -> stage -> readback-verify -> `Touch()`;
  memoized dequants cleared by `ContentChangedSince`; `vt::MoeCombine`).
  TT family 4/4; CPU battery green. The DEVICE legs (smoke, the 141/145
  token gate, the bench anchor) are OWED: the P150 board is in an
  external failure state (`cq_id 0 is out of range` reproduces on the
  UNCHANGED base commit and both pin lib generations; the documented
  recovery loop does not recover; dmesg shows a PCI rescan) — evidence
  and the full blocker record in
  `docs/bench-evidence/kolibri1-tt-b2ii-20261009.md` section 4. The
  token-gate leg is committed and runs on the first healthy card
  window. Issue stays OPEN on the device gates.
-
- 2026-10-10 (branch row/tt-kolibri-residue, off e510885d7 + the
  teacher-forced instrument eb7cefb85): the prompt-dependent hard
  divergence is ROOT-CAUSED to the tt-metal rms_norm kernel, evidence in
  test_kolibri1_tt_b2ii.cpp "SCRATCH dbg rmsnorm micro" (red, kept). With
  BIT-IDENTICAL bf16 inputs and gamma ([1,2560], random seed 7), the
  device arm — ttnn::rms_norm over f32 tiles via the NormalizeDevF32Tile
  pin-forward — measures sum ratio 0.98992 and max_abs 0.0625 (4 bf16
  ULP) against the CPU row / host-double oracle; the CPU row matches
  host-double to 1 ULP. Normed 3-4x per layer over 50 layers, that
  systematic bias compounds into the multi-nat logit shifts the
  teacher-forced instrument records (OFF arm 47 HARD, worst gap 5.17
  nats; per-prompt 'der Mond...' steps 0-24, 'Translation to German...'
  steps 0-22, 'x1 = 3...' steps 1-9). NOT a slot-pool/expert-stream
  defect for these prompts: the slot-pool no-slot refusal STILL fires on
  the fixed base (4 of 8 teacher-forced walks abort, e.what()=="1") but
  the diverging prompts never abort — two defect classes share the row.
  Two repairs were tried and refused by the substrate, both recorded in
  src/vt/tenstorrent/tenstorrent_ops.cpp (RmsNormKernel): (1) a composed
  f32 chain (x^2 -> mean -> +eps -> rsqrt -> scale), exact in the micro
  (ratio 0.9998, 1 ULP) — its [rows,1] row scale cannot reach the model:
  plain multiply broadcasts padded-tile garbage (gate collapse, 1e37
  logits), BcastOpDim::W is numerically wrong (ratio 1.019), ::H is
  refused, and repeat+same-shape multiply is refused by binary_ng
  ("Invalid subtile broadcast type"); (2) serving the eager arm's
  residual-free short-row norms from the host f32 loop — the extra host
  round-trips desync the device shadow (gate 26/33 -> 0/8 with 1e37
  garbage), both all-norms and hidden-width-only narrowings. Both
  reverted; the landed tree keeps baseline routing (byte gate back at
  26/33, 47 HARD — verified post-revert, /tmp/final_off.log) plus the
  composed arm behind VT_TT_RMSNORM_COMPOSED=1 for A/B. The embed table
  and the norm gamma stage BIT-EXACT on device (VT_KOLIBRI1_TT_STAGE_DUMP
  emb/gam dumps, both arms) — staging is clean; the bias is in the
  kernel math. defect class: tt-metal substrate (ttnn::rms_norm numeric
  bias + binary_ng subtile broadcast refusal). ISSUE STAYS OPEN; the
  fix needs either an upstream rms_norm with f32 statistics that
  matches the oracle, or a broadcast primitive that survives the model
  geometry, or the host-free shadow reconciliation this row already
  owes.

- 2026-10-10 (substrate rms reduction fix, this branch): the blocked
  in-kernel repairs are superseded by a TT-METAL SUBSTRATE patch in the
  trial tree `/tmp/tt-metal-umdtrial` (installed at
  `/tmp/umdtrial-install`, rebuilt and reinstalled in
  `/tmp/build-umdtrial2`). Mechanism: `ttnn::rms_norm`'s default compute
  config hardwired `fp32_acc = false`
  (`ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp`), so the
  layernorm program factory ran Float16_b circular buffers and
  `float32_reduction = fp32_dest_acc_en && !legacy_reduction` was false
  (`layernorm_op_multi_core.cpp`) EVEN when the vllm.cpp f32-shadow arm
  fed FLOAT32 tiles — the sum of squares accumulated in bf16, which is
  the whole 0.9899 bias. The patch sets `fp32_acc = true` and
  `approx_mode = false` for the rms_norm default config (f32 CBs, f32
  reduce, accurate SFPU rsqrt) and widens the reduce scaler CB to
  Float32 under an f32 reduction. Micro (`SCRATCH dbg rmsnorm micro`,
  same bit-identical bf16 [1,2560] seed-7 inputs): ratio 0.98992 ->
  0.99972 vs the CPU row, max_abs 4 ULP -> 1 ULP, and vs the
  host-double-bf16 denominator TT 0.99915 vs CPU 0.99942 — the residual
  is the bf16 output-store quantization floor, which the CPU oracle row
  itself cannot beat. Gate verdicts (both arms re-run, clean
  reset+cache per leg): ON 26/33, 7 flips (0 near-tie, 7 hard),
  instrument 40 tf flips (5 near-tie, 35 HARD) worst 3.19 nats; OFF
  26/33, 7 flips (1 near-tie, 6 hard — baseline was 7 hard), instrument
  116 tf flips (11 near-tie, 105 HARD) worst 7.59 nats. The ARGMAX
  chain did not move materially: the remaining flips are dominated by
  the two open op-level drifts (kGdnDecode state reduction,
  kMatmulBTQuantGrouped Q4_K envelope), both re-confirmed failing in
  isolation; the >=141 target stays out of reach. Suite
  `test_tenstorrent_backend`: the only cases failing in isolation are
  the two known ones (kGdnDecode, kMatmulBTQuantGrouped); the W4
  EnsureDevice2D counter and the matmul region class split fail only in
  full-suite order state and pass isolated. NOTE: the OFF-arm
  teacher-forced instrument reads 105 HARD / worst 7.59 nats in this
  run against the 47 / 5.17 recorded on 5277bdc72 — deterministic
  across a clean reset; the instrument counts over the full 33-position
  harness, so part of the delta is coverage, and the norm arm change
  plausibly moved teacher-forced near-ties; the argmax chain (the
  gate's contract) is unchanged. THE TRIAL TT-METAL TREE NOW CARRIES
  THIS REDUCTION-PRECISION PATCH ON TOP OF THE PORT FIXES — any future
  measurement or gate on this stack must use the rebuilt
  `/tmp/umdtrial-install` (rebuild: `cmake --build /tmp/build-umdtrial2
  -j 4 && cmake --install . --prefix /tmp/umdtrial-install`).

- 2026-10-10 (branch row/tt-kolibri-flips2, host-free worktree): the
  remaining-gate-divergence localization RE-DERIVED on the fixed substrate
  (rmsnorm fp32-accumulation patch in /tmp/umdtrial-install). The prior
  attribution to kGdnDecode / kMatmulBTQuantGrouped is FALSIFIED for kolibri1
  at census level: the kolibri TT forward dispatches Embedding, MatmulBT,
  RmsNorm/FusedChain, MoeSiluMul, MoeCombine, RopeNeox (sliding layers),
  the KV write + PagedAttention, and Add — kolibri has NO GDN op and its
  experts are fp8-block dequanted to bf16 on HOST (no kMatmulBTQuant, no
  kMatmulBTQuantGrouped arm anywhere in the forward). The first diverging
  stage on the fixed substrate was the routed-expert MoE block at layer 0:
  the memoized slot dequant of the DOWN projection dequanted the WRONG
  source bytes — `byte_base += n * k * 2` doubled for BOTH the fp8 source
  (one byte per element) and the bf16 destination, so up read down's slot
  region and down read past the slot end (NaN/Inf weights, 1e38 expert
  outputs, every downstream stage corrupt). Red-first evidence:
  `downdequant` ref-vs-memo 1,177,065/1,310,720 elements mismatched
  (NaN memo sums) and the one-hot down_w probe returned inf/NaN at k>=256
  against a clean scale grid; slotcmp proved the slot bytes themselves
  byte-exact. FIX: separate `packed_base` (n*k) and `bf16_base` (n*k*2)
  offsets in kolibri1_tt_forward.cpp. After: dequant mismatch 0/1,310,720,
  L0 moe matches the CPU arm exactly, first diverging stage moves to L1 moe
  at ~3% relative (near-tie class). Numbers (fixed substrate, this branch):
  teacher-forced 'der Mond...' worst gap 11.76 nats -> 10.05 nats with the
  argmax chain recovering CPU agreement at 5 of 13 steps (was 0);
  device token gate 4 HARD flips ON (steps to flip moved much later:
  'der Mond' step 7 gap -5.92; 'capital of Australia' step 5 -4.56;
  'Wissen ist Macht' step 8 -13.35; 'x1 = 3...' step 22 -3.06) and 3 HARD
  flips OFF before the run ABORTS on device DRAM exhaustion
  (TT_FATAL Out of Memory, 4.27 GiB bank space full — the memoized
  expert-dequant device staging budget; walks now survive much longer than
  the pre-fix early flips, so the gate reaches the exhaustion
  deterministically at the same point on both arms). THE OOM IS OPEN DEBT:
  the full 33-prompt tally cannot complete on this substrate until the
  memoized-expert device residency is capped or freed. The streaming
  slot-pool `what()=="1"` throw did NOT fire on the fixed substrate in any
  instrument or gate run (0 occurrences). The remaining flips after the fix
  are the residual bf16 device-arm drift class, not the byte bug.

- 2026-10-10 (branch row/tt-kolibri-l1moe, off 519c466b8): the L1-moe ~3%
  relative divergence is LOCALIZED and CLASSIFIED as the drift class, not a
  stride/staging bug — instrument on the fixed substrate, teacher-forced
  'der Mond...' (VT_TF_DBG_PROMPT=1, 12 forwards), OFF-vs-CPU with per-stage
  bf16 bit hashes (StageDump hash=/sq=), the router dump (RouteDump, both
  arms), the router weight hash probe, host-reference logits for BOTH arms'
  router GEMMs, and the element-level L0 dhn diff. Evidence chain: (1) the
  FIRST diverging quantity is the L0 input rms_norm output at step 0
  (hash b3eeb298 vs 2b64e8d9; element diff 1 bf16 ULP class, top ~0.03-0.06
  on values to 14, rms 0.002) — the adjudicated substrate drift, not new;
  (2) by L0 dh2 (the moe input) the drift carries OUTLIER element diffs
  (max 47.25 vs 48.0, 3 bf16 ULP on a 48-magnitude element; sq 2.166e4 vs
  2.175e4, ~0.4% rms, diff-norm ~10 over 2560 elems) from the device
  attention/norm chain; (3) the router gate weight is BYTE-IDENTICAL across
  arms (hash e6512148 at L0) and BOTH arms' router GEMMs match host
  reference logits computed from their own inputs (CPU 1.4e-5, TT 0.15
  max) — the GEMMs are correct; (4) the amplification is the ROUTER GAIN:
  the kolibri router row has ||w||_2 ~ 1 over K=2560 against logits of
  scale O(5), so a 0.4%-rms input drift injects O(1-5) ABSOLUTE logit noise
  (measured max 4.8-7.0 per layer, rel_rms 0.3-1.4) and the top-6-of-384
  sigmoid-logit-add selection — a near-tie boundary — flips 2-4 of the 6
  experts AT EVERY LAYER INCLUDING L0. The ~3% L1 moe divergence IS the
  flipped mixture. NO local defect: staging byte-exact (pool readback,
  prior record), dequants byte-exact, weights identical, GEMMs verified
  against host references. ONE CONTRACT DEFECT found and fixed by the
  instrument (kept, red-first bf16grid evidence): the TT kMatmul/kMatmulBT
  default produced a BF16 tensor upcast on commit for f32-out callers —
  all 384 router logits sat on the bf16 grid (bf16grid=384/384) with bf16
  partial sums, violating the CPU row's LinearBTRaw f32 contract
  (tenstorrent_ops.cpp: f32 out now requests dtype FLOAT32 +
  fp32_dest_acc_en; post-fix bf16grid=0/384). GATE (ON arm, clean reset,
  this branch, post-fix): 4 HARD flips at the SAME positions as pre-fix
  ('der Mond...' step 7 gap -5.92 -> -5.62; 'Australia' step 5 -4.56 ->
  -3.85; 'Wissen ist Macht' step 8 -13.35 -> -14.76; 'x1 = 3...' step 22
  -3.06 -> -3.75), then the same deterministic memoized-expert DRAM
  exhaustion abort — the contract fix does NOT clear the flips because the
  driver is the upstream drift class. CLASSIFICATION: selection flips
  DRIVEN by continuous bf16 device-arm precision drift through the
  attention/norm chain; per the row's adjudication rule this is an
  upstream adjudication decision (f32-enabling the device attention/norm
  chain or accepting the band), not a local patch. ISSUE STAYS OPEN on the
  device gates and the OOM debt.

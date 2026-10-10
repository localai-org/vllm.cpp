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

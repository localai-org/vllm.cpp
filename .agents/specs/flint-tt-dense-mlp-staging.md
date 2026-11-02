# SPEC — `FLINT-TT-DENSE-MLP-STAGING` — the dense gate/up staging produces allocation-order-dependent outliers on the P150

- Owning row: `FLINT-TT-DENSE-MLP-STAGING`
- Kind: bug (device numerics; blocks FLINT-P150-ATTENTION stage-2 adoption)
- Issue: created in the same flow (row-scoped local issue)

## Problem

The Flint P150 device arm cannot pass a model-level band gate, and the
defect is not in attention. Evidence:

- FLINT-P150-ATTENTION's stage-1 probe is healthy: a single device
  attention layer matches the host `SinksAttention` shadow on all 64
  per-head slices in both window modes, at 1/10 of the W2-derived band.
- Its stage-2 model gate is not: functionally neutral changes to DBuf
  lifetimes in flint_tt.cpp (scoping ao_dev/aoout_dev to the attention
  else-block, or staging three per-layer scalars at build) moved the
  host-shadow arm's first-token rel_rms between **2.683e-1 and
  7.224e-1**, and the device-attention arm between 4.825e-1 and 3.068 —
  with the attention legs healthy in every variant.

The suspect is upstream of attention: the W2 report (`docs/bench-evidence/
flint-tt-w2-20261025/report.md`, reading 2) already measured the dense
gate_proj as an outlier generator on device (TT sum −14168 vs CPU +15.3,
20x the CPU stream energy, silu suppressing most of the damage at −0.4%
ff re-convergence). The W2 record's own "next site to chase" note names
this staging path.

## Deliverable

1. **Attribute the allocation-order sensitivity**: instrument the dense
   gate/up/down staging (the `MatmulWeight` path through flint_tt.cpp) so
   the outlier source is *measured*, not inferred: per-tensor rel_rms
   against the CPU shadow at L0, and at L46, with the staging order
   (a) as-landed, (b) reversed, (c) with intermediate DBuf lifetimes
   extended past the consume point. If the outlier follows the staging
   order, the staging is the defect site; if it follows the *tensor*
   (same tensor, every order), the defect is in the weight upload path.
2. **Fix**: likely one of (a) staging the dense gate/up/down projections
   with explicit lifetime/commit boundaries (the kolibri1 host-free
   pattern), (b) a device-side reorder that decouples the outlier from
   the allocation order, or (c) if the defect is in the tt-metal
   allocation layer itself, a repro + record and a pin-advance note (the
   W2 report's substrate-swap finding is precedent).
3. **Verify**: the W0 band gate must hold under EVERY DBuf lifetime
   permutation the fix allows — the model-level band must stop being
   allocation-order-sensitive, and the first-token rel_rms must land
   inside the W0 band regardless of staging order.

## Gates

- The W0 model-level band gate green under every lifetime permutation
  the fix permits (this is the actual deliverable — the band must be
  allocation-order-INSENSITIVE after the fix).
- Single-layer probe still 64/64 per-head slices, both window modes.
- test_flint_paged 208/208, test_flint_gguf serving gate, MiMoV2 sink
  gates, dtype-dispatch bit-identity suite.
- check-agent-record, style, trailers, preflight (named host dotnet
  failure).

## Stop conditions

- If the outlier is measured in the tt-metal allocation layer itself
  (i.e. the defect reproduces in a minimal two-tensor staging test
  outside flint_tt.cpp), STOP and report: that is a substrate bug for
  the tt-metal tracker, with our repro attached — not something this row
  can repair in-tree.
- If the fix requires changing the W0 band, that is a re-adjudication:
  land it only with the measured before/after rel_rms in the evidence
  and the reviewer's confirmation that the change is a legitimate
  re-pin, not a widened assertion.

## Git integration

One pull request, `row/FLINT-TT-DENSE-MLP-STAGING`, spec committed first.

## Now

DONE 2026-11-02. The band is allocation-order-insensitive: all four
staging-order legs land at 2.683e-1 after the fix. The mechanism, the
permutation matrix, and the standing gates are recorded in
`.agents/evidence/flint-tt-dense-mlp-staging-20261101/README.md`
(the slot-serve fix section) and the row issue's Resolution.

## Outcome

- The defect was NOT in flint_tt.cpp's staging lifetimes. The three-matmul
  probe (leg-l-prepost-probes.log) split it: same input slot, same weight,
  correct / WRONG / correct — only the OUT block differed, and the out block
  that poisoned was a #1922 best-fit BORROW (a 40960-B block serving a
  20480-B request).
- The mechanism: `EnsureHostBytes` picked the download's host write width
  from the slot's registered CAPACITY (`s->bytes`) instead of the shadow's
  dtype, so a borrowed bf16 slot's download wrote f32 bytes that the bf16
  reader consumed word-halved. The fix writes the shadow's own dtype and
  demotes the capacity to a size check — one hunk, no Synchronize, no copy.
- Measured: the W0 band under all four lifetime legs before (2.683e-1 x3,
  7.224e-1 oproj-scoped) and after (2.683e-1 x4). L46 sh_silu rel_rms
  1.76e+0 -> 1.7655e-2 (the same borrowed-slot poison rode the shared
  expert's silu). Greedy chain identical across legs (the recorded
  hard-divergence bias class, unchanged — no band re-adjudication).
- Rejected: a persistent-buffer retire in `MarkScratchAcquired` (leg-g) —
  did not move the band; the probe matrix disproved the hypothesis.
- The W0 band value itself is UNCHANGED (2.683e-1 was already the recorded
  healthy value) — this is a repair, not a re-pin.

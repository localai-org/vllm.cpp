ID: ISSUE-LOCAL-01M5G38WAKAFKPNPQMSNM8EJ7H
Title: Kolibri-1 CPU path refuses a plain bf16 checkpoint, so the fp8 weight quantization's own error cannot be measured
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-21
Updated: 2026-10-22
Closed: 2026-10-22

## Problem

`kolibri1_weights.cpp::ResolveKolibri1WeightScheme` (the scheme dispatch
that replaced the old `CheckQuantConfig`) hard-requires the checkpoint's
`quantization_config` to be fp8 block [128, 128] dynamic and refuses a plain
bf16 Kolibri-1 artifact by name ("a bf16 Kolibri-1 artifact is not an
implemented arm").

The upstream `Aleph-Alpha/Kolibri-1-BF16` checkpoint
(`cc6d729e6a024c1e59ad48887fdef95ff37b534d`) carries no
`quantization_config` at all, so the loader rejects it and the bf16 arm is
unreachable.

Without the bf16 arm, the fp8 weight quantization's own contribution to our
divergence from the primary oracle is not separable from the rest of the
numerics. Every linear weight in the fp8 checkpoint is an F8_E4M3 block with
an f32 `weight_scale_inv` sibling, and a bf16 checkpoint's weights are
already the dequantized values, so the same forward on the same compute
class isolates the quantization's contribution.

Spec: `.agents/specs/kolibri1-bf16-arm.md`.

## Resolution

- 2026-10-22: the loader's scheme check is now a dispatch
  (`ResolveKolibri1WeightScheme`): the fp8 block [128, 128] dynamic arm is
  unchanged, a config with NO `quantization_config` is the plain bf16 arm,
  and every other scheme is refused by name with a message that names the
  arms that exist. The accounting is scheme-aware, so a bf16 checkpoint that
  ships a scale sibling and an fp8 checkpoint that loses one are both
  refused. The CPU forward resolves the bf16 arm's weights as resident bf16
  (no dequant, no scale, no cache entry) and multiplies them with the same
  compute-class rule the fp8 arm's decoded bytes ride.
  RED (before): the loader refused the bf16 fixture by name — "a bf16
  Kolibri-1 artifact is not an implemented arm" at
  `src/vllm/model_executor/models/kolibri1_weights.cpp:69`. GREEN (after):
  5/5 test cases, 12463 assertions, and the fp8 fixture and its exact bf16
  dequantization produce a worst logit gap of 0 at the prefill shape AND at
  the decode shape.
  On the REAL 50-layer model the two arms' 256 teacher-forced per-position
  logit rows are BYTE-IDENTICAL, and both score 97.7% top-1 (6 hard flips
  of 256) against the primary oracle's rows, with a 0.014-nat mean gap.
  The PROVENANCE of the local bf16 copy was settled by measurement, not by
  the byte count: it is NOT the upstream artifact (1362 of shard 1's 1375
  tensors differ from upstream; it is byte-identical to the campaign's own
  fp8->bf16 dequantization), so the upstream artifact's own gate is OWED.
  Evidence: `docs/bench-evidence/kolibri1-bf16-arm-20261021.md`.
  Spec: `.agents/specs/kolibri1-bf16-arm.md`.

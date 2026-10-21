ID: ISSUE-LOCAL-01M5G38WAKAFKPNPQMSNM8EJ7H
Title: Kolibri-1 CPU path refuses a plain bf16 checkpoint, so the fp8 weight quantization's own error cannot be measured
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-21
Updated: 2026-10-21
Closed: -

## Problem

`kolibri1_weights.cpp::CheckQuantConfig` hard-requires the checkpoint's
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

-

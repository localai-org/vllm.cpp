ID: ISSUE-LOCAL-01M687MYRKDS21SBESBG012KN7
Title: Flint attention is host-side on the P150 -- port sinks, RoPE, and the sliding window onto the device through the AttnBlock seam
Row: FLINT-P150-ATTENTION
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: -

## Problem

The Flint P150 device arm (W0/W1) runs norms and matmuls on device; attention remains host-side (SinksAttention, flint_tt.cpp:126) with activations crossing to device around it. This is the last big port of the Flint P150 campaign and the likely consumer of the non-routed-attribution row's findings. Stages: single-layer device probe (sinks + head-dim QK-norm + RoPE-as-configured + 513 window, both window modes, per-head slices, band gate from the W2 substrate-noise record) -> all-layers AttnBlock wiring with the full-model keep-quant band gate -> perf trace vs the host-attention baseline. Hazards from campaign history: SDPA only-2-heads, PA correct-but-output-wrong, device-PA-never-verified -- per-head slices checked individually, model-level token gate required before 'works'. Substrate noise: band gate, never bit-exact. Spec: .agents/specs/flint-p150-attention.md.

## Resolution

-

ID: ISSUE-LOCAL-01M5RVS7Q3EFVHGC6YBVF02A9Q
Title: Flint on Tenstorrent: W0 device forward dispatch and first device-vs-CPU correctness baseline
Row: MODEL-TEXT-FLINT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-25
Updated: 2026-10-25
Closed: -

## Problem

The Flint architecture (MODEL-TEXT-FLINT, rows MODEL-TEXT-FLINT / flint-cpu-gguf) has no Tenstorrent arm: the registry forward serves the CPU reference path only, and a Tenstorrent queue falls through to host code with no device dispatch. The kolibri1 TT campaign (kolibri1_registry.cpp kTENSTORRENT case) establishes the pattern. W0 scope: a real device forward for the bf16-resident Flint weights (GGUF DequantGgufRowToBf16 arm), registry dispatch on kTENSTORRENT, red-first refusal test, and an honest device-vs-CPU correctness baseline (first-token logits distance, greedy chain, layer-0 activation dumps) on the local P150. NOT in scope: keepquant device arm, oracle token gate, performance trace.

## Resolution

- 2026-10-25 (wave 0, branch `row/FLINT-TT-W0`, this change): the W0 device
  arm landed — the `kTENSTORRENT` dispatch case, the bf16-resident device
  forward (`flint_tt.cpp`; experts streamed at 37,748,736 B/layer against the
  70.5 GiB tier), the named per-device refusals, and the honest device-vs-CPU
  baseline on the local P150: first-token logits max_abs=3.20e+01,
  rel_rms=2.58e-01 (n=750720), greedy chain 1/8, dominated by the pinned
  substrate's ~1% systematic device rms_norm bias (op probe: matmul 1.9e-3,
  rmsnorm 7.8e-3) — the kolibri-row defect class, owed as the next wave's
  lever. Evidence: `docs/bench-evidence/flint-tt-w0-20261025/` (red.txt,
  green.txt, report.md). The dispatch was proven EXERCISED
  (`FlintTTForwardCalls` 0 -> 1 on the card). Closes when this branch lands.

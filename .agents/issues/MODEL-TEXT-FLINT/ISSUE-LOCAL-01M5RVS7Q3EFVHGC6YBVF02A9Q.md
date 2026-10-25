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

-

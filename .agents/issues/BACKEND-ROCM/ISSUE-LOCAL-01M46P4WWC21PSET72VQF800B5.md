ID: ISSUE-LOCAL-01M46P4WWC21PSET72VQF800B5
Title: rocm exl3: half-integer bit rates (K+0.5, mul1) for OrcaSAQ-class mixed-rate checkpoints
Row: BACKEND-ROCM
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-05
Updated: 2026-10-05
Closed: 2026-10-05

## Problem

Checkpoints whose trellis tensors have last-dim = 16*K+8 (K+0.5 bits per weight, mul1 codebook, e.g. orcarouter/OrcaSAQ-2-27B-EXL3-3.21bpw with 120 tensors at 56 words) fail LoadExl3's words%16==0 gate and have no kernel arm to decode them even if loaded. exllamav3 >= git-679835b7 defines the format: alternating KA/KA+1-bit positions (mask 0xAAAA, period 16), dq8_half decode.

## Resolution

Landed 2026-10-05 on rocm-gfx11-exl3-perf: bda33fd41 (frac decode + loader + dispatch) and 8cc6aa921 (reconstruct-seam repair). OrcaSAQ-2-27B-EXL3-3.21bpw loads and greedy-decodes on gfx1100; exl3 ctest suite 20/22 (2 pre-existing failures unrelated); token gate recorded as distributional in spec Outcome — integer-rate sibling diverges from the same oracle at the same class of argmax tie.

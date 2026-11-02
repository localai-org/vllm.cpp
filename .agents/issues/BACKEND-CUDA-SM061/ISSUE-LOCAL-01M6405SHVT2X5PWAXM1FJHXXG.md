ID: ISSUE-LOCAL-01M6405SHVT2X5PWAXM1FJHXXG
Title: Mirror Pascal MMQ D4 prefill activation quantization
Row: BACKEND-CUDA-SM061
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

Actual SmolLM2 Q5_0 and Q8_0 prefill projections disagree with the pinned CUDA oracle on identical inputs and packed weights. The oracle MMQ D4 layout retains F32 per32 scales and uses inverse-first rounding; the current scratch narrows scales to F16 and uses a different reciprocal route.

## Resolution

-

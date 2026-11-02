ID: ISSUE-LOCAL-01M63Z1WJK4DYN2VREM2723PJ2
Title: Use GPU Q8_1 activations for Pascal Q6_K decode
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

Actual SmolLM down projections use Q6_K with Q8_1 32-element activation scales in the GPU oracle, but the current Pascal path uses Q8_K 256-element scales.

## Resolution

-

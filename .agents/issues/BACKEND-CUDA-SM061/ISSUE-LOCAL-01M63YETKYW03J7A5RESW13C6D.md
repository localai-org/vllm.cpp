ID: ISSUE-LOCAL-01M63YETKYW03J7A5RESW13C6D
Title: Mirror Pascal Q5_0 decode activation sums from the GPU oracle
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

Pascal SmolLM Q5_0 decode uses CPU Q8_0 activation math instead of GPU Q8_1 original activation sums and diverges at generated token 8.

## Resolution

-

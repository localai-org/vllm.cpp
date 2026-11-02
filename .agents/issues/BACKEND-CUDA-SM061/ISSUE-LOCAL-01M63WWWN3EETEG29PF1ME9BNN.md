ID: ISSUE-LOCAL-01M63WWWN3EETEG29PF1ME9BNN
Title: Pin malformed dense merged GEMM descriptor contracts
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

Independent mutations remove dense descriptor guards without failing existing linear method tests.

## Resolution

29 October 2026: three malformed-input cases pin the dense descriptor guards.
The complete host suite passes 13 cases and 107 assertions. All four guard
removal mutations fail with intended exception assertions. Evidence is in
`.agents/evidence/pascal-volta-enable/quant-contract-repair/`.

The repair remains open until landing. Full model gates remain separate.

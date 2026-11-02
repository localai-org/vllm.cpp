ID: ISSUE-LOCAL-01M641CAZNCNRRYAJF8DCV86R1
Title: Mirror legacy CUDA auto FP16 dtype through shared dense execution
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

Pinned vLLM resolves BF16-configured models to F16 on Pascal and Volta, but the current engine hardcodes BF16 and shared scalar, cache and sampling seams cannot execute resolved F16 end to end.

## Resolution

-

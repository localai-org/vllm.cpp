ID: ISSUE-LOCAL-01M641H5N4MSWRZSAN07YEND4C
Title: Account for the CUDA legacy backend registrar in the DSR gate
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

The DSR gate counts the single CUDA_LEGACY backend registration key as shared device branching, while existing exact-count allowlist policy exempts the same platform descriptor keys. Add only the exact descriptor exception and prove extra branches remain rejected.

## Resolution

-

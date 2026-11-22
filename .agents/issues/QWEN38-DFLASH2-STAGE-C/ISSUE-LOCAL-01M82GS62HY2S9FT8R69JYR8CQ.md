ID: ISSUE-LOCAL-01M82GS62HY2S9FT8R69JYR8CQ
Title: Write the P150 benchmark-reference map against the Lottolabs Inkling-2 package
Row: QWEN38-DFLASH2-STAGE-C
State: OPEN
Kind: docs
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-22
Updated: 2026-11-22
Closed: -

## Problem

The developer asked to continue until we can get benchmarks and compare with other implementations. A verified local copy of Lottolabs/Inkling-2-Mini-TT-Mixed-BFP4-BFP8-P150 @ b0ad7060 now exists at /mnt/models/Inkling-2-Mini-TT-Mixed-BFP4-BFP8-P150 (482/482 SHA256 OK). Write an analysis-only reference map covering the decode-bucket scheme and what a fair trace-replay comparison must replicate, device-side exact sampling and adaptive speculative decoding versus our DFlash2 drafter, the native BFP4 tensor package versus our VT_TT_WEIGHT_RESIDENCY=bfp8 runtime staging, and a benchmark-axis table mapping our current state to Lottolabs reference numbers to the unit that would produce the comparison. No src/ changes.

## Resolution

-

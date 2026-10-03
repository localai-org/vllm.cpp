ID: ISSUE-LOCAL-01M29ATB6N2SFZD7JA6CCR2KXC
Title: The DeepSeek-V4.1-Flash EXL3 3.5bpw "Pollard" checkpoint (`bot-lab-21/...` @ `f129e31a81`, 428.49 GiB) is a HYBRID that is mostly not EXL3, and a 4-device tensor-parallel artifact this fleet cannot assemble. Verified by LFS oid rather than by its card: shards 1, 2 and 43-48 are BYTE-IDENTICAL to the deepseek-ai release (199.88 GiB of it) and only 3-42 differ; `quantized_modules` covers the routed experts alone, leaving fp8 dense and attention, MXFP4 shared experts and DSpark drafter, fp8 engram tables and native fp4 KV as the release's own bytes. So "EXL3 3.5bpw" is an average over part of one file, not a format for the file, and "Pollard" is a Hessian-aware bit-allocation RECIPE over per-expert widths, not a format -- the bytes are turboderp EXL3 trellis. It does not fit one GB10 (3.6x), the publisher's numbers are 4x DGX Spark TP4 and unreproduced here, and the exllamav3 revision that produced it is UNVERIFIED
Row: MODEL-DSV41-EXL3
State: OPEN
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-11
Closed: -

## Problem

A published quantized arm with no row, whose headline format claim does not describe most of the file.

## Resolution

-

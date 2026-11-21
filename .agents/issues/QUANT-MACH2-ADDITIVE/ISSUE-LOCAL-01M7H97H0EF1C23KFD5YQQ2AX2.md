ID: ISSUE-LOCAL-01M7H97H0EF1C23KFD5YQQ2AX2
Title: The Mach-2 packed format has no decoder, so no weight can be produced from a code stream
Row: QUANT-MACH2-ADDITIVE
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-15
Closed: 2026-11-15

## Problem

The QUANT-MACH2-ADDITIVE spec splits the campaign into five units and U1 is the decode oracle: the arithmetic that turns fn_packed_v2 code streams into weights. Nothing implements it. Without it no constant of the format can be checked against the release decoder, no fixture can be generated for U2, and the seam in U3 would be built on unverified understanding of the packing. U1 is deliberately artifact-free: the trellis state walk, the wave map, the LUT expansion, the per-tile and per-expert scales, the radix-12/20 Hadamard, the sign derivation, and the int5 and int4 group unpacking are integer code with at most one deliberate bf16 round, so they can be built and tested on a CPU with hand-derived literals before any 26.6 GB download or GPU lease.

## Resolution

U1's integer and rotation primitives landed on row/QUANT-MACH2-ADDITIVE-U1: 11 cases and 2621 assertions green under MSVC /W4 /WX, registered as test_mach2_decode. Three mutations of the implementation were each caught (spine sign bit, radix B corner, bf16 rounding bias), and the third only after a tie case was added, which is recorded. The wave map assigns one index per anti-diagonal rather than a running counter, found by a failing literal. Still owed by U1 and recorded in the spec: the _HS string tables, the Philox sign derivation, the wave-gamma tile assembly and the wscale chain.

ID: ISSUE-LOCAL-01M7HAJF0CCRRSAYN5ZEVKF8EB
Title: The Mach-2 decoder's signs and tile assembly are missing, so a code stream still cannot become a weight tensor
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

The first half of U1 landed the trellis walk, both LUT foldings, the integer groups and the radix rotation, but three pieces of decode_expert were still absent: the Philox sign expansion that leaves the expert rotation, the per-tile wave_gamma and wscale chain, and the assembly that turns gathered LUT rows into a rotated tile. Without them the oracle cannot produce a single weight, so U2 has nothing to compare against and the packing's tile layout stays unverified.

## Resolution

ExpertSigns and DecodeExpertTile landed: 13 cases and 2714 assertions green under MSVC /W4 /WX. The sign expansion is pinned to values a transcription of decode.py produced, so all four Philox constants, the round count and the truncation are covered; the assembly is pinned to a transcription-derived checksum plus four samples, and pins the operation order. Five mutations were each caught: a multiplier digit, the round count, swapped SU/SV tables, a dropped wscale, and gamma read in tile order. Two layout facts came out of the work: wave_gamma is indexed by anti-diagonal start, so it must hold at least mb+nb entries and a 1x1 tiling reads index 1 (a one-element array is out of range in the release decoder too); and a codeword's four values become four consecutive columns of one row. Still owed by U1: the _HS string tables, and the SHA-256 behind sign_seed.

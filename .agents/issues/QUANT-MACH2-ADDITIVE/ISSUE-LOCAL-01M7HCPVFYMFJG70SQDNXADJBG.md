ID: ISSUE-LOCAL-01M7HCPVFYMFJG70SQDNXADJBG
Title: HadamardExpert drops the inner stage's 1/sqrt(inner) normalisation, scaling every radix-path rotation
Row: QUANT-MACH2-ADDITIVE
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-16
Updated: 2026-11-16
Closed: 2026-11-16

## Problem

The expert rotation's radix branch ran the bare butterflies on the inner block and then the radix matrix, but decode.py recurses into a NORMALISED hadamard there, which divides by sqrt(inner). The result was scaled by sqrt(inner), and on the real expert shapes 640x2560 and 2560x640 that is exactly 64x. The U1 suite missed it because its orthonormality case only used cols=20, where the inner block is one wide and there is nothing to normalise.

## Resolution

The differential gate against the published decode.py found this as an exact 64x scale, sqrt(128*32), on both real expert shapes. HadamardExpert now recurses into the normalised transform for its inner stage. The orthonormality case covers cols 20, 40, 80, 256 and 640, and reintroducing the bare butterflies fails 758 of 3474 assertions while the fixed suite passes all 3474. decode.py itself remains unmodified.

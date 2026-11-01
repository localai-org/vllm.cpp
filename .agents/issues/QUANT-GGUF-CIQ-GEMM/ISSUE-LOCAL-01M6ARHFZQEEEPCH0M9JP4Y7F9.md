ID: ISSUE-LOCAL-01M6ARHFZQEEEPCH0M9JP4Y7F9
Title: Compose NEON Q4_K dispatch tests with the AVX2 tier
Row: QUANT-GGUF-CIQ-GEMM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: 2026-11-01

## Problem

After upstream merge, the NEON Q4_K dispatch test assumes that no x86 AVX2 tier wraps its selector. Default x86 execution fails pointer and bit-identity assertions despite valid AVX2 dispatch.

## Resolution

2026-11-01: compose both production selectors in the incoming NEON dispatch test. Default and forced AVX2 pass 5 cases and 260 assertions; portable passes 5 cases and 516 assertions. The independent quant-dot suite passes 35 cases and 285691 assertions per selection. Arm corpus remains unchanged.

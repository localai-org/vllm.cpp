ID: ISSUE-LOCAL-01M3RM2TM98FY569AZ6CEASBD3
Title: gfx11 EXL3 decode residual: Exl3GemmK keeps 90% kernel time; target >=50 tok/s
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

After the GEMV arm landed (spec rocm-exl3-gemv), serving Qwen3.5-9B-EXL3-4.00bpw on gfx1101 reaches 3.30/12.27/21.93 tok/s at c=1/4/8. rocprofv3 still attributes ~90% of kernel time to the scalar Exl3GemmK transcription: lm_head is 6bpw (uninstantiated GEMV arm, n=248320) and mid-m/GDN shapes decline the m<=8 arm. User target: >=50 tok/s decode plus max prefill speed via gfx11 hardware paths.

## Resolution

-

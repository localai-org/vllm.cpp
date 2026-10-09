ID: ISSUE-LOCAL-01M43XD57VZN22CWGNYZ5MVE5Z
Title: EXL3 6bpw cb=2 m=1 lm_head dot arm has no unit gate
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

The production lm_head path Exl3DotKImpl<6,8,2> (the arm commits 87de2c0ff/edd7765b6 rewrote: roving k-row pointers + compile-time codebook Exl3DecodePair6C<2> == Exl3DecodePairCb2Dp4a) is exercised by no test: test_exl3_rocm pins force_gemv=0 (transcription only), the only bits=6 fast-arm case in test_exl3_rocm_gemv is cb=1, and recon tests run (6,2) only through the scalar decoder at m=256. test_exl3_gemv.cpp:241 even pins HardEligible(1,5120,248320,6,2)==false (the exact serving shape) with no dot-arm case. Per reachability policy the recorded gate (test_exl3_rocm 7/7) measures a class, not the capability; a decode-pair arithmetic mutation stays green. Also stale: the (6,*) test title says 'takes the reconstruct arm at every m' while m<=8 now take the dot arm, and its header states 1.0e-3 while the CHECK uses 1.0e-2.

## Resolution

-

ID: ISSUE-LOCAL-01M686P391579S3SRN675V2XT7
Title: Two named small defects: the B2BI/B2II env silent-skip, and the preflight --staged SyntaxError
Row: TT-SMALL-FIXES
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: -

## Problem

(1) tests/vllm/models/test_kolibri1_tt_b2bi.cpp:1021 reads VT_KOLIBRI1_TT_B2BI_MODEL while every doc/instruction sets VT_KOLIBRI1_TT_B2II_MODEL -- the device leg silently skips while the suite reads 10/10 SUCCESS (TT-ANCHORING evidence; the 0.29s phantom-green ctest runs). Fix: read B2II first, fall back to B2BI, comment citing TT-ANCHORING; red-first = with only B2II set HEAD silently skips, post-fix it runs or refuses by name. (2) agent-preflight.sh --staged crashes with a Python SyntaxError around line 86 (kolibri1-GGUF implementer report, 2026-10-30); non-staged run clean. Fix + a regression case running --staged in a scratch repo asserting no SyntaxError. Spec: .agents/specs/tt-small-fixes.md. One branch, two commits.

## Resolution

-

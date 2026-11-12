ID: ISSUE-LOCAL-01M6S2M3YQZGXP63W6NXS2YK4E
Title: test_neon_q4k_dot's dispatch auto-arm fails on x86 CI -- no NEON tier exists there
Row: FLINT-NEON-Q4K-DOT-X86-GATE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-06
Updated: 2026-11-06
Closed: -

## Problem

test_neon_q4k_dot.cpp:246 asserts the auto-mode dispatch resolves to QuantQ4KNeonVecDot() -- but on the x86 CI lanes QuantQ4KNeonVecDot() IS the HWCAP probe and returns nullptr, so the assertion compares 1 == nullptr and reds build-test-cpu (run 42172515466, job 127743391331). The test's own tier-availability case (:149) already guards its body with the same check; the dispatch case's auto-arm lacks it. Fix: guard the arm on QuantQ4KNeonVecDot() != nullptr. Red = the x86 CI failure itself; green-after = the suite green on x86 AND on a DotProd host.

## Resolution

-

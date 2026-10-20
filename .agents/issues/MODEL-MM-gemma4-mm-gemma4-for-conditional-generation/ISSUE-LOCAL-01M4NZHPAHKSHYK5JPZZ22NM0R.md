ID: ISSUE-LOCAL-01M4NZHPAHKSHYK5JPZZ22NM0R
Title: Honor Gemma thinking and latest-turn reasoning end semantics
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-11
Updated: 2026-10-11
Closed: 2026-10-11

## Problem

Gemma engine inherits historical-marker detection and ignores thinking; a fresh model turn after completed thought incorrectly ends reasoning when thinking is enabled.

## Resolution

2026-10-11: Scoped Gemma factory policy and latest-tail text-literal override pass 5 cases/226 assertions. Pre-fix executable RED:30 failures including enabled historical-thought/new-turn. Five CPU regressions and28 metadata pass; factory deletion fails4 assertions. Evidence: assets/work/2026-10-11-e4b-reasoning-end/README.md. Root verification and independent review pending; prompt/served/token parity and full preflight not claimed.

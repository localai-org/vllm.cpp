ID: ISSUE-LOCAL-01M738WXZA9RW5B5AHDZ6F0T4R
Title: FlintPhaseDst formed &g_flint_phase->X on a NULL totals pointer (masked second UBSan finding in test_flint_paged)
Row: MODEL-TEXT-FLINT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-10
Updated: 2026-11-10
Closed: -

## Problem

Every FlintPhaseDst(&g_flint_phase->bucket) call site in flint.cpp performs a member access on g_flint_phase BEFORE the helper's null check; with VT_FLINT_PHASE_TIMING unset the thread_local totals pointer is null, so UBSan flags 'member access within null pointer of type struct FlintPhaseTotals'. The finding is masked on main CI because the misaligned-load abort (ISSUE-LOCAL-01M738WQD57K9D3R2AQ2KCDHND) fires first under -fno-sanitize-recover=all; fixing that defect exposes this one at flint.cpp:1151 in the same test. Fixed by passing the totals pointer and a pointer-to-member to FlintPhaseDst so the member access only happens inside the guarded branch.

## Resolution

-

## Resolution

2026-11-10, `row/FLINT-UBSAN-EMBED-ALIGN`, found in flow while gating the
misaligned-load fix. `FlintPhaseDst` now takes the totals pointer plus a
pointer-to-member (`double FlintPhaseTotals::*`), so `&totals->bucket` is
formed only inside the null-guarded branch; all 14 call sites updated
mechanically, no behavior change (the returned pointer is identical in both
arms). The finding was masked on main CI by the earlier misaligned-load
abort; it aborts test_flint_paged at flint.cpp:1151 once that defect is
fixed. GREEN: `docs/bench-evidence/flint-ubsan-embed-align-2026-11-10/green-ubsan.log`
(zero runtime errors after both fixes).

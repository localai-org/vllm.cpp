ID: ISSUE-LOCAL-01M58B5J4DHKN5AEVCRR6Y7YWE
Title: windows-msvc is red on main: unreferenced numa_guard (C4101) and a shadowed loop variable (C4456) under -WX
Row: ENG-RELEASE-WINDOWS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-18
Updated: 2026-10-18
Closed: 2026-10-18

## Problem

origin/main fails windows-msvc-cpu and windows-msvc-vulkan on every recent run (job 117743432961, and the same failure on PRs #3464/#3469/#3475/#3477/#3478). Two causes under the -WX MSVC lanes: (1) kolibri1_dequant_cache.h:243 and kolibri1_weights.cpp:225 declare 'const kolibri1_numa::PolicyGuard numa_guard', but PolicyGuard is an empty class off Linux, so MSVC reports C4101 unreferenced local and -WX makes it an error; (2) kolibri1_tt_forward.cpp:692 declares 'const int64_t e' in the residents loop shadowing the function's outer 'e' (line 532), C4456, same -WX death. Repair: [[maybe_unused]] on the two guard declarations (allocation-policy-only object, the attribute documents the intent); rename the residents loop variable to 'resident'. Linux build green, kolibri1_w2 suite 7/7 1608 assertions.

## Resolution

2026-10-18: [[maybe_unused]] on both numa_guard declarations, residents loop variable renamed to 'resident'. Linux build green; kolibri1_w2 7/7, 1608 assertions; -WX-relevant MSVC diagnostics eliminated by construction (empty-class attribute + rename). Rides PR to main.

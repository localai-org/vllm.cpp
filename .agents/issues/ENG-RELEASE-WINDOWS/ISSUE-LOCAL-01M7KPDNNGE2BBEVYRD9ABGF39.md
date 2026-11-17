ID: ISSUE-LOCAL-01M7KPDNNGE2BBEVYRD9ABGF39
Title: The Windows library build dies on three MSVC-incompatible constructs before any test runs
Row: ENG-RELEASE-WINDOWS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-16
Updated: 2026-11-16
Closed: -

## Problem

windows-msvc-cpu cannot compile vllm.vcxproj: qwen3_5.cpp:12560 uses GCC's __builtin_memcpy, which MSVC does not define (error C3861), and flint.cpp:701 plus flint_tt.cpp:1065 each shadow an outer 'v' (warning C4456) that the tree's /W4 /WX policy promotes to error C2220. The lane fails in the library, so no test target or runtime behaviour on that lane is exercised at all, and the three defects are not from any in-flight row: blame puts them on 504f75842 (2026-11-08), 090c536fb (2026-10-25) and f4725780b (2026-11-02) and all three lines are present on upstream/main.

## Resolution

-

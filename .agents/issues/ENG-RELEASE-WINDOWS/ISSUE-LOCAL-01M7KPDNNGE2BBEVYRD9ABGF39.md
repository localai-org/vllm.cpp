ID: ISSUE-LOCAL-01M7KPDNNGE2BBEVYRD9ABGF39
Title: The Windows library build dies on three MSVC-incompatible constructs before any test runs
Row: ENG-RELEASE-WINDOWS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-16
Updated: 2026-11-16
Closed: 2026-11-16

## Problem

windows-msvc-cpu cannot compile vllm.vcxproj: qwen3_5.cpp:12560 uses GCC's __builtin_memcpy, which MSVC does not define (error C3861), and flint.cpp:701 plus flint_tt.cpp:1065 each shadow an outer 'v' (warning C4456) that the tree's /W4 /WX policy promotes to error C2220. The lane fails in the library, so no test target or runtime behaviour on that lane is exercised at all, and the three defects are not from any in-flight row: blame puts them on 504f75842 (2026-11-08), 090c536fb (2026-10-25) and f4725780b (2026-11-02) and all three lines are present on upstream/main.

## Resolution

2026-11-17: fixed in 26331ec0c. qwen3_5.cpp:12560 now uses std::memcpy; the shadowing CPU-dump loops are renamed in flint.cpp (3 sites) and flint_tt.cpp (4 sites, two of them inside the staging slice-map diagnostic that the CI log never reached because error C2220 stops the TU at the first one). Verified by a from-scratch CMake+Ninja+MSVC 19.44 build of the vllm target at the repository's /W4 /WX with diarization ON and no shims: 544 steps, BUILD_RC=0; the same build failed at flint_tt.cpp with error C2220 before the change.

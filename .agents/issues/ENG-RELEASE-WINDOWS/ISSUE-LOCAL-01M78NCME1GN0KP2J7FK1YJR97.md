ID: ISSUE-LOCAL-01M78NCME1GN0KP2J7FK1YJR97
Title: The Windows Vulkan lane reds on five shadowed loop variables and an MSVC-unknown __builtin_memcpy
Row: ENG-RELEASE-WINDOWS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-12
Updated: 2026-11-12
Closed: 2026-11-12

## Problem

The windows-msvc-vulkan lane on main at de3493a37 fails in 'Build and execute the native Windows Vulkan focused gate'. cl /W4 promotes C4456 to C2220 under /WX, and five range-for loop variables re-declare a name the enclosing function already binds: flint.cpp:689 and 822 shadow the attention buffer declared at flint.cpp:562, flint_tt.cpp:1065 and 1093 shadow 'v', and flint_tt.cpp:1187 shadows 'c'. The same job also reports qwen3_5.cpp:12560 error C3861 for '__builtin_memcpy', a GCC/Clang builtin MSVC does not define. Each site is a diagnostics-only change: the shadowing names are local loop variables and the memcpy copies the same four bytes.

## Resolution

Fixed on main at de3493a37 by renaming the five shadowed range-for variables and replacing __builtin_memcpy. Red before: the windows-msvc-vulkan job 130357281689 log carries C4456 at flint.cpp:689 and 822, flint_tt.cpp:1065, 1093 and 1187, and C3861 at qwen3_5.cpp:12560. Green after: cl /nologo /Zs /std:c++20 /EHsc /W4 /WX /permissive- on VS 2022 14.44.35207 returns 0 with no diagnostics for all three translation units. Behaviour is unchanged: loop-local names and a four-byte copy whose source and destination are untouched.

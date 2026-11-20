ID: ISSUE-LOCAL-01M7W6R8RN7TY0MCXS1TMP7X3A
Title: windows-msvc lanes red on main: 4 hard errors (C3861 __builtin_memcpy, C4244, 2x C4456) + 5 latent shadow warnings under -WX in qwen3_5.cpp / flint.cpp / flint_tt.cpp
Row: ENG-RELEASE-WINDOWS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-20
Updated: 2026-11-20
Closed: -

## Problem

Both windows-msvc-cpu and windows-msvc-vulkan fail on main (run 44692868221) at exactly: qwen3_5.cpp:12902 C3861 __builtin_memcpy identifier not found; qwen3_5.cpp:4683 C2220 (C4244 float->uint64_t in the QKVZ-Z debug block); flint.cpp:701 and flint_tt.cpp:1065 C2220 (C4456 declaration of v hides previous local). Once those compile, five more warnings become errors: flint.cpp:835 (v), flint_tt.cpp:1093 (v), flint_tt.cpp:1187 (c), qwen3_5.cpp:4703 (h hides function parameter, C4457), qwen3_5.cpp:5383 (z hides parameter, C4457). All in debug/instrumentation blocks. Fix: std::memcpy instead of __builtin_memcpy, correct the z_acc conversion, rename the shadowing locals. Verify locally with gcc/clang -Wshadow -Werror on the three TUs (MSVC not available on this host; C4456/C4457 map to -Wshadow).

## Resolution

-

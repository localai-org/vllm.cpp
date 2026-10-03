ID: ISSUE-LOCAL-01M3Z28HG59E1W8DYC2JJ9EVDD
Title: windows POSIX scan flags the parakeet ggml third-party sources
Row: ENG-RELEASE-WINDOWS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-02
Updated: 2026-10-02
Closed: 2026-10-02

## Problem

The windows-msvc-vulkan and windows-msvc-cpu CI jobs fail since the diarization seam (2026-09-27): scripts/check-windows-portability.py reports 'unguarded POSIX include/call reaches Windows' on build-pr-windows-vulkan/_deps/parakeet_cpp-src/third_party/ggml/src/ggml-cpu/amx/amx.cpp:12, mmq.cpp:18, ggml-cpu.c:2542 and ggml.c:85-88 (CI run 37034253522). Those files are the vendored ggml inside the FetchContent parakeet_cpp dependency; the repository does not own them and must not patch them. Root cause: _load_codemodel_sources keeps any dependency source whose path resolves under the source root; the Windows release build configures into build-pr-windows-vulkan/ INSIDE the tree, so the FetchContent _deps sources enter the shipped-server source set. On Linux the scanner's own self-configure uses a build dir outside the root, so the same sources silently drop out and the lane difference hid the defect. The scanner's contract is OUR sources reaching Windows-incompatible POSIX, not third-party vendored code; repair by excluding FetchContent _deps/ sources from the derived source set.

## Resolution

Scanner excludes FetchContent _deps sources; red-before test reproduced the CI error verbatim, suite green after.

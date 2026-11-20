# SPEC — accept `if(WIN32)` as the central NOMINMAX guard in check-windows-portability.py

Row: ENG-RELEASE-WINDOWS (rides PR #3561). Issue: the PR's own CMakeLists
change.

## Why

`scripts/check-windows-portability.py::_has_central_msvc_nominmax` accepts
ONLY the literal `if(MSVC)` block carrying `add_compile_definitions(... NOMINMAX ...)`.
PR #3561 (Windows clang++ in GNU mode) correctly widens that guard to
`if(WIN32)` — clang++ on Windows is not MSVC (CMake's `MSVC` variable is false
for it) but includes the same Windows headers under the same min/max-macro
contract, so NOMINMAX must be defined for it too. The checker then reports
`NOMINMAX must be defined centrally ...` for every windows.h consumer
(fs_io.cpp, cpu_threadpool.cpp, ...) — a false positive against the checker's
own stated contract ("NOMINMAX must be defined centrally or by an
absence-guarded local fallback before windows.h"): `if(WIN32)` IS central,
before the first target, and BROADER than `if(MSVC)` (true for both cl.exe and
clang++ on Windows). The agent-record CI job on #3561 failed on exactly this.

## Change

- `_has_central_msvc_nominmax` renamed `_has_central_nominmax`; the contract
  regex accepts `if(MSVC)` or `if(WIN32)` (either spelling) with the
  `add_compile_definitions(... NOMINMAX ...)` inside the block and the block
  before the first `add_library/add_executable/add_subdirectory` — unchanged
  otherwise. The error message text is unchanged (it describes the contract,
  not the spelling).
- Tests both directions in tests/scripts/test_check_windows_portability.py:
  a `if(WIN32)`-guarded central NOMINMAX passes (RED before this change), the
  existing `if(MSVC)` form still passes, WIN32-without-NOMINMAX still fails,
  and a NOMINMAX definition after the first target still fails.

## Gates

- Red-before: the new WIN32 test fails against the unmodified checker
  (captured in the commit body); green-after: full
  `test_check_windows_portability.py` suite passes.
- `check-windows-portability.py` on the real tree: still green (main uses
  `if(MSVC)`), and on a tree with the `if(WIN32)` form: green.
- agent-record / commit-protocol-tag / pr-size gates on the #3561 branch.

## Git integration

Rides PR #3561 (one PR; the checker change must land in the same PR whose
CMakeLists change trips the old rule — same rule as record edits riding the
change that made them stale).

## Now

2026-11-20, branch fix/windows-clang-build (theoparis's PR #3561, rebased).

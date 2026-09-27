ID: ISSUE-GH-720
Title: `M_PI` is a POSIX extension MSVC's `<cmath>` does not define, so `windows-msvc-*` hard-error on every PR; spec [`windows-msvc-m-pi.md`](../../specs/windows-msvc-m-pi.md). MEASURED, and the intake's "three TUs" is wrong both ways: the two LTX-2.5 VAEs carry their own `#define` and compile (one of them with ZERO uses), while `tests/vllm/models/test_vocoder1d.cpp` — which `git grep -- src include` never looked at and the CI log never reached — is the second real break
Row: ENG-RELEASE-WINDOWS
State: UNKNOWN
Kind: bug
GitHub: 720
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:69`

### Frozen archive evidence

> | [#720](https://github.com/mudler/vllm.cpp/issues/720) | `ENG-RELEASE-WINDOWS` | `M_PI` is a POSIX extension MSVC's `<cmath>` does not define, so `windows-msvc-*` hard-error on every PR; spec [`windows-msvc-m-pi.md`](../../specs/windows-msvc-m-pi.md). MEASURED, and the intake's "three TUs" is wrong both ways: the two LTX-2.5 VAEs carry their own `#define` and compile (one of them with ZERO uses), while `tests/vllm/models/test_vocoder1d.cpp` — which `git grep -- src include` never looked at and the CI log never reached — is the second real break | bug |

## Resolution

-

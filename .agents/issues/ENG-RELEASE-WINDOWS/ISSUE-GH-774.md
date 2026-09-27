ID: ISSUE-GH-774
Title: `check-windows-portability.py:1710` asserted the MSVC warning policy with `token in warnings`, and `"/WX" in "/WX-"` is `True` — `/WX-` is MSVC's spelling for DISABLE warnings-as-errors, so the gate was blind to its own inversion. Measured on PR #640 commit `74ba3823f`, which shipped `/WX-` on the CXX arm while the only bare `/WX` left was on `$<COMPILE_LANGUAGE:OBJCXX>` — Objective-C++, the Metal backend, which never compiles under MSVC. Two further blindnesses fell out of the same `in`: `/W44996` answers for `/W4`, and `CMakeLists.txt:30`'s `#` comment satisfies the whole policy on its own. Repaired to a token-boundary match over the flags that reach the C/C++ compile, plus the negating spellings `/WX-` `/W0` `/w`; spec [`windows-msvc-warning-policy-tokens.md`](../../specs/windows-msvc-warning-policy-tokens.md)
Row: ENG-RELEASE-WINDOWS
State: UNKNOWN
Kind: bug
GitHub: 774
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:144`

### Frozen archive evidence

> | [#774](https://github.com/mudler/vllm.cpp/issues/774) | `ENG-RELEASE-WINDOWS` | `check-windows-portability.py:1710` asserted the MSVC warning policy with `token in warnings`, and `"/WX" in "/WX-"` is `True` — `/WX-` is MSVC's spelling for DISABLE warnings-as-errors, so the gate was blind to its own inversion. Measured on PR #640 commit `74ba3823f`, which shipped `/WX-` on the CXX arm while the only bare `/WX` left was on `$<COMPILE_LANGUAGE:OBJCXX>` — Objective-C++, the Metal backend, which never compiles under MSVC. Two further blindnesses fell out of the same `in`: `/W44996` answers for `/W4`, and `CMakeLists.txt:30`'s `#` comment satisfies the whole policy on its own. Repaired to a token-boundary match over the flags that reach the C/C++ compile, plus the negating spellings `/WX-` `/W0` `/w`; spec [`windows-msvc-warning-policy-tokens.md`](../../specs/windows-msvc-warning-policy-tokens.md) | bug |

## Resolution

-

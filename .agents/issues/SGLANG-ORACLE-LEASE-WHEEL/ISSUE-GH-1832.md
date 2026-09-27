ID: ISSUE-GH-1832
Title: **The SGLang manifest's file count `3338` is gated only against itself, and appears in no executing code.** Raised by the fresh review of PR #1831 and NOT repaired there: the number and the claim are W1's (`727efb39c`), and re-deriving them needs an `rc` job on `dgx:gpu0`, which is outside W2's scope. `3338` is quoted as MEASURED in three records -- `.agents/environment.md`, `.agents/oracles/sglang.md`, and the `SGLANG-ORACLE-LEASE-WHEEL` row of `.agents/sglang-matrix.md` ("**3338 of 3338** manifest files, 0 missing, 0 extra, 0 differing") -- while `grep -rn '3338' scripts/ tests/scripts/ .github/` returns `rc=1`. The only test that touches the count is `tests/scripts/test_sglang_lease_identity.py:54-55`, which asserts `manifest["file_count"] == len(manifest["files"])`: a self-consistency check of one JSON document, which cannot see whether 3338 is the count of the real installed tree. MUTATION, run at `85c247580`: drop `sglang/README.md` from `files` and decrement `file_count` to 3337 -- the exact shape of a mis-generated manifest -- and the suite reads `Ran 14 tests ... OK`, `rc=0`. Tree restored byte-for-byte. This matters because `sglang.__commit_id__` is `None` in the published wheel, so the manifest is the ONLY identity assertion available for this oracle, and a wrong manifest makes `IDENTITY_RC=0` a tautology one level up. A checker reading the committed JSON cannot repair it; the repair is a re-derivation -- a second independent install that REGENERATES the manifest and diffs it against the committed one. Until that runs the honest record is "3338 files, from one generation run on 2026-08-19, not independently re-derived". Listed under `## Owed` in [sglang-wheel-in-lease.md](../../specs/sglang-wheel-in-lease.md)
Row: SGLANG-ORACLE-LEASE-WHEEL
State: UNKNOWN
Kind: bug
GitHub: 1832
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:683`

### Frozen archive evidence

> | [#1832](https://github.com/mudler/vllm.cpp/issues/1832) | `SGLANG-ORACLE-LEASE-WHEEL` | **The SGLang manifest's file count `3338` is gated only against itself, and appears in no executing code.** Raised by the fresh review of PR #1831 and NOT repaired there: the number and the claim are W1's (`727efb39c`), and re-deriving them needs an `rc` job on `dgx:gpu0`, which is outside W2's scope. `3338` is quoted as MEASURED in three records -- `.agents/environment.md`, `.agents/oracles/sglang.md`, and the `SGLANG-ORACLE-LEASE-WHEEL` row of `.agents/sglang-matrix.md` ("**3338 of 3338** manifest files, 0 missing, 0 extra, 0 differing") -- while `grep -rn '3338' scripts/ tests/scripts/ .github/` returns `rc=1`. The only test that touches the count is `tests/scripts/test_sglang_lease_identity.py:54-55`, which asserts `manifest["file_count"] == len(manifest["files"])`: a self-consistency check of one JSON document, which cannot see whether 3338 is the count of the real installed tree. MUTATION, run at `85c247580`: drop `sglang/README.md` from `files` and decrement `file_count` to 3337 -- the exact shape of a mis-generated manifest -- and the suite reads `Ran 14 tests ... OK`, `rc=0`. Tree restored byte-for-byte. This matters because `sglang.__commit_id__` is `None` in the published wheel, so the manifest is the ONLY identity assertion available for this oracle, and a wrong manifest makes `IDENTITY_RC=0` a tautology one level up. A checker reading the committed JSON cannot repair it; the repair is a re-derivation -- a second independent install that REGENERATES the manifest and diffs it against the committed one. Until that runs the honest record is "3338 files, from one generation run on 2026-08-19, not independently re-derived". Listed under `## Owed` in [sglang-wheel-in-lease.md](../../specs/sglang-wheel-in-lease.md) | bug |

## Resolution

-

ID: ISSUE-GH-1649
Title: **`check-windows-portability.py` read `/w` anywhere in `CMakeLists.txt` as a project-wide negation of `/W4 /WX`, so the vendored BoringSSL targets' PRIVATE `/w` red `windows-msvc-cpu` and `windows-msvc-vulkan` on main and on every PR.** `check()` set `warnings = cmake` -- the whole file -- concatenated `cmake/CompilerWarnings.cmake` and did a flat token search, which cannot tell a global `add_compile_options(/w)` from a `target_compile_options(<vendored> PRIVATE /w)`; only the first negates the policy. The refusal lands before any translation unit is read, so the job carries no `error C####`. SECOND red, which #1649 did not record: the same defect fails this checker's own suite -- `test_real_tree_msvc_warning_policy_reaches_the_cxx_compile` with `negation='/w'` on main at `8540a2755` (78 tests, 1 failure) -- so the tree asserted the contradiction in two places at once and one fix clears both. FIXED IN FLOW: `without_foreign_target_compile_options` blanks `target_compile_options` spans whose target PROVABLY names only targets this project never declares (`add_library`/`add_executable` first arguments, with `foreach` bindings resolved); anything unresolved stays in scope, and `cmake/CompilerWarnings.cmake` is kept whole because it applies the policy through an unresolvable function parameter. The widening is bounded by three guard properties proved discriminating by MUTATION -- forcing `_target_is_foreign` to `return True` reds exactly those three and nothing else (82 tests, 3 failures), tree restored byte-for-byte. 82 tests OK; the real tree now prints `Windows portability contract OK`. RESIDUAL, stated not hidden: a negation reaching a project target through a `set()` binding rather than a `foreach()` one is still not caught; no such construct exists in the tree today. Recorded under [`gate-windows-portability-target-scope.md`](../../specs/gate-windows-portability-target-scope.md) `## 4. Risks / decisions`
Row: GATE-WINDOWS-PORTABILITY-TARGET-SCOPE
State: UNKNOWN
Kind: bug
GitHub: 1649
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:641`

### Frozen archive evidence

> | [#1649](https://github.com/mudler/vllm.cpp/issues/1649) | `GATE-WINDOWS-PORTABILITY-TARGET-SCOPE` | **`check-windows-portability.py` read `/w` anywhere in `CMakeLists.txt` as a project-wide negation of `/W4 /WX`, so the vendored BoringSSL targets' PRIVATE `/w` red `windows-msvc-cpu` and `windows-msvc-vulkan` on main and on every PR.** `check()` set `warnings = cmake` -- the whole file -- concatenated `cmake/CompilerWarnings.cmake` and did a flat token search, which cannot tell a global `add_compile_options(/w)` from a `target_compile_options(<vendored> PRIVATE /w)`; only the first negates the policy. The refusal lands before any translation unit is read, so the job carries no `error C####`. SECOND red, which #1649 did not record: the same defect fails this checker's own suite -- `test_real_tree_msvc_warning_policy_reaches_the_cxx_compile` with `negation='/w'` on main at `8540a2755` (78 tests, 1 failure) -- so the tree asserted the contradiction in two places at once and one fix clears both. FIXED IN FLOW: `without_foreign_target_compile_options` blanks `target_compile_options` spans whose target PROVABLY names only targets this project never declares (`add_library`/`add_executable` first arguments, with `foreach` bindings resolved); anything unresolved stays in scope, and `cmake/CompilerWarnings.cmake` is kept whole because it applies the policy through an unresolvable function parameter. The widening is bounded by three guard properties proved discriminating by MUTATION -- forcing `_target_is_foreign` to `return True` reds exactly those three and nothing else (82 tests, 3 failures), tree restored byte-for-byte. 82 tests OK; the real tree now prints `Windows portability contract OK`. RESIDUAL, stated not hidden: a negation reaching a project target through a `set()` binding rather than a `foreach()` one is still not caught; no such construct exists in the tree today. Recorded under [`gate-windows-portability-target-scope.md`](../../specs/gate-windows-portability-target-scope.md) `## 4. Risks / decisions` | bug |

## Resolution

-

ID: ISSUE-LOCAL-01M4FKW890MBHDWRBWBWZSSX45
Title: test_ops_paged_attn_sharedk_wmma_p1 red under ctest: CMakeLists probe path is __FILE__-relative and CWD-dependent
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

Mechanism (measured 2026-10-09, repro binaries from the sanitize-cpu lane): the 'P1 GPU binary is not registered as ordinary CTest' case (tests/vt/test_ops_paged_attn_sharedk_wmma_p1.cpp:115) fails REQUIRE( in ) — it reads tests/CMakeLists.txt via __FILE__ + '/../CMakeLists.txt'. CMakeLists.txt:66 adds -ffile-prefix-map=<source dir>=. so __FILE__ is the source-root-RELATIVE './tests/vt/test_ops_paged_attn_sharedk_wmma_p1.cpp'; the probe path './tests/vt/../CMakeLists.txt' resolves against the process CWD, and ctest runs with CWD=<build>/tests, so the open fails. Run from the source root the same binary passes 32813/32813 assertions (only this CWD-dependent case fails elsewhere; the other 6 cases are CWD-independent). Fails identically in build-test-cpu and both sanitize-cpu lanes (.agents/specs/ci-main-ctest-residue.md) — a path-portability defect in the #785 P1 host package's registration lock, not a sanitizer finding. Fix: bake the tests/CMakeLists.txt path as an absolute compile definition in tests/CMakeLists.txt and read it in the test.

## Resolution

FIXED 2026-10-09. The CTest-registration lock now reads tests/CMakeLists.txt through the absolute compile definition VLLM_CPP_TESTS_CMAKELISTS baked in tests/CMakeLists.txt (the DOTS3_NOTE_CKPT_FIXTURE_DIR convention), so the probe no longer depends on the process CWD. Evidence: BEFORE, ctest in the VLLM_CPP_SANITIZE=address,undefined tree with the CI env: the 'P1 GPU binary is not registered as ordinary CTest' case failed REQUIRE( in ) at test_ops_paged_attn_sharedk_wmma_p1.cpp:115 (log /tmp/sanitize-runs/test_ops_paged_attn_sharedk_wmma_p1.log, rc=1); the same binary passed 32813/32813 assertions when run from the source root, confirming the CWD-relative __FILE__ mechanism. AFTER the fix, same tree rebuilt: the 9-test residue ctest selection -> 100% tests passed out of 9 (test_ops_paged_attn_sharedk_wmma_p1 Passed 0.95 sec). Normal Release build (/tmp/build-c): same selection -> 100% passed out of 9.

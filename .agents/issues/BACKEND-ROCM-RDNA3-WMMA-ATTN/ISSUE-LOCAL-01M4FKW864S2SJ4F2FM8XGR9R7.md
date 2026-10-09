ID: ISSUE-LOCAL-01M4FKW864S2SJ4F2FM8XGR9R7
Title: test_compiled_gemma red under ctest: fixture dir is __FILE__-relative and CWD-dependent
Row: BACKEND-ROCM-RDNA3-WMMA-ATTN
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

Mechanism (measured 2026-10-09, repro binaries from the sanitize-cpu lane): tests/vt/test_compiled_gemma.cpp:27 fails REQUIRE( f.good() ) — the fixture directory tests/vt/fixtures/compiled_gemma/ (manifest.json + cases.bin) EXISTS in the tree but is not opened. The test resolves it as Path(__FILE__).parent_path() / 'fixtures/compiled_gemma'; CMakeLists.txt:66 adds -ffile-prefix-map=<source dir>=. so __FILE__ is the source-root-RELATIVE './tests/vt/test_compiled_gemma.cpp', and the fixture path resolves against the process CWD. ctest runs the binary with CWD=<build>/tests, so the open fails; run from the source root the same binary passes 3/3 cases, 214/214 assertions. Fails identically in build-test-cpu and both sanitize-cpu lanes (.agents/specs/ci-main-ctest-residue.md) — a path-portability defect, not a sanitizer finding. Fix: bake the fixture dir as an absolute compile definition in tests/CMakeLists.txt (the DOTS3_NOTE_CKPT_FIXTURE_DIR convention) and read it in the test.

## Resolution

FIXED 2026-10-09. The frozen torch.compile fixtures are now read through the absolute compile definition COMPILED_GEMMA_FIXTURE_DIR baked in tests/CMakeLists.txt (the DOTS3_NOTE_CKPT_FIXTURE_DIR convention), so the manifest open no longer depends on the process CWD. Evidence: BEFORE, ctest in the VLLM_CPP_SANITIZE=address,undefined tree with the CI env: the 'compiled Gemma frozen primary expressions on CPU' case failed REQUIRE( f.good() ) at test_compiled_gemma.cpp:27 (log /tmp/sanitize-runs/test_compiled_gemma.log, rc=1); the same binary passed 3/3 cases, 214/214 assertions when run from the source root, confirming the CWD-relative __FILE__ mechanism. AFTER the fix, same tree rebuilt: the 9-test residue ctest selection -> 100% tests passed out of 9 (test_compiled_gemma Passed 0.70 sec). Normal Release build (/tmp/build-c): same selection -> 100% passed out of 9.

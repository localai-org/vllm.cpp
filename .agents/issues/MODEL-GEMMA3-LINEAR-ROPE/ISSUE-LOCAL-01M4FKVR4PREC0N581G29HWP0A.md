ID: ISSUE-LOCAL-01M4FKVR4PREC0N581G29HWP0A
Title: test_linear_scaling_rope red under ctest: fixture path is __FILE__-relative and -ffile-prefix-map makes it CWD-dependent
Row: MODEL-GEMMA3-LINEAR-ROPE
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

Mechanism (measured 2026-10-09, repro binaries from the sanitize-cpu lane): tests/vllm/model_executor/layers/rotary_embedding/test_linear_scaling_rope.cpp:51 fails REQUIRE( input.good() ) — the fixture tests/vllm/model_executor/layers/rotary_embedding/fixtures/linear_rope.json EXISTS in the tree but is not opened. The test resolves it as Path(__FILE__).parent_path() / 'fixtures/linear_rope.json'; CMakeLists.txt:66 adds -ffile-prefix-map=<source dir>=. so __FILE__ is the source-root-RELATIVE './tests/vllm/model_executor/layers/rotary_embedding/test_linear_scaling_rope.cpp', and the fixture path therefore resolves against the process CWD. ctest runs the binary with CWD=<build>/tests, so the open fails; run from the source root the same binary passes 29022/29022 assertions. Fails identically in build-test-cpu and both sanitize-cpu lanes (.agents/specs/ci-main-ctest-residue.md) — a path-portability defect, not a sanitizer finding. Fix: bake the fixture file path as an absolute compile definition in tests/CMakeLists.txt (the DOTS3_NOTE_CKPT_FIXTURE_DIR convention) and read it in the test.

## Resolution

FIXED 2026-10-09. The fixture path is now baked as the absolute compile definition LINEAR_SCALING_ROPE_FIXTURE_FILE in tests/CMakeLists.txt (the DOTS3_NOTE_CKPT_FIXTURE_DIR convention) and the test reads it, so the open no longer depends on the process CWD. Evidence: BEFORE, ctest in the VLLM_CPP_SANITIZE=address,undefined tree with the CI env: test_linear_scaling_rope failed REQUIRE( input.good() ) at test_linear_scaling_rope.cpp:51 (log /tmp/sanitize-runs/test_linear_scaling_rope.log, rc=1); the same binary passed 29022/29022 assertions when run from the source root, confirming the CWD-relative __FILE__ mechanism. AFTER the fix, same tree rebuilt: the 9-test residue ctest selection -> 100% tests passed out of 9 (test_linear_scaling_rope Passed 0.89 sec, 29022 assertions). Normal Release build (/tmp/build-c): same selection -> 100% passed out of 9.

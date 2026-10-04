ID: ISSUE-LOCAL-01M43M9EEXZTM1S9KX25VSWSN5
Title: definitive CI pass on PR 3393: vulkan platform-gate grep regex, MSVC C4244 in vulkan_ops.cpp, and 17 unclassified pr-size paths
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

Three persistent failures on PR 3393 head 483374410 (run 37202846607), diagnosed from complete job logs. (1) build-test-vulkan: the build and both doctests pass; the failure is the platform-case grep step. .github/workflows/ci.yml:1314 greps the pattern 'CHECK( prio[0] == "FLASH_ATTN" ) is correct', but in a POSIX basic regular expression '[0]' is a bracket expression matching the lone character '0', so the pattern can never match the literal 'prio[0]' that doctest prints ('./tests/vt/test_vulkan_backend.cpp:557: SUCCESS: CHECK( prio[0] == "FLASH_ATTN" ) is correct!'). The step exits 1 after two green greps and a green case summary. Reproduced locally with GNU grep 3.12: unescaped pattern exit 1, escaped 'prio\[0\]' exit 0. (2) windows-msvc-vulkan: 'src/vt/vulkan/vulkan_ops.cpp(1627,37): warning C4244: argument: conversion from const int64_t to uint32_t' (Go's fourth parameter, fed the int64_t t) and 'vulkan_ops.cpp(1163,33)/(1163,83): warning C4244: initializing: conversion from uint64_t to (const) uint32_t' (groups holds a 64-bit ceil-div), fatal under the lane's -Werror. windows-msvc-cpu compiles clean and fails later at the tracked #584 test_openai_api_server.exe 0xC0000409 fast-fail, out of scope here. (3) pr-size: pwsh now installs and both evidence lanes run, but check-pr-size.py's own whole-tree sweep test_every_tracked_and_current_change_path_is_classified fails on 17 tracked paths with no class: .agents/comparators/{README.md,tensorfold.md}, nine files under .agents/evidence/bench-qwen38-tensorfold-gap/20260929T180547Z/ plus its latest symlink, docs/bench-evidence/qwen4exp-layerfp-2999-20260923-full-log.txt (flat .txt), four repro_*.cpp and one ttledger-instrument.patch under tt-* per-run directories. Log: 'FAIL: test_every_tracked_and_current_change_path_is_classified ... First list contains 17 additional elements ... First extra element 0: .agents/comparators/README.md'.

## Resolution

-

ID: ISSUE-LOCAL-01M4FK4BJFZMCSG04WD0CAY5D2
Title: ASan alloc-dealloc-mismatch in test_deepseek_v4_image_processor: test replaces operator new with malloc but not the nothrow/aligned forms, so stable_sort's temporary buffer is new-allocated and free()-d
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: -

## Problem

The sanitize-cpu ASAN+UBSAN lane is red on test_deepseek_v4_image_processor (main 41705a7a2). Repro: /tmp/build-sanitize-main with ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 UBSAN_OPTIONS=print_stacktrace=1 VT_POOL_BYPASS=1; log /tmp/sanitize-runs/test_deepseek_v4_image_processor.log (rc=1). ASan: alloc-dealloc-mismatch (operator new vs free) at tests/vllm/multimodal/test_deepseek_v4_image_processor.cpp:53 in operator delete(void*). The 864-byte block was allocated by libstdc++'s std::get_temporary_buffer (used by std::stable_sort inside vllm::OrderedRegistry at static-init time, src/vllm/model_executor/models/model_registry.cpp:143) via the NOTHROW operator new (operator new(size_t, const std::nothrow_t&)), which the test does NOT replace, and freed by the test's replaced global operator delete(void*) (line 52-54), which calls std::free. Root cause: the test replaces the throwing global operator new with a std::malloc-based one (lines 36-45) so it can count allocations, and pairs it with a std::free-based operator delete -- but it leaves the nothrow (and aligned) allocation functions unreplaced, so any allocation that goes through them (libstdc++'s temporary-buffer path in std::stable_sort is the one that fires here) is new-allocated by the default runtime and free()-d by the test: an operator-new/operator-delete mismatch ASan refuses. Fix (test-side allocation pairing): replace the COMPLETE set of replaceable global allocation/deallocation functions -- throwing, nothrow and aligned new/new[] (all routing to malloc, all counting in the probe) and the matching delete/delete[] forms (all routing to free) -- so every allocation in the binary pairs malloc with free. No product change.

## Resolution

-

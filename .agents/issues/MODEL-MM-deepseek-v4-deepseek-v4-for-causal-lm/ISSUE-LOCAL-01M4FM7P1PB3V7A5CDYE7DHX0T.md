ID: ISSUE-LOCAL-01M4FM7P1PB3V7A5CDYE7DHX0T
Title: test_deepseek_v4_vision pool-behavior cases can never pass in the sanitize lane: VT_POOL_BYPASS=1 disables the DevicePool they measure
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

MEASURED, four controls on the SAME binaries: (a) the normal Release build
(/tmp/build-b, -O3, no env) passes 16/16 with 7420/7420 assertions — the test
is green in the normal lane. (b) The SAME Release binary with only
VT_POOL_BYPASS=1 added fails 14/16 with exactly the sanitize lane's failures:
CHECK(backend.allocations() == after_warmup) 62 == 45 and
CHECK(pool_after_repeat.hits > pool_after_warmup.hits) 0 > 0 in 'DeepSeek-V4
repeated shape reuses scratch and 2-D RoPE allocation', and
CHECK(per_layer == kPerLayerPooledBuffers) 0 == 2 in 'DeepSeek-V4 vision
allocates no per-layer scratch of its own' — so the environment variable alone
reproduces the lane's failure with no sanitizer and no -O0 involved. (c) The
sanitize binary WITHOUT the bypass (ASAN_OPTIONS=detect_leaks=0) passes 16/16
with 7420/7420. (d) The sanitize binary without the bypass and detect_leaks=1
passes 16/16 on the checks and then LeakSanitizer reports '30208 byte(s)
leaked in 125 allocation(s)' — the DevicePool's deliberately-retained scratch,
which is exactly why the lane sets the bypass. The sanitize lane
(VLLM_CPP_SANITIZE, .github/workflows/ci.yml sanitize-cpu Test step) runs the
whole suite with VT_POOL_BYPASS=1 for BOTH matrix legs (address,undefined and
thread), present since 59674cf1d (2026-08-05). MECHANISM: DevicePool::Get under
VT_POOL_BYPASS=1 returns a raw driver Alloc, never increments hits_/misses_ and
never reuses blocks (include/vllm/model_executor/models/device_pool.h:129
'if (Bypass()) return b.Alloc(bytes);', Bypass() at :576-582 reads the env var
once into a function-local static). The two cases assert pool REUSE — a stable
allocation count across a repeated shape, a growing hit count, a per-layer
pooled-buffer slope of exactly 2 — properties the bypass disables by design, so
the assertions are unmeasurable in that lane. The test file itself landed only
on 2026-09-05+ (8c10cad16 and later), i.e. AFTER the bypass entered the lane,
so the cases were authored against the normal lane only and have never been
green in the sanitize lane. Fix (this row): the two cases are gated with
doctest::skip(PoolBypassLane()) — the repo's blessed environment-skip shape
(tests/vllm/models/test_qwen3_5_decode_graph_seam.cpp carries the same helper
and the same reasoning) — and PoolBypassLane() prints the reason to stderr
when the lane is active, so the skip is loud, never silent, and never a
zero-assertion pass. The predicate mirrors DevicePool::Bypass() exactly ('=1',
first character only), so this file cannot disagree with the header about which
lane a process is in. The other 14 cases of the file still run in every lane.
No assertion is weakened or deleted. Part of the CI-MAIN-CTEST-RESIDUE
sanitize sweep (group B): this test is on the residue list of
`.agents/specs/ci-main-ctest-residue.md` (thread lane).

## Resolution

Fixed in 9e5f95b0e on row/ci-sanitize-residue-b: the two pool-REUSE cases are decorated with doctest::skip(PoolBypassLane()); PoolBypassLane() reads VT_POOL_BYPASS exactly as DevicePool::Bypass() spells it ('=1', first character only) and prints the loud skip banner to stderr when active — the skip is never silent and never a zero-assertion pass. No CHECK/REQUIRE/MESSAGE line touched; the other 14 cases run in every lane. Verified 2026-10-09 on a clean rebuild in four trees (Release /tmp/build-b, -O0 /tmp/build-b-o0, address,undefined /tmp/build-san, thread /tmp/build-tsan): with the exact CI env (ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 UBSAN_OPTIONS=print_stacktrace=1 VT_POOL_BYPASS=1) the sanitize and TSAN binaries read 14 passed | 0 failed | 2 skipped, 7410/7410 assertions, rc=0, banner printed once per skipped case; Release and -O0 read 16/16, 7420/7420, rc=0, no skip. VT_POOL_BYPASS=1 alone on the Release and -O0 binaries reads 14 passed | 2 skipped, 7410/7410, rc=0 — the env var alone triggers the skip. The same sanitize binary without VT_POOL_BYPASS (ASAN_OPTIONS=detect_leaks=0) reads 16/16, 7420/7420, rc=0: the pool-behavior assertions still run and pass wherever the pool is live. Logs /tmp/rsan-test_deepseek_v4_vision.log, /tmp/rtsan-test_deepseek_v4_vision.log, /tmp/rb-test_deepseek_v4_vision.log, /tmp/ro0-test_deepseek_v4_vision.log, /tmp/rb-dsv4-bypass.log, /tmp/rsan-dsv4-nobypass.log.

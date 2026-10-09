ID: ISSUE-LOCAL-01M4FKWVM5HB565RF1YK6JPQV2
Title: test_bench_kv_cache_dtype red in every CPU lane: vllm-bench lost --kv-cache-dtype in the #293 reapply
Row: KV-FP8
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

Mechanism (measured 2026-10-09, repro binaries from the sanitize-cpu lane): tests/examples/test_bench_kv_cache_dtype.cpp execs the built vllm-bench binary; 8 of 9 cases fail with 'vllm-bench: unknown argument '--kv-cache-dtype'' (exit 2) — the flag does not exist in examples/bench/main.cpp at origin/main. 89bfd79b0 (KV-FP8 W7, #2619) added the flag, the EngineParams pass-through in both bench_core.h engine arms, and the 'KV cache dtype (requested)/(resolved storage):' report lines; the later PR #293 reapply (cd4b4c6a1 'Reapply ... onto current upstream') rewrote examples/bench/main.cpp and bench_core.h from a pre-W7 tree, silently dropping all of it while keeping the test. git pickaxe confirms no commit since removed the flag strings — the squash-merge of the stale reapply is the clobber. The test's contract is unchanged and correct: flag parses, reaches EngineParams on both arms, report names requested+resolved dtype read back out of the sized KV cache, unknown names refused by ParseCacheDType by name. Fails identically in build-test-cpu and both sanitize-cpu lanes (.agents/specs/ci-main-ctest-residue.md) — a product regression in the benchmark harness, not a sanitizer finding. Fix: re-port 89bfd79b0 onto the current main.cpp/bench_core.h (BenchConfig::kv_cache_dtype, ParseArgs arm, Usage text, params.kv_cache_dtype in both engine arms, ResolvedKvCacheDTypeName read back from LoadedEngine::kv_cache_config(), the two report lines, the stderr header field).

## Resolution

FIXED 2026-10-09. Re-ported 89bfd79b0 (KV-FP8 W7) onto the current examples/bench: the --kv-cache-dtype ParseArgs arm, BenchConfig::kv_cache_dtype, the EngineParams pass-through in BOTH engine arms of RunBench (synthetic and --model), ResolvedKvCacheDTypeName reading the sized KV cache config back through LoadedEngine::kv_cache_config(), BenchResult::resolved_kv_cache_dtype, the 'KV cache dtype (requested)/(resolved storage):' report lines, the Usage text and the stderr header field. The flag had been dropped when the PR #293 reapply (cd4b4c6a1) rewrote examples/bench from a pre-W7 tree. Evidence: BEFORE, ctest in the VLLM_CPP_SANITIZE=address,undefined tree with the CI env: 8 of 9 test_bench_kv_cache_dtype cases failed with "vllm-bench: unknown argument '--kv-cache-dtype'" exit 2 (log /tmp/sanitize-runs/test_bench_kv_cache_dtype.log, rc=1). AFTER the fix, same tree rebuilt: the 9-test residue ctest selection -> 100% tests passed out of 9 (test_bench_kv_cache_dtype Passed 8.27 sec, 9/9 cases incl. the fp8 resolved-storage line, the by-name nvfp4 refusal, and the --model checkpoint-decided polarity pair). Normal Release build (/tmp/build-c): same selection -> 100% passed out of 9 (Passed 1.35 sec).

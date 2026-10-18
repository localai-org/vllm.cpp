ID: ISSUE-LOCAL-01M4FKWV8063MP0P64ASF1RBXP
Title: test_bench synthetic-engine cases red in every CPU lane: unified KV block size is not a multiple of 16
Row: SERVE-CLI-BENCH
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

Mechanism (measured 2026-10-09, repro binaries from the sanitize-cpu lane, with and without ASAN_OPTIONS/UBSAN_OPTIONS/VT_POOL_BYPASS — identical): 6 of 15 cases in tests/examples/test_bench.cpp THROW 'No valid attention backend for device type 0 from {CPU_ATTN: [block_size not supported], FLASH_ATTN: [block_size not supported]}' at RunBench. Root cause: examples/bench/bench_core.h builds the synthetic engine with params.block_size = max_prompt + output_len + 4 (the 'unified block' for the hybrid GDN+FA KV groups), an arbitrary value (e.g. 16+16+4=36, 9+4+4=17). Since 9ecaf1bb3 registered CPU_ATTN with get_supported_kernel_block_sizes {16} (include/vllm/v1/attention/backends/cpu_attn.h:112; FLASH_ATTN declares the same), AttentionBackend::supports_block_size rejects any block size that is not a multiple of 16, and SelectAttentionBackendName throws out of engine construction. The harness predates the backend validation (57cd5afe2 landed before 9ecaf1bb3) and was never aligned. The same workload succeeds when seq_budget happens to be a multiple of 16 (e.g. input_len=8/output_len=4 with an 8-token prompt). Fails identically in build-test-cpu and both sanitize-cpu lanes (.agents/specs/ci-main-ctest-residue.md) — a harness/engine-contract defect, not a sanitizer finding. Fix: align the unified block size UP to a multiple of 16 in bench_core.h (the kernel constraint every registered attention backend declares), keeping max_model_len = seq_budget and the one-block-per-sequence intent.

## Resolution

FIXED 2026-10-09. The synthetic engine's unified KV block size is now aligned UP to a multiple of 16 ((seq_budget + 15) / 16 * 16) in examples/bench/bench_core.h, keeping max_model_len = seq_budget and the one-block-per-sequence intent. Evidence: BEFORE, ctest in the VLLM_CPP_SANITIZE=address,undefined tree with the CI env (and identically without it, and in Release): 6 of 15 test_bench cases threw 'No valid attention backend for device type 0 from {CPU_ATTN: [block_size not supported], FLASH_ATTN: [block_size not supported]}' at RunBench (log /tmp/sanitize-runs/test_bench.log, rc=1). AFTER the fix, same tree rebuilt: the 9-test residue ctest selection -> 100% tests passed out of 9 (test_bench Passed 37.88 sec, 15/15 cases). Normal Release build (/tmp/build-c): same selection -> 100% passed out of 9 (test_bench Passed 31.85 sec).

ID: ISSUE-GH-1385
Title: **CI builds 4 of 552 test targets on aarch64, and every fleet GPU box and two release bundles are aarch64.** Re-derived at `e2a9e035d`: `tests/CMakeLists.txt` defines **552** `vllm_cpp_add_test` targets plus 32 direct `add_test(NAME ...)` registrations, and the x86-64 lane ran **584** CTest entries with 3 skipped on run 32465485947. `build-test-cpu-arm64` (`.github/workflows/ci.yml:1096`) builds four ISA and kernel-tier targets and runs no `ctest` at all, so 0.72 % of the suite executes on the architecture the project measures, gates and ships on. Its own flag block (`:1110`) additionally sets `VLLM_CPP_BUILD_EXAMPLES=OFF` and `VLLM_CPP_SERVER=OFF`, which puts `examples/tokenize` and the `/v1/completions` parse segment out of reach of any target-list change. Decided in [ci-aarch64-coverage.md](../../specs/ci-aarch64-coverage.md): a curated subset is REJECTED because a stated sensitivity principle (weak memory model, `char` signedness, floating-point contraction, hash order) selects **291 of 552** targets, so half a suite costs most of a full build; a self-hosted fleet runner is REJECTED because it takes a leased box outside `rc`. Recommended: one new `build-test-cpu-arm64-full` job that builds everything and runs `ctest` serially, landing `schedule`+`workflow_dispatch` only with `continue-on-error`, then promoted per-PR once measured. Measured cost: **$0** (`timing` reports `total_ms: 0`; the repository is public), **+45-55 job-minutes** against a median of 357 per scheduled run, and **zero** added wall-clock while it finishes inside `cuda-fat-build`'s measured 123.0-minute median finish. It makes the hermetic tokenizer parity goldens execute on aarch64 for the first time, which is the `## Owed` item [prompt-token-divergence.md](../../specs/prompt-token-divergence.md) names, but the committed corpus carries only **30** combining marks over 99 lines against the 74-150 per prompt that produced the anomaly, so it is a necessary and not a sufficient probe
Row: GATE-CI-AARCH64-COVERAGE
State: UNKNOWN
Kind: bug
GitHub: 1385
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:576`

### Frozen archive evidence

> | [#1385](https://github.com/mudler/vllm.cpp/issues/1385) | `GATE-CI-AARCH64-COVERAGE` | **CI builds 4 of 552 test targets on aarch64, and every fleet GPU box and two release bundles are aarch64.** Re-derived at `e2a9e035d`: `tests/CMakeLists.txt` defines **552** `vllm_cpp_add_test` targets plus 32 direct `add_test(NAME ...)` registrations, and the x86-64 lane ran **584** CTest entries with 3 skipped on run 32465485947. `build-test-cpu-arm64` (`.github/workflows/ci.yml:1096`) builds four ISA and kernel-tier targets and runs no `ctest` at all, so 0.72 % of the suite executes on the architecture the project measures, gates and ships on. Its own flag block (`:1110`) additionally sets `VLLM_CPP_BUILD_EXAMPLES=OFF` and `VLLM_CPP_SERVER=OFF`, which puts `examples/tokenize` and the `/v1/completions` parse segment out of reach of any target-list change. Decided in [ci-aarch64-coverage.md](../../specs/ci-aarch64-coverage.md): a curated subset is REJECTED because a stated sensitivity principle (weak memory model, `char` signedness, floating-point contraction, hash order) selects **291 of 552** targets, so half a suite costs most of a full build; a self-hosted fleet runner is REJECTED because it takes a leased box outside `rc`. Recommended: one new `build-test-cpu-arm64-full` job that builds everything and runs `ctest` serially, landing `schedule`+`workflow_dispatch` only with `continue-on-error`, then promoted per-PR once measured. Measured cost: **$0** (`timing` reports `total_ms: 0`; the repository is public), **+45-55 job-minutes** against a median of 357 per scheduled run, and **zero** added wall-clock while it finishes inside `cuda-fat-build`'s measured 123.0-minute median finish. It makes the hermetic tokenizer parity goldens execute on aarch64 for the first time, which is the `## Owed` item [prompt-token-divergence.md](../../specs/prompt-token-divergence.md) names, but the committed corpus carries only **30** combining marks over 99 lines against the 74-150 per prompt that produced the anomaly, so it is a necessary and not a sufficient probe | bug |

## Resolution

-

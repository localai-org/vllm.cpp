ID: ISSUE-GH-2108
Title: **No CI runner has a GPU, so every test that appears to gate a device path is either skipped or silently running on the CPU backend — and both shapes report green.** `.github/workflows/` has no GPU job; `cuda-fat-build` compiles ten architectures and runs nothing. Found while gating SPEC-DFLASH2 W12 D1 ([#2087](https://github.com/mudler/vllm.cpp/issues/2087)), where it bit twice. `tests/vllm/v1/spec_decode/test_dflash_propose.cpp:50` builds its queue with `vt::Queue Cpu()`, so its 10/10 on a GB10 box was a CPU result taken beside an idle GPU and was briefly read as device coverage. And `test_dflash2_runner_reach` is 7-RED under CUDA: measured on `dgx:gpu0` in `vllmcpp-build:gb10` with `--gpus all`, pre-D1 `build18` gives 8 cases / 1 passed / 7 failed / 75 assertions / 18 failed and D1 `build19` gives 9 / 2 / 7 / 83 / 18 — identical failure counts, so PRE-EXISTING and not D1's, and D1's own added case passes. Representative: `:398` `REQUIRE(st_off.block_kernel_calls > 0)` fails as `0 > 0`, a CPU-path route counter that never increments when CUDA is selected; also `:278` and `:345` `REQUIRE_FALSE(blocks.empty())` and `:344`/`:390`/`:391` `CHECK(threw.empty())`. Whether the repair is backend-awareness or a real CUDA-path defect is NOT settled: `:390`/`:391` are `threw` assertions, so something is throwing on the device lane, and a case made to pass by widening its scope would be the failure AGENTS.md names under "Changing the rules or a checker". The consequence for SPEC-DFLASH2 is that `ForwardWithCtxKVDev` at `P > 1` with real device tensors — the path D1 changed — is gated by nothing but an end-to-end throughput run, which an acceptance-only defect is invisible to. Listed under `## Owed` O6 in [`.agents/specs/dflash2-batch-propose.md`](../../specs/dflash2-batch-propose.md).
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: verification
GitHub: 2108
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:814`

### Frozen archive evidence

> | [#2108](https://github.com/mudler/vllm.cpp/issues/2108) | `SPEC-DFLASH2` | **No CI runner has a GPU, so every test that appears to gate a device path is either skipped or silently running on the CPU backend — and both shapes report green.** `.github/workflows/` has no GPU job; `cuda-fat-build` compiles ten architectures and runs nothing. Found while gating SPEC-DFLASH2 W12 D1 ([#2087](https://github.com/mudler/vllm.cpp/issues/2087)), where it bit twice. `tests/vllm/v1/spec_decode/test_dflash_propose.cpp:50` builds its queue with `vt::Queue Cpu()`, so its 10/10 on a GB10 box was a CPU result taken beside an idle GPU and was briefly read as device coverage. And `test_dflash2_runner_reach` is 7-RED under CUDA: measured on `dgx:gpu0` in `vllmcpp-build:gb10` with `--gpus all`, pre-D1 `build18` gives 8 cases / 1 passed / 7 failed / 75 assertions / 18 failed and D1 `build19` gives 9 / 2 / 7 / 83 / 18 — identical failure counts, so PRE-EXISTING and not D1's, and D1's own added case passes. Representative: `:398` `REQUIRE(st_off.block_kernel_calls > 0)` fails as `0 > 0`, a CPU-path route counter that never increments when CUDA is selected; also `:278` and `:345` `REQUIRE_FALSE(blocks.empty())` and `:344`/`:390`/`:391` `CHECK(threw.empty())`. Whether the repair is backend-awareness or a real CUDA-path defect is NOT settled: `:390`/`:391` are `threw` assertions, so something is throwing on the device lane, and a case made to pass by widening its scope would be the failure AGENTS.md names under "Changing the rules or a checker". The consequence for SPEC-DFLASH2 is that `ForwardWithCtxKVDev` at `P > 1` with real device tensors — the path D1 changed — is gated by nothing but an end-to-end throughput run, which an acceptance-only defect is invisible to. Listed under `## Owed` O6 in [`.agents/specs/dflash2-batch-propose.md`](../../specs/dflash2-batch-propose.md). | verification |

## Resolution

-

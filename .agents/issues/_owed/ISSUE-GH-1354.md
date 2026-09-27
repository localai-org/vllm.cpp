ID: ISSUE-GH-1354
Title: Clock pinning is UNAVAILABLE inside an `rc` lease. `nvidia-smi -lgc 2190` returns `LGC_RC=4`, "The current user does not have permission to change clocks for GPU 0000000F:01:00.0", in three separate `rc run` jobs on `dgx:gpu0` on 2026-08-19, each running as **root** in the worker pod. `.agents/benchmarking.md` instructs "Pin the clocks before measuring, under the lock" and ships `sudo nvidia-smi -lgc 2100` as the recipe, and **every clock-pinned figure in this repository was taken over the host + `ssh` + `flock` path** that AGENTS.md now forbids for a fleet device — so the migration to leases silently removed clock pinning and no record said so. Same class as [#1265](https://github.com/mudler/vllm.cpp/issues/1265): a capability the records assume, which the current access path does not provide. Measured cost: nine timed windows across two arms recorded within-run SM-clock spreads of 12.92% to 26.36% against the 5% ceiling, `SwThermalSlowdown` active in every one and `HwSlowdown+HwThermal` in one, so `gpu_clock_state compare` returned `PAIRING_VERDICT=DISCARD` on all three Qwen3.8-27B c1 pairings **even though the cross-arm rule passed perfectly** (same boot, both arms 2489 MHz median, 0.0% offset). The cell therefore has two clean complete absolutes and no ratio. NOT fixed in flow: the fix is either an `rc` worker capability this row has no authority over, or a demonstrated settle-and-hold procedure, or a ratified different clock rule for lease-measured pairs — each its own spec, and none of them a widening of the assertion to turn a red green. Records updated meanwhile in `.agents/environment.md`, `.agents/benchmarking.md` and `.agents/benchmark-record.md`. Owed under `## Owed` in [bench-qwen38-27b-four-way.md](../../specs/bench-qwen38-27b-four-way.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1354
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:447`

### Frozen archive evidence

> | [#1354](https://github.com/mudler/vllm.cpp/issues/1354) | — | Clock pinning is UNAVAILABLE inside an `rc` lease. `nvidia-smi -lgc 2190` returns `LGC_RC=4`, "The current user does not have permission to change clocks for GPU 0000000F:01:00.0", in three separate `rc run` jobs on `dgx:gpu0` on 2026-08-19, each running as **root** in the worker pod. `.agents/benchmarking.md` instructs "Pin the clocks before measuring, under the lock" and ships `sudo nvidia-smi -lgc 2100` as the recipe, and **every clock-pinned figure in this repository was taken over the host + `ssh` + `flock` path** that AGENTS.md now forbids for a fleet device — so the migration to leases silently removed clock pinning and no record said so. Same class as [#1265](https://github.com/mudler/vllm.cpp/issues/1265): a capability the records assume, which the current access path does not provide. Measured cost: nine timed windows across two arms recorded within-run SM-clock spreads of 12.92% to 26.36% against the 5% ceiling, `SwThermalSlowdown` active in every one and `HwSlowdown+HwThermal` in one, so `gpu_clock_state compare` returned `PAIRING_VERDICT=DISCARD` on all three Qwen3.8-27B c1 pairings **even though the cross-arm rule passed perfectly** (same boot, both arms 2489 MHz median, 0.0% offset). The cell therefore has two clean complete absolutes and no ratio. NOT fixed in flow: the fix is either an `rc` worker capability this row has no authority over, or a demonstrated settle-and-hold procedure, or a ratified different clock rule for lease-measured pairs — each its own spec, and none of them a widening of the assertion to turn a red green. Records updated meanwhile in `.agents/environment.md`, `.agents/benchmarking.md` and `.agents/benchmark-record.md`. Owed under `## Owed` in [bench-qwen38-27b-four-way.md](../../specs/bench-qwen38-27b-four-way.md) | bug |

## Resolution

-

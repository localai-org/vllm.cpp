ID: ISSUE-GH-1709
Title: **`dgx:gpu0` holds 110.41 GiB that belongs to no `/proc/meminfo` category, and `rc` keeps handing out leases against it.** Measured read-only from inside a lease (`rc` job `ab12aac1-b862-4ac6-8292-9f2c641e6a8d`, 2026-08-22T16:55:46Z), because the fleet rule forbids `ssh`: `MemTotal` 119.63 GiB, `MemFree` 5.23, `AnonPages` 0.93, `Cached` 0.91, `Buffers` 0.05, `Shmem` 0.04, `Slab` 1.10, `VmallocUsed` 1.00 -- **accounted 9.22 GiB, UNACCOUNTED 110.41 GiB, 92.3% of the box.** Every ordinary owner is excluded by measurement and not by argument: the sum of `VmRSS` over every visible `/proc/*/status` is 0.0 GiB; `/dev/shm` is a 64 M mount containing nothing; `nvidia-smi` reads 0%, 11 W, `No running processes found`; the cgroup reports `memory.max=max` and `memory.current` 113 MiB; and the value is 5.0 GiB at 15:49Z, 5.1 GiB flat through 16:39Z and 4.98 GiB at 16:55Z across four leases by three submitters, on a box `up 2:33` at load 0.25. The leading explanation is a driver-held unified-memory allocation that outlived its process, which fits every observation including `Memory-Usage: Not Supported` being the one meter that would have named it -- recorded as a HYPOTHESIS, since confirming it needs host access. COST: three leases on #1612 -- `5fb9399f` lost its worker to an OOM during a build started against 5 GiB, `2ccd1acf` waited its full 1200 s at a flat 5.0 GiB and refused with exit 39, `ab12aac1` measured the box. FIXED IN FLOW, the half this row owns: `scripts/ltx25-dit-attn-flash-pixel-ab.sh` gains a `MemAvailable` start gate that waits, logs EVERY poll and refuses by name -- and logging every poll is the only reason the condition can be called persistent rather than busy. NOT FIXED IN FLOW, and named rather than folded in: a device-readiness condition in the controller, which would have parked every one of these jobs instead of spending them, and a line in the DGX profile saying a granted lease does not imply a reclaimed box. Owner: row `LTX25-DIT-ATTN-FLASH`, under `## Owed` in [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md)
Row: LTX25-DIT-ATTN-FLASH
State: UNKNOWN
Kind: bug
GitHub: 1709
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:628`

### Frozen archive evidence

> | [#1709](https://github.com/mudler/vllm.cpp/issues/1709) | `LTX25-DIT-ATTN-FLASH` | **`dgx:gpu0` holds 110.41 GiB that belongs to no `/proc/meminfo` category, and `rc` keeps handing out leases against it.** Measured read-only from inside a lease (`rc` job `ab12aac1-b862-4ac6-8292-9f2c641e6a8d`, 2026-08-22T16:55:46Z), because the fleet rule forbids `ssh`: `MemTotal` 119.63 GiB, `MemFree` 5.23, `AnonPages` 0.93, `Cached` 0.91, `Buffers` 0.05, `Shmem` 0.04, `Slab` 1.10, `VmallocUsed` 1.00 -- **accounted 9.22 GiB, UNACCOUNTED 110.41 GiB, 92.3% of the box.** Every ordinary owner is excluded by measurement and not by argument: the sum of `VmRSS` over every visible `/proc/*/status` is 0.0 GiB; `/dev/shm` is a 64 M mount containing nothing; `nvidia-smi` reads 0%, 11 W, `No running processes found`; the cgroup reports `memory.max=max` and `memory.current` 113 MiB; and the value is 5.0 GiB at 15:49Z, 5.1 GiB flat through 16:39Z and 4.98 GiB at 16:55Z across four leases by three submitters, on a box `up 2:33` at load 0.25. The leading explanation is a driver-held unified-memory allocation that outlived its process, which fits every observation including `Memory-Usage: Not Supported` being the one meter that would have named it -- recorded as a HYPOTHESIS, since confirming it needs host access. COST: three leases on #1612 -- `5fb9399f` lost its worker to an OOM during a build started against 5 GiB, `2ccd1acf` waited its full 1200 s at a flat 5.0 GiB and refused with exit 39, `ab12aac1` measured the box. FIXED IN FLOW, the half this row owns: `scripts/ltx25-dit-attn-flash-pixel-ab.sh` gains a `MemAvailable` start gate that waits, logs EVERY poll and refuses by name -- and logging every poll is the only reason the condition can be called persistent rather than busy. NOT FIXED IN FLOW, and named rather than folded in: a device-readiness condition in the controller, which would have parked every one of these jobs instead of spending them, and a line in the DGX profile saying a granted lease does not imply a reclaimed box. Owner: row `LTX25-DIT-ATTN-FLASH`, under `## Owed` in [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md) | bug |

## Resolution

-

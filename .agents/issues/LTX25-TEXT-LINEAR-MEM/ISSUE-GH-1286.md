ID: ISSUE-GH-1286
Title: #1252's threaded caption projection was reported to raise peak host memory ~79 -> 105.85 GiB and abort the full-model LTX-2.5 render on GB10. **The +26 GiB is the box's STARTING OCCUPANCY, not the change.** Both runs are retained under `/mnt/nas_share/rc/ltx25-fullmodel/out/`, and `runguard.py:236-237,260` fixes the compared column `used_gib` as the SYSTEM-WIDE `MemTotal - MemAvailable`: the pre-#1252 run began at `used = 4.741 GiB` on an idle box, the #1252 run began at `used = 31.553 GiB`, a difference of **26.812 GiB** against a claimed regression of **26.647 GiB**. Each run's own demand — peak minus its own t=0 — is **74.465 GiB before and 74.300 GiB after**, and both peaks were sampled inside a ~1900% CPU stretch, so they are the same phase class rather than two different ones. Two further columns agree independently: system `AnonPages` delta 38.012 vs 38.217 GiB, and child `VmRSS` at each peak sample 41.952 vs 42.090 GiB. Confirmed locally by an A/B of the two `Linear` arms at the shipped `1024 x 188160 x 4096` geometry swept over threadpool width; the seam's only per-call allocation is `cpu_ops.cpp:125`'s `static thread_local` widened-activation tile, `16 x K x 4` per worker = 12.04 MB at `K = 188160`. Owner `LTX25-TEXT-LINEAR-MEM`, spec [`ltx25-text-linear-mem.md`](../../specs/ltx25-text-linear-mem.md)
Row: LTX25-TEXT-LINEAR-MEM
State: UNKNOWN
Kind: verification
GitHub: 1286
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:428`

### Frozen archive evidence

> | [#1286](https://github.com/mudler/vllm.cpp/issues/1286) | `LTX25-TEXT-LINEAR-MEM` | #1252's threaded caption projection was reported to raise peak host memory ~79 -> 105.85 GiB and abort the full-model LTX-2.5 render on GB10. **The +26 GiB is the box's STARTING OCCUPANCY, not the change.** Both runs are retained under `/mnt/nas_share/rc/ltx25-fullmodel/out/`, and `runguard.py:236-237,260` fixes the compared column `used_gib` as the SYSTEM-WIDE `MemTotal - MemAvailable`: the pre-#1252 run began at `used = 4.741 GiB` on an idle box, the #1252 run began at `used = 31.553 GiB`, a difference of **26.812 GiB** against a claimed regression of **26.647 GiB**. Each run's own demand — peak minus its own t=0 — is **74.465 GiB before and 74.300 GiB after**, and both peaks were sampled inside a ~1900% CPU stretch, so they are the same phase class rather than two different ones. Two further columns agree independently: system `AnonPages` delta 38.012 vs 38.217 GiB, and child `VmRSS` at each peak sample 41.952 vs 42.090 GiB. Confirmed locally by an A/B of the two `Linear` arms at the shipped `1024 x 188160 x 4096` geometry swept over threadpool width; the seam's only per-call allocation is `cpu_ops.cpp:125`'s `static thread_local` widened-activation tile, `16 x K x 4` per worker = 12.04 MB at `K = 188160`. Owner `LTX25-TEXT-LINEAR-MEM`, spec [`ltx25-text-linear-mem.md`](../../specs/ltx25-text-linear-mem.md) | verification |

## Resolution

-

# Kolibri-1 CPU: in-process NUMA interleaving for the weight arenas
# (MODEL-TEXT-kolibri-1)

Date: 2026-10-09. Host: aarch64, 128 cores, 255 GB RAM, 4 NUMA nodes (the same
box as the 2026-10-08 records). Tree: `row/kolibri-numa-interleave` at
`1bdb9c1cc` (base `origin/main` `59425f2e0`), worktree `/tmp/vllm-kolibri-numa`,
build `/tmp/build-kolibri-numa` (Release, CPU-only, same configure flags as the
2026-10-08 records). Checkpoint `/mnt/models/Aleph-Alpha/Kolibri-1` (74 GB fp8,
32 shards). Issue: ISSUE-LOCAL-01M4GQ82X99D6JNBWDPX7ATJPY.

## What the lever is

The 2026-10-08 next-lever record measured an external `numactl --interleave=all`
wrapper at +47% decode tok/s (t32: 4.69 vs 3.2) and falsified placement lotteries
at lower thread counts. `numactl` cannot ship: production entry points do not
run under a wrapper. The landed change applies `MPOL_INTERLEAVE` over the full
weight-arena allocation and the forward call **in-process**
(`kolibri1_numa::PolicyGuard`, wired in `kolibri1_weights.cpp:225`), saving and
restoring the caller's mempolicy. `VT_KOLIBRI1_NUMA_INTERLEAVE=0` disables the
guard entirely (the knob exists so a broken host can be ruled out without a
rebuild).

## A/B result (same binary, same workload, mutex-serialised, idle host)

`test_kolibri1_decode_bench`, "incremental production shape: 128 in, 64 greedy
out", `VLLM_CPP_CPU_THREADS=32`, `VT_KOLIBRI1_DEQUANT_CACHE_MB=16384`,
`VT_KOLIBRI1_PROFILE=1`. Three interleaved pairs, one mutex hold:

| leg | decode s | decode tok/s | prefill s |
|---|---|---|---|
| OFF pair 1 | 14.96 | 4.212 | 8.38 |
| OFF pair 2 | 15.03 | 4.190 | 7.71 |
| OFF pair 3 | 15.09 | 4.174 | 7.79 |
| ON pair 1 | 12.71 | 4.958 | 6.86 |
| ON pair 2 | 12.68 | 4.968 | 6.96 |
| ON pair 3 | 12.62 | 4.992 | 6.71 |

- Decode: 4.192 ± 0.016 → 4.973 ± 0.014 tok/s, **+18.6%**.
- Prefill: 7.96 → 6.84 s mean, **−14%**.
- The ON leg beats the external `numactl` figure (4.69): the in-process guard
  also interleaves the forward-pass scratch that a wrapper's early policy covers
  inconsistently, and it restores the caller's policy instead of inheriting the
  wrapper's for the whole process.

Correctness held on all six legs: prefill anchor **101807**, greedy decode chain
**101807,109726** × 32 (md5 of the recorded golden unchanged). The lever moves
placement only; the bit-exact order-preserving contract is untouched.

## Sensitivity legs (recorded for the budget conversation)

At the cache-OFF default and 8 threads the NUMA delta collapses to ~2%
(1.258 → 1.287 tok/s): with the dequant cache off, the decode path is
memory-allocation-dominated, and interleave cannot repair an allocation-bound
loop. The lever and the 16 GiB cache budget compose; neither replaces the other.

## Claimed guarantees (for the fresh reviewer)

1. Inside `PolicyGuard`, the calling task's mempolicy is `MPOL_INTERLEAVE` over
   every allowed node; on destruction the saved policy is restored exactly.
2. Weight-arena pages touched during decode become resident on MORE THAN ONE
   NUMA node (the `numactl` lever's mechanism, now reached without a wrapper).
3. `VT_KOLIBRI1_NUMA_INTERLEAVE=0` disables the guard entirely.
4. The production decode path passes through the guard (deleting the
   `kolibri1_weights.cpp:225` construction changes the measured number and the
   page-placement test).

Mutation review 2026-10-09 (fresh reviewer): guarantees 1-3 red-first proven
(mutations: dropped policy restore, single-node mask, inverted env check —
each went red, each restored byte-for-byte). Guarantee 4 is HALF-PINNED: the
dequant-cache guard site goes red when deleted, the `kolibri1_weights.cpp:225`
site does not (no focused test reaches the loader); it is covered only by this
A/B, and the gap is recorded in the spec `## Owed`.

## Owed

- Re-baseline `docs/BENCHMARKS.md` figures with this lever (queued; figures
  moving this change + the merged NEON attention).
- The cache-budget default decision (16 vs 32 GiB) remains with the developer.

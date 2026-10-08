# Kolibri-1 CPU: the next-lever attribution — what remains hot, what is
# falsified, and the measured budget-policy proposal (MODEL-TEXT-kolibri-1)

Date: 2026-10-08. Host: aarch64, 128 cores, 255 GB RAM, Linux (the same box as
the 2026-10-07/08 records). Tree: pristine `origin/main` **bf7b654ae** (branch
`row/kolibri-perf-next`, worktree `/tmp/vllm-kolibri-perf`, build
`/tmp/build-kolibri-perf-next`, Release, `-DVLLM_CPP_CUDA=OFF
-DVLLM_CPP_TENSTORRENT=OFF -DVLLM_CPP_SERVER=OFF -DVLLM_CPP_BUILD_EXAMPLES=OFF`).
No source change was needed for the attribution; the falsification experiment
lives outside the tree (scratch micro-benchmark, `/tmp/kolibri-micro/`).
Checkpoint `/mnt/models/Aleph-Alpha/Kolibri-1` (74 GB fp8, 32 shards).
Issue: ISSUE-LOCAL-01M4EEX40G7NH571CR5GQY4CA1.

## Contention note (recorded honestly)

Two other agents share this host (a Tenstorrent repair implementer and a
serving implementer, the latter running `test_kolibri1_w3` and a 128-thread
qwen3 e2e test during this unit). Every number below is tagged with the
window it ran in. Contended legs are marked and were re-measured in a
verified-quiet window (`free -m` available > 140 GB AND load average < 12 AND
no live W3/qwen/bench process); only the quiet legs carry conclusions.

Two measurement-discipline defects were hit and are recorded because the
brief's warning about them is load-bearing: a `pgrep -f` quiet gate matches
its OWN shell's cmdline (fixed with the `[b]racket` form and exact `comm`
names), and an incompletely-killed monitor loop resumed later and raced live
legs, truncating one log — the affected leg was re-run and every surviving
leg was re-measured in the final pass (`final-legs`, all end-load < 12,
chains and counters verified per leg).

## 1. Baseline reproduction (this tree, bf7b654ae)

`tests/test_kolibri1_decode_bench`, `VLLM_CPP_CPU_THREADS=8`:

| leg | window | wall s | prefill s | decode s | decode tok/s | anchor/chain |
|---|---|---|---|---|---|---|
| 16 GiB #1 | quiet (211 GB avail) | 36.49 | 16.84 | 19.65 | **3.206** | PASS, chain exact |
| 16 GiB #2 | quiet (214 GB avail) | — | — | 19.87 | **3.170** | PASS, chain exact |
| cache OFF #1 | quiet | 69.60 | 18.37 | 51.23 | 1.230 | PASS, chain exact |
| cache OFF #2 | quiet | 63.79 | 13.72 | 50.07 | 1.258 | PASS, chain exact |
| 16 GiB (contended, load >100) | — | 48.86 | 15.08 | 33.77 | 1.865 | PASS, chain exact |

The recorded production figure reproduces: **3.17-3.21 decode tok/s at a
16 GiB budget**, 2.52× the cache-off baseline (1.24-1.26), matching
kolibri1-combined-neon-cache-20261008.md (3.06-3.17). Cache counters on both
16 GiB legs are byte-identical to the record:
`hits=71157 misses=18717 evictions=13264 decode_calls=89874`. The CHAIN is
byte-identical across the OFF and 16 GiB legs (budget invariance), last token
109726, alternating 101807/109726.

## 2. The remaining wall, attributed (VT_KOLIBRI1_PROFILE, 8 threads, 16 GiB)

Final report after 60 forwards (includes prefill):

| stage | s | share of total_forward (35.24 s) |
|---|---|---|
| linear_gemm | 17.54 | 50% |
| moe_glue | 15.63 | 44% (NESTED: the routed-expert loop wraps ExpertMlp, so ~6/7 of the expert linear_gemm+dequant time accrues here too — kolibri1_forward.cpp:369-408) |
| dequant_fp8_block | 9.31 | 26% (cold misses: 18717 × ~0.50 ms whole-matrix pool decodes) |
| attn_core + attn_rope | 6.06 | 17% |
| lm_head | 1.57 | 4% |
| norms | 0.26 | <1% |

CORRECTION to the recorded reading: the prior note treated moe_glue's 0.25 s
as a separable per-expert dequant residual. It is not: the moe_glue scope
(kolibri1_forward.cpp:369) contains the routed experts' `LinearBT` calls, so
its time is mostly the ALREADY-ATTRIBUTED nested linear_gemm and
dequant_fp8_block work. The separable glue residual (the bf16 gather/scatter
copies and the per-expert scan, :372-407) is the small remainder. There is no
unattributed 0.25 s block.

Per decode step at 16 GiB the wall is ~315 ms: ~50% expert/projection GEMV at
the NEON tier, ~25% cold-miss whole-matrix decodes, ~17% attention, ~4%
lm_head.

## 3. Lever candidates, adjudicated by measurement

### 3a. FALSIFIED: fp8-direct GEMV (dequantize fp8 weights in-register)

The idea: at t=1 the GEMV reads the cached bf16 weights (2 B/element); reading
the resident packed fp8 bytes (1 B/element) and dequantizing in-register
would halve weight traffic AND eliminate the cold-miss decode and the cache
budget entirely. Bit-exactness is achievable: the prototype composes the
landed NEON pieces (`E4m3ToF32x4` × block scale → `F32x4ToBf16x4`, the one
bf16 rounding) and keeps every output lane's K accumulation strictly
sequential in `Bt16Neon`'s order (cpu_matmul_elem.cpp:155-189).

Scratch micro-benchmark (`/tmp/kolibri-micro/gemv_micro.cpp`, single thread,
-O3 `-ffp-contract=off`, real shapes, buffers padded for the 8-byte loads):

| shape | bitdiff vs two-step reference | bf16 path | fp8-direct | speedup |
|---|---|---|---|---|
| [N=512, K=2560] (gate/up) | bf16=0, fp8direct=0 | 0.297 ms (8.8 GB/s) | 2.475 ms (0.5 GB/s) | **0.12×** |
| [N=2560, K=512] (down) | bf16=0, fp8direct=0 | 0.275 ms (9.5 GB/s) | 2.181 ms (0.6 GB/s) | **0.13×** |

Bit-exact, but **8× SLOWER**. The order-preserving contract forbids
`vfmaq`/`fmla` and any K-axis reassociation, so each element costs one
`vmulq` + one `vaddq` per output lane and the kernel is ALU-bound, not
bandwidth-bound: ~9 GB/s/thread ≈ 4.5 Gelem/s × ~2 fp ops/element ≈ the SIMD
pipe roof. The fp8 path adds ~6× the arithmetic per element
(widen → decode → scale → bf16-round → re-widen) and pays it on the ALU wall
it was trying to slip under. Falsified: halving bytes buys nothing on an
ALU-bound kernel.

### 3b. FALSIFIED: the GEMM inner loop has no bit-exact headroom

The same ALU-roof measurement answers the "make linear_gemm faster" family:
the landed tier already spends the minimum the contract allows (one mul + one
add per element per lane, `vmulq`+`vaddq`, never `vfmaq`), sits within ~25% of
the resulting roof, and every faster instruction (fused multiply-add, bf16
dot) breaks the row's bit-exactness contract. No bit-exact GEMM lever exists.

### 3c. Already optimal: the cold-miss decode path

`DecodeInto` (kolibri1_dequant_cache.h:292-301) already decodes through the
ONE pool over output rows — the exact uncached kernel, partitioned; a miss
cannot cost less than the uncached decode it replaces. The eviction policy is
already LRU with stable-identity keys. The remaining miss cost is the
working-set physics of the expert distribution, addressable only through the
budget.

### 3d. The lever that remains IS the budget policy — a developer decision

Per the task brief this is where the unit stops and delivers numbers (§4).
The cache stays opt-in (default 0) pending the policy choice.

## 4. Budget sensitivity (the measured proposal)

`test_kolibri1_decode_bench`, 8 threads, `VT_KOLIBRI1_PROFILE=1`, quiet
windows per leg, same binary:

| budget | decode s | decode tok/s | hits | misses | evictions | chain |
|---|---|---|---|---|---|---|
| 0 (OFF) #1 | 51.23 | 1.230 | — | — | — | exact |
| 0 (OFF) #2 | 50.07 | 1.258 | — | — | — | exact |
| 8 GiB #1 | 21.05 | 2.992 | 69377 | 20497 | 18321 | exact |
| 8 GiB #2 | 20.10 | 3.134 | (log lost to a concurrent-writer defect, counters not captured) | | | |
| 16 GiB #1 | 19.65 | 3.206 | 71157 | 18717 | 13264 | exact |
| 16 GiB #2 | 19.87 | 3.170 | 71157 | 18717 | 13264 | exact |
| 32 GiB #1 | 18.70 | 3.369 | 72824 | 17050 | 5043 | exact |
| 32 GiB #2 | 18.73 | 3.363 | 72824 | 17050 | 5043 | exact |

RSS: measured on this tree's W3 gate at 16 GiB: max RSS 97,158,528 KB
(92.7 GiB) — the cache adds its resident bytes on top of the ~80 GB
checkpoint working set (the 2026-10-08 combined note recorded 96.0 GiB on
the pre-bf7b654ae tree; same shape).

### Recommendation (the developer chooses the default)

The knee is NOT at 16 GiB on this evidence:

- OFF → 8 GiB: +143% decode tok/s (the cache itself is the lever).
- 8 → 16 GiB: +2-7% (1.230→3.17-3.21 vs 2.99-3.13) — near saturation of the
  8 GiB working set, evictions 18321 → 13264.
- 16 → 32 GiB: +5-6% (3.17-3.21 → 3.36-3.37), evictions 13264 → 5043, hits
  71157 → 72824 — still a real, repeatable gain from cutting evictions by
  2.6×, bought with +16 GiB of resident RSS.

Every budget's CHAIN is byte-identical (md5 `8de1463a…`, the alternating
101807/109726 chain) and the counters are byte-identical to the historical
records per budget — the budget changes only speed and RSS, never output.

Recommendation to evaluate: **32 GiB** if the production host's RSS envelope
admits ~112 GB per engine instance (96 GiB W3 working set + 16 GiB more
cache), else **16 GiB** as the value that captured 79% of calls at half the
memory. OFF (the current default) costs 2.5× decode tok/s on every production
run that does not set the var. The choice is the developer's; this unit ships
numbers, not the default.

## 5. Thread scaling

Clean-quiet legs (16 GiB budget): T=4 → 29.89 s decode, **2.108 tok/s**
(end-load 5.5); T=8 → 3.170-3.206 tok/s (§1); T=16 → 16.12-16.35 s,
**3.854-3.908 tok/s** (end-load 8-11). A T=4 leg that ended under load 44
(1.853) and a batch overtaken by the serving agent's 128-thread qwen3 e2e
test (load 134: T=4 1.372, T=8 2.620, T=16 2.195) are DISCARDED as
contaminated. Conclusion: 8 threads is the current production operating
point; 4 threads costs ~1.5×; 16 threads gains ~1.2× (3.2 → 3.9 tok/s) —
the near-linear 8-vs-4 scaling confirms the ALU-bound model (§3b): the GEMV
scales with SIMD pipes until the cold-miss DRAM decode and the eviction path
take over. 16 threads is a measured option for latency-sensitive serving at
+60% RSS-parallel pressure, not a default change; recorded, not chosen.

## Gates run on this tree (all green)

| gate | result |
|---|---|
| test_kolibri1 | SUCCESS |
| test_kolibri1_dequant | SUCCESS |
| test_kolibri1_dequant_cache | SUCCESS (no flake this pass) |
| test_kolibri1_dequant_cache_default | exit 0 |
| test_kolibri1_moe_glue | SUCCESS |
| test_kolibri1_w2 | SUCCESS |
| decode bench anchor | 109726, all legs, chain byte-identical across budgets |
| test_kolibri1_w3 at 16 GiB | SUCCESS, 900/900 assertions; ARGMAX CHAIN 141/145 (4 flips: 4 near-tie, 0 hard); FINAL-STEP FINGERPRINT worst topk diff 2.18646, worst sum diff 26763 — identical to the landed baseline; counters hits=348418 misses=519935 evictions=514482 decode_calls=868353 byte-identical to the 2026-10-07 reland record; wall 9:47.45, max RSS 97,158,528 KB (92.7 GiB) |

The Tenstorrent unit gates (test_kolibri1_tt, _tt_b2i, _tt_b2bi) are NOT
runnable in this build: `VLLM_CPP_TENSTORRENT=ON` requires a TT-Metalium
install tree; the only one on this host (`/tmp/pin-build`) belongs to the
active TT repair agent, and routing this CPU-only unit's build through their
SDK (and the TT device their W3 exercises) would collide with their row. The
change set of this unit touches no TT-path file; the TT row's gates remain
that row's obligation.

## Outcome

The next CPU decode lever does not exist as a bit-exact code change: the GEMM
tier is at its contract roof (falsified by an 8×-slower bit-exact fp8-direct
prototype), the miss path is already at the uncached decode's cost, and the
one open lever is the dequant-cache production budget — a product/policy
decision that belongs to the developer. This unit delivers the corrected
attribution, the falsifications with their measurements, and the
budget-sensitivity table backing a default recommendation.

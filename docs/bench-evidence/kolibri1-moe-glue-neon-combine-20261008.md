# Kolibri-1 CPU moe_glue: profile, attribution, and the NEON expert-output
# combine lever (MODEL-TEXT-kolibri-1)

Date: 2026-10-08. Host: thalia, aarch64, 128 cores, 255 GB RAM, Linux (the
same box as the 2026-10-07/08 records). Tree: `origin/main` **5d2f5d31a**
(the default order-preserving NEON GEMM tier + the relanded fp8 dequant
cache), worktree `/tmp/vllm-kolibri-glue`, branch `row/kolibri-glue`. Builds
in `/tmp`: Release, `-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_TENSTORRENT=OFF
-DVLLM_CPP_SERVER=OFF -DVLLM_CPP_BUILD_EXAMPLES=OFF` (configure flags matching
docs/bench-evidence/kolibri1-combined-neon-cache-20261008.md). All runs
`VLLM_CPP_CPU_THREADS=8`. Real checkpoint
`/mnt/models/Aleph-Alpha/Kolibri-1` (74 GB fp8, 32 shards).

Three binaries, one per role (all from the same tree unless noted):

- **pristine** (`/tmp/kolibri-pristine-bin/`): `origin/main` 5d2f5d31a,
  unmodified — the official per-stage profile and the A/B baseline.
- **instrumented** (`/tmp/kolibri-instr-bin/`): pristine + scratch
  fine-grained `moe_glue` sub-op scopes (never committed; reverted
  byte-for-byte before the lever) — the attribution table below.
- **lever** (`/tmp/vllm-kolibri-glue-build/`): pristine + the landed
  `MoeCombineKernel` NEON fast path — the gates and the A/B treatment.

## The question

With both landed levers stacked (NEON GEMM default tier + the opt-in 16 GiB
dequant cache), production decode is ~3.1 tok/s
(docs/bench-evidence/kolibri1-combined-neon-cache-20261008.md, on
`mine/row/kolibri-combined`). The remaining stage profile shows `moe_glue`
as a biggest remaining stage. This unit: (1) profile `moe_glue` on main
with the cache ON (16 GiB) and OFF on the decode bench AND W3; (2) split
`moe_glue` into its sub-operations; (3) land ONE sub-lever, justified by
those numbers, under the row's bit-exactness contract; (4) gate it and A/B
it. Issue: ISSUE-LOCAL-01M4D7FYY1WQZV9PNR6GBWS2R5.

## Profile on main (VT_KOLIBRI1_PROFILE=1, 8 threads)

### Decode bench (`test_kolibri1_decode_bench`, 128-in prefill + 63 t=1
### steps), per-stage totals over the 60-forward report block

Cache ON 16 GiB (instrumented binary; identical stage keys on pristine):

| stage | s / 60 fwd | ms / decode fwd | share of step |
|---|---|---|---|
| total_forward | 35.97 | 317.5 | 100% |
| attn_core | 5.31 | 89.0 | 28.0% |
| **moe_glue** | **15.53** | **81.5** | **25.7%** |
| linear_gemm (all; nests 7.17 s inside moe_glue) | 10.79 | 116.0 | 36.5% |
| — of which expert GEMMs (`linear_gemm_glue`) | 7.17 | 71.5 | 22.5% |
| dequant_fp8_block (all; 7.72 s of 8.73 s inside moe_glue) | 8.73 | 4.0 | 1.3% |
| attn_rope | 0.75 | 10.0 | 3.1% |
| lm_head | 1.92 | 15.0 | 4.7% |
| norms | 0.46 | 3.5 | 1.1% |
| silu (glue 0.19 + other 0.03) | 0.22 | 3.0 | 0.9% |
| moe_d2h + moe_topk | 0.03 | ~0.1 | 0.0% |
| unscoped remainder (embedding, pool allocs, StepInputs, dispatch) | 7.31 | 72.9 | 23.0% |

The per-forward numbers are the steady-state decode blocks (report deltas
40→60; the decode step is 317.5 ms). The whole-run moe_glue total (15.53 s)
is dominated by the ONE prefill forward (~10.6 s of it: 128 tokens x 6
experts per layer of batched expert GEMMs plus ~7.4 s of cold decodes), so
the decode-only moe_glue is 81.5 ms of the 317.5 ms step.

Cache OFF (instrumented binary, same block): dequant_fp8_block 41.9 s-class
(the uncached re-decode of every used weight per call), moe_glue 0.375 s
per forward average — see the NEON doc's table
(kolibri1-linear-gemm-neon-20261007.md) for the cache-off stage split; this
unit re-measured it at 1.249 tok/s baseline (below).

### moe_glue attribution (the fine-grained split, cache ON 16 GiB)

The scratch instrumentation adds disjoint sub-scopes inside the
`moe_glue` region of `MoeBlock` (kolibri1_forward.cpp:369-417) plus a
glue-depth tag that splits the nested `LinearBT`/`ExpertMlp` scopes into
`*_glue` (routed experts) vs plain (attention, router, shared expert).
Decode-only steady state (ms per decode forward):

| sub-operation | ms / fwd | share of moe_glue |
|---|---|---|
| expert GEMMs (`linear_gemm_glue`, 900 M=1 GEMVs) | 71.5 | 87.7% |
| expert dequant-cache lookups + cold decodes (`dequant_fp8_block_glue`) | 3.5 | 4.3% |
| expert MoeSiluMul (`silu_glue`) | 2.5 | 3.1% |
| **expert-output combine (`g_combine`)** | **2.0** | **2.4%** |
| per-expert top-k scan (`g_scan`) | 1.0 | 1.2% |
| expert scatter copies (`g_scatter`) | 0.5 | 0.6% |
| expert gather copies (`g_gather`) | ~0 | 0.0% |
| `expert_out` zero (`g_zero`) | ~0 | 0.0% |
| expert-loop overhead (per-expert vector ctors, branch) | ~1.0 | 1.2% |

(The nested items sum into `g_expertmlp` = 77.0 ms; the glue-exclusive
remainder is 4.5 ms of the 81.5 ms moe_glue.)

W3 (`test_kolibri1_w3`, teacher-forced, re-prefill per step, ~157 forwards;
stage totals at the 150-forward report, ms per forward):

| stage | cache OFF | cache ON 16 GiB |
|---|---|---|
| total_forward | 511.77 s (3411.8 ms/fwd) | 492.20 s (3281.3 ms/fwd) |
| moe_glue | 301.16 s (2007.7 ms/fwd, 58.9%) | 297.22 s (1981.5 ms/fwd, 60.4%) |
| dequant_fp8_block | 248.03 s (1653.5 ms/fwd) | 213.71 s (1424.7 ms/fwd) |
| linear_gemm | 221.32 s (1475.5 ms/fwd) | 236.62 s (1577.5 ms/fwd) |
| lm_head | 21.11 s | 21.09 s |
| attn_rope | 5.36 s | 5.34 s |
| attn_core | 5.20 s | 5.23 s |
| norms | 2.17 s | 2.18 s |
| moe_topk | 0.56 s | 0.56 s |
| moe_d2h | 0.03 s | 0.03 s |

Cache counters at 16 GiB (final): hits=348418, misses=519935,
evictions=514482, decode_calls=868353 — byte-identical to the reland and
combined records. The W3 gate shape thrashes the cache (60% miss rate:
evictions ≈ misses, the 16 GiB budget against the gate's ~10 GB working set
re-touched with fresh re-prefills per forward), so the W3 wall moves only
−3.8%; the production decode bench is where the cache pays (2.5x, below).
Both W3 legs reproduced the row's token gate exactly: 900/900 assertions,
ARGMAX chain 141/145 (4 near-tie, 0 hard), fingerprints 2.18646 / 26763.

### Reading the attribution: what is lever-able inside moe_glue

`moe_glue`'s mass is 92% NESTED work that is not the MoE glue:
the routed experts' GEMMs (71.5 ms/fwd — the shared vt GEMM, whose
per-element K-order is the row's bit-exactness contract: the K-axis lesson
forbids reordering it, and a stream-pattern rewrite is a vt-layer lever,
not a moe_glue one) and the expert dequant-cache lookups/cold decodes
(3.5 ms/fwd at decode — the cache component, budget-bound: at 16 GiB the
decode steps hit ~85% and the misses are the LRU floor, not a code defect).
The MoE-scoped glue-exclusive remainder is 4.5 ms/fwd, and its single
largest bit-exact-able item is the expert-output combine at 2.0 ms/fwd
(0.6% of the decode step; the rest: scan 1.0, loop overhead ~1.0,
scatter 0.5, gather/zero ~0).

## The lever (ONE): order-preserving NEON for the expert-output combine

`MoeCombineKernel` (src/vt/cpu/cpu_ops.cpp) computes
`out[t,h] = shared[t,h] + sum_j w[t,j] * expert_out[t,j,h]` with a scalar
col-outer / j-inner loop. **Why this sub-lever, in one paragraph:** with
the 16 GiB cache on, moe_glue is 81.5 ms of the 317.5 ms decode step, and
the attribution above splits it into 71.5 ms of nested expert GEMMs
(shared vt GEMM, K-axis-locked by the bit-exactness contract — skipped per
the GEMM K-axis lesson), 3.5 ms of nested dequant-cache work (the cache
component, budget-bound — out of this unit's scope), 2.5 ms of nested
expert silu (the shared `MoeSiluMul` op — its `std::exp` per element cannot
be vectorized bit-exactly, so it is skipped: a NEON expf is not
bit-identical to glibc's), and a 4.5 ms MoE-scoped glue-exclusive remainder
whose largest single item is the combine at 2.0 ms/fwd. The combine is
elementwise/gather-combine work over INDEPENDENT output columns — the
exact shape the landed Bt16Neon GEMM tier already proved bit-exact — so
vectorizing across columns while each lane keeps the scalar j-order
reduction is bit-exact BY CONSTRUCTION, not by tolerance.

**Bit-exactness argument (before coding):** the scalar body computes, per
output element, `acc = 0; for j: acc += w_j * LoadF32(eo_j); [acc *= scale];
[acc += shared]; StoreF32(acc)`. The NEON path runs the SAME sequence per
lane: products are `vmulq_f32` + `vaddq_f32` — NEVER `vfmaq` — so the
product rounds and the add rounds exactly as the scalar loop at the
project-pinned `-ffp-contract=off` (CMakeLists.txt:55); the scale multiply
and the shared add keep their scalar placement (one rounding each, same
order); the bf16 widening is the shift-left-16 (bit-identical to
`BF16ToF32`, the same widening the GEMM's `LoadV4<kBF16>` asserts); the
bf16 store is the RNE of `F32ToBF16` including the truncate-and-quiet NaN
branch (the same vectorized store `kolibri1_fp8_dequant.h` asserts
bit-identical over the domain). The j (reduction) axis is never vectorized
or reordered. The fast path covers the all-bf16 polarity (kolibri1's);
every other dtype combination keeps the scalar loop, unchanged.

**What was considered and skipped, with the reason:**
- expert GEMM stream-pattern rewrite (the 71.5 ms item): the shared vt
  GEMM's per-element K-order is the row's bit-exactness contract; a
  rewrite is a vt-layer lever with a whole-tree blast radius, not a
  moe_glue sub-lever — recorded as an owed follow-up with its measured
  share (22.5% of the decode step).
- expert cold decodes (3.5 ms/fwd at decode; 7.4 s of the prefill): the
  cache component; the misses are the 16 GiB LRU floor. A buffer-reuse
  lever inside `kolibri1_dequant_cache.h` (the fresh 2.6 MB entry
  allocations take first-touch page faults on every cold miss) is a
  cache-component unit, not moe_glue — recorded as an owed follow-up.
- expert `MoeSiluMul` vectorization (2.5 ms/fwd): `std::exp` per element
  cannot be made bit-exact by construction — skipped per the task's rule.
- per-expert scan / loop bookkeeping (2.0 ms/fwd combined): a CSR bucket
  build is bit-exact (pure bookkeeping) and comparable in size to the
  combine lever; the combine was picked because it is the single largest
  named glue sub-operation and its test pins a raw-uint16 arithmetic
  contract (the task's required test shape). The bucket build is recorded
  as an owed follow-up.
- dead `expert_out.Zero` skip (~0 ms at decode; 3.9 MB memset per layer at
  prefill): provably dead (every [t, top_k] slot is written exactly once —
  top-k selects distinct experts), bit-exact, but negligible — recorded.

## Red-first evidence (`tests/vllm/models/test_kolibri1_moe_glue.cpp`)

New doctest target `test_kolibri1_moe_glue` (CMakeLists.txt beside the
other kolibri1 targets), in the style of test_kolibri1_dequant.cpp: the
production op (`vt::MoeCombine`, the exact call shape kolibri1_forward.cpp's
MoeBlock uses) is compared RAW-UINT16 against an in-test scalar
transcription of the kernel math (vt::BF16ToF32 / vt::F32ToBF16 — the
converters are pinned elsewhere; the test pins the ARITHMETIC ORDER),
over the real expert geometries:

- the production decode geometry (t=1, k=6, h=2560, shared, scale 1.0);
- ragged t=7 and prefill t=128; h tails 1/3/7/13/2557 (NEON body + scalar
  tail); k tails 1/2/13; the no-shared form; the routed_scale arm;
- a 384-expert 6-of-384 route with per-expert outputs gathered into the
  [t, k, h] slot layout and the shared-expert path;
- exact-tie stores (dyadic weights x integer activations — every store is
  an RNE tie), overflow to +-inf, and the hardware default-QNaN path
  (routed +inf with a -inf shared term);
- cross-thread-count identity (1-thread vs 8-thread pool at t=128).

Input NaN patterns are deliberately EXCLUDED from the random domain (mapped
to the largest finite exponent): an input NaN's payload propagation through
`acc += shared` is compiler-scheduling-dependent (addition is commutative;
which NaN payload wins when BOTH operands are NaN is a register-allocation
accident) — measured by writing this test: the kernel and a same-source
scalar reference diverge on exactly those elements, and no bit contract can
pin them. The one deterministic NaN path (overflow → the hardware default
QNaN) IS pinned bit-for-bit.

RED → GREEN sequence (each mutation run in this worktree, scratch only,
restored byte-for-byte after, verified with `cmp`):

1. Contract pin green on the landed scalar kernel: 7/7 cases, 21/21
   assertions.
2. MUTATION (scalar kernel): j-loop reordered (k-1 .. 0) → 3/3 affected
   cases RED (18/21 assertions pass) — the test detects a reordered
   accumulation. Restored → green.
3. Lever implemented (NEON fast path) → 7/7, 21/21 GREEN — bit-identical
   over every geometry.
4. MUTATION (NEON): `vfmaq_f32` fused into the accumulation → 5 cases RED
   (11/21 assertions pass) — the test detects the changed f32 rounding.
   Restored → green.
5. MUTATION (NEON): j-loop reordered inside the vector body → 3 cases RED.
   Restored → green (byte-identical restore confirmed with `cmp`).

Shared blast radius: the kernel is the shared vt op, so the existing
bitwise pins were re-run against the NEON path —
`test_ops_moe_nongated_relu2` 81/81, `test_ops_moe` 69/69,
`test_moe_bf16_native_contract` 219/219 (all green; the f32/f16 dtype
combinations keep the scalar loop and are pinned by them).

## Gates (lever binary; cache ACTIVE and hitting at 16 GiB)

- `test_kolibri1_moe_glue`: 7/7 cases, 21/21 assertions PASS.
- `test_kolibri1`: 27/27 cases, 186/186 assertions PASS.
- `test_kolibri1_w2`: 1608/1608 PASS. `test_kolibri1_dequant`: 10/10 PASS.
  `test_kolibri1_dequant_cache`: 7/7 cases, 52/52 assertions PASS.
- Shared MoE suites: 81/81, 69/69, 219/219 PASS (above).
- `test_kolibri1_w3` at `VT_KOLIBRI1_DEQUANT_CACHE_MB=16384`,
  `VT_KOLIBRI1_PROFILE=1`, 8 threads (leg gate-w3-16g, quiet window):
  **900/900 assertions PASS; ARGMAX chain 141/145 (4 near-tie, 0 hard);
  FINAL-STEP FINGERPRINT worst topk 2.18646, worst sum diff 26763 —
  identical to the landed baseline.** Cache counters:
  hits=348418, misses=519935, evictions=514482, decode_calls=868353 —
  byte-identical to the reland/combined records. Wall 8:54 (baseline
  8:45).
- `test_kolibri1_decode_bench` anchor + chain md5 at cache off / 8 GiB /
  16 GiB (legs lever-bench-off-1/2, lever-bench-8g, lever-bench-16g-1/2):
  **anchor 109726 on all five legs; CHAIN md5
  `255e04ab66e83407588211c274eba579` on all five — identical to the
  baseline's** (and to the baseline legs at every budget).
- `scripts/check-agent-record.py`: rc=0 (see Records).
- `scripts/agent-preflight.sh`: rc=0 (background, /tmp; no NEW failures).

## A/B: production-shape decode bench, 8 threads, this tree

Baseline = pristine `origin/main` 5d2f5d31a; lever = pristine + the NEON
combine. Same host, same binaries' build flags, quiet window (see the
deviation note below). Baseline legs (pristine binary):

| leg | wall s | prefill s | decode s | decode tok/s |
|---|---|---|---|---|
| cache OFF #1 | 63.69 | 13.00 | 50.69 | 1.243 |
| cache OFF #2 | 62.67 | 13.07 | 49.60 | 1.270 |
| 8 GiB | 36.34 | 15.54 | 20.80 | 3.028 |
| 16 GiB #1 | 36.07 | 15.85 | 20.22 | 3.115 |
| 16 GiB #2 | 35.91 | 15.87 | 20.04 | 3.143 |

Means: OFF decode 50.15 s / 1.256 tok/s; 16 GiB decode 20.13 s / 3.129
tok/s (2.49x). All seven baseline legs (incl. the instrumented binary's
two) produced the anchored last token 109726 and the byte-identical CHAIN
(md5 `255e04ab66e83407588211c274eba579` of the chain string).

Lever legs (lever binary, same window discipline):

| leg | wall s | prefill s | decode s | decode tok/s | chain md5 |
|---|---|---|---|---|---|
| cache OFF #1 | 62.91 | 12.87 | 50.05 | 1.259 | 255e04ab… |
| cache OFF #2 | 63.15 | 12.95 | 50.20 | 1.255 | 255e04ab… |
| 8 GiB | 36.14 | 15.49 | 20.64 | 3.052 | 255e04ab… |
| 16 GiB #1 | 35.82 | 15.75 | 20.07 | 3.139 | 255e04ab… |
| 16 GiB #2 | 35.24 | 15.55 | 19.68 | 3.201 | 255e04ab… |

### A/B verdict

| config | decode wall base → lever | decode tok/s base → lever | moe_glue stage base → lever |
|---|---|---|---|
| OFF | 50.15 → 50.12 s (−0.06%) | 1.256 → 1.257 (+0.08%) | 22.58 → 22.51 s (−0.33%) |
| 8 GiB | 20.80 → 20.64 s (−0.75%) | 3.028 → 3.052 (+0.79%) | 15.10 → 15.00 s (−0.66%) |
| 16 GiB (primary) | 20.13 → 19.88 s (−1.24%) | 3.129 → 3.170 (+1.31%) | 15.05 → 14.63 s (−2.76%) |

Every delta is non-negative (no config regressed), and every leg's CHAIN
is byte-identical to the baseline's (md5 `255e04ab66e83407588211c274eba579`,
anchor 109726) at all three cache budgets.

**Honest reading: this is a small real win, not a measurable one.** The
run-to-run noise on this box is 0.9-2.2% (the baseline's own paired runs
span that much at every config), and the measured deltas (+0.08% to +1.31%
decode tok/s) sit inside it. The attribution-derived expectation is ~0.5%
of the decode wall: the combine is 2.0 ms of the 317.5 ms decode step
(0.63%) and the NEON body replaces the scalar loop's per-element `LoadF32`
dtype switch and strided loads, a ~4x kernel-throughput improvement, worth
~1.5 ms per decode forward plus ~24 ms of the prefill's combine. The
measured direction matches the theory at every config; the magnitude is
below the bench's noise floor. Per the standing rule the result is not
negative, so the lever LANDS — recorded with its noise context, not as a
claimed throughput win. The publishable production figure is unchanged
(~3.1 decode tok/s at a 16 GiB cache, 2.5x the NEON-only baseline); the
row's real output from this unit is the attribution: the MoE-scoped glue is
4.5 ms of the 317.5 ms decode step, and the mass is elsewhere (below).

## Outcome: what the profile says the row owes next

Measured shares of the 317.5 ms decode step (cache ON 16 GiB, this tree):

| item | ms/fwd | share | scope |
|---|---|---|---|
| attn_core (paged attention at t=1) | 89.0 | 28.0% | vt attn kernel — owed |
| unscoped forward remainder (embedding, pool allocs, StepInputs, dispatch) | 72.9 | 23.0% | unattributed — owed (needs its own scopes) |
| expert GEMMs inside moe_glue | 71.5 | 22.5% | shared vt GEMM — owed (stream pattern; K-axis-locked) |
| linear_gemm non-glue (attn q/k/v/o, router, shared expert) | 44.5 | 14.0% | shared vt GEMM — same lever as above |
| lm_head | 15.0 | 4.7% | shared vt GEMM |
| attn_rope | 10.0 | 3.1% | vt rope kernel — owed |
| MoE glue-exclusive (combine 2.0, scan 1.0, loop 1.0, scatter 0.5) | 4.5 | 1.4% | THIS unit's scope — landed lever covers the combine |
| expert cold decodes inside moe_glue | 3.5 | 1.1% | dequant cache — owed (cold-miss page faults on fresh 2.6 MB entries) |
| norms | 3.5 | 1.1% | — |
| expert silu inside moe_glue | 2.5 | 0.8% | shared op, not bit-exact-vectorizable |

The single biggest owed lever is the shared GEMM's M=1 GEMV weight-stream
pattern (Bt16Neon walks 16 weight-row streams per thread; a row-contiguous
GEMV body would stream near-peak) — 22.5% + 14% + 4.7% of the decode step
across the expert, attention and lm_head GEMVs, but it is a vt-layer
change whose per-element K order must stay sequential (the K-axis lesson)
and whose blast radius is every model's GEMM. The attn_core kernel at t=1
and the unscoped 23% remainder are the next attribution targets. None of
these is a moe_glue sub-lever; this unit's scope is complete.

## Quiet-window deviation record

The task brief's window is "poll until free -m available > 110000 AND zero
test_kolibri1_* procs". A foreign session (the kolibri-tt row, worktree
`/tmp/vllm-kolibri-tt-b2i`) held the window closed for the first ~8 legs
of this unit: its CPU sweep (W3 at 128 threads, then dequant,
dequant_cache, decode_bench) plus a STALLED device test
`test_kolibri1_tt_b2i` (1 core at 100%, memory-static 79.4 GB, device log
quiet since 10:05, >1 h) — the strict window could not open. The leg runner
therefore ran a NARROW, documented waiver for those legs: the window is
available > 110000 MB AND zero foreign kolibri MODEL procs, excluding only
the stalled tt_b2i (and the <1 GB unit suites). Safety math for the worst
leg (W3 at a 16 GiB cache, ~96-98 GB RSS): 98 + 79.4 (tt_b2i) + ~15 (other
users) = ~192 GB of 255 GB, ~63 GB headroom; thalia runs
`vm.overcommit_memory=0` (checked), so an OOM would kill the largest
process, never reboot the box. A watchdog killed any leg within 5 s if
another foreign model proc appeared (it never fired). tt_b2i exited at
~11:23 (detected by the available-memory jump 138 GB → 212 GB during leg
prof-bench-16g-2); from leg prof-w3-off onward the STRICT window held (zero
foreign test_kolibri1_* procs, available ~212 GB). Legs that ran under the
waiver: attr-bench-16g, attr-bench-off, prof-bench-off-1/2, prof-bench-8g,
prof-bench-16g-1, and the first half of prof-bench-16g-2.

## Records

- Lever: `src/vt/cpu/cpu_ops.cpp` (`MoeCombineKernel` NEON fast path +
  the two vector helpers, all under `#if defined(__aarch64__)`).
- Test: `tests/vllm/models/test_kolibri1_moe_glue.cpp` +
  `tests/CMakeLists.txt` registration.
- Issue: ISSUE-LOCAL-01M4D7FYY1WQZV9PNR6GBWS2R5 (closed by this change).
- Run logs: `/tmp/kolibri-glue-runs/` (window log + one log per leg).
- Commit: branch `row/kolibri-glue`, pushed to the `mine` fork (no PR, per
  the task brief); the landed SHA is in the session report.

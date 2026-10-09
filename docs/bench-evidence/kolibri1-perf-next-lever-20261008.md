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
Review repair 2026-10-09 (PR #3423 blocking review, records only, no product
code, no new performance claim): the whole-run table in §2 is labeled
WHOLE-RUN and is no longer applied to a decode step; §2b adds the measured
prefill/decode phase split with separate per-phase counters; the ceiling
wording is narrowed to the tested candidates everywhere (this file, the spec's
NEXT-LEVER UNIT paragraph, the issue Resolution, and the PR body).

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
| 16 GiB, profile ON #1 | quiet (load < 1) | 35.79 | 15.93 | 19.87 | **3.171** | PASS, chain exact, counters byte-identical (§2b) |
| 16 GiB, profile ON #2 | quiet (load < 1) | 35.56 | 15.68 | 19.89 | **3.168** | PASS, chain exact, counters byte-identical (§2b) |

The recorded production figure reproduces: **3.17-3.21 decode tok/s at a
16 GiB budget**, 2.52× the cache-off baseline (1.24-1.26), matching
kolibri1-combined-neon-cache-20261008.md (3.06-3.17). Two further legs with
`VT_KOLIBRI1_PROFILE=1` (the phase-split legs of §2b, 2026-10-09) reproduce
it at 3.168-3.171. Cache counters on the 16 GiB legs are byte-identical to
the record:
`hits=71157 misses=18717 evictions=13264 decode_calls=89874`. The CHAIN is
byte-identical across the OFF and 16 GiB legs (budget invariance), last token
109726, alternating 101807/109726.

## 2. The remaining wall, attributed WHOLE-RUN (VT_KOLIBRI1_PROFILE, 8 threads, 16 GiB)

WHOLE-RUN table — final report after 60 forwards (1 prefill forward + 59
decode forwards; the prefill is included in every number below). These are
WHOLE-RUN shares of a 35.24 s denominator and MUST NOT be applied to a single
decode step: the bench's own timing splits the run into 16.84 s prefill and
19.65 s decode (§1), and the phase split in §2b shows the two phases have
different stage mixes. Use §2b for decode-lever decisions.

| stage | s | share of the WHOLE-RUN total_forward (60 forwards, 35.24 s, prefill included) |
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

CORRECTION (2026-10-09 review): an earlier reading of this table applied its
whole-run shares to a single decode step ("per decode step the wall is ~315
ms: ~50% GEMV, ~25% cold-miss decodes, ~17% attention"). That attribution
was wrong: the shares above are whole-run shares over a denominator that is
~46% prefill (16.84 s of 36.49 s wall, §1). The measured decode-phase shares
are in §2b and differ materially — cold-miss decodes are ~3% of the decode
step, not ~26%, and attention is ~30%, not ~17%.

### 2b. Phase split: prefill vs decode (review repair, 2026-10-09)

The profiler accumulates globally and reports every 10 forwards; it cannot
split by phase by itself. The bench's forward structure is deterministic,
though: forward #1 is the prefill (t=128), forwards #2..#64 are t=1 decode
steps, and the reports print at forwards 10/20/30/40/50/60. Therefore
R60 − R10 is EXACTLY the 50 pure decode forwards #11..#60, and R10 is the
prefill plus the nine warming decode forwards #2..#10. The decode-phase table
below is exact; the prefill-phase numbers are DERIVED (R10 minus the nine
decode forwards costed at the steady-state per-step rate) and carry the bias
stated with them.

Legs (production config `VLLM_CPP_CPU_THREADS=8
VT_KOLIBRI1_DEQUANT_CACHE_MB=16384`, profiling ON, 2026-10-09; verified quiet
before each leg: load < 1, no test/bench/qwen process by exact `comm` name,
> 200 GB available; both PASS, chain exact — anchor 109726, alternating
101807/109726; the 60-forward counters are byte-identical to this doc and to
the historical records: hits=71157 misses=18717 evictions=13264
decode_calls=89874; the 10-forward counters are identical across both legs;
max RSS 92.3 GiB both legs):

| leg | wall s | prefill s | decode s | decode tok/s |
|---|---|---|---|---|
| profile ON #1 | 35.79 | 15.93 | 19.87 | 3.171 |
| profile ON #2 | 35.56 | 15.68 | 19.89 | 3.168 |

DECODE-PHASE attribution (EXACT: forwards #11..#60, 50 pure t=1 decode
forwards = R60 − R10; the legs agree within 0.2 s per stage):

| stage | s (leg #1 / #2) | ms per decode step | share of the decode-phase forward time |
|---|---|---|---|
| linear_gemm | 8.92 / 9.08 | 178 / 182 | 59% (the ALU-bound GEMV tier, §3b) |
| moe_glue | 3.97 / 3.98 | 79 / 80 | 26% (NESTED, same caveat as the whole-run table: the routed-expert loop wraps ExpertMlp, so most of the expert linear_gemm+dequant above accrues here too) |
| attn_core | 4.09 / 4.08 | 82 / 82 | 27% |
| attn_rope | 0.50 / 0.50 | 10 / 10 | 3% |
| lm_head | 0.70 / 0.70 | 14 / 14 | 5% |
| dequant_fp8_block | 0.49 / 0.41 | 10 / 8 | 3% (steady-state cold misses) |
| norms | 0.17 / 0.17 | 3 / 3 | 1% |
| total_forward | 15.12 / 15.18 | 302 / 304 | 100% |

The bench's own decode window is 315.3/315.7 ms per step (19.87/19.89 s over
63 steps); the ~13 ms difference is the per-step host work outside the
forward (the logits-row copy and argmax), not a stage.

Decode-phase cache counters (EXACT, same 50 forwards, identical in both
legs): hits=61717 misses=783 evictions=783 decode_calls=62500 — 15.7 misses
per decode step at steady state (7.8/step in the last block #51..#60;
37.2/step in the warmest pure-decode block #11..#20).

PREFILL-PHASE attribution (DERIVED: R10 minus the nine warming decode
forwards #2..#10 costed at the steady-state per-step rate above; the nine
warming forwards run ABOVE steady state — the cache is still filling — so
these prefill numbers are upper bounds, most biased for dequant_fp8_block;
the bench's own prefill window, 15.93/15.68 s including the one logits-row
copy, anchors the prefill total):

| stage | s (leg #1 / #2, derived) | reading |
|---|---|---|
| linear_gemm | 7.02 / 6.97 | ~44% of the prefill forward |
| moe_glue | 10.35 / 10.07 | ~62% (NESTED, same caveat) |
| dequant_fp8_block | 8.11 / 7.84 | ~51% (cold misses: the prefill touches the expert working set first) |
| attn_core | 0.48 / 0.49 | ~3% |
| attn_rope | 0.16 / 0.16 | ~1% |
| lm_head | 0.66 / 0.67 | ~4% |
| norms | 0.05 / 0.05 | <1% |
| total_forward | 16.70 / 16.40 (upper bound) | the bench prefill window (15.93 / 15.68 s) is the lower anchor; the gap is the nine warming decode forwards running ~85 ms/step above steady state |

Prefill-phase cache counters, honestly: the cumulative counters cannot be
split exactly at the prefill boundary (the nine warming decode forwards are
not steady-state — their hit rate is below steady state, so a linear
subtraction goes negative). What IS measured: 95.8% of the run's misses
(17934 of 18717) occur within the first 10 forwards (the prefill plus the
nine warming decode steps); the pure-decode window #11..#60 accrues only 783.
The prefill forward accounts for the large majority of the first-10-forward
misses (its ~16k cache lookups vs ~1250 per decode step — the prefill batches
128 tokens per expert GEMM, so it looks each expert weight up once, while a
decode step looks up ~1250), but the exact prefill/decode counter split is
NOT measurable with this profiler and is not claimed.

What the phase split implies for decode levers: (i) the GEMM family is ~59%
of the decode step — the TESTED candidates are falsified (§3a, §3b), but
GEMM scheduling/layout/threading variants are unmeasured and remain open;
(ii) attention is ~30% of the decode step and NO experiment in this unit
adjudicated it — the whole-run reading hid this, and it is a named open
decode lever; (iii) the cache budget's decode benefit is the HIT path (with
the cache OFF the decode step pays the full per-call re-dequant — measured
813 ms/step vs 315 with the cache on, §1/§4 — while the steady-state miss
cost with the cache ON is only ~8-10 ms/step), consistent with the measured
budget saturation (8→16 GiB only +2-7%); (iv) prefill-path levers are
unmeasured — the prefill forward is ~44% of the whole-run wall and about
half of it is cold-miss dequant, so the whole-run table was never decode
evidence.

## 3. Lever candidates, adjudicated by measurement

### 3a. FALSIFIED (tested candidate): fp8-direct GEMV (dequantize fp8 weights in-register)

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

Scope of this falsification: ONE implementation (in-register dequant inside
`Bt16Neon`'s exact loop shape), TWO real shapes, SINGLE thread. It falsifies
that implementation in that regime. It does not measure — and does not bound
— other GEMM scheduling, layout, or threading variants, which remain open
(AGENTS.md's no-ceiling rule: an apparent limit is an unresolved
implementation difference until traced).

### 3b. FALSIFIED (tested candidates): the landed GEMM inner loop has no bit-exact headroom

The same ALU-roof measurement answers the "make linear_gemm faster" family AS
TESTED: the landed tier already spends the minimum the contract allows (one
mul + one add per element per lane, `vmulq`+`vaddq`, never `vfmaq`), sits
within ~25% of the resulting roof, and every faster instruction (fused
multiply-add, bf16 dot) breaks the row's bit-exactness contract. No bit-exact
change to the LANDED inner loop exists. This does not measure GEMM
scheduling, layout, or threading variants, nor any other kernel — those are
unmeasured and remain open (AGENTS.md's no-ceiling rule). The decode-phase
table (§2b) puts linear_gemm at ~59% of the decode step, so those open
variants are the largest unmeasured decode lever family.

### 3c. MEASURED-AS-IS (tested path): the cold-miss decode path

`DecodeInto` (kolibri1_dequant_cache.h:292-301) already decodes through the
ONE pool over output rows — the exact uncached kernel, partitioned; a miss
cannot cost less than the uncached decode it replaces. The eviction policy is
already LRU with stable-identity keys. This measures the LANDED path against
the EXISTING uncached decoder; equality with the existing decoder does not
prove that no decoder improvement exists — decoder variants (pool
partitioning, decode kernels, prefetch) are unmeasured and remain open. What
the measurement does establish: on the landed path, the remaining miss cost
is the working-set physics of the expert distribution, addressable through
the budget — and, per §2b, it is mostly a prefill/warmup cost (95.8% of
misses in the first 10 forwards; the steady-state decode miss rate is
15.7/step).

### 3d. The next DECISION is the budget policy — a developer decision (not the only remaining lever)

Per the task brief this is where the unit stops and delivers numbers (§4).
The cache stays opt-in (default 0) pending the policy choice. The budget is
the next DECISION because it is a measured policy knob with a delivered
sensitivity table — not because the lever space is closed. Open and
unmeasured (§2b, §3a-3c): attention (~30% of the measured decode step, never
adjudicated), GEMM scheduling/layout/threading variants (~59% of the decode
step), decoder improvements, and prefill-path levers (the prefill forward is
~44% of the whole-run wall, about half of it cold-miss dequant).

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
the near-linear 8-vs-4 scaling is consistent with the ALU-bound reading of
the tested GEMM tier (§3b): the GEMV
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

What was TESTED is closed: the fp8-direct GEMV candidate (one implementation,
two shapes, single thread) is bit-exact and 8× slower — the order-preserving
kernel is ALU-bound, not bandwidth-bound; no bit-exact change to the landed
GEMM inner loop exists under the row's contract; the landed cold-miss decode
path already costs the uncached decode it replaces. What remains OPEN and
unmeasured, per AGENTS.md's no-ceiling rule (an apparent limit is an unresolved
implementation difference until traced): attention (~30% of the measured
decode step — the second-largest decode stage, never adjudicated in this
unit), GEMM scheduling/layout/threading variants (~59% of the decode step),
decoder improvements, and prefill-path levers (~44% of the whole-run wall).
The 2026-10-09 review repair adds the measured phase split (§2b): the
whole-run table is whole-run evidence only; the decode step is ~59%
linear_gemm / ~30% attention / ~5% lm_head / ~3% steady-state cold misses,
and the ~26% whole-run cold-miss share is prefill/warmup-concentrated. The
next DECISION is the dequant-cache production budget — a measured policy knob
with a delivered sensitivity table, not the only remaining lever — and that
decision belongs to the developer. This unit delivers the corrected
attribution, the phase split, the falsifications with their measurements, and
the budget-sensitivity table backing a default recommendation.

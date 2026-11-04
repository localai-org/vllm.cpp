# FLINT-CPU-DESCENT-2 — the CPU descent re-attributed post-NEON, the lm_head Q6_K NEON dot landed

Row: `FLINT-CPU-DESCENT-2` · spec: `.agents/specs/flint-cpu-descent-2.md` ·
date 2026-11-04 · worktree `/tmp/flint-cpu-descent2`, branch
`row/FLINT-CPU-DESCENT-2` (based on the FLINT-NEON-Q4K-DOT head `fedc43589`,
which this row's branch was rebased onto: the row's baseline must carry the
NEON q4_K dot). Host: thalia (aarch64, 128 cores), governor `performance` AS
FOUND. Builds on CPUs 64-127 (`-j 64`), bench legs `taskset -c 32-47` (16t) /
`-c 32-63` (32t), pool matched via `VLLM_CPP_CPU_THREADS`. Artifact:
`/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`, sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`.

## Contention record (read before the numbers)

The box was SHARED during every leg of this row. A peer session
(`flint-p150-serving-bench`, `build-p150bench`) ran its own serving bench in
the same zone; the 1-min load oscillated between ~2 and ~26 and the strict
idle gate (< 6) could only bound it, not remove it. First-sweep attribution
legs at load 13-28 (16t 4.65/4.65/5.37 tok/s) were DECLARED CONTAMINATED and
re-run. All numbers below are interleaved same-binary A/B pairs — the residual
contention hits both arms of each pair — but the ABSOLUTE levels sit below the
quiet-box records (5.93 @16t NEON-Q4K, 7.01 @16t SLIDING-ATTN) and are NOT
comparable to them. Ratios and in-situ phase deltas are the evidence.

## STAGE 1 — the re-attribution table (post-NEON-Q4K, one serving decode step)

Instrument: `VT_FLINT_PHASE_TIMING` (this row extends it with the
`shared_expert` / `moe_combine` sub-buckets of `moe_shared`). Medians over 597
steady `batch=1` steps (199 × 3 reps) per arm, `ab-*.log`:

| phase | 16t portable ms | 16t % | 16t neon ms | 16t % | 32t ms | 32t % |
|---|---:|---:|---:|---:|---:|---:|
| attention — sliding layers | 63.110 | 34.2% | 63.030 | 36.0% | 63.6 | 37.7% |
| attention — full layers | 22.876 | 12.4% | 22.452 | 12.8% | 22.7 | 13.5% |
| norms | 0.775 | 0.4% | 0.777 | 0.4% | 0.76 | 0.5% |
| dense MLP (layers 0-1) | 1.500 | 0.8% | 1.378 | 0.8% | 1.22 | 0.7% |
| MoE router + host top-k | 5.032 | 2.7% | 4.556 | 2.6% | 7.79 | 4.6% |
| MoE routed-expert loop | 68.114 | 36.9% | 63.719 | 36.4% | 54.4 | 32.2% |
| MoE shared expert | 15.773 | 8.5% | 13.694 | 7.8% | 11.2 | 6.6% |
| MoE combine (scatter) | 0.337 | 0.2% | 0.357 | 0.2% | 0.37 | 0.2% |
| lm_head (gather + untied head) | 5.744 | 3.1% | 4.550 | 2.6% | 4.6 | 2.7% |
| other in forward | 2.315 | 1.3% | 2.318 | 1.3% | 2.38 | 1.4% |
| **forward wall** | **184.66** | | **174.98** | | **168.9** | |

(New sub-buckets `shared_expert`/`moe_combine` are the row's addition; the
32t columns are the neon-arm legs, whose portable/neon delta is noise at 32t.)

### The finding that re-ranks the table

1. **The "MoE shared expert + combine" bucket is NOT a combine scatter.** The
   new sub-attribution splits the old 7.4% bucket: combine is 0.34 ms (0.2%);
   the shared expert's three bf16 GEMVs are 15.8 ms = 97% of the bucket. The
   spec's "same weights every token → hoist" premise does not apply at the
   serving batch this arm records (batch 1): one token per step means there is
   nothing to batch. The named lever for this bucket is a bf16-dot GEMV tier
   for the shared expert (the kolibri1 `vbfdotq` class), which is NOT
   bit-identical (changes accumulation order) and is a separate row.
2. **lm_head is Q6_K, not Q4_K.** `output.weight` in the published container
   is ggml type 14 (Q6_K), [150144, 2048] — the landed NEON q4_K dot does not
   cover it; it served through the scalar `vec_dot_q6_K_q8_K`. The row ported
   a NEON Q6_K dot (see STAGE 2).
3. **Router top-k is 2.6-4.6% and its selection is NOT replaceable without a
   tie-order decision** (see STAGE 2, lever 2, STOP).
4. The routed-expert loop remains the largest lever (32-37%) and keeps its
   grouped-GEMM follow-up row.

## STAGE 2 — the levers in the new table's value order

### Lever 1 (LANDED): NEON `vec_dot_q6_K_q8_K` for the lm_head

`src/vt/cpu/cpu_quant_dot_q6k_neon.cpp`, dispatched through
`SelectQuantQ6KVecDot` in `BlockVecDot(DType::kQ6_K)`,
`VT_CPU_Q6K_DOT=auto|portable|neon` (default `auto`), same house shape as the
q4_K row: one TU compiled `-march=armv8.2-a+dotprod+fp16`, runtime HWCAP
DotProd probe, portable scalar kernel as the universal fallback. The kernel
keeps the scalar's exact lane partition: per 16-value chunk the two int8x8
product halves sum in int16, widen to lanes 0..3 / 4..7, scale, accumulate in
two separate int32 accumulators; the float stage folds d per lane before the
accumulate (non-fused, `-ffp-contract=off` global); the 6-bit unpack
(ql nibble + qh pair − 32) is lane-wide but purely integer. One suite-caught
bug: the first draft loaded only qh[0..15] although a quarter reads qh[0..31]
— the second 16-byte half of each plane carried the wrong high bits
(`test_neon_q6k_dot` caught it, fixed by loading both qh halves).

Gates: `test_neon_q6k_dot` 6370/6370 (bit-identity vs the scalar kernel, f32
bit compare, randomized + adversarial grid); `test_ops_quant_dot` 271928/271928;
`test_flint_paged` 208/208; `test_flint_paged_engine` 11/11;
**keep-quant serving gate UNMOVED: 53/60 exact, 7 near-tie, 0 hard, worst
2.48648** (byte-identical to the standing record; re-run on the final binary).

In-situ lm_head: 5.744 → 4.550 ms (−21%) @16t; @32t 4.59 → 4.63 ms (noise —
the lm_head GEMV is pool-parallel at 32t and the dot is not the binding term).

### Lever 2 (STOP — tie order): parallel/radix-select router top-k

A bounded top-k insertion (`std::lower_bound` against the same comparator)
was implemented and FALSEIFIED by its own equivalence suite: the previous
path's `std::partial_sort` order among EXACT float ties is unspecified
(libstdc++ heap selection), so no replacement can be proven bit-identical
without either replicating libstdc++'s heap internals or making tie order an
explicit (index-ASC) contract in the reference forward — a numerics decision
this row does not own. Reverted; the suite
(`tests/vllm/models/test_flint_router_topk_equiv.cpp`) records the finding.
Only the trivially-identical workspace hoist (score vector out of the token
loop, same `partial_sort`, same values) landed. Router shares stay 2.6-4.6%.

### Lever 3 (attributed, not touched): shared expert + combine

Sub-attribution above: the bucket is the shared expert's bf16 GEMVs, not a
scatter. A bf16-dot GEMV tier is a numerics change (new gate class) and a
separate row. No bit-identical lever exists for this bucket at batch 1.

## Serving A/B (interleaved same-binary, `VT_CPU_Q6K_DOT=portable` vs `neon`, 3-rep medians)

| threads | pool | portable tok/s (3 reps) | neon tok/s (3 reps) | median delta |
|---:|---:|---|---|---:|
| 16 | 16 (matched) | 5.12 / 5.10 / 5.94 → **5.12** | 5.40 / 5.44 / 6.01 → **5.44** | **+6.3%** |
| 32 | 32 (matched) | 5.73 / 5.68 / 5.81 → **5.73** | 5.76 / 5.92 / 5.83 → **5.83** | **+1.7%** |

Direction consistent in every interleaved pair (neon ≥ portable in 6/6 pairs;
16t deltas +5.5%/+6.7%/+1.2%, 32t +0.5%/+4.2%/+0.3%). Absolute levels sit
below the quiet-box records because of the peer-session contention recorded
above; the +6.3% @16t is consistent with the −21% in-situ lm_head phase over
its ~3.1% share.

## Residual (ceiling-free)

Attributed after this row @16t (neon arm): routed-expert loop 36.4% (own
grouped-GEMM row), sliding attention 36.0%, full attention 12.8%, shared
expert 7.8% (bf16-dot row), router 2.6% (tie-order decision owed), lm_head
2.6%. Gap to the 26.29 tok/s llama.cpp floor: ~4.8x at the quiet-box record,
OPEN, every remaining bucket named.

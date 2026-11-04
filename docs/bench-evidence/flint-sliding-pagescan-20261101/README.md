# FLINT-SLIDING-PAGESCAN — the residual sliding-phase seq growth attributed: it is the unclipped-window KEY count, not pages; the M-A hoist is implemented and measured perf-neutral

Row: `FLINT-SLIDING-PAGESCAN`
Matrix record: `KERNEL-ATTN-PAGED-SLIDING` (kernel-matrix)
Spec: `.agents/specs/flint-sliding-pagescan.md`
Issue: `.agents/issues/FLINT-SLIDING-PAGESCAN/ISSUE-LOCAL-01M6K7HXBVC1PMDGHT5WTVS803.md`
Date: 2026-11-04. Host: thalia (aarch64 128 cores). Governor **performance** AS
FOUND (unchanged). Idle gate (1-min load < 16, `ps aux --sort=-%cpu`) before
every measured leg; one post-fix 16t triple that started under a foreign burst
(1-min load 118.8) was re-run clean (`post-16-clean-rep*`; the burst triple is
kept as `post-16-rep*` — its numbers agree with the clean triple and are NOT
cited). Artifact: `/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`, sha256
`20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`.

Recipe (per leg) — identical workload both sides, `VT_FLINT_PHASE_TIMING=1`,
fresh build `/tmp/build-pagescan` (`taskset -c 64-127 -j 64`):

```sh
VT_FLINT_KEEP_QUANT=1 VT_FLINT_PHASE_TIMING=1 VLLM_CPP_CPU_THREADS=$TH \
taskset -c $ZONE ./examples/vllm-bench \
  --model /mnt/models/flint-GGUF/flint-Q4_K_M.gguf \
  --num-prompts 1 --input-len 8 --output-len 200 --concurrency 1 \
  --seed 0 --temperature 0 --ignore-eos
```

199 steady decode steps per rep; step s attends p+1 = s+9 keys. Early window =
steps 10–20, late window = last 14 steps (medians). Analyzer: `phase_stats.py`.

## Attribution verdict

**The residual ~1.11–1.14x growth is the UNCLIPPED-WINDOW KEY COUNT, not
pages. The window (left = 512) never clips below p = 513, so window-resident
keys grow 29 → 208 across the measured run; the growth is that legitimate key
term at its (token, head)-parallel floor. None of M-A/M-B/M-C carries it.**
Per the spec's stop condition this row delivered the diagnosis; the one
addressable M-A component (the per-key btab walk) was fixed and measured
perf-neutral — its share of the per-key cost is nil.

Per-mechanism evidence:

- **M-A (the per-key btab walk) — premise REFUTED, fix IMPLEMENTED and
  PERF-NEUTRAL.** The premise (cost tracks pages while keys stay bounded) is
  false in the measured range: keys are NOT bounded below seq 513, and keys
  grow ~7x (29 → 208) while pages grow only ~7/32x (~1 → 7). The measured
  growth tracks keys, not pages. Arithmetic: the kernel does 2 btab lookups
  per key per unit (K pass, V pass) — lookups per step grow exactly with keys
  (2·32·keys per layer), i.e. 1,856 → 13,312 across the run. The hoist
  (per-unit page-slice resolution + page-major walk, identical ascending key
  order, bit-identical by construction) removed the btab lookup and the
  div/mod from both inner loops: the per-key slope is UNCHANGED
  (0.0281/0.0287/0.0281 before vs 0.0285/0.0277/0.0268 after, 16t clean) —
  the per-key cost is the dot/exp/V math itself.
- **M-B (arena scatter) — REFUTED with the decisive regime.** `base-16-ol700`
  runs 190 steps PAST the window clip (seq ≥ 513): keys constant at 513 while
  pages touched stay ~17 and the arena keeps growing — and `attn_sliding` is
  FLAT: medians 49.63 / 49.73 / 49.82 ms at steps 520–531 / 600–611 / 680–694,
  clipped-regime slope −0.00001 ms/step (`base-16-ol700`), +0.00001 after the
  hoist (`post-16-ol700`). A scatter term growing with the arena would curve
  this upward; nothing grows. The `perf stat` legs (`perf-ol30`,
  `perf-ol210`: whole-process cycles/instructions/cache-references/
  cache-misses at output-len 30 vs 210) are recorded but SUPERSEDED —
  whole-process counters include model load, which dominates the miss count
  and is not attributable to the attention region.
- **M-C (timer scope includes KV append or RoPE) — REFUTED two ways.**
  Reading the scope (`flint.cpp:1063`, `ph_attn`): it wraps qkv projections,
  head norms, RoPE (sliding layers only), the per-layer slot mapping, the KV
  write, the paged kernel, o_proj, and the residual add. Every one of those
  except the paged kernel is O(batch) = O(1) per decode step at concurrency 1
  — nothing in the scope is O(seq) except the kernel itself. And the growth
  appears EQUALLY in `attn_full` (ratio 1.147/1.158/1.165 vs sliding
  1.139/1.140/1.144), whose layers run NO RoPE and the same O(1) wrapper
  terms — a RoPE/append-scope term would bend only the sliding phase.

## The fix (M-A, bit-identical)

`src/vt/cpu/cpu_paged_attn.cpp`, scalar body (the serving default — the NEON
lane is opt-in via `VT_CPU_PAGED_ATTN_NEON=1` and is untouched):

- The page-table slice for the unit's window (`page0..page1`) resolves ONCE
  per unit into per-unit scratch (`pages`, hoisted beside `probs`/`acc`).
- Pass 1 (K) and pass 3 (V) walk `jmin..jmax` PAGE-major: per page, the block
  id and the page base (`blk*kc_blk + g*kc_hd` / `blk*vc_blk + g*vc_hd`) are
  computed once; the key loop keeps only `off` and the element math.
- `j` visits EXACTLY the same ascending order with the same per-element
  operations, so every block id, offset, and float result is unchanged —
  bit-identical by construction. The dtype suite's windowed leg
  (`tests/vt/test_ops_paged_attn_dtype.cpp:352-359`) is the proof.

## Gates (all on the fixed head, build /tmp/build-pagescan, CPUs 32-39)

| gate | result |
|---|---|
| `test_ops_paged_attn_dtype` (bit-identity, INCL. windowed leg) | 172/172 assertions, 3/3 cases PASS |
| `test_ops_paged_attn` | 1838/1838 PASS |
| `test_ops_paged_attn_neon` | 3/3 cases, 15,200,901 assertions PASS |
| `test_flint_paged` | 208/208 PASS |
| `test_flint_gguf` keep-quant serving gate | PASS — 53/60 exact, 0 hard, worst 2.486 (UNMOVED) |
| MiMoV2 sink gates (`test_mimov2_forward` 178, `test_mimov2_w1` 51, `test_mimov2_w2` 67) | ALL PASS |

## Before / after (3-rep medians, idle-gated)

| metric | before (base-16/32-rep{1,2,3}) | after (post-16-clean/post-32-rep{1,2,3}) |
|---|---:|---:|
| attn_sliding early med, ms (16t) | 35.60 / 35.64 / 35.62 | 35.56 / 35.51 / 35.65 |
| attn_sliding late med, ms (16t) | 40.56 / 40.63 / 40.74 | 40.57 / 40.53 / 40.51 |
| growth ratio (16t) | 1.139 / 1.140 / 1.144 | 1.141 / 1.141 / 1.136 |
| per-key slope, ms/step (16t) | 0.0284 / 0.0287 / 0.0281 | 0.0285 / 0.0277 / 0.0268 |
| decode tok/s @16t | 7.35 / 7.36 / 7.35 (med 7.35) | 7.38 / 7.37 / 7.40 (med 7.38) |
| growth ratio (32t) | 1.101 / 1.105 / 1.111 | 1.105 / 1.082 / 1.106 |
| decode tok/s @32t | 6.95 / 6.89 / 6.94 (med 6.94) | 6.71 / 6.96 / 6.84 (med 6.84) |
| clipped-regime slope (ol700, steps 520–690) | −0.00001 ms/step | +0.00001 ms/step |

**The residual growth ratio does NOT drop: 1.14 → 1.14 @16t.** That is the
diagnosis landing: the growth is the key count (29 → 208), which no M-A/M-B/M-C
mechanism can remove — it plateaus at seq 513 by correct window behavior
(proven flat by the clipped-regime legs). The hoist is perf-neutral (+0.4%
tok/s @16t, within noise; −1.4% @32t med, within run-to-run noise) and is kept
as a bit-identical inner-loop simplification, not as a speed lever.

## Rejected / noted

- Whole-process perf stat as M-B evidence: superseded by the clipped-regime
  experiment (model-load misses contaminate whole-run counters).
- Reverting the hoist after the neutral verdict: kept — bit-identical,
  dtype-gated, removes two 64-bit divisions and a dependent load per key per
  pass from the inner loop; the measured record shows it is not a speed lever.
- M-B's invasive fix: NOT owed — the mechanism is refuted in the measured and
  the clipped regime; nothing to fix.

## Deviations / stop condition

- The residual attributes OUTSIDE M-A/M-B/M-C (the unclipped-window key
  growth, already named as the legitimate term in
  `docs/bench-evidence/flint-sliding-attn-20261031/`). Per the spec's stop
  conditions the row delivers the diagnosis and stops; the M-A hoist rides
  with the diagnosis as its refutation evidence, not as a claimed win.
- Row records (`KERNEL-ATTN-PAGED-SLIDING` kernel-matrix row,
  `CLAIM-FLINT-SLIDING-PAGESCAN`, and the spec's record/work-breakdown
  sections) were created by this row: the FLINT spec/issue rows were never
  registered in a matrix, and the claims gate requires a stable row ID.

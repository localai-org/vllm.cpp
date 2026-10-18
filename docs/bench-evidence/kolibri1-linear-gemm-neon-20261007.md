# Kolibri-1 CPU linear_gemm — the order-preserving NEON GEMM lever is already
# landed; measurement and the falsified rejection (MODEL-TEXT-kolibri-1)

Date: 2026-10-07. Host: aarch64, 128 cores, 255 GB RAM, Linux. Tree: pristine
`origin/main` d780204dd, worktree `/tmp/vllm-kolibri-neon-gemm`, branch
`row/kolibri-neon-gemm`, build Release with
`-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SERVER=OFF -DVLLM_CPP_BUILD_EXAMPLES=OFF`
(configure log: `-ffp-contract=off` is pinned project-wide,
`CMakeLists.txt:55`). All runs `VLLM_CPP_CPU_THREADS=8`.

## The question this note answers

A prior track rejected a "NEON gemm" lever for the Kolibri-1 decode path on
the argument that any re-vectorization reorders the bf16 accumulation chain.
That argument is correct for K-axis (reduction) vectorization only. The
order-preserving form — vectorize across INDEPENDENT OUTPUTS, keeping each
output element's K accumulation strictly sequential in the scalar order,
precision, and association — was proposed as the untried lever. This note
records the verification: **that lever is already landed on main, gated, and
is the default tier on this host.** The rejection's premise is falsified by
the tree; no new kernel is warranted and re-implementing one would duplicate
landed, gated work. The prior rejection lives in
`docs/bench-evidence/kolibri1-dequant-cache-negative-20261007.md`, which is
committed on branch `row/kolibri-neon2` (pushed to the fork, `bafb7e3a4`) and
not yet on `main`, so a `main` reader needs that branch to follow it.

## Where the lever lives

- `src/vt/cpu/cpu_matmul_elem.cpp` (landed 18094ee28, 2026-07-22,
  "perf(cpu): elementwise GEMM specialized + SIMD-vectorized, BIT-EXACT"):
  `Bt16Neon` (:155) accumulates 16 independent output columns as 4 NEON
  vectors; per lane the p sequence is strictly increasing, products are
  `vmulq` + `vaddq` (NEVER `vfmaq`), so every rounding point matches the
  scalar reference at `-ffp-contract=off`. `BtM4Neon` (:198), `Nk16Neon`,
  `NkM4Neon` cover the M-blocked and [K,N] orientations; the K tail (:183)
  stays scalar in p order.
- Reachability into the Kolibri-1 path: every `LinearBT`/`LinearBTRaw` call
  in `src/vllm/model_executor/models/kolibri1_forward.cpp` resolves through
  `vt::MatmulBT` → `MatmulBTKernel` (`src/vt/cpu/cpu_ops.cpp:442`) →
  `MatmulChunked` → `MatmulOneChunk` (:206), whose tier table selects the
  NEON family by default on aarch64 (`cpu_matmul_elem.cpp:460-486`,
  `VT_CPU_MATMUL_TIER=neon|portable|ref` forces a tier).

## Profile (VT_KOLIBRI1_PROFILE=1, 8 threads, decode bench, 68 forwards)

| stage | s | note |
|---|---|---|
| dequant_fp8_block | 41.9 | #1; already NEON (kolibri1_fp8_dequant.h) |
| moe_glue | 25.5 | CONTAINS nested dequant+gemm of routed experts (kolibri1_forward.cpp:335 scope wraps ExpertMlp); not separable |
| linear_gemm | 17.3 | the lever's stage, NEON tier active |
| attn_core | 5.6 | |
| lm_head | 1.7 | |

The task's two candidate stages overlap: `moe_glue`'s accumulator nests the
per-expert `linear_gemm`/`dequant` scopes, so `linear_gemm` is the named
stage and its remaining headroom is measured below. `moe_glue`'s separable
residual (the bf16 gather/scatter copies) is the follow-up.

## A/B: same host, same binaries from this tree, 8 threads

`test_kolibri1_decode_bench` (production shape 128 in → 64 greedy out), tier
forced per run. The CHAIN is byte-identical across all three tiers
(anchor 109726, chain alternates 101807/109726) — the bit-identity holds
end-to-end, not only in the unit tests.

| tier | wall s | prefill s | decode s | tok/s |
|---|---|---|---|---|
| neon (default) | 71.5 | 15.8 | 55.7 | 1.130 |
| portable (16-accumulator scalar C++) | 83.7 | 24.3 | 59.3 | 1.062 |
| ref (one serial f32 accumulator per output) | 267.7 | 152.6 | 115.2 | 0.547 |

- The landed NEON lever vs the TRUE scalar (ref tier): **2.07× decode tok/s**
  (1.130 vs 0.547) and **3.74× whole-run wall** (267.7 → 71.5 s, includes
  prefill).
- vs the portable 16-independent-accumulator tier (the pre-SIMD state of the
  art in this file, the best scalar tier): **14.6% less wall** (83.7 → 71.5 s)
  = **1.171× whole-run**, **1.064× decode tok/s** (1.130 vs 1.062).
- Host noise is real: an earlier same-binary neon run measured wall 81.0 s /
  0.96 tok/s under concurrent load; the A/B pairs above ran back-to-back
  idle. The tier ORDERING was stable in every pair measured.

W3 gate wall (neon, this host, isolated): 803 s process wall,
`test_kolibri1_w3` 900/900 assertions
PASS with the recorded ARGMAX chain and fingerprints (see Gates).

## Red/green and the mutation check

The bit-identity gate already exists and fails before any order break:
`tests/vt/test_ops_matmul_elem.cpp`, 689 assertions — raw-uint16 byte
compare vs `RefGemm` over every dtype × orientation × 14 shapes including
ragged M/N/K tails (`Shapes()` covers N = 1/7/16/17/33/48/64 and
K remainders 1/2/3), row-strided activations, cross-thread-count identity,
the forced-tier-is-the-run-tier check, and the x86/portable fallback via
`VT_CPU_MATMUL_TIER=portable` (689/689 green under it here).

Mutation check performed in this worktree, scratch only: swapping the two
`vaddq_f32` steps at p and p+1 in `Bt16Neon` (reordering the accumulation)
and rebuilding `test_ops_matmul_elem` → **3 test cases / 39 assertions RED**
(bit-identity, widening, and inf/nan cases all detect it). Restoring the
file byte-for-byte and rebuilding → 689/689 green again. The tests detect a
swapped accumulation order.

## Gates (this tree, this host, 8 threads)

| gate | result |
|---|---|
| test_ops_matmul_elem (neon default) | 689/689 PASS |
| test_ops_matmul_elem (portable fallback) | 689/689 PASS |
| test_kolibri1_dequant | 10/10 PASS |
| test_kolibri1 (186-assertion suite) | 186/186 PASS |
| test_kolibri1_w3 | 900/900 PASS, fingerprints match (first run OOM-killed under concurrent build load; isolated rerun clean) |
| test_kolibri1_decode_bench anchor | last token 109726, all tiers |

## Outcome

The order-preserving NEON GEMM is landed, reachable, gated, and worth 2.07×
decode tok/s over the true scalar (3.74× whole-run wall, prefill included) on
the production decode shape. The prior rejection is
closed by this evidence. Follow-ups recorded, not owed here: the separable
`moe_glue` copy residual, and the fp8-block dequant-cache question already
carried by kolibri1-dequant-cache-negative-20261007.md.

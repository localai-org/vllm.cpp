# Kolibri1 CPU decode GEMV dispatch batching (t32, NEON, 32 GiB dequant cache)

Row: `MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm`
Branch: `row/kolibri-gemv-batch` (base `row/kolibri-gemv-overhead` @ `467f26418`)
Date: 2026-10-15
Parent evidence: `kolibri1-gemv-overhead-20261015/report.md` (the attribution this
restructure was priced from)

## What landed

`vt::MatmulBTBatch` (OpId::kMatmulBTBatch, CPU provider, `src/vt/cpu/cpu_ops.cpp`):
N independent MatmulBT jobs in ONE `Threadpool::Run`. Each job's chunk grid is
built exactly as `MatmulChunked` builds it alone; every output element is served
by the same `MatmulOneChunk`, whose K reduction is local and strictly
sequential — bit-identical to the per-job loop by construction.

Call-structure changes (all behind `VT_KOLIBRI1_BATCH_GEMV`, default ON, `=0`
restores the old structure in the same binary):

- (a) routed experts: gate+up of all active experts ride one batch dispatch,
  down another — 18 kicks/layer → 2.
- (b) attention q/k/v: one batch dispatch — 3 kicks/layer → 1.
- (b sweep) shared expert gate+up: one batch dispatch.

## Bit-identity evidence

- Unit gate `tests/vt/test_ops_matmul_batch.cpp`: the batched run is
  BYTE-IDENTICAL (memcmp) to the per-job `vt::MatmulBT` loop across ragged
  decode shapes and worker counts {1,2,4,8,17}. Mutation-verified: perturbing
  the batch dispatch to swap two rows' outputs turns the gate red (observed:
  1 test case failed, 2 assertions failed; reverted).
- End-to-end: the decode bench greedy chain is `101807,109726,` alternating ×64
  (anchor last token 109726) before and after every stage — bit-identical.

## Measurement

Recipe: `taskset -c 32-63` (host load 17–143 contaminated unpinned runs), 32
threads, NEON paged attention, 32 GiB dequant cache, median of 3, interleaved
same-binary A/B (`VT_KOLIBRI1_BATCH_GEMV` 0/1).

| arm | tok/s runs | median |
|---|---|---|
| stage (a) off | 4.875 / 4.833 / 4.903 | 4.875 |
| stage (a) on | 5.030 / 5.056 / 5.098 | 5.056 (+3.7%) |
| +(b) off | 4.915 / 4.930 / 4.930 | 4.930 |
| +(b) on | 5.108 / 5.131 / 5.113 | 5.113 (+3.7%, +1.1% stacked) |
| final (+(b) sweep) off | 4.898 / 4.912 / 4.913 | 4.912 |
| final on | 5.112 / 5.097 / 5.125 | 5.112 (+4.1% over off) |

Pre-change unpinned baseline was 4.59 / 4.56 / 4.33 (median 4.56); the pinned
off-arm is the honest denominator.

Kicks (VT_GEMV_PROF, atexit window, per decode step):

| site | before | after |
|---|---|---|
| routed gate/up/down | 900 | 100 |
| attn q/k/v | 150 | 50 |
| shared gate/up/down | 150 | 150 (gate+up share one dispatch; the shared_gate window now also counts the silu + down calls) |
| GEMV total | 1301 | ~401 true pool kicks (451 window-counted) |

GEMV-stage wall in the profiled run: routed 30.8 → 23.5 ms/step (stage a);
whole GEMV stage 128.5 → 95.5 ms/step at the end.

## Priced vs measured

The attribution priced ~20 ms/step (11%) for (a)+(b)+(c). Stages (a)+(b) land
+4.1% (≈8 ms/step). The priced pool mixed true dispatch with load imbalance and
cache effects (the report's own caveat: "an upper bound on pure dispatch"); the
recoverable share is the smaller one.

## What remains

- (c) one sync per layer (persistent parallel region / `vt::FusedChain` across
  the layer's linears) is NOT landed: it is an op-dispatch architecture change,
  not a call-structure change, and cannot ride this lever. Remaining measured
  dispatch+idle is ~8 ms/step (GEMV wall 95.5 vs busy-summed/32 87.2), the pool
  it can still recover.
- Kernel rate (attribution finding 3): per-worker busy is still ~1.3 Gelem/s
  vs the 10.5 Gelem/s vbfdotq micro ideal — the bigger remaining headroom,
  gated on the bit-exactness policy, unchanged by this work.
- lm_head, attn_o, router remain single large calls (dispatch ~0.1 ms each);
  no batching lever left there.

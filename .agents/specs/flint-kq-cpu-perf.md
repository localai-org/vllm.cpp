# SPEC — `FLINT-KQ-CPU-PERF` — the batch-1 quantized expert compute class: descend from 1.20 tok/s toward the 26.29 floor

- Owning row: `FLINT-KQ-CPU-PERF`
- Kind: performance (measurement-driven compute-class descent)
- Issue: created in the same flow (row-scoped local issue)

## Baseline and target (both measured, both committed)

- Baseline: the keep-quant paged arm serves `flint-Q4_K_M.gguf` at
  **1.20 tok/s @16t** (2.46 @32t) — `docs/benchmarks/flint-kq-cpu.md`,
  CIQ compute class (one tiny kernel call per expert projection,
  activation re-quantized per call).
- Reference points on the same artifact/box: the **dequant-at-load bf16
  paged arm measures 3.25 tok/s @16t** — the packed-CIQ path is 2.7x
  SLOWER than dequantizing everything at load. The gated llama.cpp floor
  is **26.29 tok/s @16t** (llama-cpp-flint oracle).
- So the first finding of this row is already known: at batch 1, the
  fastest correct class may not be "keep everything packed" — it may be
  "dequant experts at load + one grouped bf16 GEMM", with packed-int
  dots only winning above a batch/expert-reuse threshold. Measure, do
  not assume.

## The three named levers (from the campaign's evidence)

1. **Grouped expert GEMM**: the in-tree
   `kMatmulBTQuantGrouped` / `MoeGateUpSwiGLUGrouped` path — one call
   for all selected experts instead of per-expert-projection launches.
2. **llama.cpp-style Q8_K block-dot gathers**: quantize each activation
   row to Q8_K ONCE, then integer vec_dot per expert row — the move
   behind mistral.rs's routed-phase 63→23 ms drop (reference context;
   their kernel map is recorded in the campaign notes).
3. **MoE gather/scatter fusion**: the gather/ExpertMlp/scatter sequence
   around the routed tier.

## Scope — measurement-driven descent, not a big-bang rewrite

Stage 1 (micro): build a scratch harness that times the three compute
classes for the routed-expert phase at batch 1 (and batch 8 for
threshold visibility) on the same weights: (a) current CIQ dispatch,
(b) dequant-at-load + bf16 grouped GEMM, (c) Q8_K activation block-dot
gathers (if the harness can reuse an existing NEON q8 kernel; do not
write a new quant format). Record a table. The dequant-at-load memory
cost (~2.9x the packed tree for Q4_K) is part of the record — the
packed tree exists to bound memory, so a win for (b) that costs 60+ GiB
RSS is a trade to adjudicate, not to hide.

Stage 2 (adoption): wire the winning class into the paged arm through
the SHARED seam (the existing dispatch path; a new compute class
registers there, no parallel path). The serving gate
(`test_flint_gguf` keep-quant gate) re-pins to the new class's measured
token record — the gate's band/contract follows the compute class,
re-adjudicated if the class change moves tokens (the adjudication
principle: quantized arms are measured records, and a compute-class
change is a new record).

Stage 3 (record): serving numbers at 16/32 threads next to the 1.20
baseline and the 26.29 floor; docs updated; the residual gap stays
open with the next lever named (no ceiling).

## Non-goals

- No TT-path change. No new quant format. No CUDA/GPU arm. No claim of
  reaching the floor — the floor is a direction, and llama.cpp's
  gather kernel is itself the named next lever after ours exhausts the
  three above.

## Red-first

The 1.20 tok/s record IS the red: the row's green is a measured,
reproducible improvement over it with the token gate in-band. Any
compute-class change must re-run the keep-quant serving gate and land
its re-adjudicated pin in the same commit (the gate is the correctness
contract; throughput never trades against it).

## Gates

check-agent-record, style, trailers, env-doc (if any new env), preflight
(named host dotnet failure), benchmark-index (if docs/BENCHMARKS.md
numbers update). Serving legs: bench zone, idle-gated, claims file,
governor logged. Unit gates for any new grouped/dot kernel: the existing
kMatmulBTQuant* suites.

## Stop conditions

- If stage 1 shows NO available class beats 1.20 tok/s at batch 1
  (i.e. the bottleneck is elsewhere — memory bandwidth, routing, the
  attention tail), record the profile split and STOP: the row delivered
  the attribution, and the next lever is a different row.
- If the winning class cannot hold the token gate in-band after honest
  re-adjudication, STOP: numerics first.

## Git integration

One pull request, `row/FLINT-KQ-CPU-PERF`, spec committed first. The
stage-1 harness may land as a scratch/typed tool or test fixture — the
measurement table it produces is the deliverable of stage 1.

## Outcome

Measured 2026-10-31 (evidence:
`docs/bench-evidence/flint-kq-cpu-perf-20261030/README.md`):

- The row's premise is falsified. The CIQ class is the fastest available at
  decode batch sizes on the aarch64 box: routed-expert phase 0.458 ms/layer at
  batch 1 vs 1.535 for dequant-at-load bf16 (4.394 vs 11.768 at batch 8). The
  "Q8_K block-dot gathers" lever is not a separate class to adopt — the CIQ
  kernel already quantizes the activation to Q8_K once per call and runs one
  integer `vec_dot` per weight row; what it lacks is a NEON dot (m=1 decode
  takes the scalar generic port), which is the named next lever, not a new
  quant format.
- The 1.20 tok/s baseline was threadpool oversubscription: the pool defaults
  to `hardware_concurrency()` (128) under a 16-cpu serving budget. With
  `VLLM_CPP_CPU_THREADS` matched to the cpus, the unchanged class serves
  5.51 @16t / 5.75 @32t (from 1.20 / 2.46), floor 26.29 / 26.38 — residual
  4.8x, open. Gate re-run byte-identical to the committed pin (53/60, 0 hard,
  worst 2.486): no re-adjudication, because no compute-class change landed.
- Rejected: dequant-at-load adoption — 3.3x slower at batch 1 AND +~71 GiB RSS
  (measured per-layer dequant cost, not estimated).
- Default: `VLLM_CPP_CPU_THREADS` keeps its documented
  `hardware_concurrency()` default; docs now carry the matched-pool serving
  recipe. Making the default affinity-aware owes its own spec and red-first
  gate (a product decision, not this row).

## Now

Row `DONE`: the measurement landed, the public record carries the new numbers
and class, the issue closed with the falsifying evidence. Next work is new
rows: the NEON `vec_dot_q4_K_q8_K` port (llama-cpp oracle mirror), the
non-routed 88% serving-time attribution, and the affinity-aware pool default.

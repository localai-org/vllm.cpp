# Pascal MMQ D4 activation correction

Issue: `ISSUE-LOCAL-01M6405SHVT2X5PWAXM1FJHXXG`.
Spec-first: `9fb365e29`. Base: `561ee7b38`. Date:29 October2026.

The exact-input diagnosis proves Q5_0/Q8_0 prefill uses the pinned CUDA oracle's
D4 activation representation: F32 scales per32, inverse-first rounding, signed
integer weights. Pin llama.cpp `62318cd57dc63900ecb153272c0b10f78550855d`.
Source chain: `ggml-cuda.cu:1955-2010`, `mmvq.cuh:3`,
`mmvq.cu:318-380`, `mmq.cu:355-445`, `mmq.cuh:60-76,564-581`,
`quantize.cu:457-550,574-602`. The executing oracle and exact-input history
are documented in `/tmp/lumen-smol-diagnosis.md`, section
`Exact remaining prefill quantization mismatch and correction boundary`.

The new Pascal hardware header enters through registered MatmulBTQuant.
It requires sm61/sm62, a compiled DP4A image,48KiB shared memory, M>8,
K divisible by32, contiguous activations and canonical original NK weights.
CPU/Q8-aligned/elementwise repacks do not enter. F32/BF16 carriers widen before
quantization and round only the final store. Packed weights remain resident.
M1/M8,sm60,Volta,modern CUDA,grouped/MoE and other formats retain their paths.
The shared format-neutral packer is available to the separately owned Q6
prefill repair; this change contains no Q6 prefill implementation.

## Red, green and mutations

The initial registered test passes M1/M8 controls, then fails Q8,M9,K32:
the old F16-scale scratch produces0 where D4 expects9.10777e-7
(`red-run.log`). The actual Q8 V projection also fails on identical oracle
inputs and unchanged packed weights: row0,col0 actual.024987612 versus
D4.024993284 (`actual-red-run.log`). No test calls an internal kernel.

The final synthetic suite passes80 cases: Q5/Q8,M1/8/9/16/128,K32/576,
ragged N7,F32/BF16 input/output. It includes zero and tiny activations,
zero/subnormal/signed weight scales and reciprocal-sensitive input bits
`0x3fa25c7f`/`0x3c7575a9`. The latter gives D4 product1.5 versus old-route
1.4999999. Independent scalar reference weights decode directly from original
packed bytes. The test uses a forward-error bound proportional to sum of
absolute block terms for valid reduction reassociation, plus final BF16 error.

Three independent scratch mutations compile and link, then fail:

- Narrow the D4 activation scale to F16: M9 tiny-vector output becomes0
  instead of-2.02914e-6 (`halfscale-run.log`).
- Use old inverse arithmetic: reciprocal-vector output-9.44472e-6 differs
  from-9.51437e-6 (`reciprocal-run.log`).
- Delete production D4 dispatch: the registered suite fails at the same
  tiny-scale guarantee (`dispatch-run.log`).

The production source remains unchanged during mutations. Restored green
passes all80 cases (`restored-green-run.log`). The bounded full quant suite
passes20 cases and254813 assertions (`suite-bounded-run.log`); the existing
published-MoE geometry stress case is excluded with its established bounded
command. No new tolerance or skipped assertion is added to that suite.

## Actual same-input projections

The approved checkpoint is
`bartowski/SmolLM2-135M-Instruct-GGUF` revision
`09816acd5d99df7be770d85ea30822623dab342c`,
`SmolLM2-135M-Instruct-Q4_K_M.gguf`,105454432 bytes,
SHA256 `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`.
The fixture manifest records each actual input/output capture and unchanged
packed tensor hash. Exact captured oracle inputs enter registered MatmulBTQuant
with the original tensor weights. Actual output comparison passes:

| Tensor | M,N,K | RMSE | Maximum absolute error |
|---|---|---|---|
| layer0 attn_v Q8_0 |16,192,576|4.76507e-9|5.96046e-8|
| layer0 ffn_up Q5_0 |16,1536,576|4.24459e-8|4.76837e-7|
| layer0 ffn_gate Q5_0 |16,1536,576|5.42358e-8|4.76837e-7|

Raw input and model files remain outside Git. Reproduce with
`test_cuda_pascal_mmq INPUT.f32 WEIGHT.packed ORACLE.f32 M N K IS_Q5`.
Fixture paths and hashes are in `actual-fixtures.json`. Output logs are
`attn_v-actual-green.log`, `ffn_up-actual-green.log`, `ffn_gate-actual-green.log`.

## Build and remaining gates

`/tmp/pascal-mmq-driver.py` extracts the existing Release compile commands,
substituting source roots and scratch outputs. NVCC12.4 compiles the changed
TU for sm60/sm61/sm70; all three images succeed in `quant-green-compile.log`.
Mutation builds target sm61. Thin archive `isolated.a` references existing
archives without a new full archive copy and excludes prior quant/attention
objects. Explicit reviewed Q5/Q6 decode, Tiny output-head fix, and attention
objects preserve integration behavior. GPU runs hold `/home/user/gpu.lock`.
No original build-cache object is replaced and no full build is claimed.

The operator reruns the actual public Smol p16,g16 gate twice with the candidate
`/tmp/pascal-mmq/actual-model`. It still emits357 at index8, matching the existing
unresolved whole-model mismatch. The operator log is
`/tmp/lumen-smol-d4q58-candidate.log`; warm time is.6866s. This correction is
necessary but does not establish whole-model token parity. Q6 prefill remains
separately owned. There is no correctness waiver or model completion claim.

Offline record validation passes ANCHOR-ROT=0; `git diff --check` passes.
Startup preflight exits124 after the45-second bound, not a green full gate.
Fresh independent review and operator gates remain required. No merge or
remote write occurs. The canonical issue remains OPEN until landing.

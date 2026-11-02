# Registered CUDA FP16 scalar operations

## Scope

This staged slice implements Wave 1 of `legacy-cuda-fp16-dense.md`.
The registered operations accept typed FP16 output through the selected provider.
The existing `OpProvider` layout and registration signatures do not change.
Automatic model dtype selection remains unreached until Wave 4 of BACKEND-CUDA-SM061.
ISSUE-LOCAL-01M641CAZNCNRRYAJF8DCV86R1 owns that wiring.
ISSUE-LOCAL-01M641XWGMDJA4CBEJC5FF91JM owns this slice.
ISSUE-LOCAL-01M6434Z4NXA27S36QYE271NDG owns the existing modern BF16 SiLU defect.
All issues remain open until landing.

## Source anchors

The primary revision is vLLM `a7c23ac96d7806e7c7e7d862eadbce5a33529b94`.

- `csrc/libtorch_stable/activation_kernels.cu:30-44,113-155,165-177` rounds the SiLU intermediate to the gate dtype before multiplication.
- `csrc/libtorch_stable/layernorm_kernels.cu:13-216,325-373` defines the scalar normalization and residual kernels.
- `vllm/ir/ops/layernorm.py:9-21,40-66` defines the native expected values and FP16 tolerances.
- `vllm/ir/tolerances.py` defines the F32 and BF16 default tolerances.
- `csrc/libtorch_stable/pos_encoding_kernels.cu:8-112` defines cache loading, both rotation conventions, and optional keys.
- `tests/kernels/core/test_layernorm.py:21-132` supplies the complete dtype, token, hidden-size, stride, residual, and weightless matrix.
- `tests/kernels/core/test_activation.py:137-221` supplies the SiLU matrix and exact tolerance.
- `tests/kernels/core/test_rotary_embedding.py:29-93` supplies the cache matrix. This port adds FP16 to its F32 and BF16 source contract.

## Harness adaptations

The test uses registered `vt` operations without PyTorch. Seed 0 selects `std::mt19937` normal inputs.
This preserves the distribution and seed intent. The generated values differ from PyTorch's generator.
Weightless normalization uses explicit unit gamma because the registered operation requires a weight tensor.
The native normalization reference keeps its unrounded F32 residual variance and upstream tolerances.
A separate exact fixture detects the executing CUDA kernel's typed residual rounding before variance.
The independent CUDA activation reference ports the primary scalar source with CUDA scalar wrappers.
It preserves the primary zero absolute and relative tolerance for all F16, BF16, and F32 cases.
The dense GEMM cases use small orientation and row-stride fixtures with FP32 accumulation.
Quantized embedding tests compare all 19 resident codecs against their existing CPU decoders, with both ID widths.
The upstream distributed embedding test requires masked rank-local vocabulary and rank-sum interfaces.
Those interfaces do not exist in `vt::Embedding`. This slice tests the applicable single-device gather contract.
It makes no tensor-parallel embedding claim.

## Environment and recipes

The runtime device is a Quadro P620, compute capability 6.1, with 2 GiB device memory.
All GPU commands take `/home/user/gpu.lock` through `flock`.
Compilation uses at most four simultaneous jobs.
`build.py` records the exact compile and link adaptation of `/tmp/lumen-build-legacy/compile_commands.json`.
The build uses CUDA 12.4, GCC 13 for CUDA, and the configured host compiler for C++.
The thin archive references unchanged base objects. Four missing loose objects come from the original archive.
`baselines.py` relinks the existing CUDA and legacy tests against the same modified objects.
The source base is `e1bea7113`. Preimplementation amendments are `d6813eadc`, `d390cc67e`, and `466a0682a`.
The first amendment has an invalid `Following-Agents-Protocol: yes` trailer.
The existing ref remains unchanged at the coordinator's instruction. Integration owes a correctly messaged new commit.

## Verification

`registered-red.log` records nine failing registered-operation cases before implementation.
`focused-green.log` records 11 passing cases and 332 passing assertions.
`modern-bf16-red.log` records exit 1 and 12 failing assertions among 72.
That experiment forces the existing modern SiLU source branch on the Pascal test device.
It demonstrates the source defect without claiming modern hardware runtime coverage.
Actual modern and Volta runtime measurements remain PENDING because those devices are unavailable.
The primary full model runtime is PENDING. The independent scalar reference does not replace that gate.
The startup preflight was interrupted before its compile gate completed and is not a passing full gate.
Its existing base failures include test registration, CPU floor harness, gate commands, controller preflight, and device leakage.
The final test-registration checker passes with the new CUDA-guarded registration.
The coordinator records independent fixes for device leakage and gate-command branding on its integration branch.
No checker baseline changes belong to this slice.

The full scalar matrix passes: 14 cases and 2084 assertions.
The explicit `VT_FUSED_TIER=1` fixture passes: 1 case and 4 assertions.
The final focused run passes: 11 cases and 332 assertions.
The existing CUDA operation suite passes: 14 cases and 451 assertions.
Its old all-FP16 refusal fixture now checks the still-unsupported mixed FP16 and BF16 combination.
The fresh current-source bounded quant suite passes: 20 cases and 254813 assertions.
The published GLM MoE geometry is excluded because it cannot fit this 2 GiB device.
An earlier cached test object produced stale Q5_0 expectation failures. Recompiling the current test resolves them.
A concurrent executable relink invalidated one bounded comparison attempt with a permission error. That attempt supplies no gate evidence.
The recorded recipe uses a distinct executable path per suite to prevent that collision.
The legacy matrix suite passes 28 registered workloads.
The synthetic legacy public API suite passes all nine CPU and CUDA comparisons.
The native EXL3 legacy refusal suite passes. The existing standalone intrinsics binary passes 4096 signed and unsigned DP4A cases.
The provider suite passes: 14 cases and 545 assertions.
The real-artifact C API harness returns 77 without a checkpoint argument. Its model gate remains PENDING for Wave 4.
All six changed CUDA translation units compile for sm60, sm61, and sm70.
The two SiLU translation units also compile for sm80 and sm90. Compilation is not hardware runtime evidence.
Two DFlash citations in `.agents/kernel-matrix.md` move with `cuda_ops.cu`.
The rows are KERNEL-ATTN-DFLASH-BLOCK and KERNEL-ATTN-DFLASH-PAGED-BLOCK. Unrelated rows remain byte-identical.
The original independent review found a test-matrix gap and stale comment.
The scoped repair review passes at `4c416296c`, resolving both findings.
The operator rerun passes 14 cases and 3428 assertions. Full preflight is not green:
the original run was interrupted before compilation; the unchanged standalone
checker completes with 44 of 829 compilation failures. See the
[final gate disposition](final-gates/README.md) for exact failures, later integrated
record fixes, retained review reports, source hashes and unresolved model gates.

## Implementer mutation evidence

Seven scratch mutations each exit 1 through the registered operation tests:
selected-provider admission, deleted FP16 reverse-cast registration, reverse-cast payload format,
residual rounding before variance, output payload format, merged SiLU rounding, and split SiLU rounding.
`mutations.py` records the exact source edits, link recipe, and focused cases.
The tracked source remains byte-identical because mutations use scratch copies.
These results supplement the fresh review. They do not replace independent mutation review.
`architecture-compiles.json` records the object hashes and matching source hashes for each architecture compile.
The final preflight initially saw an undeclared role marker. The helper claim now exists.
`role-only.log` records the corrected role gate. The original full preflight keeps its original result.

# FP16 Wave 1 independent review

Verdict: FAIL. One P2 upstream test-port finding requires a fresh implementer.
No numerical product defect was found in the tested Wave 1 operations.

Reviewed immutable head: 6c785c4a83b1934a276c74cc4444ee1a24976185.
Product commit: 077772ef591cb71769a8bf85714b7e0c18101c12.
Parent spec/base: e1bea7113; preimplementation amendments d6813eadc, d390cc67e, 466a0682a.
Scratch: /tmp/lumen-pascal-kernel-review, detached at the reviewed head.
Evidence and runnable review scripts: /tmp/fp16-independent-review.
The implementer's subsequent evidence-only commits were not substituted for this immutable review head.

## Findings

P2 — Preserve the upstream F32/BF16 cached RoPE matrix in the committed regression test.
Location: tests/vt/test_cuda_fp16_scalar.cpp:306-348, especially :316 and :327-328.
Pinned tests/kernels/core/test_rotary_embedding.py:37 enumerates F32 and BF16.
The spec requires preserving upstream dtype parameters and adding source-grounded FP16 coverage.
The port instead hardcodes FP16 through input, cache, allocation, and expected output.
It therefore omits all 192 original dtype/shape combinations, while covering the 96 added FP16 combinations.
Existing smaller CUDA controls do not preserve that full upstream matrix.
A fresh implementer must loop F16/BF16/F32 over the same existing shape matrix, preserve old-mode checks, and update evidence.
Do not change the public operation's rank-3 ABI. The upstream alternate flattened representation requires a documented reshape/view harness adaptation; it does not authorize adding a rank-2 product arm.

This is mutation-proven, not a hypothetical coverage concern.
Scratch mutation removed only the BF16 RopeFromCache production call site in src/vt/ops.cpp.
The committed cached RoPE test still passed (exit 0), while the independent all-format matrix failed (exit 1).
Logs: BF16-gap-committed.log and BF16-gap-independent.log.
The unmodified independent all-format matrix passed all 288 dtype/shape combinations.

P3 — Remove the stale gamma-only assertion beside the expanded FP16 load.
Location: src/vt/cuda/cuda_ops.cu:77-79.
The comment says FP16 is admitted only as gamma and that activation/output remain F32/BF16.
That statement is false after the new Tin/output FP16 dispatches below it.
Replace only the invalid current-state claim; preserve useful historical provenance.
No separate numerical defect is implied by this comment.

## Independently executed verification

Fresh source compilation and linking used the configured CUDA12.4/GCC13 legacy recipe.
All eight changed C++/CUDA product objects, scalar tests and reference were compiled from the immutable scratch source.
An independent thin archive references unchanged base objects; only four missing base members were extracted.
The implementer's binaries and mutable caches were not used as review gate results.
CUDA compilation used two concurrent jobs, coordinated with the implementer's two jobs, never more than four total.
Every GPU command ran through flock /home/user/gpu.lock on the actual Quadro P620 sm61.

- Full scalar suite: 14 cases, 2084 assertions, all passed, no skipped cases.
- Focused scalar suite: 11 cases, 332 assertions passed; the three expensive matrix cases were deliberately excluded only in this focused rerun.
- Explicit VT_FUSED_TIER=1 residual-norm fixture: 1 case, 4 assertions passed.
- Independent supplementary driver: 5 cases, 1171 assertions passed. Includes all 288 RoPE dtype/shape combinations, separate norm/residual row strides, all 65536 FP16 reverse-cast words with exceptional-value classification/sign checks, FusedChain SiLU/RoPE recipes, selected-provider wrapper refusal and capability-cache invalidation.
- OpProvider size/alignment and all four field offsets statically match a legacy four-field struct. Existing aggregate initialization compiled unchanged.
- Existing CUDA operations: 14 cases, 451 assertions passed.
- Fresh current-source bounded quant: 20 cases, 254813 assertions passed. The published GLM MoE geometry remains explicitly excluded for the 2 GiB allocation limit; this is not a full unbounded quant verdict.
- Existing provider suite: 14 cases, 545 assertions passed.
- Legacy registered matrix suite: 28 workloads passed.
- Legacy synthetic public API suite: all nine CPU/CUDA comparisons passed.
- Legacy DP4A intrinsics: 4096 signed/unsigned cases passed.
- Native EXL3 pre-sm80 named refusal suite passed.
- check-test-registration.py passed; git diff --check passed.

All six changed CUDA translation units independently compiled for sm60, sm61 and sm70.
The merged/split SiLU translation units also independently compiled for sm80 and sm90.
cuobjdump verified the compiled architecture images in all 16 additional architecture-specific objects.
The sm61 runtime objects were independently freshly compiled as well.
The modern SiLU source now rounds the intermediate on every architecture, matching the pinned primary scalar and packed source.
F32 arithmetic and the Pascal/Volta BF16 branch preserve their prior scalar rounding.
Compilation establishes no unavailable sm60/sm70/sm80/sm90 runtime result.

## Independent mutation guarantees

All mutation variants were independently compiled in external scratch files.
The production scratch checkout was never edited by a mutation.
The following 21 controls each exited 1 with the focused test failure intended:

1. Disable registered CastF32 production dispatch.
2. Disable registered Add production dispatch.
3. Disable registered RmsNorm production dispatch.
4. Disable registered ordinary Embedding production dispatch.
5. Disable registered EmbeddingQuant production dispatch.
6. Disable registered merged SiluAndMul production dispatch.
7. Disable registered split MoeSiluMul production dispatch.
8. Disable registered direct RopeNeox production dispatch.
9. Disable registered cached RopeFromCache production dispatch.
10. Disable registered Matmul production dispatch.
11. Disable registered MatmulBT production dispatch.
12. Disable registered Tier1 FusedChain production dispatch.
13. Disable the production FusedChain composite route.
14. Let the selected-provider capability query inherit a different provider's mask.
15. Store BF16 payloads through the FP16 scalar output pointer.
16. Remove FP16 residual narrowing before variance.
17. Remove merged SiLU intermediate narrowing.
18. Remove split SiLU intermediate narrowing.
19. Decode FP16 reverse cast as BF16 at identical byte width.
20. Misroute MatmulBT transpose orientation.
21. Store quantized embedding output as BF16 at identical byte width.

Three additional controls failed in the independent supplemental driver:
selected-provider wrapper refusal (the incapable provider must not be entered),
FusedChain's production SiLU call site, and FusedChain's production cached RoPE call site.
The separate BF16-cache mutation above proves the remaining committed-test gap.
Mutation sources, compile/link logs, focused logs and mutation-results.json are retained under the evidence directory.
A first quantized-store mutant did not compile because that translation unit lacks the named BF16 intrinsic; it supplies no mutation evidence.
The corrected runnable mutant uses the existing BF16 store specialization and failed the exact half-output checks as intended.

## Source authority, restoration and limits

Primary pin: a7c23ac96d7806e7c7e7d862eadbce5a33529b94.
Preparation independently fetched and byte-checked ten pinned upstream files, plus read IR norm/tolerance and scalar-conversion sources.
The exact SiLU matrix uses the separately compiled source port of primary activation_kernels.cu:30-44,165-177.
Residual tests distinguish native IR tolerance reference from the executing CUDA scalar-rounded residual-before-variance contract.
Provider review found no OpProvider layout/C++ signature break, new global FP16 admission, stale selected-provider cache result, or demonstrated fallback defect.
The third-party higher-priority incapable provider is refused before its body executes; disable/re-enable and device-cap invalidation controls pass.

source-byte-clean.json proves all 13 scoped source/test files match git show at 6c785c4a8 byte-for-byte.
The complete tracked tree has no diff from that head, and git status --porcelain is empty.
After all mutations, the clean focused 11/332, explicit Tier1 1/4, and independent 5/1171 drivers were rerun and passed.
All unmodified tests above ran against clean immutable objects; mutation executables have distinct names and never replace the scalar/baseline executables.

The original implementer's full preflight is still pending and is not accepted as green by this report.
Known base checker repairs remain coordinator-owned; no checker was weakened.
Full primary FP16 model runtime is unavailable on the current sm61 lane and remains PENDING.
No primary runtime, real-model token gate, model default enablement, performance ratio, memory acceptance, or unavailable-board runtime PASS is claimed.
Automatic model dtype/loader propagation remains the staged Wave4 obligation of BACKEND-CUDA-SM061 and ISSUE-LOCAL-01M641CAZNCNRRYAJF8DCV86R1, explicitly named in the product commit body and spec.
Fresh repair, fresh scoped review, original full-gate evidence, and operator rerun remain required before integration acceptance.

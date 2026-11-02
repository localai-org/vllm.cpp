# Pascal and Volta enablement

Row: `BACKEND-CUDA-SM061`
Related rows: `BACKEND-CUDA-SM060`, `BACKEND-CUDA-SM070`.
Issue: `ISSUE-GH-146`.

## Now

ACTIVE. User approved implementation on 29 October 2026. Base: `db1d71c9b7352a07983b408b2e755e853b4534f0`.

## Scope

Implement production CUDA execution for Pascal 6.0, Pascal 6.1, and Volta 7.0. Preserve existing public interfaces and newer GPU dispatch. Optimize Pascal using the local Quadro P620, 2048 MiB, sm_61, driver 580.178.04. CUDA 12.4.131 is installed. Hardware access is approved. Volta hardware validation and optimization remain TODO until the GPU arrives. P100 runtime validation remains pending without that hardware.

## Upstream chain

Use `.agents/upstream-sync.md` for the primary pin and `.agents/oracles/llama-cpp.md` for the registered secondary pin. Pinned vLLM excludes Pascal from its CUDA target list. Volta is toolkit-dependent: `CMakeLists.txt:115-132` includes 7.0 with CUDA 12.4 and excludes it with CUDA 12.8 or newer. `vllm/platforms/cuda.py:256-263` selects FP16 for compute 6.x and 7.x. Volta primary-oracle runtime eligibility remains PENDING until hardware arrives. Read the matching pinned llama.cpp CUDA source, including capability dispatch, BF16 conversions, float GEMM, quantized GEMV, and attention. Re-anchor the superseded `237ad9b96` citations in `cuda-arch-breadth-fp16.md` before relying on them. The executing chain must cite actual source paths and lines.

## Our baseline

`CMakeLists.txt:202` selects architecture 121a by default. `cmake/CudaArchFeatures.cmake` resolves per-source architecture features. `src/vt/cuda/cuda_paged_attn.cu` contains guarded BF16 WMMA bodies and scalar fallbacks. `src/vt/cuda/cuda_matmul.cu` routes BF16 work to cuBLASLt, which requires a real legacy execution path. Existing historical breadth work is scoped in `cuda-arch-breadth-fp16.md`; `ISSUE-GH-146` remains open.

## Design

Use capability-based dispatch behind the existing CUDA backend. Add architecture-specific files for legacy hardware where new kernels are required. Route all model execution through the existing C API, loader, ModelRegistry::Forward, attention block, fusion, merged GEMM, and sampler. Preserve existing tensor storage formats unless an explicit documented adaptation is unavoidable. Keep all current `vllm_*` symbols, flags, configuration keys, filenames, and defaults on newer GPUs.

Audit and fix compile and execution blockers in the portable CUDA translation units. CUDA 12.x accepts the requested legacy targets; diagnose incompatible newer targets or instructions explicitly. Do not claim runtime support from successful compilation alone. Add narrow refusals naming missing quantized operations where a model arm cannot run.

## Tests to port

Port the applicable pinned-oracle capability, conversion, matrix, quantization, and attention cases with revision anchors. Add production-entry-point regression coverage that reaches the selected old-architecture implementation. Host-only dispatch tests supplement actual GPU tests. Record failing tests before code, focused green, emitted SASS, and production-call-site mutation failures.

## Gates

Full scoped gate: legacy CUDA configure/build with tests enabled; focused CUDA regression tests on P620; CMake architecture-selection tests; existing CPU and CUDA focused tests for changed operators; `scripts/agent-preflight.sh --quiet` for repository checks. Run builds with at most four jobs. Record exact commands and unchanged baseline failures separately.

Build sm_60, sm_61, and sm_70 with CUDA 12.x and inspect emitted device code. Execute all GPU commands under `${GPU_LOCK:-$HOME/gpu.lock}` on this non-fleet local device. Never bypass a fleet lease if the selected device changes.

Use real pinned checkpoint weights fitting 2 GiB including KV cache, activations, scratch, and runtime overhead. The initial candidate is OPT-125m if reachable through the current public entry point; a supported smaller dense checkpoint is an allowed builder choice. A quantized small Llama or Qwen checkpoint covers a compressed arm when authorized storage and downloads are available. Record artifact revisions, SHA256, memory, prompt, and all output tokens. Model downloads await explicit approval and storage choice. No synthetic fixture substitutes for this gate.

Declare workload axes before optimizing: batch 1, concurrency 1, prompt lengths 16 and 128, generated lengths 16 and 64, greedy sampling. If model limits or CLI reachability require adaptations, record the exact reason before execution. Correctness must pass first. Compare matching pinned-oracle execution on the identical device, artifact, and workload. Required performance floor: throughput ratio >=1, latency ratio <=1, memory ratio <=1 on every declared axis, reproduced on an idle device with the same binary. Unmet axes remain open, with a next traceable hypothesis. No ceiling claims.

## Dependencies

Pinned source checkouts and model storage values are not configured. Source downloads needed to read the registered oracle are authorized task preparation; large checkpoint downloads still need user approval. Do not copy another developer's paths. The GPU is small, so broad model support cannot be inferred from a single fitting checkpoint. BF16, half, atomic, shared-memory, and tensor-core assumptions need separate inspection.

## Work breakdown

1. Establish failing legacy build and production execution tests.
2. Implement the minimum complete legacy operator paths and safe dispatch.
3. Build all three architectures, run Pascal gates, and optimize measured bottlenecks.
4. Document supported arms, evidence, and exact outstanding hardware TODOs.

## TODO after Volta GPU delivery

- Record GPU identity, VRAM, driver, and toolkit.
- Run real-checkpoint generation and token correctness through the public API.
- Confirm FP16 tensor-core dispatch and memory formats with source and trace evidence.
- Optimize attention, prefill GEMM, and decode GEMV on actual hardware.
- Measure throughput, latency, and peak memory against the pinned oracle on matching workloads.

Volta stays build-verified and runtime-unverified until these gates run.

## Integration and stop conditions

One issue-owned branch and one PR carry the spec before implementation. No push or merge is authorized. Independent implementer and fresh mutation reviewer are mandatory. The operator reruns the scoped gates. Missing external resources leave the affected gate PENDING, without stopping independent implementation or local verification. Stop acceptance on correctness failures or invalid GPU exclusion.

## Evidence

Implementation and review evidence belongs in `.agents/evidence/pascal-volta-enable/`.

## Pending gates

- Owning issue `#146`: pinned real-checkpoint correctness and all declared performance axes. Model storage and downloads await developer approval.
- Owning issue `#146`: P100 runtime validation and Volta hardware validation, including the toolkit-compatible pinned vLLM primary path.
- Owning issue `#146`: native EXL3 GEMM and fused MoE for Pascal and Volta. The current Tensor Core kernels require sm_80 and refuse older GPUs. This does not refuse the portable GGUF k-quant arm.

## Implementation decisions

CUDA 12.4 cuBLASLt executes BF16 storage on P620 using CUDA-core float GEMV and `magma_sgemmEx_kernel`, confirmed by an Nsight Systems CUDA trace. Preserve that production path instead of adding a speculative replacement. The demonstrated runtime gap is batched F32 inputs with BF16 output. Only that lane on GPUs below sm_80 uses the new tiled CUDA fallback. Its FP32 accumulator follows the input type and stores BF16 directly, without a wider persistent weight or output buffer.

Pinned llama.cpp `common.cuh:712-753` supplies the exact sm_60 DP4A emulation. The unsigned overload additionally preserves EXL3 byte sums. Both overloads accumulate modulo 2^32. The native DP4A instruction remains selected on sm_61 and newer.

## Port map

| Pinned source | Local destination | Adaptation |
|---|---|---|
| llama.cpp `62318cd57`, `ggml/src/ggml-cuda/common.cuh:712-753` | `src/vt/cuda/cuda_legacy_intrinsics.cuh` | Exact signed-byte DP4A emulation on sm_60; unsigned byte sums preserve EXL3 decoding. |
| llama.cpp `62318cd57`, `ggml/src/ggml-cuda/ggml-cuda.cu:1672-1690` | `src/vt/cuda/cuda_legacy_matmul.cuh` | FP32 compute adaptation for the measured cuBLASLt F32 batched-to-BF16 gap; preserves tensor storage and shared dispatch. |
| vLLM `a7c23ac96d`, `csrc/libtorch_stable/activation_kernels.cu:30-44,165-167` | `src/vt/cuda/cuda_ops.cu` | Legacy SiLU rounds its intermediate to input scalar type before multiplying, matching the scalar and packed upstream chain. |
| Existing guarded local WMMA bodies and native EXL3 dispatch | `src/vt/cuda/cuda_paged_attn.cu`, `cuda_gdn.cu`, `cuda_exl3.cu` | Guard file-scope aliases; refuse unavailable EXL3 tensor instructions before launching. |

## Owed

No rowless issue is transferred to this row. The row-owned issue and Pending gates above retain all incomplete work.

## Risks/decisions

Preserve the existing BF16 cuBLASLt lane because the actual P620 trace confirms it executes. Restrict the F32-to-BF16 batched fallback to older capabilities and the measured unsupported lane. Guard native tensor instructions and reject unsupported EXL3 operations before launch. Runtime results on P620 do not validate Volta or P100. Correctness and performance debt remain explicit in Pending gates.

## Production attention selection expansion

Public-loader regression on P620 on 29 October 2026 fails before generation:
`FLASH_ATTN: compute capability not supported`. This is tracked by
`ISSUE-LOCAL-01M63MWPJR5Q5R2D24BDRS5RA5`.

Keep upstream FLASH_ATTN's sm_80 minimum intact. Add a separate registered
`CUDA_LEGACY` descriptor in a new hardware-specific source, with the existing
NHD cache geometry and dense decoder-only portable kernels. The CUDA platform
priority adds this descriptor only for major 6 or 7; its capability predicate
accepts exactly sm_60, sm_61, and sm_70. Newer priorities remain byte-identical.
The backend advertises only the BF16/F32 storage exercised by portable kernels,
auto/BF16 KV storage, supported head geometry, and decoder attention. It does
not advertise FP16 tensor-core optimization, MLA, sparse, sink, or quantized KV
capabilities. All execution continues through the existing runner,
`dense_attn::AttnBlock`, `vt::PagedAttention`, cache, and sampling seams.

Secondary source: pinned llama.cpp `ggml/src/ggml-cuda/fattn.cu:757-775`
selects portable tile/vector kernels as well as tensor instructions. This is a
local descriptor adaptation for kernels already present here; no claim is made
that vLLM implements a Pascal backend or that our descriptor ports FlashAttention.
The actual model storage contract remains unchanged by selection.

The synthetic public CAPI fixture must load and generate the same greedy tokens
on CPU and CUDA for prompt lengths 1, 16, and 37, capture finite full logits,
and refuse host reference fallback. Delete the legacy priority or registrar in
a scratch build and require public load to reproduce the refusal. Host tests
pin exact allowed capabilities and unsupported modes, and preserve sm_80+
selection. This supplements, and never substitutes for, published-model gates.

## Gate record reconciliation

The shared spec supplies runnable gates for `BACKEND-CUDA-SM060`,
`BACKEND-CUDA-SM061`, and `BACKEND-CUDA-SM070`. The exact runnable inventory
must include all three rows after their transition to `ACTIVE`. Keep the
classifier and its exact-set assertion unchanged. A campaign record commit
names each related row so the live audit can identify the same unmerged work.
This provenance records shared sm_60, sm_61, and sm_70 build work. It does not
claim P100 or Volta runtime validation, or real-checkpoint performance.

Repair issue: `ISSUE-LOCAL-01M63NAHP61SNGD5ZBQM9BGCA8`.

## Legacy prefill shared-memory repair

`ISSUE-LOCAL-01M63NES5TYK3E3WH5B50XVAPP` records the independent P620 failure
for a valid head dimension of 256. The portable prefill stages 32 keys and
values as F32, requiring 65536 bytes against the queried 49152-byte limit.

Keep the existing portable online-softmax kernel and its key accumulation order.
On a valid pre-sm_80 device, halve the runtime key tile until both F32 staging
buffers fit the queried per-block shared-memory limit. Preserve the default tile
on sm_80 and newer. The existing capability probe and launch guard remain the
source of the hardware limit. Pinned llama.cpp `fattn.cu:757-775` at
`62318cd57dc63900ecb153272c0b10f78550855d` selects tile/vector implementations
for non-MMA paths. This repair adapts the already executing local tile kernel.

Parameterize the existing synthetic public-loader fixture with an optional head
dimension, preserving the original 64 default. Exercise 64, 128, and 256 heads
through CPU and CUDA public completion for prompt lengths 1, 16, and 37.
The model width is 256, so a head dimension of 192 cannot define an integral
head count and is not a valid variant of this fixture. Capture finite full
logits, equal greedy tokens, and no CPU reference fallback. Reverting the tile
selection must reproduce the public completion failure. Real-checkpoint gates
remain separate and pending. Full sm_60/61/70 build and operator verification
remain required after this repair is integrated.

## Legacy metadata regression repair

`ISSUE-LOCAL-01M63NZRF5VEMZHJ9F3NQM27FF` records five mutation gaps in the
registered `CUDA_LEGACY` metadata tests. Keep production predicates unchanged.
Exercise valid head dimensions 8 through 256 and refuse nonpositive, unaligned,
and oversized dimensions through the production selector. Pin block multiples
of 16, the inherited zero-as-unspecified rule, explicit auto and bfloat16 KV
acceptance, KV connector refusal, and invalid cache block allocation refusal.
Keep legacy capability, dtype, attention mode, and newer GPU controls.
Compile the complete host selector suite and mutate each claimed metadata
predicate in scratch objects. Require each mutation to fail the focused test.
This host gate supplements the public CUDA execution and real-model gates.

## Final local verification

The complete Release build and final independent local gate pass at combined source `011dbdc97`. See [final verification](../evidence/pascal-volta-enable/final-verification/README.md). Published-checkpoint correctness and performance remain PENDING approved weights. P100 runtime and Volta runtime remain PENDING hardware. Volta optimization remains TODO. This evidence does not move the row to DONE or close its issues.

## Quantized dense projection shards

`ISSUE-LOCAL-01M63VRMM7HCJJ1MN1QSEVT88E` owns the dense projection representation gap exposed by the real Llama GGUF gate. Keep each Q, K, V, gate, and up shard in its original block encoding. The published TinyLlama artifact has Q4_K Q/K and Q6_K V. SmolLM2 has Q5_0 Q/K and Q8_0 V. A homogeneous merged tensor cannot represent either group.

Append optional `qkv_proj_parts` and `gate_up_proj_parts` owners to the shared dense weight container. Empty parts preserve every existing path. The Llama GGUF adapter alone populates the new parts. The adapter owns inverse Q/K row permutations, moving whole packed rows rather than decoding blocks.

Add a dense realization of `vt::MergedGemmGroup` with contiguous per-shard products. Route each contraction through `vt::MatmulBT`, which already selects registered quantized operators by weight dtype. QKV has arity 3 and no epilogue. Gate/up has arity 2 and the existing split SwiGLU epilogue. A new `MlpGateUpMethodBase` method owns the gate/up products and BF16 activation boundary. The existing attention block consumes Q/K/V products. Do not add a parallel model forward or GPU-specific model branch.

Pinned llama.cpp `62318cd57` executes original quantized matrices through `llama_model_loader` and the `GGML_OP_MUL_MAT` CUDA chain. The existing shared `MatmulBTQuant` implementation already ports those block-dot semantics. This slice changes representation and wiring, not quant arithmetic. Test heterogeneous shard products and gate/up equivalence through the methods, then require the Llama public real-model gate to reach the parts. Deleting the production parts call must fail its focused regression. Compare stored bytes and dtypes with the checkpoint. Full real-model correctness and performance remain open until measured.

## Legacy split SwiGLU input-width boundary

`ISSUE-LOCAL-01M63W80J6R0M1JXTQ5P1ENWGN` owns the split SwiGLU rounding mismatch. Pinned vLLM `a7c23ac96d` activation kernels return the gate scalar type from SiLU before multiplying the up scalar. The CPU split kernel and the repaired legacy merged kernel implement that boundary. On pre-sm75 devices, narrow CUDA `MoeSiluMul`'s SiLU result through its gate input type before multiplication. Preserve sm75 and newer behavior. Add a registered-op BF16-gate, BF16-up, F32-output comparison against CPU, then disable the narrowing in a scratch production CUDA translation unit and require failure. This supplies the numerical contract for the new quantized shard MLP method without copying and concatenating activation buffers.

## Llama GGUF adapter for the real-model gate

`ISSUE-LOCAL-01M63VJCK05BTSAMDXVWT60E2T` records the public loader refusal for
the approved SmolLM2-135M and TinyLlama-1.1B checkpoints. Implement an explicit
Llama GGUF adapter into the existing registered Llama model and shared dense
forward. Build a Llama HF-shaped config from the llama metadata prefix. Reuse
the existing dense GGUF tensor ownership, quantized residency, merged QKV, and
merged gate/up loader through a named shared helper. No Q/K norm is introduced.
Undo the converter's Llama Q/K row permutation before the shared HF NeoX RoPE
path. Preserve tied output detection and GQA geometry.

Pinned source anchors: llama.cpp `62318cd57`, `conversion/llama.py:175-181,
255-264` converts HF Q/K rows into interleaved GGUF order. The inverse maps
HF row `a * D/2 + b` from GGUF row `2*b+a`, independently for each head.
vLLM `a7c23ac96`, `vllm/model_executor/models/llama.py` owns the shared QKV,
SwiGLU, RMSNorm, and HF RoPE behavior. The first-party GGUF plugin remains the
primary GGUF reference where it executes. Pascal comparison uses the pinned
llama.cpp secondary because the pinned vLLM excludes Pascal targets.

Initial accepted GGUF geometry is the plain full-RoPE Llama family represented
by the two approved checkpoints. Refuse scaling metadata and rope-frequency
sidecars, Q/K norm tensors, and bias tensors with precise missing-arm messages,
rather than silently accepting unimplemented semantics. These broader arms
remain owed under this issue. Existing HF Llama and Qwen3 paths stay unchanged.

The failing test enters through public `vllm_engine_load` and completion using
the approved real artifacts. Add a focused deterministic adapter test for config,
GQA, tied output, inverse Q/K rows, quantized ownership, and refusal cases.
Mutation removes the production dispatch and separately the Q/K inverse and
requires focused tests to fail. Build changed production translation units with
the existing Release recipe. Run real-model public completion on P620 under the
GPU lock, then let the operator run identical pinned-oracle token gates.

## Published GGUF tokenizer repair

Issue: `ISSUE-LOCAL-01M63W5W3MTWY1KXRRD77FBG5D`.
The approved SmolLM2 checkpoint carries `smollm`; TinyLlama carries the `llama`
SentencePiece family. Load both through the existing tokenizer surface. Pin
llama.cpp `62318cd57` `src/llama-vocab.cpp:111-246,362-371,2097-2107,2564-2595,3579-3633`.
SmolLM isolates each Unicode number before the existing GPT-2 pretokenizer.
SentencePiece uses score-ranked adjacent merges, highest score then leftmost,
with UTF-8 symbols and byte fallback after merging. Reuse existing metaspace
and special-token surfaces, retaining SPM prefix and raw-byte decode semantics.
Honor GGUF BOS/EOS metadata for `EncodeWithSpecialTokens`; `Encode` adds none.
Test synthetic score order, whitespace, byte fallback, special tokens, malformed
metadata, and digit splitting. Compare actual checkpoint encode/decode against
the pinned oracle. Mutate each new production branch and require failure.
No model forward, quantization, or CUDA kernel changes belong to this repair.

## Dense shard descriptor contract regression

`ISSUE-LOCAL-01M63WWWN3EETEG29PF1ME9BNN` records four guard mutations that
escaped the valid-only dense shard tests. Preserve production behavior. Add
malformed descriptors through `vt::MergedGemmDense` for count and array
contracts, unsupported fused providers, epilogue arity, and epilogue outputs.
Use allocated, valid operands where a removed guard can continue safely.
Require the exact guard diagnostic and exception type. Each removed guard
must fail the focused host suite, then restore the complete passing suite.
The shared production implementation and real-model gates remain unchanged.

## Untied original-format dense output head repair

`ISSUE-LOCAL-01M63XH2ED7MGFNBNWG8BKEEB6` records the real TinyLlama forward
failure after successful public load. Its untied GGUF output head retains
original `[vocab, hidden]` NK layout and quantized storage. The shared dense
forward incorrectly assumes every untied head is transposed `[hidden, vocab]`.
Select `vt::MatmulBT` for tied or NK-owned heads, preserving `vt::Matmul` for
existing transposed HF heads. Keep the resident weight and registered quant
contraction seams, without a copy or reshaping the mismatched tensor.

Add a small untied quantized GGUF public completion test. The production
loader, registry, forward, and sampler must reach the fixed projection.
Require completion and finite full logits. Reverting the head dispatch must
reproduce the inner-dimension error, including on the pinned TinyLlama model.
Preserve tied-head and existing non-NK paths. Full model oracle and performance
gates remain separate obligations after this correctness repair.


## Exact DSR registrar accounting repair

Issue: `ISSUE-LOCAL-01M641H5N4MSWRZSAN07YEND4C`.
Base: `9f4b15d93f98035043a26b82ee3e09b333bd6c64`. Scope: the DSR gate and its synthetic-tree regression suite only. No product dispatch or DSR baseline changes.

The new `src/vllm/v1/attention/cuda_legacy.cpp` declares one CUDA_LEGACY backend registration key. It contains capability metadata rather than a shared-model device branch. The existing checker policy at `scripts/check-device-leakage.py:360-366` states that an integer allowlist entry means exactly that many platform-definition references; adjacent backend and GDN registrar entries already use that policy. Add one exact-count `kcuda` entry for this file, with the registration rationale. Do not allow the whole file or increase the shared-layer baseline.

First extend the existing synthetic tree with the same registrar and observe its current gate fail. Add regressions proving that an extra device branch in this descriptor file fails the exact-count guard and that removing its sole registration fails. The existing planted shared-model CUDA branch mutation must continue to fail. Run the complete device-leakage mutation suite and the live gate. A missing environment value remains pending; no environment or GPU value is needed for this offline checker repair.

Evidence must include the red result, focused green result, live gate, record/style checks, and final immutable head for independent review. Full preflight is invoked before edits; its unrelated baseline failures are not this repair's scope. Stop if the registrar is not a legitimate platform descriptor or if the change requires broad checker-semantic relaxation. The issue remains open until landing; integration remains with the operator.


DSR registrar repair evidence (29 October 2026): the new synthetic descriptor first fails with `kcuda=2 > baseline1`. The exact one-reference allowlist makes all58 DSR mutation tests pass. Explicit mutations for an added shared-model CUDA branch, an added descriptor branch, and a removed registrar key all fail as required. The live gate passes at DSR32, equal to the unchanged baseline32. Evidence: `.agents/evidence/pascal-volta-enable/dsr-registrar/`. Full campaign correctness and unrelated preflight failures remain outside this checker repair.

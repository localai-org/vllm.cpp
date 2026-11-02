# Legacy CUDA FP16 dtype through shared dense execution

## Now

SPECIFIED. This commit contains the issue and design only. No implementation is authorized by this spec commit alone.
Issue: ISSUE-LOCAL-01M641CAZNCNRRYAJF8DCV86R1.
Task base: 9f4b15d93. Row owner: BACKEND-CUDA-SM061, within the Pascal and Volta campaign.
The operator dispatches fresh implementers after this spec becomes base-reachable.
Use one pull request by the repository default unless the developer records a different integration shape.

## Problem and required result

Pinned vLLM resolves an automatic BF16 model configuration to FP16 on Pascal and Volta.
The local CUDA platform advertises BF16 first on every architecture. The dense model path also allocates BF16 directly.
Several registered operations reject FP16 output. Other CUDA dispatches interpret every non-F32 output as BF16.
Changing a configuration string alone therefore cannot implement the reference behavior safely.

Resolve one model dtype at the existing shared configuration and platform seams.
Every model-path weight conversion, activation, residual, projection, rotary cache, and default KV allocation inherits that resolved dtype.
Keep original GGUF block encodings resident. Route execution through existing shared operations and the public model registry.
FP16 stores must round as FP16. A two-byte allocation never establishes the scalar format by itself.

This correction is necessary primary behavior. It is not evidence that FP16 alone repairs the actual token gate.
SmolLM remains failing. The separate Q6_K prefill candidate is not accepted, and its TinyLlama regression remains failing.
Do not make that candidate an unreviewed dependency or alter its goldens to open this campaign.

## Reference authority and pins

| Reference | Pin | Authority in this campaign |
|---|---|---|
| vLLM | a7c23ac96d7806e7c7e7d862eadbce5a33529b94 | Model dtype, scalar semantics, defaults, shared model structure, tests |
| First-party GGUF plugin | d4c1f0d082fc7cd4350da56689109a01c1f29d6c | Primary GGUF format and activation/output dispatch where it serves |
| llama.cpp | 62318cd57dc63900ecb153272c0b10f78550855d | Secondary executed GGUF output and hardware floor only where primary cannot execute |

Read `.agents/oracles/vllm.md`, `.agents/oracles/vllm-gguf-plugin.md`, and `.agents/oracles/llama-cpp.md` before implementation.
The oracle files record the active pins. This spec does not advance a pin.
The sparse source snapshot and diagnostic copies are evidence, not substitute authority for the recorded revisions.

Primary anchors:

- `vllm/config/model.py:2340-2378`, `_resolve_auto_dtype`: preserve a supported config dtype, otherwise select the platform preference. Automatic F32 models downcast to the preferred dtype.
- `vllm/config/model.py:2381-2430`, `_get_and_verify_dtype`: preserve explicit dtype parsing, validation, and conversion warnings. Do not apply auto fallback to every explicit request.
- `vllm/platforms/cuda.py:255-264`: capability80 and newer prefer BF16 and accept BF16, FP16, F32. Capability60 through79 accept FP16 and F32, preferring FP16.
- GGUF plugin `config.py:46-51`: accepts FP16, BF16, F32 activations and minimum capability60.
- GGUF plugin `linear.py:43-66`: quantized projection output follows the input dtype. Read its selected custom operation before choosing a quantized execution format.
- `vllm/model_executor/models/llama.py`: existing shared QKV, attention, MLP, residual, and final norm chain. Do not create another Llama forward implementation.
- `csrc/libtorch_stable/activation_kernels.cu:30-44,165-167`: SiLU intermediate and multiplication use the input scalar contract.

The implementer reads each additional pinned executing kernel and generated specialization before porting it.
Record exact file and line anchors in the wave evidence. A generic torch formula does not prove the executed reduction chain.
Primary semantics outrank the secondary oracle when their formats or accumulation policies differ.

## Existing gap and reachability inventory

Line anchors describe task base 9f4b15d93 and must be refreshed against each wave base.

| Shared surface | Existing gap | Required extension |
|---|---|---|
| Platform metadata | `src/vllm/platforms/cuda.cpp:110-113` always returns BF16 first | Mirror capability-dependent supported dtype order through existing platform metadata |
| Model config | `src/vllm/model_executor/models/llama_gguf_weights.cpp:92` forces BF16; `llama_registry.cpp:77` calls BF16-returning ActDType | Resolve checkpoint config, explicit request, and actual platform once, then carry the result |
| Registered op validation | `src/vt/ops.cpp:28-29` admits FP16 input but not FP16 output | Extend only fully implemented operation/provider combinations; keep unsupported providers' precise refusals |
| Hidden and residual | `qwen3.cpp:150,167,203,313,317,328,338,424` hardcodes BF16 | Propagate the resolved dtype through the existing dense forward |
| Shared MLP | `include/vllm/model_executor/layers/linear.h:99-101` hardcodes shard products to BF16 | Route FP16 gate/up/output through MlpGateUpMethodBase and MergedGemmGroup |
| Dense attention buffers | `include/vllm/model_executor/models/dense_attn_block.h:580,815,839` chooses or casts BF16 | QKV, attention output and o_proj inherit the model dtype |
| Ordinary GGUF weights | `qwen3_gguf_weights.cpp:194-214` DqBf16 and OwnBf16 always produce BF16 | Resolve expanded norm, embedding and ordinary dense values into the model dtype |
| Embedding | `ops.cpp:1788-1803` rejects FP16 output; CUDA float and quantized embedding dispatch lacks FP16 stores | Implement ordinary and kept-quant embedding through the registered Embedding seam |
| Norm and fused residual norm | `ops.cpp:1157-1174`, `cuda_ops.cu:495-545` lacks FP16 activation, output and residual | Typed FP16 scalar dispatch and residual rounding, including the existing FusedChain realization |
| Add | `ops.cpp:1764-1784`, `cuda_layernorm.cu:250-291` supports F32 and BF16 output | Typed FP16 addition and output store through registered Add |
| Split SwiGLU | `ops.cpp:1136-1146`, `cuda_moe.cu:852-883` lacks FP16 gate/up/output | Explicit FP16 SiLU intermediate and multiply rounding through the existing operation |
| RoPE | `ops.cpp:1838`, `cuda_ops.cu:1195-1202` and cached path near1307 lack FP16 activation dispatch | Direct and cached FP16 rotation with the primary's cache/scalar semantics |
| Elementwise GEMM | General MatmulBT rejects FP16 output; `cuda_matmul.cu:1344-1358` has a separate FP16 cuBLAS utility | Reach the typed utility from registered Matmul and MatmulBT with correct NK orientation and compute policy |
| Kept-quant GEMM | ActDT already reads FP16 near `cuda_quant_dot.cu:2324`; output branches choose F32 or BF16 | Explicit FP16 output in every reached Q4_K, Q5_0, Q6_K, Q8_0 and shared quantized dispatch |
| KV conversion | CastF16 and two-byte cache copy already exist | Ensure source dtype and cache spec agree before copying bits |
| Reverse cast | `ops.cpp:5588`, `cuda_glue.cu:103-121` CastF32 accepts BF16 only | FP16-to-F32 registered conversion for logits, callbacks and existing consumers |
| KV specification | `kv_cache_interface.cpp:450` deliberately refuses float16; dense attention guard near562 rejects it | Wire FP16 allocation, storage, reader and accounting before removing this precise stale refusal |
| Paged attention | `ops.cpp:5224-5255`, `cuda_paged_attn.cu:3049-3129,3206-3231` lacks FP16 query/KV/output dispatch | Typed generic and eligible legacy kernel paths, preserving masks, strides, paging and accumulator policy |
| Logits and sampling | Public logits and on-device sampler consume F32 | Preserve the documented F32 logits exception while consuming FP16 final hidden correctly |

Existing primitives include DType::kF16, half codecs, CastF16, FP16 quant activation readers, two-byte cache copy, and FP16 cuBLAS GEMM.
None proves the full model path is reachable.
Audit FusedChain, merged projections, gather, tensor views, cached RoPE and every residual cast before enabling auto FP16.
A global IsOutFloat change must not accidentally admit FP16 to an unimplemented provider or operation.
Use existing provider and platform capability seams. Extend a seam if it cannot represent the typed contract.
Do not add ad hoc per-layer architecture predicates or a parallel model path.

## Dtype, compatibility, and memory contract

1. Preserve the public C ABI, symbol names, struct layouts, old model config keys, and existing entry points.
2. Distinguish the configured checkpoint dtype, explicit runtime request, resolved model dtype, storage dtype, and accumulator dtype.
3. Mirror automatic legacy resolution only after the complete FP16 arm passes its gates. No temporary config-only switch can ship.
4. Preserve explicit dtype behavior according to the pinned primary source. Do not delete BF16 software fallbacks or existing explicit modes.
5. Preserve automatic and explicit behavior on newer CUDA hardware and other backends. Add controls for capability80 and newer.
6. Default KV storage inherits the resolved model dtype. Existing explicit cache overrides retain their declared behavior and errors.
7. Norm accumulation, GEMM compute, softmax, logits, and sampler buffers retain primary-defined F32 exceptions. Annotate model-path F32 allocations with their reason.
8. Preserve compressed quantized weight bytes, shapes, and NK ownership through loader, merged MLP, projections, and head.
9. Do not reinterpret BF16 and FP16 payloads. Every conversion and output store is typed and covered by a rounding-sensitive test.
10. Record activation, residual, weight, scratch and KV byte counts against the executing oracle. Two-byte dtype equality does not excuse format divergence.

No global F32 default, GGUF-only F32 override, or llama-specific norm replacement is authorized.
The secondary oracle's F32 activation and FP16 cache chain is not the primary's legacy FP16 model policy.

## Implementation waves and file ownership

Every wave has a fresh implementer worktree pinned to the previous reviewed wave head.
Implementation waves are sequential. No two waves edit a shared file concurrently.
New tests live in per-wave files. Evidence lives in one file or directory per wave.
The operator owns integration records and reruns each gate. Implementers never edit a shared global status surface.

The first implementer branches from this immutable spec head, whose product base is 9f4b15d93.
Later waves branch from the operator's reviewed integration head, never another implementer's mutable tree.

| Wave | Exact product authority | Tests and records |
|---|---|---|
| 1 | src/vt/ops.cpp; src/vt/cuda/cuda_ops.cu; cuda_embedding_quant.h; cuda_layernorm.cu; cuda_moe.cu; cuda_glue.cu; cuda_matmul.cu; existing FusedChain realization only if required by its typed contract | New tests/vt/test_cuda_fp16_scalar.cpp, scoped existing cases, tests/CMakeLists.txt, per-wave issue/spec evidence |
| 2 | src/vt/cuda/cuda_quant_dot.cu and its reached quantized helper headers; narrow ops.cpp quant guard only after Wave1 handoff | New quant FP16 test file, scoped quant cases, CMake registration, per-wave evidence |
| 3 | src/vllm/v1/kv_cache_interface.cpp; related cache spec/allocator files; src/vt/cuda/cuda_cache.cu; cuda_paged_attn.cu; ops.cpp attention/cache guards; include/vllm/model_executor/models/dense_attn_block.h cache validation only | New cache/attention FP16 tests and per-wave records |
| 4 | src/vllm/platforms/cuda.cpp; existing model config/dtype resolver; llama_registry.cpp; llama_gguf_weights.cpp; qwen3_gguf_weights.cpp; qwen3.cpp; qwen3_5.cpp ActDType seam; include/vllm/model_executor/layers/linear.h; dense_attn_block.h model buffers | Public API dtype/model tests, compatibility tests, per-wave evidence and required public usage projection |

Each CUDA filename in this table is under src/vt/cuda unless its full path is given.
A necessary product file outside a dispatched wave's exact authority requires an operator scope amendment before editing.

Waves1 through3 must ship reachable registered operations, but may leave legacy automatic model selection unreached until Wave4.
For each such staged slice, its commit and pull request bodies name the unreached auto-model wiring, BACKEND-CUDA-SM061, and ISSUE-LOCAL-01M641CAZNCNRRYAJF8DCV86R1.
The Owed section below owns that wiring. A test-only constructor or direct kernel launch cannot substitute for a registered entry point.

### Wave 1: shared scalar and dense projection capability

Own `src/vt/ops.cpp`, scalar CUDA files, embedding providers, `cuda_matmul.cu`, and corresponding typed helper headers.
Implement registered FP16 norm, fused residual norm, Add, Embedding, SiLU multiply, direct/cache RoPE, Matmul, MatmulBT and reverse cast.
Use ordinary and quantized embedding arms. Audit CPU and other providers before relaxing any shared wrapper guard.
Keep model allocation and auto dtype selection unchanged in this wave.
A public registered-op test must reach each typed production call site.
Where the primary uses fused operations, test FusedChain against the unfused FP16 contract and preserve scalar rounding boundaries.

### Wave 2: shared quantized FP16 projections

Own `src/vt/cuda/cuda_quant_dot.cu`, reached quantized helper headers, and scoped quant tests.
Start only after Wave1 commits its final guard and scalar contract.
Add explicit FP16 output to generic and reached Pascal decode/prefill kernels without changing weight encodings or batch dispatch policy.
Read primary GGUF plugin and executing custom kernel eligibility for FP16 input/output before selecting MMVQ, MMQ or cuBLAS adaptations.
Do not reuse the secondary F32 MMQ D4 path for FP16 merely because the storage width is now supported.
Preserve existing F32/BF16 branches byte-for-byte where possible. Remove no software BF16 fallback.

### Wave 3: KV and paged attention capability

Own cache interface/specification, cache routing, `cuda_paged_attn.cu`, scoped dense attention validation, and cache tests.
Start after Wave2. Acquire shared `ops.cpp` ownership only after the previous wave releases it.
Add FP16 query/KV/output instantiations and correct format-specific vector loads/stores.
Wire model-dtype auto cache resolution and existing explicit float16 cache requests through allocator, store and reader.
Remove the named float16 refusal only after the exact production path passes.
Keep FP8, BF16, existing architecture policy, and environment overrides intact.
Retain the registered PagedAttention seam and on-device scheduling.

### Wave 4: one model dtype and loader-to-sampler propagation

Own platform dtype metadata, model configuration, GGUF ordinary-weight conversion, shared dense model buffers and shared layer allocations.
Start only after all required operator and cache waves pass independent review.
Resolve one dtype using existing model config and platform interfaces. Carry it through loader, context, ModelRegistry::Forward and dense_attn::AttnBlock.
Make ordinary converted weights and shard gate/up products inherit the resolved dtype.
Keep quantized bytes resident and reuse MlpGateUpMethodBase, MergedGemmGroup and FusedChain.
The public include/vllm.h client must load, prefill, decode, produce F32 logits and sample on device.
Do not add an example-only forward or hidden alternate engine.
Enable the legacy auto default only when this same commit passes the complete public gate.

### Wave 5: operator acceptance and measured execution

A fresh reviewer mutates each claimed guarantee at immutable heads. Fresh implementers repair any findings.
The operator runs the full build and the exact model gates itself, then performs idle same-binary measurements.
No additional product changes belong in this evidence wave.
Do not declare the campaign complete while an applicable accuracy, latency, throughput or memory axis remains failing or unavailable.

## Test-first and upstream test ports

Before each wave, port the smallest upstream case that fails at the registered production entry point.
Capture its red exit and exact failure. A config object or constructed private kernel does not prove runtime reachability.

Pinned upstream tests already located:

| Upstream test at a7c23ac96d | Required port |
|---|---|
| `tests/kernels/core/test_layernorm.py:21-89`, test_rms_norm | FP16/BF16/F32, tokens7/83/4096, hidden8/768/769/5120/5125/8192, residual on/off, stride cases, seed0 and upstream tolerances1e-2 |
| `tests/kernels/core/test_layernorm.py:91-132`, test_rms_norm_weightless | Preserve weightless and residual parameters and tolerance source |
| `tests/kernels/core/test_activation.py:137-221`, test_act_and_mul | Preserve activation parameters, FP16/BF16/F32, shape matrix, seed and per-dtype tolerances; include split and merged production seams |
| `tests/kernels/core/test_rotary_embedding.py:29-65`, test_rotary_embedding_opcheck | Preserve both RoPE conventions, optional key, head strides, dimensions and position limits; add FP16 source-grounded adaptation because this upstream opcheck enumerates F32/BF16 only |
| `tests/kernels/core/test_vocab_parallel_embedding.py:88-118` | Preserve masked gather, original/padded vocab and rank-sum behavior for applicable single-device cases |
| `tests/kernels/attention/test_cache.py` and `tests/kernels/test_cache_kernels.py` | Port applicable store/copy/paging/tail cases, explicit storage dtype, slot ownership and untouched slots; preserve listed dtype parameters and add source-grounded FP16 coverage where absent |
| `tests/v1/sample/test_head_dtype.py:42-97` | Preserve explicit head dtype and model-dtype behavior, with the public F32 logits contract separately checked |
| `tests/v1/sample/test_sampler.py` and `test_topk_topp_sampler.py` | Preserve applicable greedy/sampling, mask, seed, penalty and invalid-input cases; retain on-device execution |

The implementation reads each full test and fixture before porting it. Do not turn this summary into a reduced parameter list.
Document unavoidable harness adaptations only. Do not tighten or weaken upstream tolerances to hide storage or reduction differences.
Fetch remaining exact primary dtype-resolution and paged-attention tests at the same pin before implementing those seams.
Supplement with registered-op FP16 fixtures that detect BF16 reinterpretation, cancellation-sensitive residual rounding, underflow, overflow, odd strides and tail blocks.

Model tests start through include/vllm.h with a small pinned fixture and then the approved actual checkpoints.
The initial public red must show unsupported FP16 or incorrect dtype propagation before implementation.
Use internal test diagnostics to observe dtype/bytes without adding a public ABI field solely for tests.
Test auto BF16-config on sm60/sm61/sm70 metadata, explicit FP16/BF16/F32 modes, and unchanged modern automatic BF16.
The device metadata test is independent from real hardware execution and cannot replace it.

## Mutation review guarantees

A fresh reviewer uses a scratch copy and restores bytes after each mutation:

- Remove each registered FP16 production call site. Its focused case must fail.
- Replace an FP16 load/store with BF16 at the same byte width. Format-sensitive tests must fail.
- Remove scalar rounding at residual Add or SiLU intermediate. The corresponding rounding fixture must fail.
- Change one norm or MLP allocation back to BF16. The public model dtype/byte trace must fail.
- Force supported_dtypes to modern BF16 order on a legacy device. Automatic resolution tests must fail.
- Force legacy FP16 selection on modern hardware metadata. The compatibility control must fail.
- Skip FP16 KV conversion or relabel cache dtype without conversion. Exact stored bytes and attention tests must fail.
- Misroute FP16 MatmulBT orientation, a quantized output store, or a merged gate/up projection. Registered numerical cases must fail.
- Delete the production loader or model registry wiring. The public API model gate must fail.
- Force a host sampling readback. The declared device-residency or trace gate must fail.

A passing private helper test after deleting production wiring is a review failure.

## Hardware, oracle execution, and full acceptance

The available P620 sm61 is authorized for GPU work under `/home/user/gpu.lock`.
Use the existing CUDA12.4/GCC13 legacy recipe, build at most four jobs, and reuse bounded scratch archives.
Compile changed CUDA translation units for sm60, sm61 and sm70. Verify typed code and compiled images, not only host success.
Run on real Pascal before accepting its default. P100/sm60 remains hardware-pending until executed.
Volta implementation is in scope, but runtime optimization remains TODO until the GPU arrives.
Do not report a metadata or compilation test as Volta runtime validation.

The primary oracle is globally gateable, but its exact Pascal/Volta FP16 GGUF production workload is not yet measured in this session.
The implementer must attempt the pinned primary and plugin with matching model, dtype, production configuration and actual device eligibility.
Record every build/run command and the concrete unavailable dependency or architecture error if it cannot run.
A source-level minimum capability declaration is not proof of an executable oracle lane.
Do not use enforce-eager as a throughput denominator.
When primary execution is unavailable, use the registered pinned secondary only for the admitted output question and retain the primary execution gap explicitly.
Never claim secondary F32 activations prove primary FP16 numerical equality.

Approved actual checkpoint pins remain the ones in docs/USAGE.md:

- `bartowski/SmolLM2-135M-Instruct-GGUF` at `09816acd5d99df7be770d85ea30822623dab342c`, `SmolLM2-135M-Instruct-Q4_K_M.gguf`, 105454432 bytes, SHA256 `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`.
- `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` at `52e7645ba7c309695bec7ac98f4f005b139cf465`, `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`, 668788096 bytes, SHA256 `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.

Quantized arms are mandatory. Cover each encoding actually reached by these mixed checkpoints, including Q4_K, Q5_0, Q6_K and Q8_0.
Cover ordinary expanded weights and norm weights separately. Preserve stored packed bytes, shard ownership and head orientation.
Name any unimplemented applicable arm and its missing shared operation explicitly; the campaign remains incomplete for that arm.

Run the complete configured build, required CPU/offline gates, each focused CUDA gate, bounded full quant gate, and public ABI compatibility tests.
Report the P620's measured large-MoE allocation blocker rather than silently removing it.
Run actual SmolLM and TinyLlama p16/p128 by t16/t64, identical prompt IDs, deterministic sampling, token counts and repetitions on each side.
Use exact tokens where deterministic. A distributional gate requires explicit ratification, never an inferred near-tie exception.
Current failing candidates do not become new goldens. FP16 is not presumed sufficient for exact tokens.

Only after correctness, record idle same-binary A/B throughput, latency, peak GPU memory, model-weight bytes, activation/residual bytes, scratch bytes and KV bytes.
Use identical workloads and matching traces. Record revisions, compiler flags, model hashes, contention, warmup and repetitions.
Any axis below its required floor remains an open gap with the next traceable hypothesis.

## Risks and stop conditions

Mixed precision affects residual rounding, quantizer thresholds, attention history, and greedy ties.
A local operator improvement can regress a previously matching full-model sequence.
A default change is blocked until the complete production chain and the exact model gate pass.

Stop and return NEEDS_CONTEXT when a required shared seam cannot represent upstream behavior or its binding contract is absent.
Stop for a required public ABI break, undocumented explicit-mode change, new unowned provider admission, or unpinned model artifact.
Stop performance acceptance on any correctness regression. File and repair newly found defects through fresh issue/spec/review flow.
Do not repair the coordinating implementer's own review findings.

## Owed

Wave4 owns the staged auto-model wiring for BACKEND-CUDA-SM061. The row-owned canonical issue is named above.
Implementation waves, upstream test ports, real FP16 primary-oracle eligibility, independent mutation review, actual-model token gates and all memory/performance measurements remain owed.
No implementation or acceptance result is implied by this spec-only commit.

## Wave 1 scope amendment, 29 October 2026

The operator authorizes two required extensions before Wave1 implementation.
`LaunchEmbeddingQuant` lives in `src/vt/cuda/cuda_quant_dot.cu:3044-3077`.
Wave1 owns only the quantized embedding output store and launch dispatch there.
Wave2 retains every quantized GEMM kernel, policy, and activation format.

The existing provider supports predicate describes a device, not a scalar type.
A shared wrapper cannot admit FP16 safely without identifying its selected provider.
Extend `include/vt/op_provider.h` and `src/vt/op_provider.cpp` with additive scalar capability registration and query functions.
Keep the `OpProvider` layout, aggregate initialization, and existing signatures unchanged.
Unregistered scalar capabilities remain false. Publish FP16 only for implemented CUDA scalar operations.
Test selection with a higher priority provider that does not publish FP16.
No device-name branch or global FP16 output admission belongs in the shared wrappers.

## Wave 1 exact activation test reference, 30 October 2026

The operator authorizes `tests/vt/cuda_fp16_scalar_reference.cu` and its test target source registration.
Port the independent scalar SiLU computation from pinned vLLM `csrc/libtorch_stable/activation_kernels.cu:30-44,165-167`.
Use CUDA `expf` on both sides to preserve the exact tolerance in `tests/kernels/core/test_activation.py:137-221`.
The helper consumes test tensors only. It never calls a production operation or production scalar helper.
Compare registered merged and split operation results with this independent reference at zero absolute and relative tolerance.
Preserve the original FP16, BF16, and F32 shape matrix. Add a fixed half intermediate rounding fixture.
A separate mutation removes the half intermediate rounding and must fail that fixture.

## Wave 1 BF16 activation correction, 30 October 2026

The required exact activation matrix exposes an existing newer-target defect.
Split and merged SiLU round BF16 intermediates only when `__CUDA_ARCH__ < 750`.
Pinned primary scalar and packed kernels round that intermediate on every target.
The operator authorizes removal of this architecture condition only for these two owned SiLU computations.
Preserve F32 computation and existing Pascal and Volta BF16 results.
The full exact upstream test remains unconditional. Do not scope the test away from newer hardware.
Modern hardware execution remains PENDING because this host has only the P620.
Compile a modern specialization when this toolkit accepts it. Do not present compilation as runtime evidence.

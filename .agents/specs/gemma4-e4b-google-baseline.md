# Original Google E4B multimodal baseline

## Scope

Complete `google/gemma-4-E4B-it` uncompressed BF16 inference in vllm.cpp for text, tool calls, images, audio input, and video frames. Audio/video generation is not implied. Preserve PLE, KV sharing, partial RoPE, sliding/full attention, softcaps, tokenizer behavior, and trained clipping semantics. Do not substitute another runtime, community checkpoint, quantized checkpoint, or preprojected embeddings for acceptance. No LittleBit-2, training, or compression implementation belongs in this milestone. Later LittleBit-2 must preserve every modality and tool behavior.

## Ownership and integration

Owner: `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`.
Existing canonical local record: `.agents/issues/MODEL-MM-gemma4-mm-gemma4-for-conditional-generation/ISSUE-GH-2173.md`. It is `State: UNKNOWN`, `Availability: METADATA_ONLY`, not proof of an open actionable issue. It covers vision, not this complete milestone.
Canonical milestone issue: [ISSUE-LOCAL-01M4KSETGY63TDEDPF6EHQHRA8](../issues/MODEL-MM-gemma4-mm-gemma4-for-conditional-generation/ISSUE-LOCAL-01M4KSETGY63TDEDPF6EHQHRA8.md), OPEN. The historical record remains UNKNOWN/METADATA_ONLY. No remote state is inferred.
Historical designs: [Gemma-4 multimodal](gemma4-multimodal.md) and [vision reachability](gemma4-vision-reachability.md). Their captures remain historical evidence, not acceptance for the original Google checkpoint.

Shared seam owner: `ENG-MM-INPUT-PIPELINE`. Historical `ISSUE-GH-2300` remains UNKNOWN/METADATA_ONLY. Commit `c0649889d` added the tower and merge runner seam. Reuse and test that seam. Do not create a parallel runner or take ownership of shared changes without coordination.

## Now

T00 establishes the local issue and canonical design before product implementation. Independent review and the operator rerun remain pending. No original-checkpoint inference or product implementation is claimed.

The owning matrix lifecycle remains READY. The coordination audit dated 2026-08-06 retired the old Gemma claims and moved this row from ACTIVE to READY. A stale ACTIVE token in its historical evidence made the derived live view report ACTIVE. The scoped metadata correction removes that ambiguity without changing historical test results. The local helper role owns only this bounded records task on `row/MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`.

## Git integration

One PR is the repository default. No PR, push, merge, or remote write is authorized for T00. Commit this canonical design before product code. Planning artifacts and run state remain untracked.

## Upstream chain

Inspected local base: `6c43232260b300e6992f9375811f782fe24c172a`.
Google revision: `ee0ef6023621cff504d758262d4e04895a5af4a2`, independently confirmed by the public HF revision API. API reports ungated, 7,996,156,490 BF16 parameters, one `model.safetensors` file. Weight bytes were not downloaded or hashed; theoretical parameter storage is 15,992,312,980 bytes, not measured runtime residency.
The planning audit captured small public metadata files. The source is [the pinned Google checkpoint](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2). Config SHA256 `33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4`; processor SHA256 `32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c`.

Text config: 42 layers, width 2560, PLE width 256, 18 KV-shared layers, 8 query heads, 2 KV heads, local/global dimensions 256/512, vocabulary 262144, sliding window 512. Vision: 16 layers, width 768, patch 16, pooling 3, 280 image soft tokens. Audio: 12 layers, width 1024, projection 1536, trained clipped linears. Frontend: 16 kHz, 128 features, FFT 512, frame 320, hop 160, mel floor 0.001. Video processor declares 32 frames and 70 maximum soft tokens, distinct from image defaults. Do not collapse video processing into repeated default image processing.

Primary oracle: vLLM `a7c23ac96d7806e7c7e7d862eadbce5a33529b94` from `.agents/upstream-sync.md:101`; runtime/distribution `0.3.0.dev267+ga7c23ac96d`, FlashInfer `0.6.15.post1`. Fetched `vllm/model_executor/models/gemma4_mm.py`: `Gemma4ProcessingInfo` at 196, `Gemma4MultiModalProcessor` at 580, `_get_mm_fields_config` at 784, `Gemma4MultimodalEmbedder` at 950, `Gemma4ForConditionalGeneration` at 1015, `_process_image_input` at 1299, `_process_audio_input` at 1581, `embed_multimodal` at 1617. These anchors identify the source to port, not proof that the oracle runs this exact checkpoint.
Secondary preprocessing oracle: transformers 5.14.1 per `.agents/oracles/transformers.md:39-40`. Its record refers to an older vLLM environment. The public v5.14.1 annotated tag resolves to commit `a08ace4bbd97e721c98751deec37d87b026acadc` (tag object `ec710763db3fd53921a54c6eccf693619abbe866`), verified with git ls-remote. The processor source at that commit is captured in the source manifest. T01 must confirm compatibility with the current primary before fixture capture. Do not silently update the global oracle pin. Exact-Google gateability remains PENDING.

## Our baseline

`src/vllm/model_executor/models/gemma4_registry.cpp::kGemma4Factory` at 171-178 lacks both MM hooks. `src/vllm/model_executor/models/model_registry.cpp::ModelRegistry::SupportsMmInputs` at 781-783 requires both. Existing registered `ForwardMm` is not an input pipeline. `include/vllm/model_executor/models/gemma4_audio.h:37-42` explicitly declares host-F32 reference math and leaves production BF16/device execution owed.
The audit child scanned 1110 production files and found no production tower invocation or Gemma processor registration at this SHA. Root independently verified only the three critical source files: gemma4_registry.cpp, model_registry.cpp, and gemma4_audio.h. These are static findings, not executed tests. Direct Gemma MatmulBT calls remain; inventory their dtypes and shared-seam obligations without changing model math or starting a performance rewrite.

Design: model loader retains all checkpoint tower/projector weights; model processor expands modality placeholders and emits validated typed fields; factory encode/embed hooks invoke actual towers and merge in original sequence order through the existing runner. Public ABI and server route raw inputs into that chain. Bound encoder-cache lifetime, request ownership, cancellation, batch isolation, and placeholder count. Reject unsupported inputs explicitly, never fall back to text or silently drop a modality. Preserve text-only behavior and PLE token identities at MM positions.

## Gates

Implement serially: metadata and strict validation, loader accounting, chat and tools, image processing, audio processing, video processing, vision encoding, audio encoding, hook and runner wiring, public serving, strict gate harness, exact-checkpoint reference validation, then production reachability. Separate three evidence levels:
1. CPU/offline: config parsing, original template/tool tokens, processor numerics, masks, counts, loader accounting, synthetic-shape tower math, and request routing. No runtime inference claim follows from this level.
2. Production reachability: raw image/audio/video-frame requests through include/vllm.h and serving, invoking the real processor, tower, hook, runner, and decode path. Independently delete each production call site in scratch; the gate must fail. Tiny fixtures can establish routing but not exact full-checkpoint inference.
3. Exact-checkpoint reference: both actual engines load the pinned original Google artifact, verify every tensor and artifact hash, run identical prompts/media/sampling, and compare intermediate stages plus generated token IDs. Require text/tool/image/audio/video and mixed-input cases, prefill/decode, streaming/nonstreaming tools, batched isolation, and text regression. Require zero token divergence by default. Any stochastic/distributional exception needs separate prior ratification, not the legacy permissive image predicate.

Before collecting results, freeze processor/tower tolerances from the pinned upstream tests or an independently measured oracle repeatability envelope. Record exact values and justification. Missing thresholds block capture acceptance; do not choose thresholds after seeing C++ errors. Integer tokens, masks, lengths, placeholder order, and positions require exact identity. Require finite numeric outputs and explicit per-stage max-absolute and relative error bounds.

Every required gate fails on absent assets, identity mismatch, zero executed cases, skipped cases, nonfinite outputs, or mismatch. Existing Unsloth captures and skip-on-no-checkpoint tests remain regression evidence only. Report per modality executed/passed/failed/skipped counts; a process exit 0 alone is insufficient.

## Dependencies

No hardware or download authority is granted here. Root obtains approved checkpoint staging, fixture provenance, environment values, sufficient disk/RAM/VRAM, an idle-device lease, and required mutex before full acceptance. No host names or secrets belong in deliverables. Record build/runtime identities and resolved dtypes. CPU builds require CMake >=3.24, C++20, Python3, and available third-party headers. Network-disabled configure must fail rather than fetch dependencies silently.
Missing environment does not block records or CPU unit design. Oracle pin compatibility, float normalization, video timestamps, audio padding, device memory residency, and stale historical ownership are explicit risks. No claim of speed parity or complete port is allowed while applicable repository performance axes remain pending. This milestone is prerequisite evidence for LittleBit-2, not a waiver for quantized arms: preserve existing arms and record new compression work as deferred.

## Risks/decisions and review

Root dispatches one bounded task at a time. A fresh independent reviewer checks each immutable implementation head and mutates critical guarantees in scratch. Root reruns each Verify command, then requires a separate independent final reviewer for the whole baseline. Findings return to a fresh fixer; never lower a gate or discard correct work. Current evidence: static inspection and metadata audit only. No product implementation, original-checkpoint inference, or accepted runtime evidence exists for this milestone.

## T00 verification and pending gates

Run after the canonical spec commit:

```sh
python3 scripts/check-agent-record.py && python3 scripts/check-symbol-anchors.py && git cat-file -e HEAD:.agents/specs/gemma4-e4b-google-baseline.md
```

The original setup preflight (bg-51) exited 1 with `tools suites` failing. The later pre-edit run (bg-82) also failed: `test_check_symbol_anchors`, `test_tower_skip_rss_arm`, `tools suites`, and `check-windows-portability.py`. All three runs reported five argument-dependent skips: `check-arm-isa-build.py`, `check-cpu-isa-build.py`, `check-cuda-fat-gencode.py`, `check-pr-size.py`, and `check-triton-aot-multiarch.py`. The staged run (bg-97) exited 1 with three failures: `test_check_symbol_anchors`, `tools suites`, and `check-windows-portability.py`. Its tower test did not fail, and `check-now-current.py --staged` passed. These attempts do not establish a green full preflight.

Diagnosis 395d3ad5 confirmed symbol-test Git discovery contamination through a one-variable A/B. Future scratch runs inside the enclosing checkout need `GIT_CEILING_DIRECTORIES` set to the scratch parent. Temporary directories must sit outside the E4B source tree. Source inspection attributes the Windows failure to generated, then deleted temporary headers inside that tree; no Windows rerun confirmed this diagnosis. The focused tower test passed, but its original failure cause remains UNKNOWN. These findings do not prove a general concurrent-contamination explanation.

The oracle guard requires 20 GiB free; the operator measured 18.75 GiB, so that blocker remains. Diagnostic visibility-variable mismatches came from setting device variables empty rather than leaving them unset. They do not establish the original tools failure's cause. Full preflight remains PENDING, not waived. No further full preflight runs are authorized for this follow-up. Do not alter checkers or delete files to bypass the guard. A preflight attempt is not runtime inference evidence. No translation units in scope means no C++ build was proved.

## Owed

T03C5 tokenizer bridge is not wired into serving or prompt timing. T10 under
MODEL-MM-gemma4-mm-gemma4-for-conditional-generation owns that integration;
The row-owned T03C5 issue remains open for this staged obligation.

The OPEN milestone issue owns the original-checkpoint baseline and its remaining gates. Numeric processor and tower tolerances are PENDING upstream-test enumeration or independent oracle repeatability measurement. Freeze justified values before accepting C++ comparisons. Do not invent numeric bars. Exact integer identity and zero token divergence remain the default requirements.

Original checkpoint hashes, complete tensor inventory, primary-oracle gateability, processor fixtures, runtime dtypes, device execution, and production-call-site mutations remain PENDING. Environment-dependent gates remain PENDING until explicit resource and environment authority exists. Missing values are not defaults.

GGUF and other quantized arms remain obligations of the broader model port. This BF16 milestone does not complete or waive them. Preserve existing arms and explicitly refuse unsupported paths. LittleBit-2, compression, training, and performance rewrites are outside this milestone. All unmeasured performance axes remain open.

## Port map

| Upstream surface | Local destination and obligation |
|---|---|
| Gemma4ProcessingInfo and Gemma4MultiModalProcessor | Register model processing through the shared processor factory. Preserve image, audio, and video defaults separately. |
| Gemma4MultimodalEmbedder | Load original tower and projector tensors. Preserve model BF16 and trained clipping. |
| Gemma4ForConditionalGeneration.embed_multimodal | Fill both factory hooks and reuse the existing encoder and merge runner. |
| Gemma-4 text backbone and chat template | Preserve PLE, KV sharing, rotary positions, text regression, and tool streaming behavior. |
| Public raw-media requests | Route through include/vllm.h and serving to the same model hooks. |

## Tests to port

Before implementation, enumerate tests at the pinned upstream revisions and preserve their parameters, fixtures, modes, tolerances, and failure cases. Exact upstream test symbols and measured float bars remain PENDING, not inferred from historical tests.

Required coverage includes original config and tensor accounting, canonical chat and tools, image patches and positions, audio features and masks, and video sampling and timestamps. Tower comparisons require independent stage references. Hook and runner tests require batching, cancellation, cache lifetime, count mismatch refusal, and call-site deletion mutations. Synthetic tests establish routing or math only. Final reference and reachability gates require original Google weights and all modalities, with no skipped cases.

## Work breakdown

T00 creates the OPEN issue and commits this design. T01 pins metadata, validates evidence, enumerates upstream tests, and freezes justified numeric bars. T02 accounts for config and weights. T03 preserves chat and tools. T04, T05, and T06 implement image, audio, and video processing. T07 and T08 wire the vision and audio encoders. T09 wires hooks and the existing runner. T10 exposes raw inputs through the public ABI and serving. T11 creates the strict acceptance harness. T12 measures original-checkpoint reference equivalence. T13 proves production reachability and completes independent review. Tasks remain serial except for the root-approved T01 prerequisite split below. T00 authorizes none of the product edits.

## T01 metadata contract and remaining prerequisites

T01 is PARTIAL. The standard-library validator and tracked original metadata
fixtures implement the executable CPU scope. Independent SHA-256 pins reject
self-attested replacement bytes. Strict report validation rejects missing cases,
skips, nonfinite metrics, identity changes, and integer or token mismatches.
Offline success explicitly says metadata-only and leaves original-model
gateability PENDING. Runtime acceptance fails closed even for zero errors.
No caller-supplied threshold can override this barrier.

The fixture [README](../../tests/fixtures/gemma4_e4b_google/README.md) records
license, attribution, reproduction, and the report schema. The
[test inventory](../../tests/fixtures/gemma4_e4b_google/upstream-tests.json)
enumerates actual declared test symbols and decorators at both immutable pins.
It includes hashes and source URLs for all Gemma4-named Python test files in the
pinned repository trees. This is source inspection, not test execution.

Relevant vLLM processing tests include
`test_gemma4_image_schema_accepts_variable_patch_counts`,
`test_gemma4_image_batching_keeps_variable_patch_counts_unstacked`,
`test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens`,
`test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens`,
`test_get_mm_max_tokens_per_item_respects_configured_video_num_frames`,
`test_get_prompt_updates_respects_nested_max_soft_tokens`,
`test_video_cache_is_independent_of_image_kwargs`, `test_limit_mm_per_prompt`,
and the five `test_encoder_chunk_*` cases. Their source is
`tests/models/multimodal/processing/test_gemma4.py` at the pinned vLLM commit.
Its `GEMMA4_MODEL_ID` is Google E2B, not E4B. These test declarations do not
establish exact-E4B runtime compatibility.

At the Transformers pin, `tests/models/gemma4/test_image_processing_gemma4.py`
checks defaults, resize dimensions, soft-token budgets, positions, and padding.
`test_processing_gemma4.py` checks vision-token counts, image tokens, truncation,
multimodal counts, and audio-token counts. It conditionally skips unsupported
counting capabilities and explicitly skips video-frame sampling.
`test_modeling_gemma4.py` contains text, image, multi-image, cache-sharing, and
sliding-window integration tests. Its integration checkpoint is E2B. Its tester
skips audio and video feature tests because input dictionaries lack those media.
The inventory preserves the decorators, including skip reasons. Unified and
assistant model suites are separate variants, not replacement E4B evidence.

Static compatibility evidence: pinned vLLM `requirements/common.txt` requires
`transformers >= 5.10.4`. Version 5.14.1 satisfies that declared lower bound.
Pinned vLLM `gemma4_mm.py` imports `Gemma4AudioFeatureExtractor`, `Gemma4Config`,
`Gemma4Processor`, and `Gemma4VisionConfig`. Pinned Transformers
`src/transformers/models/gemma4/__init__.py` exposes configuration, feature
extraction, image, modeling, processing, and video modules through lazy imports.
These source declarations are not an import smoke test or proof that both
engines load and execute the original checkpoint together. The checkpoint's
`transformers_version` is `5.5.0.dev0`, not an installed environment record.

Numeric processor and tower bars remain PENDING. No inspected test supplies a
complete, justified E4B cross-engine per-stage error envelope. A tolerance for
another model, backend, or test is not a valid substitute. Before enabling
runtime acceptance, obtain an authorized pinned environment, verify imports and
original checkpoint loading, and measure independent oracle repeatability for
each required stage and dtype. Ratify explicit absolute and relative bounds
before comparison capture. Keep exact integer identity and zero token divergence.
T01B cannot be marked done until these prerequisites are resolved and reviewed.

The first focused test run failed because the new validator did not exist.
After implementation, 18 CPU tests passed without skips. They use original
metadata and synthetic reports only. No C++ runtime, GPU, weights, or upstream
runtime tests were executed. Full preflight remains PENDING under the existing
T00 environmental blockers and the prohibition on another full-suite attempt.

## Root-approved T01 dependency split

Root accepted T01A metadata tooling at `e1759f0c0637a5be4df0735be17b2668c170d2cf`.
Root supplied passing verification `bg19728tests` and independent review
`reviewer9fce`, with all findings fixed. T01A is DONE for metadata tooling only.

T01B retains the PENDING measured oracle compatibility and justified frozen
numerical bars. T02 synthetic CPU loader/config work depends on T01A only.
T01B is a HARD dependency for every numeric/runtime acceptance, including T12
and T13. Synthetic fixtures never establish original-Google inference.
This scheduling clarification changes no runtime barrier or acceptance criterion.
Full preflight remains PENDING under the recorded failures. No rerun is authorized.

## T02 bounded output-head slice

The first executable slice validates tied/untied output-head loading through
real safetensors and Gemma loader sources. The original Google config is tied.
Pinned vLLM `vllm/model_executor/models/gemma4.py` at
`a7c23ac96d7806e7c7e7d862eadbce5a33529b94` constructs `ParallelLMHead` with
vocabulary and hidden dimensions, and ties it only when configured.
The downloaded source SHA-256 is
`205e9c543b263b2cebf0a8567de629597e179713f02e505cc4f94fb5168f566a`.
This is source grounding, not oracle execution.

CPU tests reproduce missing and malformed untied heads before the fix.
The expanded red run executed 8 cases with 8 failed assertions and zero skips.
The fix requires the untied head and its exact configured shape, and rejects
nonboolean nonnull tie flags. Absent/null defaults and explicit tying remain
unchanged. Positive controls check BF16 transposition and both loader overloads.
The green run executed 8 cases and 38 assertions with zero failures or skips.

T02 remains PARTIAL pending tower/projector inventory, clipping-scalar loading,
modality-config validation, and original-shape tower fixtures. Reduced text
fixtures and original-config refusal tests do not prove those guarantees.
PLE, KV topology, text math, tower execution, and runtime gates are unchanged.
Root rerun and independent review of this immutable slice remain required.
The owning milestone issue remains OPEN.

## T02 modality projector slice contract

The next bounded slice retains both modality projectors in the actual decoder
weight bundle. Full tower loading is deferred: the existing vision type stores
BF16 vectors, while the audio type is an F32 reference, not a production BF16
encoder representation. Do not populate that reference and claim production audio.

Source: pinned vLLM `vllm/model_executor/models/gemma4_mm.py`,
`Gemma4MultimodalEmbedder.__init__` (lines 964–997) and
`Gemma4ForConditionalGeneration.hf_to_vllm_mapper` (1045–1058), at
`a7c23ac96d7806e7c7e7d862eadbce5a33529b94`. Local source SHA-256:
`ee9b33f5f3d76bfec371dfc65292a915cc5b4741367ba0b28a450a5af8c40f9f`.
The executing constructor uses weightless pre-projection RMSNorm and a biasless
linear, despite its stale class docstring saying post-projection norm.
Checkpoint names are `model.embed_vision.embedding_projection.weight` and
`model.embed_audio.embedding_projection.weight`. Source shapes are
[text hidden, vision hidden] and [text hidden, audio output_proj_dims]; the
constructor falls back to audio hidden for absent/null/zero output_proj_dims.
The original metadata supplies 2560, 768, and 1536 respectively.

For this staged loader contract, absent/null modality config preserves legacy
text-only loading; false and other nonobject values are errors, not absence.
An object must supply a positive integer hidden_size and positive finite
rms_norm_eps. Audio output_proj_dims may be absent/null/zero (hidden fallback)
or a positive integer. Reject malformed values before projector allocation.
Retain the full modality config alongside an owned BF16 raw [out,in] projector,
including clipping flags and encoder fields for subsequent tower work. Validate
exact source shape and use the existing dense materializer, preserving its
scaled F8_E4M3 support rather than adding a global BF16-only guard.

Tests enter both decoder loader overloads using small real safetensors.
Cover config refusal, required tensor accounting, rank/orientation errors,
BF16 byte preservation, F16 refusal, scaled FP8 preservation, ownership after
shard destruction, and original-config shape rejection without full allocation.
This is projector accounting, not tower accounting or numerical equivalence.
T07–T09 must connect the retained projectors to encoder execution and hooks;
MODEL-MM-gemma4-mm-gemma4-for-conditional-generation and OPEN
ISSUE-LOCAL-01M4KSETGY63TDEDPF6EHQHRA8 own that wiring. Complete tower inventory,
clipping-scalar loading, full modality validation, original-shape tower fixtures,
assistant/tools/image/audio/video acceptance, and T12/T13 remain owed.
Full preflight remains PENDING; another full run is prohibited for this slice.

### Projector slice verification

Both real decoder loader overloads now retain validated config and owned BF16
projectors in raw checkpoint [out,in] layout (no transpose). Scaled F8_E4M3
passes through the unchanged dense materializer. The red-before run failed
84 assertions across three new cases; the original nine cases passed.
Green passes 16 cases / 192 assertions, zero skips, using the minimal CPU
target. CTest first rejected the stale nine-case expected count; updating it
to exactly sixteen restored its zero-skip gate. The 28 metadata regressions
and offline CLI pass. Evidence: control-room
`assets/work/2026-10-10-e4b-projectors`.

Root supplied rootbg271 and review24858d PASS for the prior output-head slice
(9 cases / 44 assertions plus 28 metadata cases). This projector slice awaits
independent immutable-head review and root verification. T02 remains PARTIAL:
tower inventory, clipping, full modality validation and encoder/hook wiring
are owed. No full preflight, GPU, weights download or original-model inference
ran. Assistant/tools/image/audio/video acceptance is unchanged.

## T02 vision patch-embedder retention slice

Inspection found no production vision weight loader: `Gemma4VisionWeights`
exists only in forward code and synthetic/reference tests. Bound this slice to
its two patch-embedder tensors; encoder blocks, typed full tower configuration,
clipping scalars and audio tower remain pending. An empty blocks vector is not
an executable tower and must not be interpreted as encoder readiness.

Pinned Transformers `modeling_gemma4.py` at
`a08ace4bbd97e721c98751deec37d87b026acadc`, lines 579–588, defines
`patch_embedder.input_proj.weight` [hidden, 3*patch_size^2] and
`patch_embedder.position_embedding_table` [2, position_embedding_size, hidden].
The vLLM prefix mapper cited above maps `model.vision_tower` unchanged apart
from the model prefix. Retain these in the existing vision representation,
with raw BF16 input projection and the representation's documented F32 position
lookup exception (exact widening of materialized BF16, not a new F32 policy).
Require positive integer patch_size and position_embedding_size in present
vision configs, checked multiplication, exact tensor shapes, and the unchanged
dense BF16/scaled-FP8 materializer. Absent/null vision stays optional. The full
JSON config remains retained by the projector slice; no invented defaults.

Tiny original-structure tests exercise both actual decoder loader overloads,
missing tensors, malformed shapes/dtypes/config, ownership and raw layout.
Preserve text/projector regressions, update the exact CPU case count, compile
affected shared-header consumers, and run the 28 metadata tests. Full preflight
remains pending and prohibited by the current task. No inference claim follows.

The parent reports projector rootbg296 PASS (16 cases / 192 assertions and 28
metadata tests) and independent reviewer7892d92a PASS (13 mutations, copy-mode
and header checks). That slice is reviewed; T02 remains PARTIAL.

### Patch inventory verification

Both decoder loader overloads retain owned input-projection BF16 bits and
exactly widened position-table values in `vision_patch_weights`. Config remains
in `vision.config`. This partial representation deliberately leaves blocks and
embed_projection empty; the projector remains in its existing owned field.
The dense materializer is unchanged: scaled FP8 matrices work; rank-three FP8
position tables remain explicitly rejected, not newly supported.

Red: 18 cases, 48 failed assertions, original 16 cases passing. Green: 20 cases,
356 assertions, zero skips, exact-count CTest PASS. Both overloads cover missing
inventory, malformed config/shapes/dtypes, overflow, absent modality, ownership
past shard destruction, raw layout and nonunit FP8 scaling. Seven direct shared
header consumers passed syntax-only compilation. All 28 metadata tests, record,
anchor and whitespace checks pass. Evidence: control-room
`assets/work/2026-10-10-e4b-vision`; captured pinned Transformers source SHA-256
`ccab8e2dd80b71e9ca34e2c87291e17c40a27c755006e554da2ebf70d6616916`.

T02 remains PARTIAL. Independent review and root verification of this patch
slice are pending. Encoder blocks, full typed tower config, clipping scalars,
audio tower and executable encoder/hook wiring remain owed. No inference,
original-weight, GPU or full-preflight claim follows from retention.

## T02 vision encoder-block retention

The parent reports rootbg363 and reviewf3c57a74 PASS for the patch slice:
20 cases, 358 assertions, 28 metadata tests; overflow mutation closed.
T02 remains PARTIAL. Retain all seven matrices and six learned norms per
encoder layer through both decoder loader overloads. Retention does not
establish encoder execution, clipping, audio, or runtime acceptance.

Pinned Transformers a08ace4bbd97e721c98751deec37d87b026acadc
`modeling_gemma4.py:168-215,692-701,927-954,990-1002,1036-1038`
defines `model.vision_tower.encoder.layers.{i}.` with attention
`self_attn.{q,k,v,o}_proj.linear.weight` and MLP
`mlp.{gate,up,down}_proj.linear.weight`. Shapes are respectively
[heads*head_dim,H], [kv_heads*head_dim,H] (K and V),
[H,heads*head_dim], [I,H] (gate and up), and [H,I].
The four layer norms have shape [H]; `self_attn.{q,k}_norm.weight`
have shape [head_dim]. V normalization has no learned scale. No biases,
V norm, rotary buffers, or projector norm tensors are invented. The pinned
vLLM `gemma4_mm.py:1046-1055` mapper preserves this tower structure.

Keep the existing raw BF16 block representation, including BF16 learned
norm scales from the BF16 checkpoint (the forward widens for RMS arithmetic).
Preserve dense matrix scaled-FP8 conversion. Norms require BF16, as the
existing representation requires; no new F32 checkpoint policy is introduced.
Validate positive integer layer count, intermediate size, head count,
KV head count and head dimension, head divisibility, and checked tensor and
block-vector byte sizes before allocation. Use source dimensions rather than
assuming H equals heads*head_dim. Full execution config remains pending.

Tiny two-layer original-structure fixtures must distinguish each tensor's
layout and each layer, survive shard destruction, and reject missing,
malformed, unsupported-dtype and overflow inputs through both overloads.
No original weights or large allocations are needed. Preserve patch/projector
behavior and absent vision. Full preflight remains pending and prohibited.

### Block retention verification

Both real decoder loader overloads now retain all 13 learned tensors per layer
in `vision_patch_weights.blocks`, using original `.linear.weight` names.
Validated local integer dimensions determine shapes, including asymmetric
query/KV widths; the full JSON config remains owned. This does not make the
existing MHA-only forward execute GQA, or supply clipping scalars. Full typed
execution config and encoder/hook wiring remain pending.

The initial red run failed 97 assertions; a malformed fixture initially failed
in the safetensors reader rather than the loader. After correcting its payload,
the complete suite against the original loader failed 251 assertions (23 cases,
19 passing). Restored implementation passes 23 cases / 1492 assertions, no
skips, through the exact-count CTest gate. The two-layer fixture distinguishes
every matrix and norm, raw layout, BF16 and nonunit scaled FP8 matrices,
shard-independent lifetime, missing inventory, dtype, config and byte overflow.
Nine direct header consumers pass syntax-only compilation with actual
compile_commands flags. All 28 metadata tests and offline CLI pass; runtime
acceptance remains false. Record and symbol-anchor checks pass.

Evidence: `assets/work/2026-10-10-e4b-vision-blocks`. Only the existing minimal
CPU target was built with two jobs. Full preflight remains prohibited/pending;
no GPU, original weights, install, remote operation or inference ran. The
implementation awaits independent review and the parent rerun. T02 is PARTIAL.

## T02 clipping and typed vision configuration

Prior block slice at 54e1150c1cebf1ac4aaecb60b9f085965b5f1576 is reviewed:
parent reports rootbg422 and review0d43 PASS, 23 cases/1492 assertions and
28 metadata tests. T02 remains PARTIAL.

Pinned Transformers a08ace4bbd97e721c98751deec37d87b026acadc
`modeling_gemma4.py::Gemma4ClippableLinear` (168-194) registers scalar
input_min, input_max, output_min, output_max on each of seven block linears
when use_clipped_linears is true; defaults are negative/positive infinity.
`configuration_gemma4.py::Gemma4VisionConfig` supplies false as the clipping
default, pooling 3, RMS epsilon 1e-6, default RoPE theta 100, tanh GELU,
no attention bias and no standardization. Google explicitly enables clipping.
Pinned vLLM a7c23ac96d7806e7c7e7d862eadbce5a33529b94
`gemma4_mm.py::Gemma4ForConditionalGeneration` constructs the Transformers
vision tower, replaces inner linears and maps the model.vision_tower prefix.

Header-only HTTP ranges from Google's immutable model.safetensors confirm
448 bounds (16 layers x 7 linears x 4), all BF16 scalar shape []. Names are
model.vision_tower.encoder.layers.{i}.self_attn.{q,k,v,o}_proj.{bound} and
model.vision_tower.encoder.layers.{i}.mlp.{gate,up,down}_proj.{bound}.
No bounds belong to patch/projector plain linears. The 281040-byte header's
SHA256 is 3d3bfac0fbdf7bd65981df4033efb7690240a8fe6f92eb2f3e3846b289a6f994;
metadata evidence resides in assets/work/2026-10-10-e4b-vision-clipping.
No tensor payload was fetched. This is inventory, not numerical acceptance.

Both real loaders shall require all bounds when clipping is enabled and
retain each as a float scalar for existing Clip consumers. BF16 widening is
exact and costs four bytes per bound, not matrix widening. Disabled clipping
requires no bounds and uses true infinite no-op defaults. Reject wrong shape,
wrong dtype, NaN or reversed bound pairs; permit infinities, including upstream
initialization defaults, and equal endpoints. The NaN/order checks are local
fail-closed policy against undefined/unintended clamp behavior.

Retain optional typed vision configuration alongside weights: all existing
forward dimensions, text width, epsilon, pooling, theta, clipping flag and
standardization/activation/bias/dropout semantics. Validate typed scalar inputs,
float epsilon conversion and pooling-square overflow. Reject unsupported
activation, bias, standardization and RoPE type rather than silently ignoring
them. Preserve GQA dimensions for inventory without claiming the MHA forward
executes them. No new encoder hook or execution support is in this slice.

Real-path tiny fixtures shall prove both overloads, layer-specific bounds,
shard-independent lifetime, defaults, shape/dtype/NaN/order/config/overflow
refusal and preserved BF16/scaled-FP8 matrix layout. Red precedes product
changes. Run only the minimal CPU target, actual-flags header syntax checks,
28 metadata regressions and focused record/style/trailer checks. Full preflight
and audio, hooks, original-checkpoint numeric/runtime gates remain pending.

### Clipping/config verification

Both real loader overloads now retain source-named BF16 scalar bounds and an
optional typed vision config. Disabled/absent clipping leaves infinite no-op
bounds; enabled clipping requires all four scalars for every linear. NaN,
reversed order, non-scalar shapes and non-BF16 bounds fail closed. Existing
matrix loading, including scaled FP8 conversion, is unchanged. Typed config
retains GQA dimensions but does not extend the existing MHA execution path.
Activation, bias and standardization are restricted to the existing forward's
semantics; dropout is retained for inventory but inference does not apply it.

Red before product changes: 27 cases, 23 passing, 922 failed assertions out of
2416; original 23 cases passed. Final green: 29 cases, 2648 assertions, zero
skips, plus exact-count CTest 1/1 PASS. Fixtures cover every bound on both
layers through both loaders, lifetime, infinite/equal endpoints, explicit false,
missing configuration, overrides, malformed types, NaN/Inf config and overflow.
Existing BF16/scaled-FP8 layout tests remain green. Nine direct header consumers
pass fresh syntax compilation using actual compile_commands flags. All 28
metadata tests and offline metadata CLI pass; runtime_accepted remains false.
Record and symbol-anchor checks pass. Evidence is in the clipping scratch
folder above (under 1 MiB); only the reused minimal CPU target was built, -j2.

No preflight was launched on resume. Process inspection found no remaining
process from the earlier mistaken, timed-out preflight; none was killed.
Full preflight is pending, not green. T02 is PARTIAL: audio inventory and
encoder/hook wiring remain owed. Original-checkpoint inference, numerical
acceptance, GQA execution and runtime prerequisites are not established.
Root verification and independent review of this slice remain pending.

## T02 audio frontend inventory refinement

Vision clipping at 8c94b97e4e54a357304232cee8c3597209c6f6fb is reviewed:
parent reports rootbg452 and review5ec77653 PASS (29 cases/2648 assertions,
28 metadata tests). T02 remains PARTIAL.

This bounded slice retains the complete subsampling frontend, not the Conformer
blocks or output projection. Pinned Transformers a08ace4bbd97e721c98751deec37d87b026acadc
`modeling_gemma4.py::Gemma4AudioSubSampleConvProjection` and
`Gemma4AudioSubSampleConvProjectionLayer` define two biasless 3x3 Conv2d weights,
two biasless LayerNorm weights (plain scale, not 1+w), and a biasless input
linear. `configuration_gemma4.py::Gemma4AudioConfig` defaults channels to
[128,32], epsilon to 1e-6. The input linear width is explicitly
(channels[0] // 4) * channels[1]; do not guess a feature_size config field.
Pinned vLLM a7c23ac96d `gemma4_mm.py::Gemma4ForConditionalGeneration` maps
model.audio_tower to audio_tower and constructs this Transformers tower.

The already-downloaded Google header (SHA256 above) confirms all five weights
are BF16: model.audio_tower.subsample_conv_projection.layer0.conv.weight
[128,1,3,3], layer0.norm.weight [128], layer1.conv.weight [32,128,3,3],
layer1.norm.weight [32], input_proj_linear.weight [1024,1024]. All suffixes
share that prefix. No frontend biases or clipping scalars exist. This slice
must not change block clipping defaults or imply their inventory is loaded.

Both actual decoder loaders shall retain these tensors in the existing
Gemma4AudioWeights host-F32 representation, preserving row-major Conv2d and
[out,in] linear layout. This is reference storage, twice the BF16 payload size,
not production BF16 execution. Retain separately typed frontend dimensions and
epsilon, with owned full JSON already on audio.config. Reject malformed or
missing required hidden_size/epsilon, non-two-element/nonpositive channels,
zero projected width, nonrepresentable float epsilon, wrong shape/dtype and
checked F32 size overflow before tensor lookup/allocation. Missing channels
use source defaults; explicit null is invalid. Optional absent audio remains
empty. Output projector fallback semantics and scaled-FP8 connector behavior
remain unchanged. Frontend weights require actual BF16, not guessed FP8 support.

Tiny original-structure fixtures precede implementation and exercise both
entrypoints, each tensor's contents/layout and shard-independent lifetime,
missing tensors, shape/dtype/config and exact overflow diagnostics. Preserve
all prior loader tests, actual-flags header checks and 28 metadata regressions.
Full preflight remains pending and prohibited for this bounded task.

Remaining source inventory: 746 of 751 audio_tower tensors: output_proj.weight
[1536,1024], output_proj.bias [1536], plus 744 tensors across 12 Conformer
layers (62 each, including clipping). The existing embed_audio connector is
already retained separately. Relative positional encoding has no persistent
header tensor. Typed block/output config, block and output weights, encoder
hooks, production dtype execution and all numeric/runtime gates remain owed.

### Audio frontend verification

Both decoder overloads now reach LoadAudioFrontend through their shared loader.
The five source-named BF16 tensors are widened exactly into owned host-F32
vectors in audio_frontend_weights. Typed audio_frontend_config records only
validated frontend fields; full JSON remains in audio.config. No frontend
clipping was invented. Existing connector fallback and scaled-FP8 behavior,
text and vision tests remain passing after adding required frontend fixtures.

Red before loader implementation: 33 cases, 29 passing, 67 failed assertions.
Final CPU green: 33 cases / 3002 assertions, zero skips; exact-count CTest PASS.
Tests exercise both entrypoints, distinct tensor/element contents, row-major
layout, lifetime after fixture destruction and keepalive reset, missing/rank/
dtype errors, channel types/defaults, missing config fields, epsilon narrowing,
and exact frontend tensor-size overflow diagnostics before tensor lookup.
Nine affected consumers pass syntax checks with actual compile_commands flags;
28 metadata tests and offline CLI pass. Record and symbol-anchor checks pass.
Evidence: assets/work/2026-10-10-e4b-audio-loader. Only the reused eight-TU CPU
target was built with -j2; incremental scratch stays below 256 MiB.

T02 remains PARTIAL, awaiting root verification and fresh independent review.
The remaining 746 audio_tower tensors, typed block/output config, encoder hooks,
production BF16 execution and numerical/runtime gates are not supplied here.
Full preflight remains pending and was not run, as authorized for this slice.

## T02 complete audio tower retention contract

Frontend b5360a5fb0ecaa2e9e4242245776b66f720623fe is reviewed per parent
rootbg37/review62e: 34 cases, 3004 assertions, 28 metadata tests PASS.
T02 remains PARTIAL; this change completes inventory, not inference.

Pinned Transformers a08ace4bbd97e721c98751deec37d87b026acadc
modeling_gemma4.py::Gemma4AudioLayer, Gemma4AudioFeedForward,
Gemma4AudioAttention, Gemma4AudioLightConv1d, Gemma4ClippableLinear and
Gemma4AudioModel define the following inventory. configuration_gemma4.py::
Gemma4AudioConfig defines defaults. Local source artifacts are in
assets/work/2026-10-10-e4b-vision and -vision-clipping; pinned vLLM
planning/e4b/sources/vllm-gemma4_mm.py constructs the Transformers audio tower and
maps model.audio_tower to audio_tower. The existing downloaded header is
assets/work/2026-10-10-e4b-vision-clipping/header.json (no payload read).

Under model.audio_tower.layers.{0..11}, H=1024, D=128, K=5:

| Suffix | Shape | Count/layer |
|---|---|---|
| feed_forward{1,2}.ffw_layer_1.linear.weight | [4H,H] | 2 |
| feed_forward{1,2}.ffw_layer_2.linear.weight | [H,4H] | 2 |
| feed_forward{1,2}.{pre,post}_layer_norm.weight | [H] | 4 |
| self_attn.{q,k,v}_proj.linear.weight | [heads*D,H] | 3 |
| self_attn.post.linear.weight | [H,H] | 1 |
| self_attn.relative_k_proj.weight | [heads*D,H] | 1 |
| self_attn.per_dim_scale | [D] | 1 |
| lconv1d.linear_start.linear.weight | [2H,H] | 1 |
| lconv1d.linear_end.linear.weight | [H,H] | 1 |
| lconv1d.depthwise_conv1d.weight | [H,1,K] | 1 |
| lconv1d.{pre_layer_norm,conv_norm}.weight | [H] | 2 |
| norm_{pre_attn,post_attn,out}.weight | [H] | 3 |
| each of ten clippable linears: input_min/input_max/output_min/output_max | [] | 40 |

These 62 tensors per layer total 744. output_proj.weight [1536,H] and
output_proj.bias [1536] complete the remaining 746; with the five frontend
weights, 751 source audio tensors are accounted for. Every listed tensor is
BF16 in the header, including scalars. Accept only BF16 and widen exactly to
owned host-F32 reference vectors, preserving source row-major layout, without
transposes, 1+w transforms or finite sentinel substitution. This deliberate
reference-storage exception is not a device dtype claim. All norms have plain
learned scale and no bias. Only output_proj has bias. GLU is activation gating,
not an extra learned gate. Relative positions and softcap are nonpersistent.
No new representation is needed for these weights; fill the existing
Gemma4AudioWeights on audio_frontend_weights, plus optional typed audio_config.

Retain validated source defaults for layers(12), heads(8), kernel(5),
chunk(12), left(13), right(0), residual(0.5), softcap(50), invalid logits(-1e9),
gradient clipping(1e10), activation(silu), clipping(true). Frontend dimensions
and epsilon remain validated by its existing contract. Retain output width
using the existing connector compatibility fallback: absent/null/zero means H;
a positive output_proj_dims sets the width for both output and connector.
Head dimension is source integer H/heads (require nonzero); retain dimensions,
not a claim that the existing host forward supports every such configuration.
Validate numeric types, finite F32 narrowing, positive dimensions and checked
matrix/layer/context products before lookup or allocation. Preserve full JSON.
When clipping is false, bounds stay true +/-infinity; when true require all
four BF16 rank-zero bounds, allow infinities/equality, reject NaN or reversed
pairs with tensor-specific diagnostics. Reject incorrect shapes/dtypes/missing
weights through both real decoder loaders. Optional absent audio stays empty.

Tests precede behavior: multiple tiny source-structure layers, all weights and
bounds distinct across layers/elements, lifetime after shard destruction,
precise negative diagnostics for every inventory entry, clipping and config
errors/overflow. Reuse only eight-TU CPU target with -j2, actual compile-command
header checks, 28 metadata regressions and record/anchor/style/trailer checks.
Scratch assets/work/2026-10-10-e4b-audio-blocks has 256 MiB incremental budget,
external TMPDIR and Git ceiling; stop below 4 GiB free. Full preflight remains
pending and prohibited in this bounded task. Commit implementation then request
root verification and independent review. Encoder hooks, preprocessing,
production device BF16, numeric/runtime acceptance and original-checkpoint
execution remain owed even if all 751 tensors can be retained.

### Complete audio inventory verification

Both decoder entrypoints now call LoadAudioBlocks after their shared frontend
loader. The existing audio_frontend_weights owns all tower vectors; its name is
historical. Optional audio_config retains validated block/output fields and full
JSON stays on audio.config. No separate disconnected inventory container or
execution hook was introduced. BF16-only matrices, norms, scale, bias and bounds
are exactly widened without layout changes. Disabled bounds use true infinity.

Corrected red: 36 cases, all original 34 passing; 757 failed assertions out of
3761. The initial malformed-shape fixture changed byte count and was rejected by
the safetensors reader; appending a singleton dimension preserves byte count
and reaches the actual loader shape guard. The old frontend empty-block checks
were retained for that red, then changed to require the newly retained layers.
Final green: 41 cases / 6866 assertions, zero skips, exact-count CTest 1/1 PASS.
Two layers have distinct matrix/norm/scale/bias elements and every clipping
bound; a separate tiny twelve-layer fixture tests source layer/head defaults.
Every one of the 126 block/output tensors in the two-layer fixture is tested
missing, wrong rank and wrong dtype through both overloads. Every bound gets
NaN rejection, every pair gets reversed-order rejection, and disabled/infinite/
equal bounds pass. Config tests cover defaults, overrides, type/numeric refusal,
F32 narrowing and checked size/context overflow before block lookup/allocation.

Nine prior direct header consumers plus the audio source and audio tower test
pass syntax checks using actual compile_commands flags. All 28 metadata tests,
offline CLI, record and symbol-anchor checks pass. Evidence and reproduction
commands: assets/work/2026-10-10-e4b-audio-blocks. The reused CPU target alone was
built with -j2; scratch is below 1 MiB and reused build below 70 MiB; free disk
remains above 13 GiB. Full preflight was not run and remains pending.

The 744 block plus two output tensors complete the source inventory of 751
including the frontend, but no original checkpoint payload was loaded. T02
remains PARTIAL. Production BF16, encoder hooks, preprocessing, numerical gates
and original-checkpoint runtime acceptance remain pending. Root rerun and fresh
independent review of the implementation commit are required.

## Loaded audio execution slice

Complete retention at 220df15a3b7287475f990e26d353ba61f0d438c1 is reviewed:
root supplied rootbg68 and reviewer aa1b137c PASS (41 cases / 6866 assertions,
28 metadata tests; 751-header inventory matched). This is reported evidence,
not a new rerun and not runtime acceptance.

Pinned HF a08ace4bbd97e721c98751deec37d87b026acadc Gemma4AudioModel.forward and
Gemma4AudioSubSampleConvProjection downsample the input mask twice; pinned
vLLM a7c23ac96d Gemma4MultimodalEmbedder.forward applies weightless RMSNorm
BEFORE its linear despite contradictory class prose. _process_audio_input
selects the output mask's valid rows. Local artifacts are the previously
recorded modeling_gemma4.py and planning/e4b/sources/vllm-gemma4_mm.py.

Actual seam gaps: kGemma4Factory has neither encode_mm nor embed_mm;
AudioKwargs is mel-major while Gemma4AudioForward is frame-major; the retained
connector is BF16 OwnedTensor while the forward requires host-F32 embed_proj.
The forward does not apply its input padding mask to attention and assumes
right context zero. Vision uses backend operations and still has a GQA/MHA
execution mismatch. Audio is the bounded next execution dependency.

Implement the registered audio encoder arm using the shared MmEncoderOutput
carrier, with owned CPU F32 output. Keep embed_mm absent: SupportsMmInputs must
remain false until a working merge hook exists. The processor/server milestone
is not claimed by a directly callable encoder hook. Expose the same encoder
implementation to a tiny loaded-weight test, not a disconnected alternate
helper. Unsupported modalities and non-CPU queues fail closed.

Add an explicit optional AudioKwargs frame mask (empty means all valid), with
strict boolean, exact-length, nonempty contiguous-valid-prefix validation.
Support one chunk only. Transpose mel-major input. For causal/right-zero
attention, trim trailing padding before forward; valid results cannot depend
on future frames. The output row count is ceil(valid_frames/4), equivalent to
mask[::2][::2] selection; require the placeholder count to match. Reject
unsupported activations, uneven heads, odd hidden dimensions, nonzero right
context, empty past window, and incompatible mel/input-projection widths.
Do not silently execute unsupported loaded configurations. Preserve clipping
bounds. Convert only the connector from owned BF16 to F32 for the existing
forward, without copying the full tower. Keep production BF16 execution owed.

T02 remains partial. This necessary encoder dependency is split ahead of the
original T03 template / T04 image / T05 audio / T06 video chain; none is marked
done. Full raw-input processing, merge hooks, decoder execution, numerical
acceptance (T01B), and production dtype remain pending. Tiny source-derived
scalar fixtures establish bounded connector arithmetic and call reachability,
not full encoder numerical parity. Tests cover wrong inputs, meaningful
outputs, padding equivalence, and deletion kills. Build only real required TUs
in the existing minimal CPU target; record actual commands, regression counts,
header checks and mutations in assets/work/2026-10-10-e4b-encoder-integration.
Full preflight remains pending and is prohibited for this bounded task.

### Encoder execution verification

Spec 7831faccd preceded code. The production factory now copies its MM hooks
from constant-initialized kGemma4MmHooks, whose audio hook calls the existing
Gemma4AudioForward with the loaded tower and separately widened connector.
Gemma4LoadedModel moved unchanged in ownership semantics to gemma4_loaded.h;
BorrowedWeightsTag is forward-declared, used by reference, and remains defined
only in qwen3_5_common.h. No test stubs or alternate encoder were linked.

Linked intentional red: 42 cases, original 41 passing, execution case throws
"not implemented". Earlier missing-tag / NextQueueId build errors were setup
failures, not runtime red. Final exact-count CTest: 43 cases / 6903 assertions,
zero skips (the original 41 / 6866 included). The two new cases dispatch through
ModelRegistry::EncodeMm using the actual production MM hook table and real
loaded models from both loader variants. One fixture independently checks
[1,2,3,4,5] bias, RMS mean-square 11 and two connector selectors. A sparse
center-tap fixture checks sign-dependent frontend output against sqrt(5),
layout sensitivity, padding removal and changed features. These do not prove
full Conformer numeric parity. Invalid masks/shapes/chunks/lengths/modalities,
nonfinite input, non-CPU queues and unsupported loaded attention configs fail.

Four scratch-only compiled and linked mutations are killed: remove production
MM hook, corrupt mel layout, remove connector RMSNorm, zero frontend output.
The production registry TU passes actual-flag syntax checking, but its entire
text factory and decoder are NOT linked into the bounded test. Thus the proof
boundary is the production hook table plus shared registry dispatch, not
RegistrationFor resolution, engine admission or decoder execution. embed_mm
stays null and SupportsMmInputs remains false; this is a necessary executable
encoder dependency, not a served multimodal capability.

Only the minimal loader target was built with -j2, adding real encoder, audio,
registry, tensor and backend TUs with section collection. Eleven actual-command
header consumers, 28 metadata tests, offline metadata CLI, records, anchors and
whitespace pass. Scratch is under 1 MiB; reused build is 71 MiB, free disk above
13 GiB. Evidence and reproducible commands are in
assets/work/2026-10-10-e4b-encoder-integration. Full preflight remains pending
and was not run. T03-T06, merge hook, production BF16, T01B and original-model
runtime acceptance remain pending. This slice awaits root verification and
fresh independent review; previous retention review is not reused as its review.

### Audio epsilon correctness refinement

Review 2b63b rejects e1c27f951: positive configured epsilon is accepted, but
Gemma4AudioForward incorrectly applies it to every norm. Pinned HF
`a08ace4bbd97e721c98751deec37d87b026acadc`, modeling_gemma4.py:198,
423-424 and 535-537 uses default **1e-6** for FF pre/post norms (both FFs),
layer pre/post-attention and output norms. Keep configured rms_norm_eps for
subsampling LayerNorm (:372, :394-399), light-conv RMS (:499-500), and
weightless connector RMS (pinned vLLM Gemma4MultimodalEmbedder.forward).
Do not reject otherwise valid nondefault epsilon or blanket-change all norms.

Before behavior changes, add a small nonzero sparse Conformer fixture through
loaded weights and the shared registry hook. An independent double scalar
oracle must propagate frontend LayerNorm/ReLU, both FF residuals, single-token
attention, light-conv GLU/SiLU, layer output norm and connector. Use small
exactly BF16-representable weights and distinct configured epsilon; compare
numeric outputs, not the production forward against itself. Preserve all 43
existing cases and 28 metadata tests. Capture actual red mismatch then green;
compiled scratch mutants must kill reversion of fixed norms and changing
configured frontend/light-conv/connector norms to fixed epsilon. Keep the
minimal 15-real-TU CPU build at -j2 and scratch below 256 MiB. Evidence belongs
in assets/work/2026-10-10-e4b-audio-epsilon-fix. No headers or capability changes
are needed. Full preflight, T03-T06 and runtime acceptance remain pending;
no GPU, weights, installation, remote operation or full preflight is authorized.

### Audio epsilon correction verification

Spec ea30d690a precedes this correction. Five block callsites now use fixed
1e-6; frontend, light-conv and connector retain configured epsilon. The new
loaded scalar oracle uses two sparse nonzero blocks, one token, BF16-exact
weights, eps=1e-4/1e-2 and both loader ownership paths through shared registry
dispatch. Initial red: 43/44 cases pass, 6 numeric assertions fail (7139 total).
After increasing the light-conv output weight to expose its epsilon contract,
all eleven compiled/linked scratch mutants fail numeric assertions: all five
fixed callsites reverted together, each separately, each frontend norm, each
light-conv norm and connector independently switched to the wrong epsilon.
An initial mutant compile error from the unused constant is not a kill; the
scratch reversion removes that constant without changing compiler flags.

Final minimal 15-real-TU CPU gate and exact-count CTest pass 44 cases / 7139
assertions, zero skips. All original 43 cases remain. The 28 metadata tests,
offline CLI, records and symbol anchors pass. No headers changed. Evidence:
assets/work/2026-10-10-e4b-audio-epsilon-fix. The fixture proves these epsilon
contracts, not general multi-token Conformer parity. SupportsMmInputs remains
false; embed_mm is absent. Full preflight, original-checkpoint execution,
production BF16, T03-T06 and runtime acceptance remain pending. Await root
verification and independent cumulative review 2b63b of the committed head.

## T05A executable audio preprocessing prerequisite

The loaded CPU audio slice at d021e05980db9157f4324d5b05e3c016f4e6c78a is
reviewed per parent rootbg166 and review2b63: 44 cases/7139 assertions,
28 metadata tests and 15 mutation kills. This is supplied acceptance evidence,
not a claim of a rerun here. encode_mm exists for CPU host F32 only;
embed_mm remains absent, SupportsMmInputs false. Full registry resolution and
served runtime have not been established.

Plan change before implementation: T05A depends on T01A metadata and the
accepted audio consumer, not T04 image preprocessing. Its typed waveform to
AudioKwargs transformation has no image/template code dependency. T03 template,
T04 image and T06 video remain PENDING. T05 stays PARTIAL until prompt expansion,
shared factory/server routing, resampling policy parity and original upstream
oracle gates are complete. This prerequisite uses the existing audio processor
ProcessWaveform interface and shared AudioKwargs, not a standalone executable
API. No image factory or served capability is fabricated.

### Exact source contract

Transformers a08ace4bbd97e721c98751deec37d87b026acadc,
`src/transformers/models/gemma4/feature_extraction_gemma4.py`,
Gemma4AudioFeatureExtractor.__init__, _extract_spectrogram and __call__:
- 16000 Hz, 128 filters, 320-sample periodic Hann cast to float32; hop 160,
  FFT512. Float32 waveform/window multiplication precedes double FFT.
- Left 160 zeros, no centered reflect padding. Unfold windows have **321**
  samples, discard the last sample before window/FFT; right zero pad to 512.
- Magnitude (NOT power); HTK mel centres, triangles in Hz, norm=None.
  Double filter coefficients, natural log(magnitude-mel + 0.001), not clamp.
- No dither, preemphasis, waveform scaling or per-bin normalization for the
  pinned processor_config.json. No inference from Whisper/Parakeet semantics.
- Single-item longest padding rounds waveform length up to multiple 128.
  End sample i*160+320 in left-padded waveform determines mask validity;
  invalid mel rows are zeroed after float32 conversion. Transpose upstream
  [frames,128] into shared AudioKwargs [128,frames]. num_samples is retained
  input length; num_tokens = ceil(valid_frames/4), max 750.
- Source __call__ defaults truncate at 480000 samples. Pinned vLLM
  gemma4_mm.py over-long audio warning derives the same 30 seconds from
  750*40ms and 16000 Hz. This bounded API rejects lengths beyond 480000
  instead of silently truncating; this is an explicit validation divergence.
  Other rates are rejected, not silently relabeled or passed through the
  shared scipy resampler (which is not default PyAV parity). Rate conversion
  belongs before this frontend and remains pending. Empty and <=160-sample
  inputs return zero valid tokens; negative length, null nonempty buffer,
  nonfinite samples and unsupported rate throw before allocation/processing.

Source helpers at the same SHA: audio_utils.py::mel_filter_bank,
_create_triangular_filter_bank, hertz_to_mel, mel_to_hertz and window_function;
feature_extraction_sequence_utils.py::SequenceFeatureExtractor._pad/_truncate.
Retrieved source SHA256s (scratch retains exact bytes):
feature_extraction_gemma4.py 44101c7ce4286e2e37ca10629b97a5c09c53b3862ad21ce6f9ae35d3d87d76d4;
audio_utils.py 1289c67d1b63a8a56c26ece6ee9f5c9e9e319433f095e492fb8bb3a530349875;
feature_extraction_sequence_utils.py db6b08c4847ba853194066ae00d438bed672acbefcad6011a424824dca04e12d.

### Shared seams and tests

Existing Whisper/Parakeet frontends use private scalar DFTs with incompatible
framing. w2v_fbank.cpp has a reusable private radix-2 double FFT: extract its
unchanged arithmetic into a shared header and use it from both callers, adding
a power-of-two guard. Extend shared mel_filter_bank with double HTK norm=None
without altering Slaney callers. No new runtime dependency is required.

Add a bounded CPU target compiling actual processor/mel TUs. Independently
compute scalar DFT and mel triangles in tests (no production helpers for
expectations), test silence, impulse, deterministic multitone, short/empty,
128/160 framing boundaries, trailing-mask zeros, 30-second cap and malformed
inputs. Exercise returned data through existing loaded encoder consumer where
feasible; synthetic tests do not establish full oracle parity. Retain and run
44/7139 loader regression and 28 metadata tests. Check changed-header consumers
with actual compile_commands flags. External scratch:
assets/work/2026-10-10-e4b-audio-preprocessing; -j2, incremental <256 MiB,
external TMPDIR and Git ceiling; stop below 4 GiB free. Full preflight remains
prohibited/PENDING. No GPU, weights, installs, broad library builds or remote
writes. Implementation must await root verification and fresh review.


### T05A implementation evidence

Executable ProcessWaveform returns existing AudioKwargs, including masks and
valid-token count, and is exercised through the loaded CPU encoder hook.
Shared FFT extraction leaves w2v arithmetic unchanged; existing golden suite
passes 6 cases/20973 assertions. New frontend suite passes 5 cases/5225
assertions: independent scalar DFT, analytic silence/boundary table, malformed
inputs, cap, FFT and 640 pinned-source NumPy values. Source capture uses
unmodified AST method bodies, not full processor __call__; explicit padding
is supplied. The source fixture and scalar tests are not full oracle parity.

Loader suite passes 45 cases/7149 assertions, including all original 44/7139,
zero skips; metadata 28/28 and exact-count CTest 2/2 pass. Consumer fixture
uses 128 mel channels with constant output projection to independently assert
rows/scalars, placeholder mismatch and no-valid-frame refusal; waveform-driven
encoder numerical parity is not inferred. Eight actual-command header consumer
checks pass. Three compiled scratch mutations fail: power instead of magnitude,
clamp instead of additive floor and off-by-one mask. No red-before run was
captured; these are post-implementation mutation checks. Evidence and exact
reproduction are in assets/work/2026-10-10-e4b-audio-preprocessing/README.md.

Source citation correction: the earlier `_get_max_audio_len` name was not an
upstream symbol; the 480000 cap is directly __call__'s max_length default,
consistent with the processor's 750*40ms budget. It is NOT the dummy input
profiling length (which returns fft_length for this extractor). Rate refusal
and over-cap refusal are explicit bounded API choices, not upstream parity.

Await root verification and fresh review. No capability, T03/T04/T06, runtime,
production-BF16, full-registry or full-preflight gate was promoted.

### T05A prospective process correction (2026-10-11)

Independent review found technical PASS but process FAIL at 23c1b906:
required preimplementation RED was not captured. That historical gap remains
FAIL, not waived. Neither review mutations nor this correction retroactively
satisfy it. Original spec and implementation commits remain unchanged.
ISSUE-LOCAL-01M4KSETGY63TDEDPF6EHQHRA8 stays OPEN; independent review and root
policy disposition are required before acceptance.

The fresh corrective implementer verified HEAD is exactly 23c1b906 and no
tracked changes exist. The prospective sequence is: commit this plan, roll
back only T05A production behavior, retain pinned tests/fixtures, capture
buildable RED, commit the rejection state, then implement the same complete
source contract and capture GREEN. Retain processor declarations and CMake
plumbing so tests link. Replace processor and shared FFT behavior with explicit
unavailable exceptions; restore private w2v FFT and pre-T05A mel construction.
The rejection state is temporary remediation, not accepted runtime behavior.
Valid-waveform and valid-FFT tests must fail at those explicit refusals, not
at compiler/linker errors. No test, fixture, expected value or tolerance changes
are permitted. Restoration may reuse reviewed arithmetic; it is not a claim
of independent rediscovery or a replacement for historical RED.

Evidence: assets/work/2026-10-10-e4b-audio-correction. Use only named CPU gates,
-j2 maximum, external TMPDIR/Git ceiling, under 256 MiB incremental scratch and
stop below 4 GiB free. Run both frontend/loader gates, metadata 28, w2v goldens,
actual-command header checks and focused records/style/trailers. Full preflight
is explicitly prohibited and remains PENDING, as do the original oracle,
served capability and T03/T04/T06. No GPU, weights, installation or remote
operation. Existing untracked plans remain untouched. This is a new corrective
chronology only; status remains awaiting independent review, process gap NOT
waived.

Corrective RED captured before repair: run.py red freshly compiled and linked
both actual test targets (max two compiler processes), then each executable
returned 1. Frontend: 0/5 cases pass, all 11 assertions pass; all five fail at
explicit missing frontend/shared FFT exceptions on valid input. Loader:
44/45 cases pass, all 7139 assertions pass; only the waveform-consumer case
fails at the frontend refusal. Thus existing loader behavior remains green
while required missing functionality is RED, without broken includes or links.
Logs and exact red patch are in the correction evidence directory. Invalid-input
CHECK_THROWS passing under blanket refusal proves no validation guarantee;
only valid-input failures constitute this RED evidence. No repair has yet been
applied, and historical missing RED remains unsatisfied.

Counter correction: 78ea4cbb0 incorrectly added case exceptions to assertion
denominators. Actual RED logs contain 11 and 7139 passing assertions, zero
failed assertions, and five/one case exceptions respectively. This correction
is recorded in a new commit, without amending the RED-state history.

### T05A corrective repair verification

Plan ae1b9055e preceded RED-state commit 78ea4cbb0; both remain in history.
After that captured RED, the repair restores the complete bounded frontend,
HTK mel construction and shared radix-2 FFT, including w2v's shared caller.
Reviewed arithmetic is reused byte-for-byte from 23c1b906, not independently
rediscovered. There are no remaining rejection placeholders and no changes to
tests, fixtures, tolerances or CMake. Relative to 23c1b906 the final product and
test trees are identical; the corrective history and this record are new.

Fresh real-TU builds and direct runs pass frontend 5 cases/5225 assertions,
loader 45/7149 and w2v 6/20973, all with zero skips. Named CMake targets built
at -j2 and exact-count CTest passed 2/2. Metadata passes 28/28; all eight header
consumers pass with actual compilation flags. Records, symbol anchors and diff
whitespace checks pass. Evidence and reproduction commands remain under
assets/work/2026-10-10-e4b-audio-correction. Scratch is 19 MiB (existing bounded
build separately 72 MiB); available disk is 14 GiB. No broad build, full
preflight, GPU, weights, installation or remote operation ran.

Status: corrective implementation awaits root verification and independent
review of the new repair commit. Original missing preimplementation RED is
still historical process FAIL, NOT waived and NOT retroactively satisfied.
The open issue, T05 PARTIAL, full preflight and original runtime/oracle gates
remain unchanged. The final policy disposition belongs to root after review.

### Bounded shared embedding merge (prospective)

T05A corrective scope is reviewed per rootbg242/review39093660: frontend
5/5225, loader 45/7149, metadata 28. Historical 23c1b906 missing RED remains
FAIL; no further rollback or retrospective waiver is intended.

This prerequisite uses the existing ModelRegistry::EmbedMm and ForwardMm ABI,
not another runner. The shared ENG-MM-INPUT-PIPELINE seam is extended only by a
factory readiness boolean defaulting true. SupportsMmInputs must additionally
require readiness. Gemma sets it false: CPU reference encode/merge callable
hooks are not served-runtime capability. Other factories retain their existing
semantics. Direct dispatch remains callable for bounded verification.

Pinned source: planning/e4b/sources/vllm-gemma4_mm.py embed_input_ids maps MM
positions to PLE id zero before get_per_layer_inputs; the existing text
ForwardBody and EmbedScaledBf16 implement BF16-rounded sqrt(hidden) scaling.
Merge emits owned BF16 [T,H] and I32 [T] PLE buffers for existing ForwardMm.
Text lookup is scaled and rounded before audio overwrites, audio is rounded to
BF16 without text scale, PLE text ids outside vocab_size_per_layer_input map
to zero. CPU only; stale device ids, nonempty MRoPE, invalid token ids, masks,
cardinality, dtype, shape, stride, nonfinite values and empty steps fail closed.
Ordered encoder row slices are already chunk-relative; merge does not apply
prefill offsets again. Multiple ordered slices are supported.

MmEmbedInputs has no original spans or modality metadata. Duplicate indices,
overlap and source range are upstream admission/gather responsibilities, not
claims of this merge. Tests will exercise represented slice/cardinality errors
and chunk slices; original-span admission coverage remains a separate shared
pipeline gate. Image/video encoding and CUDA remain refused. No full media
capability is advertised.

Fresh executable RED must precede behavior, using the existing real-TU CPU
loader target. Then verify independent text/audio/PLE expectations, ownership,
malformed inputs and capability default compatibility. Attempt real production
factory resolution without stubs; document concrete linkage limits if bounded
linking cannot reach it. Retain frontend, loader and metadata regressions and
actual-flags header consumers. Scratch assets/work/2026-10-10-e4b-embedding-merge
is capped at 256 MiB, external TMPDIR/Git ceiling, -j2, stop below 4 GiB free.
Full preflight, GPU, weights, installs and remote operations prohibited. T03,
T04, T06, device BF16, original oracle and served-runtime acceptance pending.

### Shared merge implementation evidence

Fresh RED after 7489c1de9: loader 44/45 cases, one valid-input exception at
missing embed_mm, 7146 passing assertions; frontend 5/5225 PASS. GREEN:
46/7194 loader, 5/5225 frontend, 28 metadata, exact-count CTest 2/2, 144 actual
flags consumers PASS. Production hook dispatch now returns owned BF16 merged
embeddings and I32 PLE ids; readiness remains false. Ordered offset slices,
independent text/audio values, PLE masking and malformed represented inputs
are tested. Original-span validation remains upstream and unproven here.

Actual production registry TU compiles but bounded linking fails on decoder,
resident-expert and KV-spec dependencies; see factory-link.log. No stubs or
full-library build. Thus full factory resolution/ForwardMm execution is not
claimed. Evidence/reproduction: assets/work/2026-10-10-e4b-embedding-merge.
Await root verification and fresh review. T03/T04/T06, production device BF16,
original oracle, served runtime and full preflight remain pending.

### T04A decoded RGB prerequisite (prospective)

Shared merge is reviewed per rootbg311 + reviewefcd: loader 46/7196,
frontend 5/5225 and metadata 28 PASS. SupportsMmInputs remains false and
full factory execution remains unproven. T03 template stays PENDING. This
prerequisite consumes already-decoded RGB bytes, not messages or placeholders;
its patch transform has no dependency on template expansion. It does not
complete T04 or bypass T03 for a full processor/prompt pipeline.

Source audit: Transformers a08ace4bbd97e721c98751deec37d87b026acadc,
image_processing_gemma4.py::get_aspect_ratio_preserving_size,
Gemma4ImageProcessor.aspect_ratio_preserving_resize/_preprocess,
convert_image_to_patches and pad_along_first_dim; image_processing_backends.py::
TorchvisionBackend.rescale_and_normalize/rescale. Google metadata stays pinned
to ee0ef6023621cff504d758262d4e04895a5af4a2. Defaults: patch 16, pooling 3,
280 soft-token budget => 2520 padded patches. Aspect ratio target uses square
root area scaling then floors each axis to multiples of 48, with thin-axis
fallback capped at budget*48. No crop. Resize is torchvision tensor BICUBIC
with antialias=True, NOT shared Pillow's fixed-point bicubic or Qwen's
non-antialiased algorithm. torchvision is absent locally. Do not install or
substitute: support exactly the upstream identity-resize branch, refusing
all other dimensions with an explicit missing antialiased-resize error.
All five upstream budgets (70,140,280,560,1120) are supported; patch and pooling
are fixed to checkpoint values. RGB conversion/decoding, batching and optional
normalization/rescale overrides are outside this decoded uint8 API.

Use the shared HWC uint8 convention with a sized span (exact length checked),
and reuse ImageKwargs for owned float32 patch storage and grid. Add a Gemma
result wrapper for xy positions and soft-token count; no Qwen temporal patch
packing, normalization or BF16 cast. Patch rows are raster y,x; within each
row pixels are y,x,channel. Multiply by float32 1/255; do not pre-apply the
tower's 2*(x-.5). Padding is float zero and (-1,-1); there is no upstream
separate attention mask. Positions define the tower's contiguous valid prefix,
with soft-token count valid_patches/9 (not always the budget).

Extract the existing tower valid-prefix input boundary into a shared callable
validator used by the actual Gemma4VisionForward and tests. Validate shapes,
finite [0,1] values, raster xy positions, pooling divisibility and zero trailing
padding before the tower creates a queue. This provides processor-to-actual
input-contract coverage, not backend queue/tower numerical inference. Preserve
existing shared image callers without altering their layouts or resampler.
Explicit bounds: decoded dimensions <= 65536 and <= 64 Mi pixels; reject
nonpositive, mismatched length, overflow-sized dimensions, unsupported budgets
and required resize before allocating output. Output max ~31 MiB at 1120.

Fresh executable RED after this spec commit must fail valid identity inputs
through the production processor interface, not compilation/linking. Then
implement, retaining independent scalar patch expectations and pinned source
geometry fixtures, including portrait/landscape, all budgets, exact values,
layout, padding/mask/count, malformed and unsupported inputs. Port applicable
upstream test_image_processing_gemma4.py tests (defaults/budgets/positions/zero
padding); generalized patch/pooling, full resize, PIL/numpy/tensor wrappers and
batch tests are deferred, not silently skipped. No production inference claim.

Use new named CPU real-TU target, -j2, external TMPDIR/Git ceiling in
assets/work/2026-10-10-e4b-image-preprocessing, <=256 MiB incremental scratch,
stop below 4 GiB free. Retain loader 46/7196, frontend 5/5225, metadata 28.
Check actual-command header consumers, records, anchors, style and trailers.
Full preflight explicitly prohibited by current task (PENDING, not waived),
no GPU/weights/install/full-library/remote/push/PR/merge. Commit runnable code
and hand immutable SHA to root for verification and fresh independent review.
Original vision execution, video, template, oracle and served-runtime gates
remain pending. Open milestone ISSUE-LOCAL-01M4KSETGY63TDEDPF6EHQHRA8 unchanged.

### T04A implementation evidence — awaiting review

After spec 0c64366e3, fresh real-TU RED built and exited 1: 0/2 cases,
6/17 assertions passing, valid image rejected by explicit missing behavior.
The subsequent implementation replaces that placeholder with source identity
geometry, float32 1/255 rescaling, raster HWC patches, xy positions, zero/-1
padding and valid soft-token counts. Shared ImageKwargs storage is reused;
shared RGB callers and PIL resampler are unchanged. All required resize is
explicitly refused, not approximated. Bounds are checked before allocation.

GREEN: new target 3 cases/138 assertions, zero skips; nine supported images
include all budgets, portrait/landscape, default square and thin no-padding
cases. Every pixel and position is independently checked, plus malformed
inputs and tower-contract failures. Tiny source geometry fixture executes the
exact pinned function body, not full HF processor. Tests exercise the same
ValidateGemma4VisionInput invoked by Gemma4VisionForward before CreateQueue;
no backend/tower inference or numerical output is claimed. New validator
checks shapes, raster positions, pooling divisibility, pixel range/finiteness
and trailing padding. Actual tower TU and five other direct consumers pass
actual-command syntax checks. Loader 46/7196, frontend 5/5225, metadata 28,
exact-count CTest 3/3, records and anchors PASS. clang-format unavailable;
whitespace and commit style/trailers checked. Evidence/reproduction under
assets/work/2026-10-10-e4b-image-preprocessing, well below 256 MiB, -j2,
external TMPDIR/Git ceiling, 13 GiB free. Full preflight remains prohibited
and PENDING. No GPU, weights, install, full-library build or remote operation.

Missing torchvision blocks local resize oracle comparison, NOT inherently
C++ resampler implementation. General antialiased tensor bicubic remains an
explicit required milestone, not a permanent waiver. Next capture approved
pinned-source up/downscale, noninteger, asymmetric and extreme-aspect fixtures
with compatible Torch/torchvision versions/dtypes recorded; compare resized
bytes and final patches/positions/counts, including rounding/clipping and edge
support. T03 template, video, full processor/prompt pipeline, numerical vision
execution, full factory and served runtime remain pending. SupportsMmInputs
stays false. Immutable implementation awaits root verification and fresh review.

### T04B tensor uint8 antialiased resize (prospective)

T04A is reviewed bounded-only per rootbg339/review24fda: 3/138 and 5025
source-geometry AST comparisons; not full processor or vision execution.
Installed Torch 2.11.0+cu130 (70d99e998b4955e0049d13a98d77ae1b14db1f45)
executes CPU uint8 bicubic AA. Qualify torchvision v0.26.0 source paired with
Torch 2.11: v2 resize_image keeps CPU bicubic uint8, calls interpolate with
align_corners=False and antialias=True. No float cast/round occurs on this arm.
Use this installed ATen oracle, not Pillow or float interpolate. Fetch only
small immutable sources, recording hashes. ATen computes double Keys -0.5
weights, clipped and normalized support, per-axis int16 precision selected from
maximum positive weight, signed half-away coefficient rounding, horizontal then
vertical uint8 rounding/clamping. Shared Pillow uses fixed 22-bit precision and
cannot be reused unchanged. Add a private bounded source-backed kernel without
changing shared resampling or normalization. Stream horizontal rows per vertical
window so extreme axes cannot allocate input-height times output-width storage.
Retain dimension/area bounds, fixed patch16/pool3, identity shortcut and budgets.

Before behavior, generate independent Torch uint8 golden bytes and require
actual processor output to match all patch scalars, positions/padding/count and
actual tower input validator. Cover up/down, asymmetric noninteger, extreme
axes, overshoot/clipping patterns, all five budgets. Genuine executable RED
must precede implementation. CPU -j2 only; scratch <=256 MiB, stop below 4 GiB;
external TMPDIR/Git ceiling. Retain image, audio 5/5225, loader 46/7196 and
metadata 28; actual-flags headers, records/anchors/style/trailers. Full preflight
prohibited by task, pending. No installation, GPU, weights or remote operation.
Full HF processor, vision execution and served claims remain pending.

### T04B implementation evidence — awaiting root verification and fresh review

Spec 391a75659 preceded behavioral RED: actual target built after correcting a
CAPTURE macro compile error; actual tests executable refused valid resize.
RED 2/4 cases, 137/138 assertions passing, no skips. Wrong-path attempt and
compile error are not counted as RED. Implemented bounded private tensor AA
kernel; shared Pillow and normalization unchanged. ProcessImageRGB routes to
the production ProcessImage implementation. Identity path remains unchanged.
Independent pinned Torch CPU fixtures check every resized byte and exact float
rescaling through patches, then positions/padding/count via actual tower input
validator. Nine fixtures, all budgets; fixture README records source pins,
hashes, generation and version qualification. No threshold relaxation.

GREEN 4 cases/201 assertions, loader 46/7196, audio 5/5225, metadata 28;
exact-count CTest 3/3. Six actual-flags header consumers pass. Records and
anchors pass; whitespace checked. No clang-format available. Kernel uses one
accumulator row rather than input-height*output-width temporary; existing
65536-axis/64Mi-pixel bounds apply before allocation. Coefficients bounded by
O(input axes + output axes); output <=31MiB patches plus <=8MiB resized bytes.
Evidence: assets/work/2026-10-10-e4b-image-resize. Full preflight prohibited and
pending. Full HF processor, numerical vision and serving remain pending.

### T04C CPU vision execution (prospective)

T04B is reviewed per parent rootbg378/review472eff: image 4/201,
36 independent CPU oracle cases (byte exact) and 13 mutation kills; loader
46/7196, audio 5/5225 and metadata 28. This does not establish vision execution.

Wire decoded RGB processor output through the production shared encode_mm hook,
loaded vision blocks and raw BF16 [out,in] connector into Gemma4VisionForward,
then the existing embedding/PLE merge. Use the actual CPU backend and op
providers, not link stubs or a parallel test runner. First determine the exact
bounded translation-unit closure; stop with measured evidence if the 256 MiB
incremental scratch limit cannot accommodate it. No full-library build.
Default original E4B config is MHA (12 heads, 12 KV heads, width 768,
head_dim 64). Reject GQA or incompatible execution configurations before
allocation, while retaining the broader loader inventory contract.

Pinned Transformers modeling source retained in the earlier vision scratch
owns pixel scaling, axis positions, sandwich RMSNorm, per-projection clipping,
2D RoPE, noncausal scale-1 attention, tanh GeGLU, spatial pooling and weightless
connector normalization. Do not assume Q/K/V or gate/up clipping bounds match.
Use independently generated nonzero small Torch CPU source/equation fixtures
for intermediate and final values. Explicitly qualify reduced-size BF16 CPU
fixtures: they are not original-checkpoint or full HF runtime acceptance.

Commit this spec before a genuine behavioral RED through shared registry encode
and loaded weights. Then correct production behavior and prove GREEN including
image metadata and PLE merge, malformed inputs, unsupported modality/device and
configuration refusals. Preserve SupportsMmInputs=false, audio behavior and
CUDA/video refusal. Public factory/full runtime reachability remains unproven.
Run the three existing targets, metadata 28, actual-header-flags consumers,
record/anchor/style/trailer checks, and the new exact-count CPU target -j2.
External scratch assets/work/2026-10-10-e4b-vision-execution owns logs and README;
external TMPDIR/Git ceiling, 256 MiB incremental limit, stop below 4 GiB free.
No GPU, weights, installation, full preflight, remote, push/PR/merge or cleanup
of other work. Full preflight remains prohibited/PENDING. T03, video and full
runtime remain pending. Hand the runnable commit SHA to root for fresh review.

### T04C implementation evidence — awaiting root and independent review

Spec 11bfd142a preceded behavior. Real CPU closure was measured rather than
stubbed: backend, op registry/wrappers, CPU ops/threadpool, element GEMM and its
ISA tiers, quant-dequant (retained registrar dependency), and layernorm/Add.
Initial closure linked after the three x86 tier TUs were added; missing Add
provider was corrected by linking its real layernorm TU. Neither setup failure
is counted as numeric RED. red2.log then shows 0/2 cases: five assertions pass,
three fail (encoder .390625, pool .226562, connector .00976562 max errors), plus
actual shared registry audio-only rejection. Independent nonzero Torch BF16
fixtures and tolerances existed before production correction.

Production shared hook dispatch now accepts image metadata and loaded vision
weights, calls the actual CPU backend/tower and raw BF16 connector, then existing
merge/PLE. Each Q/K/V and gate/up projection receives its own clipping bounds.
Pool now rounds the average to BF16 before float32 scaling, as source does.
Execution rejects GQA/head-width mismatch and malformed metadata/weights;
loader inventory remains broader. Queue destruction is exception-safe.
No readiness gate was enabled; audio and text regression behavior is retained.

GREEN: vision 2 cases/661 assertions; loader 46/7196; audio 5/5225; image
4/201, zero skips. Metadata 28 and offline metadata-only CLI pass. Four
scratch-only compiled/linked mutations are killed: shared K clipping, shared
up clipping, missing positions and bypassed connector norm. Seven direct
header consumers compile with actual target flags. New stage captures cover
attention and MLP separately. Fixture README records source SHA, generator,
Torch revision, reduced-shape qualification and non-exact BF16 tolerances.

Scratch 22 MiB; reused build 105 MiB versus initial74 MiB, comfortably below
256 MiB incremental. CPU -j2, external TMPDIR/Git ceiling, 13 GiB free.
Evidence/reproduction: assets/work/2026-10-10-e4b-vision-execution/README.md.
Full preflight prohibited/PENDING. No GPU, original weights, install, remote,
push/PR/merge. Full factory, runtime, T03 and video remain pending. This proves
production hook-table dispatch, not public-factory or served reachability.

### T04C review coverage repair (prospective)

ISSUE-LOCAL-01M4N83NGMK17ZZ608E631NX2R tracks two surviving mutations,
not known production bugs. Add a separate independent Torch CPU equation
fixture at width12, three MHA heads of dimension4, with sqrt(12) scaling.
Use nonzero dyadic patch/position inputs and isolate pooling by zeroing the
attention/MLP residual output projections. Compare encoder and pooled BF16
values exactly, without the broad full-block tolerance. Retain source-rounded
and deliberately late-rounded oracle outputs and assert they differ.
No original-shape, general-theta or full-HF claim follows from this fixture.

Add under/oversized storage cases for each Q/K/V/O, MLP, norm, patch,
position and connector field and block count. Check raw registry connector
rank, dtype, dimensions and short/excess bytes. Compile/link scratch-only
pool-no-round and >= attention-size mutations with the real CPU providers;
require runtime assertion failures, then rerun the unmodified four targets
and metadata tests. Production behavior and admission flags stay unchanged.
Reuse the existing build, -j2 and external TMPDIR/Git ceiling. Keep total
incremental storage below256MiB and stop below4GiB free. Full preflight is
prohibited/PENDING; no network, installation, GPU, weights or remote writes.

### T04C review repair evidence — awaiting root and fresh review

Spec e37dacc31 preceded the test repair. Independent width12/three-head MHA
Torch equations distinguish pre-scale BF16 average rounding in three of24
pooled values. Encoder and pool assertions retain zero tolerance. A fixed
+0.25 dyadic position offset removes near-zero float32 cancellation from the
first fixture; no expected values came from C++ and no tolerance was widened.
The fixture README records this adjustment and source provenance.

Scratch-only real compiled/linked mutations now fail: pool-no-round has one
assertion failure (max error .015625); oversized attention storage has four
failures, one each for Q/K/V/O. Actual recompiled baseline passes5/1172.
Short/excess storage coverage includes all tower fields and block count (34
assertions), plus nine raw connector rank/dtype/dimension/byte failures through
the shared registry. No production code or admission behavior changed.

Final exact-count CTest4/4: vision5/1172, loader46/7196, audio5/5225,
image4/201, zero skips; metadata28 and offline CLI pass. Records and anchors
pass. Evidence and exact commands: assets/work/2026-10-10-e4b-vision-repair.
Build106MiB versus initial74, execution22, review19, repair5: incremental78MiB
of256MiB; disk13GiB free, -j2, external TMPDIR/Git ceiling. Full preflight
prohibited/PENDING. SupportsMmInputs remains false; full factory, original
shape/checkpoint, full HF, general theta, T03, video and runtime remain pending.

### Reconciled vision repair acceptance

Root rootbg480 PASS: loader46/7196 audio5/5225 image4/201 vision5/1172 and metadata28. Independent29b61b94 PASS: pool-no-round one failure, oversized Q/K/V/O four failures, connector dtype one failure; actual pinned HF fixture exact. Evidence: assets/work/2026-10-10-e4b-vision-repair-review/README.md. Scoped coverage acceptance only; T02/T03/runtime and full preflight remain pending.

### T03A original Google rendering (prospective)

Issue: ISSUE-LOCAL-01M4NT3N9NX3RT6JX11S8RCA0Y. Reuse the existing
protocol -> MakeChatTemplatePromptFn -> minja renderer, not a new renderer.
Google checkpoint ee0ef6023621cff504d758262d4e04895a5af4a2 fixture
tests/fixtures/gemma4_e4b_google/chat_template.jinja has SHA256
0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5.
Its captured_content and tool_body branches consume ordered typed arrays.
The retained vLLM renderer tests target examples/tool_chat_template_gemma4.jinja,
not this original: their empty thought-channel expectations must not replace
Google bytes. Use installed Jinja2 3.1.2 ImmutableSandboxedEnvironment with
trim_blocks/lstrip_blocks true, keep_trailing_newline false, no autoescape.
Generate expected bytes independently of C++ using this exact fixture.

Bounded compatibility policy: select this one template by full SHA256 using
the existing sha256_bytes utility, once at parse. Changed templates and all
other models retain their existing flattened-string adapter byte behavior.
No general content-format detector or new API switch. Preserve original wire
content separately during decoding without changing processor content_parts.
Only the selected adapter validates structure: strings/null or arrays of text,
image/image_url, audio/input_audio, video objects as the Google template names.
Bare string elements normalize to text (existing protocol convention). Text
requires a string; unknown types, non-object elements, scalar content and
media in system/developer arrays fail explicitly instead of disappearing.
Media payload fields remain opaque and unchanged: rendering is not media
validation, fetching, token expansion, or a claim of supported wire video/audio
processing. No invented audio_url/video_url mapping. Retain reasoning_content
as a scoped fallback to reasoning, as the Google source does.

Test real JSON request decode and prompt factory against generated exact bytes:
strings, system/developer, assistant/tool continuations, multiple tools, nested
and escaped arguments, thinking unset/false/true, historical reasoning, ordered
image/audio/video/mixed/tool media. Cover scoped refusal and other-template
compatibility. Reuse existing test_chat_template TU in standalone target if
bounded linkage permits. Direct-source test_gemma4_e4b_template links actual
chat_template.cpp, protocol.cpp, gguf_reader.cpp, hash utility and vllm_test_main
with section GC; no libvllm, wholearchive, fake symbols or shared stub.
Capture executable RED before product edits, then exact-count/no-skip GREEN.

Production callers remain server_main, serving_chat and C API chat_prompt via
MakeChatTemplatePromptFn. This proves their shared renderer, not full serving:
T10 full-chain execution/mutations remain owed. No readiness, raw processing,
tokenizer, parser or C ABI fallback changes. T03B missing vocabulary/token
parity and T03C parsers remain pending; full T03 and runtime are not complete.
Full preflight prohibited/PENDING, missing .env does not block CPU slice.
Use existing build, max -j2, VT_CPU_THREADS=1, external TMPDIR and Git ceiling.
Stop below 4GiB free or 256MiB incremental build plus scratch; no GPU/network/
install/weights/push/merge or cleanup. Root rerun and fresh review required.

T03A discovery before implementation: the exact Google fixture also fails minja
parsing at adjacent string literals in raise_exception (row259). Preserve that
executable failure as a real production render defect, not a compiler failure.
For this exact hash only, join the three static diagnostic literals before
minja parsing (Jinja/Python implicit literal concatenation). No vendored parser
or other-template semantics change. Test original bytes as input throughout.
Then separately rerun old flattened BuildMessages after syntax admission to
prove marker-loss/refusal RED before the content implementation.

Independent oracle further exposes minja empty-map truthiness: Google's
`if params['properties']` emits `properties:{}` when Jinja omits it. The
120-case oracle has 30 exact failures after content retention, all containing
this empty schema. Within the exact-hash source adapter only, qualify the
properties condition with `| length > 0`, preserving Python truth semantics
for the source-defined mapping field. No generic engine change or oracle edit.

### T03A implementation evidence — awaiting root and fresh review

Original JSON -> prompt factory -> minja passes120 independent Jinja byte
comparisons. Standalone target includes the45 existing template cases and
three new cases:48 cases/355 assertions, zero skips. Actual direct-source
closure adds read_only_file_mapping.cpp and existing kv_cache_utils.cpp;
only vllm_test_main archive, no libvllm/stubs. CTest5/5 also passes loader46/7196,
audio5/5225, image4/201, vision5/1172. Metadata28 and offline CLI pass;
runtime_accepted=false. 59 direct header consumers compile with actual flags.

Old-code syntax RED:48 cases/45 passing,132 failed assertions. Syntax-only
adapter then exposes content RED:96 failed assertions including missing media
markers and refusal. Empty-schema follow-up RED:30 exact byte failures, fixed
only in the pinned source adapter. Generator reads no C++ outputs and exact
regeneration passes. Evidence, commands, setup failures, limitations and actual
link command: assets/work/2026-10-11-e4b-template/README.md.

All unrelated templates retain flattened strings. Even a whitespace-modified
Google template is outside identity admission and retains its old parse error.
Media payload fields remain opaque; only container/type/text shape is validated.
Programmatic media without retained original content is refused. Raw media,
T10 full-chain reachability, T03B accepted, T03C parsers, full T02/T03 and
runtime acceptance remain pending. Full preflight prohibited/PENDING.

### T03A F1 empty schema repair (prospective)

ISSUE-LOCAL-01M4NX9D8CDAB2GXHNJE40HNG4 tracks independent reviewer660d7ac2
F1 at f340f218. Original Google line26 omits empty array items; minja emits
items:{}. Add independently generated Jinja2 3.1.2 cases for empty items,
nested object/array properties, nonempty controls, empty parameters and empty
properties. Demonstrate executable byte-comparison RED before product edits.
Audit mapping truth tests in the unchanged exact fixture, including line95
params. Preserve mapping type tests (lines36/66/131/170/248): an empty map
there is intentionally still a mapping, not a false branch. Schema description,
type, required, enum and nullable fields use their string/list/bool contracts;
this repair does not redefine malformed schema types or rewrite minja.

Only the existing full-SHA source adapter may qualify mapping truth conditions
with nonzero length. Original fixture and Python oracle semantics remain
unchanged. Other templates, typed content and readiness are unchanged. Reuse
actual standalone closure, max-j2 and VT_CPU_THREADS=1. Run template and four
CPU targets with exact cases/no skips plus28metadata. Evidence lives in
assets/work/2026-10-11-e4b-template-fix; total incremental budget256MiB includes
writer/review scratch (~72MiB already), stop below4GiB free. Full preflight
prohibited/PENDING; no GPU/network/weights/install/full-library/push/PR/merge.
Inherited scoped helper role retained. Root verification then fresh independent
mutation review remain owed; full T03/T10/runtime readiness remains false.

### T03A F1 repair evidence — awaiting root and fresh review

Prospective8c32404ac preceded executable oracle RED:24 failures across empty
items, object-nested items, array/object-nested items and empty parameters.
The targeted RED excluded47 unrelated cases, not a final no-skip gate.
The bounded exact-SHA adapter now qualifies all three schema mapping truth
tests (params, properties, items); mapping-only branches remain unchanged.
Original Google fixture bytes and oracle provenance remain unchanged.

Independent Jinja2 3.1.2 oracle180 regenerates identically. Final CTest5/5:
template48/415, loader46/7196, audio5/5225, image4/201, vision5/1172,
all zero skips; metadata28 passes. Actual production TU closure retained,
no stubs/libvllm. Evidence/reproduction: assets/work/2026-10-11-e4b-template-fix.
Budget remains below256MiB aggregate incremental; free disk above4GiB.
F1 issue closed for this repair, not scope acceptance. Root verification and
fresh independent mutation review remain owed. Full T03/T10/runtime, original
weights and full preflight remain pending; readiness remains false.

### T03C1 Gemma engine reasoning end (prospective)

ISSUE-LOCAL-01M4NZHPAHKSHYK5JPZZ22NM0R owns this bounded repair.
T03A is accepted at1f8e4a529 per rootbg615 (five CPU targets and28metadata)
and independent14d27efa:180 oracle rows,12 semantic matches, one refusal and
nine compiled mutation kills. This does not complete T03.

Port only the text-literal virtual is_reasoning_end contract through the real
get_parser_engine factory. Pinned a7c23ac96d7806e7c7e7d862eadbce5a33529b94
vllm/parser/gemma4.py:297-304,418-437 and engine/parser_engine.py:615-674
are authoritative: CONTENT initially even with thinking; reverse scan returns
true at channel close or tool start, false at channel opener, and stops at the
latest turn or tool-response boundary. With no decisive marker in that tail,
return not thinking. Omitted Gemma thinking is false. Preserve the existing
true default for every other family with a scoped factory overload, retaining
the existing explicit bool signature and Gemma tokenizer constructor calls.
Do not change generic engine semantics or assembly state.

Before product edits, port the retained canonical new-turn test's three
parameters and capture executable RED for enabled thinking after historical
completed thought. Add empty/plain tails, both implicit/explicit ends, boundary
precedence, recent marker precedence and other-family defaults. Exercise the
base virtual interface returned by the actual factory, not a test substitute.
Use a synthetic tokenizer, explicitly not original vocabulary. Expected values
come from retained Python tests/source, never C++ output. Test initial CONTENT
for both thinking modes. Prove factory-deletion sensitivity with a compiled
mutation after GREEN. Prompt/streaming tests remain separately owed.

Standalone test_gemma4_e4b_reasoning_end links the13 actual provider TUs listed
in assets/work/2026-10-11-e4b-parser-sources/README.md plus the test and
vllm_test_main. Use platform-guarded section GC, no libvllm/stubs or deleted
factory branches. Resolve missing real providers as harness setup, never RED.
Run exact-count zero-skip new target, all five existing CPU targets and28metadata;
syntax-check changed-header consumers with actual compiler flags. Run focused
record/anchor/style/trailer/whitespace checks, not full preflight.

No prompt hooks, reasoning registry, serving or text-tool parser changes.
Preserve Gemma serving's deliberate text-tool branch and bare-call support.
T03B accepted. T03C prompt state and served reasoning, T10, original weights
and full preflight remain pending; SupportsMmInputs remains false. Missing .env
is not a CPU-slice blocker. Retain scoped helper role. Evidence goes in
assets/work/2026-10-11-e4b-reasoning-end with exact commands and resources.
Max-j2, VT_CPU_THREADS=1, external TMPDIR, Git ceiling assets/work; stop below
4GiB free or256MiB aggregate campaign growth (about119MiB used at entry).
No GPU/network/install/download/full-library/push/PR/merge or unrelated cleanup.
Root verification and fresh independent review follow implementation commit.

### T03C1 implementation evidence — awaiting root and fresh review

Prospective c0f304521 preceded executable RED:4 cases/208 assertions,
30 failed, including canonical enabled new-turn true-versus-false mismatch.
The factory's one-argument overload now selects false only for Gemma; existing
explicit bool calls and tokenizer-first Gemma construction remain compatible.
The Gemma override evaluates the latest complete literal, equivalent to the
pinned reverse scan for these distinct markers. Generic engine/config and
serving/text-tool code remain unchanged. Initial assembly state stays CONTENT.

Final new target:5 cases/226 assertions, zero skips. Three canonical new-turn
parameters, all nine retained streaming end-detector parameters in text form,
24 tail rows across omitted/false/true, plain-content checks and nine-family
default regressions pass. This is not a port of streaming/prompt lifecycle.
Full CTest6/6 passes: template48/415, loader46/7196, audio5/5225,
image4/201, vision5/1172 and reasoning-end5/226. Metadata28 passes.
33 changed-header consumers pass actual-flags syntax checks. Scratch factory
construction deletion compiles/links and fails4 assertions (5 cases, zero skips).
Real13-provider standalone closure links without extra providers or stubs.

Evidence and reproducible commands: assets/work/2026-10-11-e4b-reasoning-end.
Initial build invocation needed CMake regeneration; that setup error is not RED.
Aggregate campaign growth is about142MiB of256MiB; free space exceeds4GiB.
Root verification/fresh review remain pending. T03B, prompt preparation, served
reasoning, T10, original weights and full preflight remain owed. Full T03 is not
complete; SupportsMmInputs remains false.

### T03C2 prompt state (prospective)

ISSUE-LOCAL-01M4P1A9D5ED9CY2YT4229BNB9 owns engine-local prompt preparation.
T03C1 accepted at01035eef6 per rootbg684 (six CPU targets, zero skips,
28 metadata) and reviewer3876d13e PASS (5/226,39216 Python comparisons,
nine mutants). T03A accepted at1f8e4a529. Review evidence lives in
assets/work/2026-10-11-e4b-reasoning-review/README.md.

Pinned vLLM a7c23ac96d7806e7c7e7d862eadbce5a33529b94 gemma4.py:418-446,
480-526 and engine/parser_engine.py:195-218,446-461 define this contract.
Retained nine-file manifest lives in assets/work/2026-10-11-e4b-parser-sources.
Add a nullable borrowed prompt-ID overload while preserving four-argument
parse_delta. Absent does not consume preparation; present empty does. Call a
public virtual base no-op hook once before feed, then mark prepared AFTER the
hook, since hooks may reset. Reset clears preparation. Gemma resolves supplied
tokenizer IDs without guessed fallbacks. Reverse scan opens reasoning only if
channel opener precedes any close, tool start, new turn or tool response in
reverse order. Thinking does not gate this. Reset only the underlying engine
to REASONING and protect it from default initialize_streaming; preserve assembly
bookkeeping, initial CONTENT elsewhere and existing prefix heuristics.

Prospective TDD permits an inert compatible prompt-carrying overload as minimal
test enablement before behavioral RED. A missing-overload compile error is not
RED. Real-factory tests use synthetic IDs, not original vocabulary. The decisive
case supplies [channel,thought,newline] then 'Sure' before any close: every
preclose byte is reasoning, postclose Hello is content. Whole-batch alone is
invalid because the existing heuristic injects an opener on a first-chunk close.
Port retained streaming PromptOpenReasoning276-360, PreInit382-469 and
ChannelLess478-545 with chunk1/2/10/whole, both thinking modes, generated and
redundant openers, fresh turns and nonstream comparison, all boundary orders,
missing IDs/null tokenizer, repeated prompt after close, absent versus empty,
reset/prefix state, prepare then initialize, finish not a new request. A small
subclass checks reset-capable base-hook lifecycle; production factory is primary.
Expected values derive from pinned source/tests, never C++ output.

Reuse test_gemma4_e4b_reasoning_end and its13 production TUs; add focused tests
and existing assembly regressions if closure permits. Older goldens are regression
coverage, not new-pin parity. Capture executable assertion RED, then exact-count
zero-skip GREEN. Run all six CPU targets,28 metadata and actual-flags header
consumers including serving/adapters. Audit function-pointer uses: overloads
preserve call source compatibility but untyped address-taking becomes ambiguous;
new virtual/layout requires rebuilding, not blanket binary compatibility.
Review mutations must remove/move/repeat preparation, set flag before a reset
hook, omit reset clearing/init protection, remove boundaries and gate on thinking.

No registry, adapter routing, Gemma text-tool branch or HTTP prompt claim.
T03B accepted. Served reasoning, T10, original weights and full preflight remain pending;
readiness remains false. Retain helper role; full preflight prohibited, not rerun.
Evidence: assets/work/2026-10-11-e4b-prompt-state (commands, red/green/resources).
Max-j2, VT_CPU_THREADS=1, external TMPDIR and Git ceiling assets/work. Entry
aggregate148MiB of256MiB; stop at256MiB aggregate or below4GiB free. Reuse
mutant objects sequentially and preserve fresh-review space. No GPU/network,
download/install/weights/full library/push/PR/merge/agents/unrelated cleanup.
Root verification and fresh review follow the committed implementation.

### T03C2 implementation evidence — awaiting root and fresh review

Prospective fcc0b0125 preceded inert overload/test enablement and executable
RED:9 cases/5276 assertions,12 failed,zero skips. The first 'Sure' delta leaked
to content in both thinking modes. Compilation/link succeeded with all13 real
providers and existing assembly regression TU; no stubs or new library closure.

The borrowed nullable overload now prepares once before feed and marks after
the virtual hook. Reset clears preparation. Gemma scans actual vocabulary IDs,
resets only its underlying engine and protects initialization. Four-argument
calls retain their symbol; no untyped parse_delta function-pointer uses were
found. New virtual/layout changes require consumer rebuilding, not ABI mixing.
No registry/adapter routing or text-tool branch changed; HTTP is not covered.

Final CTest6/6 passes in403.51s: reasoning15/5416, template48/415,
loader46/7196, audio5/5225, image4/201 and vision5/1172; all zero skips.
Reasoning target includes seven new prompt cases and three older assembly
regressions, which are not new-pin parity claims. Metadata28 and44 actual-flags
header consumers (including serving and both adapters) pass. Eleven compiled
mutants fail executable assertions: call removed53, after-feed50, repeated26,
flag-before-reset-hook2, reset-clear1, initialization-protection4, thinking36,
and each of four boundaries4. These are implementer sensitivity checks, not
fresh review. Initial missing configs include and unused mutant parameter were
harness compile errors, corrected and never counted as RED or mutation kills.

Evidence and exact commands: assets/work/2026-10-11-e4b-prompt-state/README.md.
All nine retained source hashes match. Resource growth stays below256MiB and
free disk above4GiB. Full preflight not rerun; T03B/served reasoning/T10,
original weights and full readiness remain pending. Root rerun and fresh
independent review are required against the implementation commit.

### T03C3 reasoning registry and adapter (prospective)

ISSUE-LOCAL-01M4P2X9JQ50HW5D9JMYDS7MF5 owns this bounded slice.
T03C2 accepted at c79b3f120 per rootbg7416 (six CPU targets and28 metadata)
and reviewer153f803a PASS: baseline15/5416, probes17/10632,12 mutants,
44 header consumers and nine retained hashes. T03A/C1 remain accepted.

Pinned a7c23ac96d7806e7c7e7d862eadbce5a33529b94 reasoning/__init__.py:55-58,
engine/registered_adapters.py:43-46, adapters.py:68-69 and gemma4.py:418-435
require Gemma4 registration, default thinking false and direct prompt forwarding.
Add Gemma4ParserReasoningAdapter and one enumeration entry. Preserve the old
name-only factory symbol and other-family defaults. Add an explicit bool and
borrowed EngineTokenizer factory overload for Gemma only; other names retain
the name-only construction policy. Tokenizer lifetime must cover the engine.
Add a public virtual default-noop ABC prompt hook, forwarding directly through
ParserEngineReasoningAdapter. Prompt vectors are borrowed only during the call.
Caller owns once-only invocation; do not use parse_delta's preparation flag.
Preserve extraction through extract_reasoning_streaming and skip-tool guard.
Request kwargs interpretation remains owed; do not borrow Kolibri effort rules.
New virtuals require rebuilding consumers; overloads can make untyped function
address-taking ambiguous. No binary compatibility guarantee is made.

First establish actual-provider link closure on the existing reasoning-end
target, reusing its13 providers and adding the actual reasoning factory closure.
No libvllm, stubs or deleted factory branches. Missing dependencies are setup,
not RED. Capture registry resolution/enumeration RED, then independently capture
ABC prompt forwarding RED with a synthetic open prompt and first small Sure
before a closer, both thinking modes. An inert compatible hook/context overload
is permitted as test enablement. Registration failure must not mask that RED.
Port retained TestAdapterExtractReasoning1112-1150 expectations independently.
Cover tool text, skip flag restoration, content-only output, boundaries, missing
IDs/null tokenizer, initialization protection, prefixes and other families.
Already-correct behavior is regression coverage, not a newly invented bug;
compiled mutations qualify sensitivity of registration, forwarding and skip guard.
Synthetic vocabulary is not original Google vocabulary.

Final gate: six existing CPU targets with exact counts and no skips,28 metadata,
actual-flags changed-header consumers, record/anchor/style/trailer/whitespace.
Full preflight prohibited/PENDING. Reuse build-cpu, max-j2, VT_CPU_THREADS=1,
external TMPDIR and Git ceiling assets/work. Campaign entry charge182346092
of268435456 bytes; reserve20971520 for independent review. Measure build growth,
scratch and transient link peak; stop below4GiB free or writer ceiling247463936.
Evidence: assets/work/2026-10-11-e4b-reasoning-adapter/README.md.
No GPU/network/download/weights/install/full-library/push/PR/merge/agents.

Serving MakeReasoningParser/callers, Gemma text-tool selection and engine-tool
exclusion, tokenizer bridge and HTTP loops stay unchanged. This is staged
registry/ABC reach, not served prompt integration. MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
owns the remaining wiring under this spec's T03/T10 scope. Full T03/T03B/T10,
original weights and full preflight remain owed; readiness remains false.
Root verification and fresh independent review follow the implementation commit.

### T03C3 implementation evidence — accepted

Prospective5fdbca989 preceded actual-provider linking and both executable REDs.
Registry RED:16 cases/5431 assertions,2 failures,zero skips. With registration
working and an inert hook, forwarding RED:17/5440,4 failures,zero skips;
first Sure was content in both modes. No setup failure counted as RED.

The actual factory now constructs Gemma4ParserReasoningAdapter, defaults false,
and accepts explicit thinking plus borrowed tokenizer context. ABC forwarding
is direct and caller-timed. Existing extraction and skip guard remain unchanged.
The old factory symbol remains; typed address-taking is tested. No existing
untyped factory address-taking was found; new overloads and virtuals require
consumer rebuilding and are not a binary-compatibility promise.

Final six CPU targets pass in400.51 seconds, zero skips: reasoning37/5789,
template48/415, loader46/7196, audio5/5225, image4/201, vision5/1172.
Metadata28 passes,53 actual-flags header consumers compile, nine source hashes
match, and record/anchor checks pass. Existing Qwen3/Kolibri and assembly tests
are older-pin regression coverage, not new-pin parity. Real closure adds13
reasoning TUs to the existing13 providers, without libvllm/stubs or duplicated
provider targets. Four compiled scratch mutants fail assertions: substituted
registration1, removed forwarding24, disabled skip guard28, missing restoration2.

Root bg818 accepted at02a73685. Fresh reviewer d476e952 PASS41/6286, seven
compiled mutants killed, nine retained source hashes verified.
Evidence: assets/work/2026-10-11-e4b-reasoning-adapter/README.md.
Review evidence: assets/work/2026-10-11-e4b-reasoning-adapter-review/README.md.
T03B accepted. Full T03/T10 and full readiness remain pending.

### T03C4 prospective request-aware reasoning construction

ISSUE-LOCAL-01M4PADQGY0M2FCM1TCQVYDZYV owns this bounded repair.
T03C3 at02a736854 is accepted per rootbg818 (six CPU targets, zero skips,
28 metadata) and freshreviewd476e952 PASS41/6286, seven compiled mutants and
nine retained hashes. Review bg826 was a scratch unused-parameter setup failure;
corrected bg830 supplies the behavioral results, not that failed attempt.

Retained vLLM a7c23ac96d7806e7c7e7d862eadbce5a33529b94 parser/gemma4.py:418-429
reads kwargs.get('chat_template_kwargs', {}) or {}, then enable_thinking with
false default; engine/adapters.py forwards kwargs. Interpret JSON values with
Python truthiness: missing/null/false/numeric zero/empty string/array/object
are false; nonzero and nonempty values are true, including string 'false'.
No Kolibri effort rules, boolean-only extraction or new rejection policy.
ChatCompletionRequest normalization in protocol.cpp retains only outer objects;
leave it unchanged. Template defaults merge privately in MakeChatTemplatePromptFn's
closure; this slice reads the request, not that merged map. It makes no claim
of equivalence to server-default template settings.

Add a small production request-aware construction function in its own TU,
called by OpenAIServingChat::MakeReasoningParser(request). Gemma delegates to
the accepted explicit-thinking factory; all other names delegate to name-only
construction. Preserve both existing factory symbols. No tokenizer bridge,
prompt timing, HTTP loop changes, new virtuals or generic model factory edits.
Keep Gemma text-tool routing and bare engine-tool exclusion unchanged.

TDD uses test_gemma4_e4b_reasoning_end's 26 real production providers plus the
new TU, never libvllm, stubs or removed factory branches. Inert API scaffolding
may delegate the old factory for executable behavioral RED. True/nonempty
request values must fail through real adapter reasoning-end/tail behavior;
setup failures do not count. Cover all false/empty values, other-family defaults,
unknown/disabled names and typed legacy factory access. Derive an independent
expected JSON truth table with installed Python standard library; synthetic
text fixtures do not prove original Google vocabulary parity. Compile bounded
kwargs-ignored/default/truthiness/non-Gemma mutants sequentially.

Final six existing CPU targets require exact counts and no skips, plus28
metadata tests, actual-flags changed-header consumers (including serving),
record/anchor/style/trailer/whitespace checks. Max-j2, VT_CPU_THREADS=1,
external TMPDIR, no network/GPU/download/install/weights/full preflight/library,
push/PR/merge or agents. Monitor from before RED setup, including shared growth
and transient files. Entry charge204521975, cap268435456, review reserve20971520,
writer ceiling247463936; retain an allowance for unsampled transients and stop
below4GiB free. Evidence: assets/work/2026-10-11-e4b-reasoning-kwargs/README.md.

Claims are actual request-aware construction tested and serving caller wired
and syntax-compiled, NOT end-to-end HTTP or served reasoning. Full T03B/T03/T10,
original tokenizer/weights, full preflight and readiness remain false/pending.
Root exact-SHA verification and separate fresh review follow the code commit.

### T03C4 implementation evidence — accepted

Prospective71ef7af7b preceded inert production API and test compilation.
Initial setup bg862 failed on a two-argument doctest CAPTURE; no RED claimed.
Original preimplementation bg865 linked40 cases/6235 assertions:48 genuine
kwargs failures at line42 and20 malformed-channel fixture failures at line43.
This is mixed RED, not a clean TDD result. Production bg868 reduced failures
to8, all from that fixture. The corrected inert replay bg870 happened after
the implementation was saved; its48 failures are post-fix sensitivity, NOT
prospective RED. The valid implementation was restored without resetting Git.
Corrected green bg871 passes40/6235, no skips. Full chronology is retained.

MakeRequestReasoningParser now applies Python JSON truthiness for Gemma and
preserves other-family defaults. MakeReasoningParser(request) calls it in the
real serving consumer. Existing factory symbols, normalization, tokenizer,
prompt timing, HTTP loops and Gemma tool routing remain unchanged.
The gate links27 real production providers (the accepted26 plus request.cpp),
seven test TUs and real vllm_test_main, never libvllm or stubs.

Final six CPU gates pass with exact counts and zero skips: loader46/7196,
audio5/5225, image4/201, vision5/1172, template48/415, reasoning40/6235.
Metadata28,29 actual-flags header consumers, records and anchors pass.
Seven compiled scratch mutations fail: ignored kwargs48, wrong default3,
null truthiness4, empty containers8, strings16, numeric zero12, non-Gemma131.
Nine retained source hashes match. Final serving compilation is separately
retained after the include-order edit; no HTTP traversal claim follows.
Evidence: assets/work/2026-10-11-e4b-reasoning-kwargs/README.md.
No clang-format executable was used. Full preflight prohibited/PENDING.
Request kwargs differ from template defaults merged inside MakeChatTemplatePromptFn:
that closure ignores null/'auto' overrides. No merged-default parity is claimed.
Full T03/T10, original tokenizer/weights and readiness remain false/pending. T03B accepted.

### T03C4 accepted scope and T03C5 prospective tokenizer bridge

Root rootbg8916 accepted six CPU targets and28 metadata at c8d9c1528.
Fresh reviewer f839 passed42/7680 with zero skips and seven compiled kills.
Mutants deliberately filtered2/1445 with40 excluded, not zero-skip runs.
Evidence: assets/work/2026-10-11-e4b-reasoning-kwargs-review/README.md.
The original RED remains mixed:48 genuine failures plus20 bad-fixture
failures. The postfix replay is sensitivity, not a clean first-pass RED.

T03C5 is authorized by ISSUE-LOCAL-01M4R157C4X783JF2XYZH9S19M.
Add a production borrowed EngineTokenizer adapter for tok::Tokenizer.
Caller owns the immutable tokenizer and bridge longer than the parser; neither
may be moved or replaced while borrowed. No owning parser context, serving
change, prompt-timing integration, download or original-Google vocabulary claim.
SupportsMmInputs remains false. T03B, T10 and full readiness remain pending.

Local adapter behavior derives from
`include/vllm/parser/engine/token_id_scanner.h::EngineTokenizer` and
`src/vllm/tokenizer/tokenizer.cpp::Tokenizer::FinalizeTables`,
`src/vllm/tokenizer/tokenizer.cpp::Tokenizer::Decode`,
`src/vllm/tokenizer/tokenizer.cpp::Tokenizer::HasToken` and
`src/vllm/tokenizer/tokenizer.cpp::Tokenizer::IsSpecial`.
Iterate [0,VocabSize), skipping !HasToken; store TokenText in a stable const
map, delegate singleton Decode, and enumerate final IsSpecial state in ID order.
HasToken currently excludes empty stored strings, so the bridge excludes them
rather than inventing valid empty tokens. Unknown IDs retain Decode exceptions.
Reject identical text across distinct assigned IDs: an explicitly approved local
inverse-map safety restriction, NOT an upstream duplicate-vocabulary policy.
Repeated same-ID added records use final IsSpecial, not AddedTokens history.

Retained nine Python sources remain pinned to vLLM
a7c23ac96d7806e7c7e7d862eadbce5a33529b94; verify their manifest offline.
They specify existing parser prompt/scanner behavior, not the local bridge.
Synthetic genuine HF JSON follows
`tests/vllm/test_tokenizer_metaspace_split.cpp::MakeJson` schema and loads with
Tokenizer::FromHfJsonBytes. Test metaspace, byte fallback, literal added tokens,
holes, unknown IDs, final specials and ambiguous inverse-map rejection.
Exercise the explicit-thinking borrowed factory and ABC prompt hook with Gemma's
first small Sure before the closer, both modes, boundaries, reset, and scanner
drop behavior. No fake tokenizer, fake provider or replaced factory symbol.

Before behavioral RED, measure actual full-provider compile/link feasibility.
Reuse in place the existing27 production objects, seven test TUs and real
vllm_test_main from test_gemma4_e4b_reasoning_end. Compile four real tokenizer
providers (tokenizer, bpe, pretokenizer, unicode_data) with actual target flags.
Reuse template gguf_reader and read_only_file_mapping objects without copying;
choose minimal portable CMake integration only after this gate. Include bridge
and test objects in the budget. No libvllm build, branch deletion or fake closure.
An inert bridge may have the final API, return empty decode/vocab/special lists,
and retain the borrowed reference solely to enable executable behavioral RED.
It must not implement correct behavior before the first executable test.
Existing correct behavior is regression coverage; scratch mutation is not new RED.
Compile/link errors are setup failures and must be repaired within budget.

Resource entry219315142 bytes; shared build baseline143465952. Charge positive
shared growth plus new scratch and evidence growth. Exact writer ceiling239075328
preserves20MiB review and8MiB unsampled allowance under268435456. Monitor at250ms
from before setup, including TMPDIR, objects and link transients; stop below4GiB
free or before predicted breach. Serial compile/link preferred, maximum-j2.
A genuine budget blocker stops before broad builds with exact observed and
predicted bytes and preserved evidence, without claiming RED/GREEN.

If feasible: six bounded CPU targets with exact adapted counts and zero skips,
28 metadata tests, actual-flags consumers and record/anchor/style/trailer checks.
Sequential compiled mutations cover decode-as-TokenText, missing markers,
special enumeration, ambiguous acceptance and prompt forwarding as budget permits.
Evidence and commands: assets/work/2026-10-11-e4b-tokenizer-bridge/README.md.
No network/GPU/install/weights/full preflight/full library/push/PR/merge/agents.
Root verification and fresh review follow the committed implementation SHA.

### T03C5 implementation evidence — accepted

Prospective017ec82b0 precedes the inert bridge. Actual-provider budget gate
compiled tokenizer2166640, bpe377360, pretokenizer122120, unicode_data79272
bytes and linked5235136 bytes using the27 existing parser providers, seven
existing test TUs, new bridge/test TUs, real test main and template GGUF/mapping
objects in place. No fake symbols, omitted factory branches or libvllm.
The final closure adds five production TUs and one test TU, reusing two real
GGUF providers. CMake explicitly lists the new bridge in the production target.
Unix Makefiles reuse template object paths; other generators use the real
GGUF/mapping sources. Only the existing Unix Makefiles lane was executed.

Initial missing FromHfJsonBytes diagnostic argument and vexing parse were
compile setup errors, not RED. HF JSON rejects empty vocabulary strings;
the first executable failed two fixture loads and one intended ambiguity check,
not a valid whole-suite RED. Removing that invalid entry BEFORE implementation
produced43/6318 with34 failed assertions. After implementation,12 failures
exposed an invalid fixture reuse: boundary scenarios reused a live parser rather
than constructing one parser per independent request. Correcting that setup
produced43/6319, zero skips. Original RED is mixed, not clean first-pass TDD;
no postfix rollback was used. Real decode/map/special/ambiguity failures precede
implementation, as do first open-prompt failures. Explicit prompt reset coverage
remains on the same parser after a completed close/content sequence.

Six CPU targets pass: loader46/7196, audio5/5225, image4/201, vision5/1172,
template48/415, parser43/6319;28 metadata and two actual-flags new-header
consumers pass. Five compiled mutants fail unfiltered: decode-as-stored2,
missing markers22, all-added-special2, ambiguous acceptance1, prompt forwarding44.
Final restored scratch closure43/6319 passes with zero skips. Nine retained
source hashes match. Full preflight remains prohibited/PENDING.
Evidence: assets/work/2026-10-11-e4b-tokenizer-bridge/README.md.

Root bg966 accepted at 766ffea2. Fresh reviewer 35e9e8bb PASS: 46/6518,
five compiled mutants killed (3 cases, 43 filtered, special-enumeration 82
assertions fatal). Review evidence:
assets/work/2026-10-11-e4b-tokenizer-bridge-review/README.md.

The bridge is caller-owned and immutable by contract; no serving stack bridge,
HTTP test, original Google tokenizer claim or owning context was introduced.
SupportsMmInputs remains false. T03B/T03D accepted. T10 and full readiness remain pending.

## T03B: config.json special token ID parity (accepted)

Owning issue: ISSUE-LOCAL-01M4VH638BS1P682RC4QMQD22Z
State: accepted (fbca85528, reviewer 90aa862f PASS, 14 mutants)

### Gap

The real Google `tests/fixtures/gemma4_e4b_google/config.json` defines nine
special token IDs that the Python model reads from `config` attributes
(`boi_token_id`, `eoi_token_id`, `boa_token_id`, `eoa_token_id` /
`eoa_token_index`, `image_token_id`, `audio_token_id`, `video_token_id`,
`bos_token_id`, `eos_token_id`, `pad_token_id`). The C++ `HfConfig` struct
has no typed fields for these; `ParseGemma4ForConditionalGenerationConfig`
is a no-op. The loader test (line 197) validates `hidden_size`,
`vocab_size`, and `num_hidden_layers` from the real config but does not
validate any special token ID.

### Scope

Add typed read-only accessors for the Gemma-4 multimodal special token IDs
from `config.raw`. Add a test case that loads the real config.json fixture
and verifies each ID against its expected value. No production wiring,
serving, or forwarding changes. No tokenizer vocabulary changes.

### Expected values from the real fixture

```
audio_token_id: 258881
boa_token_id: 256000
boi_token_id: 255999
eoa_token_id: 258883
eoa_token_index: 258883
eoi_token_id: 258882
eos_token_id: [1, 106]
image_token_id: 258880
video_token_id: 258884
bos_token_id: 2
pad_token_id: 0
vocab_size: 262144
```

### RED before fix

A test that calls the accessor and checks the expected values must fail
before the accessor exists. A missing target or compile failure is setup,
not RED.

### Out of scope

Tokenizer vocab entries for parser markers (`<|channel>`, `<channel|>`,
`<|tool_call>`, `<|turn>`, `<|tool_response>`) — those are added tokens in
the tokenizer, not config.json fields. Span overlap/ordering/provenance.
Serving integration. Full preflight.

## T03D: video token replacement sequence (accepted)

Owning issue: ISSUE-LOCAL-01M4VPQYYNYHKHYBMA5S203KAH
State: accepted (1d01f933b, reviewer 3cec0fd8 PASS, 5 mutants)

### Gap

The Python `get_video_repl` in `gemma4_mm.py:412-447` builds the
interleaved token sequence for video frames. For each frame it formats
an `mm:ss` timestamp, encodes the prefix string via the tokenizer,
then appends `boi_token_id`, `video_token_id * N`, `eoi_token_id`.
The C++ side has no equivalent.

### Algorithm (from Python source)

```
for i, (ts, n_tokens) in enumerate(zip(timestamps, num_soft_tokens_per_frame)):
    minutes = int(ts // 60)
    seconds = int(ts % 60)
    ts_str = f"{minutes:02d}:{seconds:02d}"
    prefix = f" {ts_str} " if i > 0 else f"{ts_str} "
    ts_token_ids = tokenizer.encode(prefix, add_special_tokens=False)
    all_token_ids.extend(ts_token_ids)
    all_token_ids.append(boi_token_id)
    all_token_ids.extend([video_token_id] * n_tokens)
    all_token_ids.append(eoi_token_id)
```

Constants:
- `_VIDEO_MAX_SOFT_TOKENS = 70` (soft tokens per video frame)
- `_VIDEO_MAX_FRAMES = 32` (max sampled frames per video)

### Scope

Add a C++ function that builds the video replacement token ID sequence.
The function takes:
- `std::vector<double> timestamps` — frame timestamps in seconds
- `std::vector<int32_t> num_soft_tokens_per_frame` — soft token count per frame
- `int32_t boi_token_id`, `eoi_token_id`, `video_token_id`
- `std::function<std::vector<int32_t>(std::string_view)> encode` — text-to-token-ID callback

Returns `std::vector<int32_t>` — the full replacement sequence.

The encode callback abstracts the tokenizer so the test can use a
simple mock (e.g., per-character ASCII encoding). No real tokenizer
files are needed.

### Test cases

1. Single frame at t=0: produces `encode("00:00 ") + [boi, video*N, eoi]`
2. Two frames at t=0 and t=5: produces `encode("00:00 ") + [boi, video*N1, eoi] + encode(" 00:05 ") + [boi, video*N2, eoi]`
3. Timestamp formatting: 65 seconds → `01:05`, 0 seconds → `00:00`, 125 seconds → `02:05`
4. Empty frames list → empty result
5. Timestamp with fractional seconds: int truncation (e.g., 5.7 → `00:05`)

### Out of scope

Frame sampling/extraction from video data. Video pixel processing.
Vision tower forward pass for video frames. Serving integration.
Real tokenizer encoding. Span overlap/provenance.

## T09: wire hooks and runner — flip mm_inputs_ready (accepted)

Owning issue: ISSUE-LOCAL-T09-MM-READY
State: accepted
Commit: `3f3500877` (fix), `9d9a478dd` (spec)
Review: `9cbd01cb` PASS — 48/7214 loader, 5/1172 vision, 28 manifest, 3/3 mutants caught

### Gap

The Gemma4 multimodal path is fully implemented at the component
level: `EncodeMmGemma4` (image), `EncodeMmGemma4Audio` (audio), and
`EmbedMmGemma4` (merge) are all implemented and tested in isolation.
The `kGemma4MmHooks` factory sets `encode_mm` and `embed_mm` to these
functions.

However, `kGemma4MmHooks.mm_inputs_ready` is `false`
(`gemma4_audio_encoder.cpp:53`). The runner gates multimodal dispatch
on `ModelRegistry::SupportsMmInputs()`, which requires all three:
`mm_inputs_ready && encode_mm != nullptr && embed_mm != nullptr`.
Since `mm_inputs_ready` is false, the runner skips all multimodal
processing. The API path never processes audio/image/video.

### Scope

Flip `mm_inputs_ready` from `false` to `true` in `kGemma4MmHooks`
(`gemma4_audio_encoder.cpp:53`) so the runner dispatches multimodal
inputs through the implemented encode_mm → embed_mm → forward path.

`kGemma4Factory` (`gemma4_registry.cpp:157-166`) copies
`kGemma4MmHooks` as its base, so the flip propagates to the factory
the registry uses. No other source changes are needed.

Update four existing `CHECK_FALSE(SupportsMmInputs(model))`
assertions that were sentinels for the gate being closed:

- `test_gemma4_e4b_loader.cpp:1110` — audio encoder test
- `test_gemma4_e4b_loader.cpp:1345` — audio merge test
- `test_gemma4_e4b_loader.cpp:1402` — readiness contract test
- `test_gemma4_e4b_vision_execution.cpp:103` — vision execution test

These become `CHECK(SupportsMmInputs(model))` because the model now
advertises MM support through the hooks.

The readiness contract test block
(`test_gemma4_e4b_loader.cpp:1404-1414`) tests the contract logic with
a default factory and stays unchanged — it verifies that
`SupportsMmInputs` requires all three of `mm_inputs_ready`,
`encode_mm`, and `embed_mm`.

### RED before fix

A new test case in `test_gemma4_e4b_loader.cpp` constructs a model
through `kGemma4Factory` (the factory the registry uses, not
`kGemma4MmHooks` directly) and asserts
`SupportsMmInputs` returns true. Before the flip, this fails because
`kGemma4Factory` copies `mm_inputs_ready = false` from
`kGemma4MmHooks`. After the flip, it passes.

### Out of scope

Serving integration (T10). Full e2e API path with real weights.
Video frame sampling/extraction. Span overlap/provenance.
Original weights inference and per-capability quality gates.
Production device audio/BF16.

## T10A: register Gemma4 mm chat seam (accepted)

Owning issue: ISSUE-LOCAL-T10A-MM-CHAT-SEAM
State: accepted
Commit: `b2f449ba3` (fix), `58f9c1a5d` (spec)
Review: `3c85635e` PASS — 4/8, 2/3 mutants caught, mutation 3 coverage gap fixed with additional test

### Gap

T09 flipped `mm_inputs_ready` to `true`, so the server now treats
Gemma4 as a multimodal model and calls
`MultiModalChatRegistry::MakeSeam(ctx)`. No registration exists for
`Gemma4ForConditionalGeneration`, so `RaiseForUnregistered` throws
and the catch block installs a generic refusing seam.

A registered factory with a Gemma4-specific refusal message is the
prerequisite for all serving-level multimodal work. It validates the
install context (tokenizer, config, prompt renderer, image codec)
and returns a seam that refuses multimodal requests with a clear
message naming the architecture and the pending modalities.

### Scope

Create `src/vllm/entrypoints/openai/mm_chat_gemma4.cpp` with:
- A `MakeGemma4ChatSeam` factory that validates the install context
  and returns a `MultiModalChatSeam` whose `chat_fn` refuses
  multimodal requests with a clear message
- `REGISTER_VLLM_MM_CHAT(gemma4, "Gemma4ForConditionalGeneration", ...)`
- Add the source to `tests/CMakeLists.txt`

### RED before fix

A test case that calls
`MultiModalChatRegistry::Find("Gemma4ForConditionalGeneration")`
and asserts the registration exists. Before the registration, this
returns null.

### Out of scope

Image/audio/video processing in the chat path. Full serving
integration. Real model weights. Placeholder expansion.

## T10B: image processing in the Gemma4 chat seam (accepted)

Owning issue: ISSUE-LOCAL-T10B-IMAGE-CHAT
State: accepted
Commit: `833ac90af` (fix), `bff0d136e` (RED), `198232156` (spec)
Review: `636be6e2` PASS — mm_chat 4/8, mm_chat_image 2/264, 3/3 mutants caught

### Gap

T10A registered the mm chat seam but the `chat_fn` refuses all
multimodal requests. Image processing through the serving path is
not implemented.

### Scope

Upgrade `MakeGemma4ChatSeam` in `mm_chat_gemma4.cpp` to process
image content parts through the full serving path:

1. Factory resolves `boi_token_id`, `eoi_token_id`,
   `image_token_id` from `Gemma4SpecialTokenIds` (T03B, reads
   config.json)
2. Factory validates `prompt_fn` and `codec` are present
3. `chat_fn`:
   a. Walks messages, extracts image_url parts and text
   b. Replaces each message's content_parts with the text-only
      content (images become their `\t<|image|>` placeholder in
      the rendered prompt)
   c. Renders the prompt via `prompt_fn`
   d. Tokenizes via `tokenizer->EncodeWithSpecialTokens`
   e. For each image: decodes via `codec`, processes via
      `Gemma4ImageProcessor`
   f. Expands each `image_token_id` in prompt_ids to
      `[boi] + [image_token] * N + [eoi]` where N =
      num_soft_tokens
   g. Builds `MultiModalInputs` with `MultiModalFeatureSpec` per
      image
4. Audio and video content parts still refuse with a clear message

### Out of scope

Audio processing in the chat path. Video frame sampling/extraction.
Real model weights. Full e2e inference. The `default_output_length`
config field (C++ Gemma4VisionConfig does not carry it; the image
processor defaults to 280 which matches the E4B model).

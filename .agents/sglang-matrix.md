# SGLang parity matrix — the whole SGLang runtime surface, classified

**Status:** canonical SGLang-surface inventory for the SGLang parity PROGRAM
(`CLAIM-SGLANG-PARITY-PROGRAM`). This is the SGLang analogue of
[model-matrix.md](model-matrix.md) / [engine-matrix.md](engine-matrix.md): every
row has a stable `SGLANG-*` ID, the SGLang source anchor `file:line`, our
implementation mapping/anchor, evidence, the owning spike/spec, and a
**classification** in place of a lifecycle state. The gate methodology (SGLang as
correctness + performance oracle) is [specs/sglang-parity-oracle.md](specs/sglang-parity-oracle.md).

**Pinned SGLang source:** tag `v0.5.15`, commit
`f63458b5beaceabbd9d749b9fc956370e1b649e6`, cloned to `/home/mudler/_git/sglang`;
every `file:line` below is from that tree (paths relative to `python/sglang/srt/`).
**Mirror source (behavior truth):** vLLM `555967922` / 0.26.0.dev0 — SGLang is a
COMPETITOR engine, not the mirror source (see the oracle spec's policy caveat).

**Why this program exists (user-directed 2026-07-27):** "we need the same vLLM
approach there for reaching parity." SGLang is elevated to a full parity target
under `ROAD-V1-A` (the SGLang floor). The honest finding of this inventory: a
**large fraction of SGLang's surface is FUSED** — we and SGLang both descend from
the same paged-attention / continuous-batching / prefix-sharing ideas, and our
vLLM-derived implementation already expresses it. The VALUE of this matrix is the
precise map of the minority that is genuinely **SGLANG-DISTINCT** and worth an
opt-in, separated from what is already covered and what is research/niche.

## Classification legend (this matrix's "state" axis)

- **FUSED** — already covered by our vLLM-parity implementation; the row cites
  the `file:line` proving functional equivalence. No new code owed (an SGLang-name
  ALIAS at most).
- **SGLANG-DISTINCT** — a genuinely-different behavior our vLLM-derived design
  cannot currently express; scoped as an OPT-IN (flag/knob), never a fork of the
  engine.
- **INVENTORIED** — known, not yet scoped (needs its own spike before implement).
- **OUT-OF-SCOPE** — research / niche / large-scale-cluster / benchmark-track;
  not a behavior we adopt for the single-box gate.

## Rollup

| Classification | Rows |
|---|---|
| FUSED | 23 |
| SGLANG-DISTINCT | 8 |
| INVENTORIED | 7 |
| OUT-OF-SCOPE | 8 |
| **Total** | **46** |

**Headline SGLANG-DISTINCT items (the opt-in worklist), ranked in the oracle
spec §6:** cache-aware **LPM scheduling** (`--schedule-policy=lpm`,
`schedule_policy.py:155`); **in-batch prefix-collision** de-prioritization
(`schedule_policy.py:76`); **radix eviction strategies** lfu/slru/priority
(`server_args.py:739`); **jump-forward** constrained decoding
(`outlines_jump_forward.py:182`, deferred); **custom logit processors**
(`custom_logit_processor.py:24`); **batch-invariant deterministic** inference
(`batch_invariant_ops/`); **PD prefill/decode disaggregation**
(`server_args.py:2273`); **two-batch compute-comm overlap** for EP
(`batch_overlap/two_batch_overlap.py`). Everything else is FUSED (both engines
share the idea) or OUT-OF-SCOPE.

---

## KV cache / prefix sharing

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-KV-RADIX` | RadixAttention — radix TREE of KV prefixes, longest-prefix match with mid-node split, LRU-over-tree eviction, `lock_ref`, `extra_key` tenant isolation | `mem_cache/radix_cache.py:280` (`RadixCache`), `:355`/`:648` (match), `:563` (evict), toggle `server_args.py:755` | our block-hash APC == the fused equivalent (`src/vllm/v1/core/kv_cache_utils.cpp:259`); engine row `KV-SGLANG-RADIX-CACHE` (`SPIKE`); verdict in [sglang-radixattention.md](specs/sglang-radixattention.md) §1; **EXPOSED (`CLAIM-SGLANG-ABI-DOCS`):** C-ABI `vllm_model_params.enable_prefix_caching` tri-state (ABI v7), server `--[no-]enable-prefix-caching` / alias `--[enable\|disable]-radix-attention`; docs [docs/SGLANG-COMPAT.md](../docs/SGLANG-COMPAT.md) §1 | FUSED | Only delta = token/page vs `block_size` granularity, bounded + output-neutral. `--enable-radix-attention` is an ALIAS for the APC toggle, no distinct path. |
| `SGLANG-KV-EVICT` | radix eviction strategies lru / **lfu / slru / priority** | `server_args.py:739` (`radix_eviction_policy`); `mem_cache/radix_cache.py:276` (`__lt__`) | our APC is block-LRU only (`kv_cache_manager.cpp:124`); engine row `ENG-SGLANG-BEHAVIOR-FLAG` SW4 | SGLANG-DISTINCT | Opt-in `--radix-eviction-policy` knob over the block pool; minor, output-neutral. |
| `SGLANG-KV-HICACHE` | HiRadix / HiCache — multi-tier GPU→CPU→remote-store radix cache | `mem_cache/hiradix_cache.py:76` (`HiRadixCache`), storage init `:175`; `enable_hierarchical_cache` `server_args.py:1991` | our KV-offload track (`ENG-KV-OFFLOAD` / LMCache); [KV-OFFLOAD.md](../docs/KV-OFFLOAD.md) | FUSED | Same tiered-offload idea; covered by the KV-offload rows (separate track). |
| `SGLANG-KV-CONNECTOR` | pluggable KV storage-backend factory (Mooncake / HF3FS / NIXL / LMCache / AIBrix / S3 / redis) | `mem_cache/storage/backend_factory.py:16,66`; `connector/` (`s3.py`, `redis.py`, `azure.py`) | our KVConnector ABI + LMCache `lm://` client (verified vs a live `lmcache.v1.server`) | FUSED | Same connector abstraction; our LMCache client is byte-verified. Extra remote backends are INVENTORIED breadth under the KV-offload track. |
| `SGLANG-KV-MAMBA-RADIX` | Mamba/hybrid-SSM radix cache (separate full + Mamba LRU state) | `mem_cache/mamba_radix_cache.py`; `hi_mamba_radix_cache.py`; registry `mem_cache/registry.py:78-131` | engine row `KV-MAMBA-ALIGN` (`SPIKE`); vLLM `mamba_cache_mode=align` | FUSED | Scoped under `KV-MAMBA-ALIGN` — the align-retention path is the cache-ON precondition for `BACKEND-GATE-CUDA-SGLANG-PREFIX`. |
| `SGLANG-KV-CHUNK` | ChunkCache off-path fallback (intra-request KV only) | `mem_cache/chunk_cache.py`; `registry.py:83-94` | our cache-off no-prefix coordinator (`kv_cache_coordinator.cpp:260`) | FUSED | Equivalent: cross-request sharing disabled ⇒ intra-request only. |
| `SGLANG-KV-SESSION` | session radix cache (per-serving-session prefix tree) | `mem_cache/session_radix_cache.py`; `enable_session_radix_cache` `server_args.py:1107` | none | OUT-OF-SCOPE | Serving-session feature, not requested; no vLLM analogue. |

## Scheduler

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-SCHED-LPM` | **cache-aware LPM scheduling** — reorder the waiting queue by longest matched prefix; DFS-weight; large-queue auto-fallback | `managers/schedule_policy.py:155` (`SchedulePolicy`), `:139` (`CacheAwarePolicy`), `:176`/`:205` (calc/sort), fallback `:229` | our scheduler is FCFS/priority only, NO cache-hit ordering (`scheduler.cpp:63`, `request_queue.cpp:30,181`); engine row `ENG-SGLANG-BEHAVIOR-FLAG` SW1; **EXPOSED (`CLAIM-SGLANG-ABI-DOCS`):** C++ `EngineParams::policy=kLPM`, C-ABI `vllm_model_params.scheduling_policy="lpm"` (the concurrent session's ABI v9 **string** field — no duplicate int knob), server `--scheduling-policy lpm`; observable via `Scheduler::policy()`; docs [specs/sglang-enablement.md](specs/sglang-enablement.md) + [docs/SGLANG-COMPAT.md](../docs/SGLANG-COMPAT.md) | SGLANG-DISTINCT | The ONE default-adjacent throughput lever. SGLang's OWN default is `fcfs` (`server_args.py:692`) so the default is already covered; `--schedule-policy=lpm` is opt-in. Token-neutral. |
| `SGLANG-SCHED-INBATCH` | in-batch prefix-caching de-prioritization (avoid colliding on the same not-yet-cached prefix) | `schedule_policy.py:253` (`_compute_prefix_matches`), thresholds `:76`/`:83`, sort key `:311` | LANDED inside `kLPM`: `maybe_reorder_waiting_for_lpm` (`scheduler.cpp:183-243`) de-prioritizes the second in-batch collider via OUR block-hash APC keys (no second trie); thresholds `scheduler.h:kInBatch{Check,Deprioritize}Threshold=32`; gate `tests/vllm/v1/test_scheduler_lpm.cpp`; engine row `ENG-SGLANG-BEHAVIOR-FLAG` SW2 | ACTIVE (order-only) | Admission-ORDER parity, output-neutral (RED-first gate). Throughput lever is **NOT-APPLICABLE** in our engine: our APC caches blocks at ALLOCATION time (`kv_cache_manager.cpp:267`), so the second same-step collider already HITS the first's just-cached prefix — the redundant prefill SGLang avoids (its radix updates post-forward) never occurs here. |
| `SGLANG-SCHED-OVERLAP` | overlap / zero-overhead scheduler — CPU processing of batch N-1 overlapped with GPU compute of batch N | `managers/scheduler.py:1563` (`event_loop_overlap`), `:344`; default ON `server_args.py:776` | `ENG-ASYNC-SCHED` `DONE`, default-ON (`async_scheduler.cpp`; `VT_ASYNC_SCHED`) | FUSED | Our async/overlap scheduler is the fused equivalent; default-ON in both. |
| `SGLANG-SCHED-CHUNKED` | chunked prefill | `server_args.py:655` (`chunked_prefill_size`); `schedule_batch.py` | `ENG-CHUNKED-PREFILL` (`scheduler.cpp:225,548`) | FUSED | Already covered. |
| `SGLANG-SCHED-CONTINUOUS` | continuous batching (running-first, `get_next_batch_to_run`) | `managers/scheduler.py:2598` | `ENG-SCHED-CORE` | FUSED | Already covered. |
| `SGLANG-SCHED-PRIORITY` | priority scheduling | `server_args.py:693` (`enable_priority_scheduling`) | our `SchedulingPolicy::kPriority` (`scheduler.h:56`) | FUSED | Already covered. |

## Attention backends

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-ATTN-BACKENDS` | per-phase attention-backend zoo (flashinfer / triton / fa3 / torch_native / flex / aiter / cutlass-mla / trtllm-mla), selected separately for prefill vs decode | registry `layers/attention/attention_registry.py:23`; selectors `server_args.py:1277` (`attention_backend`), `:1285` decode, `:1293` prefill; instantiation `model_executor/model_runner.py:2575` | our vendored FA2 + paged-attn dispatch (`KERNEL-ATTN-FA2`; `src/vt/cuda/cuda_paged_attn.cu`) selected via the backend seam | FUSED | Same "pick the fast attention kernel per phase" idea; our dispatch mirrors vLLM's `get_attn_backend`. Which specific kernel is faster is the benchmark track, not a behavior toggle. |
| `SGLANG-ATTN-SPARSE` | DeepSeek NSA/DSA sparse attention + DeepSeek-V4 compressed-KV pipeline | `layers/attention/dsa_backend.py`, `deepseek_v4_backend.py`; `mem_cache/deepseek_v4_*` | our MLA/DeepSeek track (`KERNEL-ATTN-MLA`; DeepSeek-V2 gated 8/8) covers standard MLA; sparse variants not yet | INVENTORIED | Standard MLA is FUSED; the NSA/DSA sparse + V4 compressed-KV variants are a distinct model/kernel scope, spike-first. |

## Quantization

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-QUANT` | scheme registry: fp8, modelopt_fp4/fp8 (NVFP4), w8a8_int8, w8a8_fp8, w4afp8, awq, gptq, compressed_tensors, marlin, mxfp4, gguf, bitsandbytes | registry `layers/quantization/__init__.py:73`, lookup `:143`; modelopt `modelopt_quant.py:39` | our [quantization-matrix.md](quantization-matrix.md) (82 rows: NVFP4 W4A4/W4A16, FP8, GGUF CIQ, Marlin, compressed-tensors) | FUSED | Same scheme families; selection logic mirrors vLLM. SGLang's hardware-specific extras (w4afp8, qoq, mxfp_w4a8 NPU, mlx, petit) are INVENTORIED breadth in our quant-matrix, not distinct behaviors. |

## Speculative decoding

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-SPEC-EAGLE` | EAGLE / EAGLE3 / NEXTN(MTP) / STANDALONE / multi-layer-eagle draft-verify | enum `speculative/spec_info.py:28`, `create_worker` `:193` (eagle `:225`, standalone `:229`); `server_args.py:1583` | our `SPEC-MTP` `DONE` (NEXTN/MTP == our MTP), draft/verify machinery | FUSED | Same draft/verify family. An EAGLE-specific drafter is a future MODEL port on the existing machinery, not a new mechanism. |
| `SGLANG-SPEC-NGRAM` | n-gram speculative (native C++ ngram) | `speculative/ngram_worker.py`; `speculative/cpp_ngram/`; `spec_info.py:235` | our ngram path | FUSED | Already covered. |
| `SGLANG-SPEC-DFLASH` | DFlash diffusion-style draft worker | `speculative/dflash_worker_v2.py`; `spec_info.py:200`; `server_args.py:1585` | `SPEC-DFLASH` `DONE` (ours: correctness-complete + at/above vLLM throughput, `CLAIM-DFLASH-D14`) | FUSED | Both engines have DFlash; ours is landed and gated. |

## MoE

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-MOE` | fused MoE: topk/router + grouped GEMM runners (triton / cutlass / deepgemm / flashinfer / marlin), EP dispatch | topk `layers/moe/topk.py:360`; dispatcher `layers/moe/fused_moe_triton/layer.py:87`; EP `layers/moe/ep_moe/layer.py:49`; cutlass `layers/moe/cutlass_moe.py` | our Marlin/cutlass MoE (Qwen3-MoE gated STRICT 6/6; GDN-MoE gate models) | FUSED | Same grouped-GEMM + top-k routing. The specific fastest runner is the benchmark track. |
| `SGLANG-MOE-EPLB` | EPLB expert-parallel load balancer + expert-location rebalancing | `eplb/eplb_manager.py:16` (`rebalance` `:55`), `eplb/expert_location.py:163`; `enable_eplb` `server_args.py:1819`; routing hook `topk.py:95` | none (single-box gate does not do multi-node EP rebalancing) | INVENTORIED | Large-scale expert-parallel feature; spike-first if multi-node EP is ever targeted. |
| `SGLANG-MOE-TBO` | two-batch / single-batch compute-comm overlap for EP all-to-all | `batch_overlap/two_batch_overlap.py`, `single_batch_overlap.py` | none | SGLANG-DISTINCT | Overlaps expert all-to-all comm with compute; genuinely-distinct EP-scale lever, opt-in, only matters at multi-GPU EP. |

## Constrained decoding

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-CONSTRAIN-BACKENDS` | grammar backends xgrammar / outlines / llguidance + reasoner-aware wrapper | selector `constrained/base_grammar_backend.py:223`; default xgrammar `server_args.py:4585` | our `StructuredOutputManager` per-step bitmask (`structured_output/manager.cpp:50`), xgrammar-style; `json_schema_to_gbnf` | FUSED | We ship vLLM's xgrammar-style per-step masking. outlines/llguidance backend variety is breadth, not a distinct behavior. |
| `SGLANG-CONSTRAIN-JUMP` | jump-forward decoding — emit FSM-forced token runs without model steps | `constrained/outlines_jump_forward.py:182,146,159` | SAFE SUBSET LANDED (`CLAIM-SGLANG-SW3`): forced-token detection hook `StructuredOutputGrammar::forced_token()` (`include/vllm/v1/structured_output/backend_types.h`; native impl `src/vllm/v1/structured_output/backend_native.cpp` — reuses the trie×FSM DFS, short-circuits on a 2nd valid token) + driver `DrainForcedTokens` (`src/vllm/v1/structured_output/jump_forward.{h,cpp}`, opt-in `VT_ENABLE_JUMP_FORWARD`, default OFF); gate `tests/vllm/v1/structured_output/test_jump_forward.cpp`; engine row `ENG-SGLANG-BEHAVIOR-FLAG` SW3; **EXPOSED (`CLAIM-SGLANG-ABI-DOCS`):** C++ `EngineParams::enable_jump_forward` (`std::optional<bool>`), C-ABI `vllm_model_params.enable_jump_forward` tri-state int (ABI **v10**, appended after the v9 fields), server `--[enable\|disable]-jump-forward`; `VT_ENABLE_JUMP_FORWARD` retained as override (`JumpForwardEnabled(optional<bool>)`); observable via `LoadedEngine::jump_forward_enabled()`; `vllm_abi_version()`==10; docs [specs/sglang-enablement.md](specs/sglang-enablement.md) + [docs/SGLANG-COMPAT.md](../docs/SGLANG-COMPAT.md) | PARTIAL (token-unique subset) | Output-neutral speed lever. LANDED = the TOKEN-UNIQUE forced run (exactly one grammar-valid token at a non-accepting state ⇒ the constrained sampler's only finite-logit token ⇒ argmax under ANY params ⇒ PROVABLY byte-identical to per-token decode, no re-tokenization). RESIDUAL (named): the general byte-forced-but-multi-tokenizable span (SGLang's re-tokenize + boundary rollback, `outlines_jump_forward.py:146-172` / `schedule_batch.py`@935cda944b^:503-544) is DELIBERATELY not jumped (forced_token→nullopt ⇒ falls back to normal decode); and production scheduler splice (jumped tokens have no computed KV ⇒ needs the KV-recompute path). Opt-in `--enable-jump-forward`; stays default-off until the production splice lands. |

## Sampling

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-SAMPLING` | sampling batch info + penalty orchestrator (freq/presence/repetition/min-new-tokens) | `sampling/sampling_batch_info.py:24,77`; `sampling/penaltylib/orchestrator.py` | our sampler (greedy/temp/top-k/p/min-p/penalties/seed/stop/logit-bias in vLLM's exact order) | FUSED | Same sampling controls and order. |
| `SGLANG-SAMPLING-CUSTOM` | per-request serializable `CustomLogitProcessor` (e.g. thinking-budget) | `sampling/custom_logit_processor.py:24`, `:61` (`ThinkingBudgetLogitProcessor`); gate `sampling_batch_info.py:134`; `enable_custom_logit_processor` `server_args.py:2593` | ACTIVE (`CLAIM-C7-CUSTOM-LOGITS`): per-request host callback through the C-ABI (`vllm_logits_processor`, ABI v8 — `include/vllm.h:186`), wired at the sampler's non-argmax-invariant stage (`src/vllm/v1/sample/sampler.cpp:308`; impl `src/vllm/v1/sample/logits_processor/builtin.cpp:75`); engine row `SAMPLE-CUSTOM-PROCESSORS`; **EXPOSED (`CLAIM-SGLANG-ABI-DOCS`):** C-ABI `vllm_sampling_params.logits_processor` + `logits_processor_user_data` (ABI v8, per-request programmatic hook — no server flag); docs [docs/SGLANG-COMPAT.md](../docs/SGLANG-COMPAT.md) §4 | FUSED | Same capability (user-supplied per-request logit processor). We mirror vLLM's `SamplingParams.logits_processors` structure/ordering exposed as a C-ABI callback (fn ptr + user_data); SGLang's Python/dill-serialized callable + server flag is the delivery difference, not a distinct behavior. Residual: single C callback (not a batched plugin graph), no Python-side registration. |

## Multimodal / tokenizer / LoRA

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-MM` | auto-registered per-arch mm processors + ViT CUDA-graph runner + video/audio extraction | dispatch `managers/multimodal_processor.py:13,44`; base `multimodal/processors/base_processor.py:180`; ViT graph `multimodal/vit_cuda_graph_runner.py` | our multimodal track: image+video e2e STRICT 32/32 on Qwen3.6-27B / Qwen3-VL-4B; audio (Voxtral) decoder token-exact | FUSED | Same mm-processor → vision-tower → backbone pipeline; ours is gated on the gate models. Per-arch processor breadth is model-matrix territory. |
| `SGLANG-TOKENIZER` | tokenizer as a separate async manager w/ dynamic-batch tokenization + multi-tokenizer mixin | `managers/tokenizer_manager.py:244`; `managers/async_dynamic_batch_tokenizer.py`; `tokenizer/tiktoken_tokenizer.py` | our byte-level BPE + SentencePiece + GGUF vocab, byte-exact vs the vLLM oracle | FUSED | Same tokenization result; async-process packaging is an architecture choice, not an output behavior. |
| `SGLANG-LORA` | LoRA manager + triton/torch/chunked/ascend backends, MoE-LoRA, overlap loader, eviction | `lora/lora_manager.py:57,100`; `lora/backend/`; `lora/lora_moe_runners.py` | our LORA track (`LORA-*` rows) | FUSED | Same LoRA-adapter application. MoE-LoRA + overlap-load are breadth items under the LoRA track, not distinct engine behaviors. |

## OpenAI server / entrypoints

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-SERVER` | OpenAI HTTP server: completions / chat / embeddings / rerank / score / classify / responses / transcription | routes `entrypoints/http_server.py:1606,1614,1624`; serving modules `entrypoints/openai/serving_*.py` | our OpenAI server subset: `/v1/completions`, `/v1/chat/completions`, SSE, `/v1/models`, `/tokenize`, `/detokenize`, `/metrics`, `/reset_prefix_cache` | FUSED | Core generation surface covered. Embedding/rerank/score/responses/transcription endpoints are INVENTORIED breadth (mirror vLLM's own surface where it has them). |
| `SGLANG-TOOLPARSE` | model-specific tool-call parser matrix (~26 detectors: qwen25/qwen3_coder, deepseek v3/v31/v32/v4, glm4/45/47, kimi_k2, hermes, mistral, llama3, gemma4, gpt-oss, pythonic, step3, minimax-m2, ...) | registry `function_call/function_call_parser.py:60`; `function_call/<name>_detector.py` | our 36 parser families / 40 accepted names (every vLLM tool parser at the pin except 3 Rust/Harmony) | FUSED | We already ported vLLM's overlapping set, held to the upstream test suites. Selection via `--tool-call-parser`. |
| `SGLANG-REASONING` | reasoning parsers (deepseek-r1, harmony, think-tag) | `parser/reasoning_parser.py`, `parser/harmony_parser.py` | our 7 reasoning parsers (think_auto, deepseek_r1, mistral, minimax_m2, step3, olmo3), streamed as `reasoning` deltas | FUSED | Same reasoning-split-before-tool-parse model. |
| `SGLANG-GRPC` | gRPC serving front-end (+ Anthropic / Ollama compat, Realtime/ASR) | `entrypoints/grpc_server.py:156`; `entrypoints/anthropic/`, `entrypoints/ollama/` | none (we ship the OpenAI HTTP + C-ABI surfaces) | OUT-OF-SCOPE | vLLM has no gRPC front-end either; niche protocol breadth, not a parity behavior. |

## Graph capture and runtime execution

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-BCG` | **Breakable CUDA Graph** — one forward captured as a SEQUENCE of `torch.cuda.CUDAGraph` segments split at eager break points, with no `torch.compile`; plus per-phase backend selection (`FULL`/`BREAKABLE`/`TC_PIECEWISE`/`DISABLED`, decode defaults `FULL`, prefill defaults `BREAKABLE` on CUDA) and graph-executable dedup via `cudaGraphExecUpdate` | `runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:204-241` (segment split), `:244-260` (replay), `:14-23,156-169` (shared mempool + weak-ref intermediates); `runner_backend/breakable_cuda_graph_backend.py:14-17` ("No torch.compile"); break points `layers/radix_attention.py:256`, `attention_forward_methods/forward_mla.py:1092`, `layers/radix_linear_attention.py:159`, `models/nemotron_h.py:1240`, `dsa/dsa_indexer.py:2414`; config `cuda_graph_config.py:38-45,95-112`; dedup `runner_backend/cuda_graph_dedup_mixin.py:219-242,358` | our capture primitive `src/vt/cuda/cuda_backend.cu:203-232`; bucket set `include/vllm/model_executor/models/decode_graph_sizes.h:32-41`; pure-decode routing `src/vllm/v1/worker/gpu/runner.cpp:1338-1341`; NINE hand-rolled drivers (`Qwen3_5DecodeGraph`, `Qwen3_5DenseDecodeGraph`, `Qwen3MoeDecodeGraph`, `Qwen3DenseDecodeGraph`, `DeepseekV2DecodeGraph`, `VoxtralDecodeGraph`, `deepseek_v4.cpp`, `laguna.cpp`, and the DFlash draft graph `src/vllm/model_executor/models/qwen3_dflash.cpp:771,870,1038,1091,1095,1106` — count corrected 2026-08-18, [#1179](https://github.com/mudler/vllm.cpp/issues/1179); `9bc4d7f44` recorded eight, and the DFlash driver is file-local with no header declaration, which is why it was missed) | INVENTORIED | **The CAPABILITY is vLLM-derived, not SGLang-distinct**: vLLM's v1 default is already `FULL_AND_PIECEWISE`, splitting at `splitting_ops` (`vllm/config/compilation.py:60-63,517,615,630` @ `555967922`), so it is in our benchmark denominator. What SGLang landed FIRST is the CONSTRUCTION — piecewise coverage from runtime stream capture plus a decorator, no Dynamo/Inductor/FX. That construction is the part worth mirroring, because we have no compiler either. **Prefill capture is REFUTED as a lever here** (GB10 2026-07-09: prefill GPU-idle-between-launches 3.8%, GPU-busy >96%, 27B gap 92.5% non-GEMM GLUE) — do not re-derive. Exec dedup is unported (`cudaGraphExecUpdate` appears nowhere in `src/` or `include/`). Spec [sglang-breakable-cuda-graph.md](specs/sglang-breakable-cuda-graph.md); issues [#1161](https://github.com/mudler/vllm.cpp/issues/1161), [#1162](https://github.com/mudler/vllm.cpp/issues/1162), [#1163](https://github.com/mudler/vllm.cpp/issues/1163), [#1164](https://github.com/mudler/vllm.cpp/issues/1164). |

## Structurally distinctive / research (vLLM has no native analogue)

| ID | SGLang surface | SGLang anchor (`file:line`) | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-PD` | prefill/decode DISAGGREGATION as a first-class scheduler mode + pluggable KV transfer (Mooncake / NIXL / Ascend / Mori) + encode role | mode `server_args.py:2273`; enum `disaggregation/utils.py:60`; transfer `:409`; roles `disaggregation/prefill.py`, `decode.py` | our engine is single-role; the KV-connector ABI is the nearest primitive | SGLANG-DISTINCT | vLLM has an experimental PD/connector path we partly mirror; SGLang's is deeper. Opt-in, cluster-scale; spike-first, not for the single-box gate. |
| `SGLANG-BATCH-INVARIANT` | batch-invariant / deterministic-inference ops (bitwise reproducible across batch composition) | `batch_invariant_ops/batch_invariant_ops.py`; `enable_deterministic_inference` `server_args.py:2508`; radix-safe deterministic attn subset `server_args.py:325,329` | none (our determinism story is token-exact-vs-oracle, not batch-invariant kernels) | SGLANG-DISTINCT | Genuine feature: reproducible logits regardless of batch grouping. Opt-in; useful for our own gate determinism. Spike-first. |
| `SGLANG-ELASTIC-EP` | elastic expert-parallel scaling + expert backup | `elastic_ep/elastic_ep.py`, `expert_backup_manager.py` | none | OUT-OF-SCOPE | Multi-node EP autoscaling; cluster-scale research. |
| `SGLANG-MULTIPLEX` | PD-mux — SM partitioning between prefill and decode on one GPU | `multiplex/pdmux_context.py`, `multiplex/multiplexing_mixin.py` | none | OUT-OF-SCOPE | SM-carving; niche, hardware-specific. |
| `SGLANG-DLLM` | diffusion language-model decoding | `dllm/config.py`, `dllm/algorithm/` | none | OUT-OF-SCOPE | Different decoding paradigm (research); not a vLLM parity behavior. |
| `SGLANG-CHECKPOINT-ENGINE` | live weight sync / RL rollout weight updates | `checkpoint_engine/checkpoint_engine_worker.py`; `weight_sync/tensor_bucket.py`; routes `http_server.py:1264` | none | OUT-OF-SCOPE | RLHF-serving weight hot-swap; not in the inference-parity scope. |
| `SGLANG-CUSTOM-KERNELS` | `sgl-kernel` custom CUDA/attention/MoE kernels | external `sgl-kernel` package | our vendored FA2 / Marlin / cutlass / GDN | OUT-OF-SCOPE | Kernel parity is the BENCHMARK track (`BACKEND-GATE-CUDA-SGLANG*`), measured not toggled. |
| `SGLANG-INTROSPECT` | KV-canary token-oracle sweeps + state-capturer (routed-expert / indexer-topk introspection) | `kv_canary/`; `state_capturer/routed_experts.py` | our SACRED gates + parity harness | OUT-OF-SCOPE | Dev/QA introspection tooling, not a runtime behavior. |

## Oracle stand-up (the program's own INVENTORIED work)

| ID | Item | Anchor | Our mapping / anchor | Class | Notes |
|---|---|---|---|---|---|
| `SGLANG-ORACLE-PERF` | stand up the SGLang PERF oracle on GB10 (arm64 cu130 image), wire into the existing preflight harness, re-pin P1→v0.5.15, close P2 image/model/GPU classification | image `lmsysorg/sglang:v0.5.15-cu130@sha256:d0a667e`; `bench_serving.py:874-887` | reuse `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` (`tools/bench/*serve_low*`, `scripts/dgx-sglang-low-concurrency.sh`) | **STOOD UP + FIRST FLOOR MEASURED** (`CLAIM-SGLANG-PERF-BENCH`, 2026-07-28) | SGLang v0.5.15 arm64 cu130 image PULLED + RAN the 27B-NVFP4 gate model on GB10 sm_121a ("server is fired up and ready to roll", CUDA graphs captured) — **no from-source build, as predicted**. Cache-neutral c8/c16 floor measured (see § "Perf oracle results" below). Residuals: 35B, c1/c2/c4, and the shared-prefix cache-ON arm. |
| `SGLANG-ORACLE-LEASE-WHEEL` | reach the pinned SGLang oracle from an `rc` lease through the PyPI wheels, because the 2026-07-28 image route is now forbidden and `rc run` has no `--image` | wheels `sglang-0.5.15-cp312-cp312-manylinux_2_34_aarch64.whl` sha256 `1c2d2602b4ba04c6a71d2f3bf2e3654da53987536f0d65dbe4f57cdc65c9812e` and `sglang_kernel-0.4.4-cp310-abi3-manylinux2014_aarch64.whl` sha256 `727e4bc53abeade20260186f99199200320b9fa51f8de7af90c01524cff73e5d`; upstream `docker/Dockerfile:210,242` installs the same kernel wheel from the same index at `CUDA_VERSION=13.0.1` | identity manifest [sglang-wheel-in-lease.json](specs/sglang-wheel-in-lease.json), 3338 files of the installed `sglang/` tree; spec [sglang-wheel-in-lease.md](specs/sglang-wheel-in-lease.md) | DONE | **W1 specified it; W2 built the driver and RAN it. `gateable` is `yes` again, on the wheel route.** Static analysis on 2026-08-19 settled the question that could have killed the route: `sgl_kernel/sm100/common_ops.abi3.so` holds 56 fatbins of 6 cubins each, architecture set 90/100/103/110/120/121, **zero fatbins without 121**, and `cuobjdump -lelf` (13.0, V13.0.85) names the 121 cubins **`sm_121a`**, the accelerated variant, one per container; the plain ones are the 90 cubins, which `sgl-kernel/CMakeLists.txt:127,221` explains exactly. **W2, 2026-08-23, two `rc run` jobs on `dgx:gpu0` and nothing else.** Job `86282a1a` installed both PyPI wheels and hashed the bytes that landed (`WHEEL_SHA_OK=1` twice), installed `flashinfer-jit-cache==0.6.12+cu130` so the first leg would not measure the JIT compiler, and asserted `IDENTITY_RC=0` from `cd /`: **3338 of 3338** manifest files, 0 missing, 0 extra, 0 differing, at pin `f63458b5`. No compiler ran for any of 199 resolved packages. Job `b9e7709d` re-asserted that identity, served `Qwen3_5ForConditionalGeneration` (Qwen3.8-27B bf16, 55.6 GB) to readiness in **454 s** with decode CUDA graphs captured, and the server's OWN log names `attention_backend='flashinfer'` and `moe_runner_backend='auto'`. One timed c1 leg: **6/6 completed, 0 failed, 768 output tokens = 6x128 exactly**, 173.23 s, via `/v1/completions`. The discarded warmup differed by 0.05%, which is the measurement that the warm cache worked. **Owed:** the c8 leg is VOID because the BOX REBOOTED DURING it at 22:27:00Z, and it is NOT a concurrency-8 datapoint of any kind -- it had issued only its single `warmup_requests=1` request, `sglang-c8.log` is 0 bytes, and ONE request was in flight at the moment of death (the next job on `dgx:gpu0` read `boot_id=26394f62` against this row's `02d5a76f`, PID counter 3510 -> 594, in `/mnt/nas_share/rc/gdn-moe-packed-ba/logs/gate-ab.log` lines 1-3 and 18-20; the 5,000 MB `MemAvailable` watchdog never fired because the machine died with 15,449 MB available, so the job never printed its own teardown assertion and `scripts/rc-sglang-lease-reap.sh` made it separately in job `0f84b66d` at 23:10:48Z: `COMPUTE_APPS=0`, `SGENV_PROCS=0`, `/tmp/sgenv` gone with the reboot -- that job archived nothing itself, so read it back with `rc logs 0f84b66d-1c30-4de5-bdb8-ee7b058f284a`, copied post-hoc to `/mnt/nas_share/rc/sglang-w2/out/reap-20260823T231048Z/job.log`), and why the box went down stays OPEN, the clock spread was **7.59%** against the 5% ceiling at 84 C so **no ratio** is derivable, and `moe_runner_backend` is NAMED but UNEXERCISED on a dense model. The replacement driver is `scripts/rc-sglang-oracle-lease.sh` with the gate at `scripts/sglang_lease_identity.py` and its mutation suite at `tests/scripts/test_sglang_lease_identity.py`; `scripts/dgx-sglang-low-concurrency.sh` stays unrunnable and was NOT patched. [#1265](https://github.com/mudler/vllm.cpp/issues/1265) stays open for the owed items. |
| `SGLANG-ORACLE-CORRECT` | SGLang greedy token-capture correctness cross-check on a shared dense model | oracle spec §3 | new parity capture reusing the SACRED harness | INVENTORIED | A regression net + divergence detector; vLLM already binds correctness, so lower urgency. |
| `SGLANG-ALIAS` | `--enable-radix-attention` CLI alias + C-ABI `enable_prefix_caching` tri-state field | `server_args.py:755` | CLI `examples/server/main.cpp:185`; C-ABI `include/vllm.h:100` (field to add); engine row `KV-SGLANG-RADIX-CACHE` RW1 | INVENTORIED | Trivial ergonomics over the shipped APC; reuses `tests/parity/test_qwen3_apc_e2e.cpp`. (Counted under INVENTORIED as unstarted implementation work.) |

## Perf oracle results — cache-neutral, 27B-NVFP4 (`CLAIM-SGLANG-PERF-BENCH`, 2026-07-28)

First measured SGLang-vs-ours competitor-floor comparison. vllm.cpp `7e9ffbff`
(main HEAD) vs SGLang `v0.5.15-cu130` arm64 (`@sha256:d0a667e`, arch digest
`ce9667f4`) in SGLang's production config (overlap scheduler + radix cache
default-on, CUDA graphs captured, `--mem-fraction-static 0.30`). **Model:**
`unsloth/Qwen3.6-27B-NVFP4` snapshot `890bdef7` (dense hybrid mamba+attention,
byte-identical weights both arms). **Workload:** deterministic custom corpus
(seed 0, 80 prompts × 1024 in / 128 out tokens exact, common-prefix ≤32 =
cache-neutral), greedy (temp 0, top-p 1, `ignore_eos`), `--request-rate inf`,
KV capacity 20480 tokens matched both arms. **Client:** `sglang.bench_serving`
(`sglang-oai` vs `vllm` OpenAI-completions path). **Box:** idle GB10 sm_121a,
one `flock $GPU_LOCK`, engines strictly sequential (SGLang fully torn down — GPU
freed + MemAvailable returned to baseline — before ours launched). **Reps:** 3
each after an untimed warmup; all within run-noise (CV mostly <0.1%). Equivalence
proof: both arms drove the identical corpus, both emitted exactly 80×128 output
tokens with 0 errors, identical peak concurrency (c16→21, c8→13). Evidence
(dgx): `~/work/sglang-bench-HEAD/evidence/raw/27/{sglang,ours}/c{8,16}-r{1,2,3}.jsonl`.

Median of 3 reps; higher-is-better ↑, lower-is-better ↓; ratio normalized so
`>1.0` = ours wins:

| Axis | c16 SGLang | c16 ours | c16 ratio | c8 SGLang | c8 ours | c8 ratio |
|---|---|---|---|---|---|---|
| Total throughput tok/s ↑ | 367.3 | **812.5** | **2.21× WIN** | 367.2 | **529.1** | **1.44× WIN** |
| Output throughput tok/s ↑ | 40.8 | **90.3** | **2.21× WIN** | 40.8 | **58.8** | **1.44× WIN** |
| Requests/s ↑ | 0.319 | **0.705** | **2.21× WIN** | 0.319 | **0.459** | **1.44× WIN** |
| Mean TTFT ms ↓ | 33425 | **2980** | **11.2× WIN** | 11289 | **1775** | **6.4× WIN** |
| Median TTFT ms ↓ | 33867 | **2714** | **12.5× WIN** | 18158 | **1638** | **11.1× WIN** |
| P99 TTFT ms ↓ | 49561 | **7220** | **6.9× WIN** | 18198 | **3589** | **5.1× WIN** |
| Mean TPOT ms ↓ | **104.0** | 154.4 | 0.67× GAP | **104.0** | 122.9 | 0.85× GAP |
| Mean ITL ms ↓ | **105.6** | 154.4 | 0.68× GAP | **105.7** | 122.9 | 0.86× GAP |
| Median ITL ms ↓ | **105.5** | 125.2 | 0.84× GAP | **105.5** | 109.9 | 0.96× ~tie |
| Peak mem GB (proxy) ↓ | **~54.9** | ~56.4 | ~1.03× ~tie | — | — | — |

**Honest verdict.** Ours decisively BEATS the SGLang v0.5.15 floor on aggregate
throughput (2.21× at c16, 1.44× at c8) and on TTFT/prefill (6–12× lower) on the
cache-neutral burst — our scheduler admits and drains the burst with near-zero
queue vs SGLang's 33 s / 11 s queue waits. SGLang WINS the steady-state
per-token decode-latency axis (TPOT/ITL 1.18–1.49× lower than ours): our higher
throughput comes from packing larger decode batches, which raises per-request
per-token latency. **This TPOT/ITL deficit is a reproduced OPEN GAP — a candidate
future lever, recorded honestly, not explained away.**

> **UPDATE 2026-07-28 (`CLAIM-DECODE-LATENCY-EXPLORE`) — hypothesis CONFIRMED by
> direct measurement; the gap is BATCH-COMPOSITION, not a per-token kernel
> deficiency.** On OUR engine, sweeping the decode batch B∈{1,2,4,8,16} (27B-NVFP4,
> `--max-num-seqs 16`) gives ITL(1)=**101.75 ms** (already ≤ SGLang's op-point
> 104–105 ms) rising monotonically to **158.5 ms** @ B16 (median 101.7→131.1).
> nsys (`decode_b{1,16}.nsys-rep`) shows every hot kernel (NVFP4 nvjet/cutlass
> GEMMs, GDN mamba) is **sub-linear** in batch (1.6–1.8× for 16× tokens → per-token
> GPU cost ↓~10×) — "more requests × cheaper-per-token", not "each token costs
> more". SGLang's **effective decode concurrency ≈4** (40.8 tok/s ÷ 1000/105 ms),
> not 16 — its 33 s admission queue (prefill-first, non-mixed steps,
> `managers/scheduler.py:2700`) keeps few requests decoding at once. **Throughput
> win and ITL loss are the same lever.** Knob = `max_num_seqs` /
> `max_num_batched_tokens` (both exist); latency-oriented point `max_num_seqs≈8`
> = ITL −21% at 1.38× SGLang throughput. Default stays throughput-oriented
> (unchanged). Full curve + repro: [specs/decode-latency-lever.md](specs/decode-latency-lever.md).

Peak memory (system
MemAvailable delta on the unified pool; nvidia-smi is N/A) is comparable, ours
~1.5 GB higher — indicative not binding. **Residuals (named):** 35B-A3B-NVFP4 not
run; low-concurrency c1/c2/c4 not run (SGLang c1 ran ~13.3 s/it → impractical for
3-rep reproduction this pass); shared-prefix cache-ON arm
(`BACKEND-GATE-CUDA-SGLANG-PREFIX`, RadixAttention vs our APC) not run; vLLM arm
not re-measured here (this is the SGLang floor specifically; vLLM stays the
behavior oracle); greedy token-exact cross-check (`SGLANG-ORACLE-CORRECT`)
separate/pending. Not the full P2 binding grid (which additionally requires the
vLLM arm + full c{1,2,4,8,16}).

---

## Honesty roll-up

- **FUSED (23):** RadixAttention, HiCache, KV connectors, Mamba-radix, ChunkCache
  fallback, overlap/chunked/continuous/priority scheduling, attention-backend
  dispatch, quantization scheme registry, EAGLE/ngram/DFlash speculation, fused
  MoE, xgrammar constrained decoding, sampling + penalties, multimodal, tokenizer,
  LoRA, OpenAI server core, tool + reasoning parsers. We and SGLang share the
  idea; our vLLM-derived code already expresses it.
- **SGLANG-DISTINCT (8):** LPM scheduling (SW1 LANDED), in-batch prefix
  de-prioritization (SW2 LANDED), radix eviction strategies, jump-forward
  decoding (SW3 — safe TOKEN-UNIQUE subset LANDED, `CLAIM-SGLANG-SW3`; the
  general re-tokenization span is a named residual), custom logit processors,
  batch-invariant determinism, PD disaggregation, two-batch EP overlap. Each is
  an OPT-IN over our vLLM-derived design, never a fork.
- **INVENTORIED (5):** sparse/NSA attention + DeepSeek-V4 compressed-KV
  (`SGLANG-ATTN-SPARSE`), EPLB expert rebalancing (`SGLANG-MOE-EPLB`), and the
  oracle-standup + alias work (`SGLANG-ORACLE-PERF`, `SGLANG-ORACLE-CORRECT`,
  `SGLANG-ALIAS`) — spike-first before implementing.
- **OUT-OF-SCOPE (8):** session radix cache, gRPC/Anthropic/Ollama front-ends,
  elastic-EP, PD-mux, dLLM, checkpoint-engine, sgl-kernel (benchmark track),
  introspection tooling.

See [specs/sglang-parity-oracle.md](specs/sglang-parity-oracle.md) for the gate
methodology (SGLang as correctness + performance oracle), the board assessment
(dgx GB10 via the arm64 cu130 image; no from-source build needed), and the
ranked execution plan.

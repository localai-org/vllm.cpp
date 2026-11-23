# Kolibri-1.1 support — ASSESSMENT (2026-11-23)

Row: `MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm`

Assessment only. No forward implementation lands in this unit. It enumerates the
released GGUF, states the delta against our kolibri1 (`Kolibri1ForCausalLM`)
support, pins the oracle landscape, estimates the port, and verifies how the
current tree treats a Kolibri-1.1 request.

Upstream artifacts checked 2026-11-23:

- Model repo `Aleph-Alpha/Kolibri-1.1`, revision `e9515166e22131ef` (config read
  from HF; a local copy is not on `/mnt/models` yet). GGUF repo
  `Aleph-Alpha/Kolibri-1.1-GGUF`, Q4_K_M single file, 45,510,000,000+ bytes,
  expected sha256 `ee511452dceb9171…` (LFS oid), downloading to
  `/mnt/models/Aleph-Alpha/Kolibri-1.1-GGUF/Kolibri-1.1-Q4_K_M.gguf`.
- Blog: aleph-alpha.com/en/blog/kolibri-1-1/ (2026-11-23).

## Config delta vs Kolibri-1 (read from config.json)

Identical to kolibri1 unless listed:

| Field | Kolibri-1 | Kolibri-1.1 |
|---|---|---|
| architectures / model_type | `Kolibri1ForCausalLM` / `kolibri1` | `Kolibri1_1ForCausalLM` / `kolibri1_1` |
| hidden / layers / heads / KV heads / head_dim / vocab / rope_theta | 2560 / 50 / 48 / 4 / 128 / 128000 / 10000 | same |
| MoE | 384 experts, top-6, moe_inter 512, shared expert 512, norm_topk_prob false | same |
| layer_types | 4 × `sliding_attention` + 1 × `full_attention` repeating | 4 × `sliding_attention` + 1 × **`indexed_attention`** repeating |
| `mlp_only_layers` | absent (all MoE) | `[0, 1]`, `mlp_only_intermediate_size` 3584 (dense FFN prefix) |
| `index_layers` | — | `[4, 14, 19, 29]` (the four indexers; the other six indexed layers are reindexers) |
| `kv_source_layers` | — | 50-entry map; layers 9, 14 share layer 4's K/V; 34, 39, 44 share 29's; 49 shares 4's (per-layer KV reuse) |
| `index_topk` | — | 2048 |
| `index_pool_size` / `index_compression` | — | 16384 / 16 (pool = 1024 blocks × 16 positions) |
| `index_num_local_tokens` / `index_num_sink_tokens` | — | 128 / 8 |
| `index_num_heads` | — | 12 (= 48 query heads / 4 KV heads) |
| `kv_cache_qat` | — | true (NVFP4-quantized KV cache, QAT) |
| `head_dtype` | — | `float32` (attention head compute), model dtype bf16 |
| quantization_config | fp8 dynamic, blocks 128×128 | same fp8 recipe; attention projections + lm_head excluded (bf16) |
| sliding_window / use_sliding_window | 513 / true | same |
| max_position_embeddings | 262144 | 262144 (blog claims up to 1M) |

Parameter count moves 78B → 75B: the densified prefix plus the dropped per-layer
K/V projections of the shared-KV layers more than offset the added indexer
projections.

## indexed_attention — the exact mechanism

Source: llama.cpp PR #32891 (open, 2026-11-23, +979 −105 in 27 files), diff read
in full this session; file:line references are the PR's post-image of
`src/models/kolibri1.cpp` and the converter `conversion/kolibri1.py`.

1. **Routing.** Each sparse (indexed) layer routes with the 12 query heads of
   the FIRST KV head (`n_route = n_head/n_head_kv`,
   `build_attn_sparse` @ src/models/kolibri1.cpp, PR #32891). K and V are
   rounded to NVFP4 (`ggml_round_nvfp4`) before they enter the cache, so the
   cached first-KV-head key IS the routing key.
2. **Score.** score(q, k) = mean over the 12 routing heads of
   `ReLU(q·k)/sqrt(head_dim)` with q and k rounded to NVFP4 — the upstream
   `ggml_lightning_indexer` op (already in mainline for Qwen QSA; the PR only
   widens its test matrix to `nh = 12`). Weights vector: `1/(12·sqrt(128))`.
3. **Selection set per query at position p.** Always: the first 8 sink tokens
   (`KOLIBRI_N_SINK = 8`) plus the last 128 tokens (`KOLIBRI_N_RECENT = 128`).
   The indexer selects top-2048 (`index_topk`) additional keys from the
   candidates `[8, p-128]`. Final mask width = 2048 + 8 + 128 = 2184
   (`n_attended`, test-backend-ops sparse-attn cases 2184).
4. **Pool for long contexts.** While `n_kv ≤ 16384` the candidate pool is the
   whole context. Beyond that, the layer's **indexer** scores 16-token blocks
   (`index_compression = 16`) by the mean of their NVFP4-rounded routing keys
   and keeps the top 1024 blocks (`index_pool_size/index_compression`),
   gathering block keys member-by-member from the cache
   (`ggml_get_rows` + per-member slab add, then round + scale).
   **Known upstream deviation**: training averages keys BEFORE NVFP4 rounding;
   llama.cpp averages after (round(mean(round(keys)))) — flagged in the PR and
   in `build_attn_sparse` as negligible by evals. Any our-side port must pick a
   side and record it.
5. **Indexer vs reindexer.** `index_layers = [4, 14, 19, 29]` are indexers: they
   compute a fresh pool. The other six indexed layers (9, 24, 34, 39, 44, 49)
   are reindexers: they reuse the key pool of the closest preceding indexer
   (`pool` threaded through the graph, `key pool of the closest preceding
   indexer`). Converter invariants (conversion/kolibri1.py, PR #32891): a
   sliding layer is never an indexer and never shares K/V; every reindexer's KV
   source is an earlier full layer with own KV.
6. **KV sharing.** `kv_source_layers` maps each layer to the layer whose K/V it
   reads. Shared-KV layers keep ONLY a `wq` projection — the converter drops
   their `k_proj`/`v_proj`/`k_norm` from the GGUF ("Checkpoints retain K/V
   projections that shared-KV layers do not use"). In llama.cpp this rides the
   existing KV-reuse hook (`hparams.kv_source`, reused from GEMMA3N/GEMMA4,
   llama-model.cpp create_memory) plus a new `get_cells()` accessor and a
   hard requirement for the unified KV cache when several sequences are live
   (`--kv-unified`), because the sparse inputs are per-sequence position maps.
7. **Attention itself.** Standard GQA flash attention over the 2184-wide mask,
   NoPE (sparse layers are the former "full" layers), kq_scale 1/sqrt(128).
8. **Dense prefix.** Layers 0–1 replace the MoE with a dense FFN
   (intermediate 3584, silu); llama.cpp keys it on
   `LLM_KV_LEADING_DENSE_BLOCK_COUNT` and `MODEL_TENSOR.FFN_GATE/UP/DOWN`
   under the kolibri1 tensor family (constants.py addition in the PR).

## What transfers from kolibri1 (extensive inventory)

Confirmed identical, directly reusable from the kolibri1 rows
(MODEL-TEXT-kolibri-1 and its ~10 follow-ups, docs/bench-evidence/kolibri1-*):

- Tokenizer (qwen2 pre-tokenizer, 128k vocab), reasoning/tool-call parsers,
  chat template — unchanged. The official 1.1 GGUF now carries
  `tokenizer.ggml.pre = "qwen2"` and `expert_gating_func = 2`, the exact
  values the kolibri1 community files got wrong (see the walk's KV diff).
- Sandwich norms, residual structure, QK-RMSNorm, NeoX RoPE on sliding layers
  only, sliding-window 513 layers and the iSWA split (4:1 → now 4:1 with the
  "full" slot indexed).
- MoE block: 384 experts, top-6, sigmoid routing on logits + expert_bias, no
  renorm, shared expert — byte-identical config; the bfdot/NEON GEMV arms,
  dequant cache, NUMA work carry over.
- fp8-dequant checkpoint arm (same quantization recipe, attention in bf16).
- GGUF k-quant reading: the walked Q4_K_M uses ONLY ids 0 (F32, 391),
  12 (Q4_K, 413) and 14 (Q6_K, 71) over 875 tensors — zero untabulated type
  ids; `gguf_reader.cpp`'s table covers them all (the Q5_1-class trap does not
  fire for this file; GSQ-RCO quants live in the repo's IQ1_M/IQ2_* tiers).

## What is NEW (the port)

- `indexed_attention`: the lightning-indexer routing path, the candidate mask
  builder (sink/recent bookkeeping, per-ubatch, host-side today upstream), the
  block-pool builder, and top-k mask application feeding the existing paged
  attention with a 2184-wide mask.
- NVFP4 KV cache for the sparse layers (`kv_cache_qat`): round K/V (and the
  routing queries) to NVFP4 and store the 4-bit cache; our
  `gguf_reader.cpp` already tabulates NVFP4 (id 40, block 64/36) for weights,
  but a KV cache in NVFP4 is a new cache dtype for the paged attention path.
- Shared-KV cache plumbing: per-layer KV source indirection in the KV-cache
  spec (`v1::KVCacheConfig` groups) and in decode, plus the unified-cache /
  single-stream constraint upstream needed for multi-sequence.
- Dense-FFN prefix (layers 0–1): trivial dense FFN arm next to the MoE.
- Refusal hardening: see "Refusal behaviour" — the GGUF arm must refuse a 1.1
  file by name.

## Effort estimate (units, human-scale implementation sessions)

| Unit | Content | Size |
|---|---|---|
| U1 loader/enumeration | GGUF KV + tensor-map extension for the 1.1 file (new KVs: indexer types, kv source map, indexer top-k, candidate block size/top-k blocks; dense prefix tensors; Q-only shared-KV layers), config parser for `kolibri1_1` (reuse `ParseKolibri1Params`, add indexed layer kind + index fields), HF-safetensors loader for the fp8 checkpoint | M (2–4 sessions; ~80% reuses kolibri1_gguf_weights/kolibri1 weights) |
| U2 sliding layers | The 40 sliding layers: kolibri1's iSWA path unchanged → FREE except layer-type plumbing (`indexed_attention` strings and the KV-source map touching the cache spec) | S (≤1 session) |
| U3 indexed_attention kernel + retrieval | The new work: NVFP4 round op (CPU NEON first), lightning-indexer score path for 12×4 heads at head_dim 128, candidate mask builder, block pool, top-2048 selection, mask application into the paged attention; correctness gate vs goldens | L (6–10 sessions; the mask/pool bookkeeping is the long pole) |
| U4 shared-KV plumbing | kv_source indirection in `MakeKolibri1KVCache` + decode, unified-cache constraint, per-sequence position maps | M (2–3 sessions) |
| U5 paged integration + serving | Paged KV in NVFP4, batching, prefix-cache interactions, registry + `include/vllm.h` surface, USAGE docs | M (2–3 sessions) |
| U6 quantized arms + gates | Q4_K_M arm gate vs llama.cpp/official GGUF, fp8 HF arm gate vs plugin goldens, port the upstream tests (llama.cpp test-backend-ops sparse shapes 2184; plugin kernel tests where portable) | M (2–4 sessions) |

Total ≈ 15–25 focused sessions for the CPU bf16/Q4_K_M arms, dominated by U3.
A token-exact CPU gate is achievable against the official GGUF (greedy decode,
llama.cpp b-series once #32891 merges) and against plugin v2.0.0 goldens only on
CUDA SM90/SM100 (plugin constraint — see oracle landscape).

## Risks

- **NVFP4 QAT numerics.** The sparse layers were TRAINED with 4-bit KV; a
  bf16/f32-KV "simplification" is not a fallback, it is a different model. The
  round op must be bit-exact (upstream pins ties-to-even, E4M3 scale = amax/6).
- **The upstream rounding-order deviation** (pool keys averaged after rounding)
  must be mirrored or consciously diverged and recorded; token gates near ties
  will expose it.
- **GGUF arch collision** — `general.architecture` stays `kolibri1` for 1.1
  files (converter reuses MODEL_ARCH.KOLIBRI1). See "Refusal behaviour".
- **Host memory.** Q4_K_M is 45.5 GB on disk; the fp8 HF checkpoint is ~31
  shards (size TBD at walk; bf16 upgrade ~150 GB). CPU gate needs a box with
  headroom over 45.5 GB + KV + activations (the kolibri1 rows already ran the
  78B model on this class of host).
- **Oracle lag.** The only executable CPU oracle (llama.cpp) is an UNMERGED PR;
  the CUDA-only plugin pin cannot gate our CPU arm. Until #32891 merges, the
  gateable CPU denominator is the PR head built from source (a scratch
  implementation by the registry's meaning — record it as such).
- **128-sink/recent bookkeeping in paged decode** interacts with our paged
  cache block layout; upstream needed a per-sequence cell→position map and a
  unified-cache restriction, and we should expect the same constraint.

## Oracle landscape (pins; nothing implemented here)

| Oracle | Record | Status for kolibri1_1 |
|---|---|---|
| llama.cpp (primary CPU reference for the GGUF arm) | `.agents/oracles/llama-cpp.md` pin `9df0b73c7079` (b11765) and `.agents/oracles/llama-cpp-kolibri1.md` | NEITHER covers kolibri1_1. Support = PR #32891, OPEN 2026-11-23, head `5f4d222898dd`, base `a0677c341ba6` (master at PR creation), +979 −105. Unmerged ⇒ unpinnable; when it merges, the first release tag carrying it is the pin advance (same recipe as the b11765 advance). GGUF day-one: `Aleph-Alpha/Kolibri-1.1-GGUF` (converted with the PR's own converter). |
| aleph-alpha-inference (primary, model author) | `.agents/oracles/aleph-alpha-inference.md` pin v1.0.0 `049a6a7bd240` | v2.0.0 released 2026-11-23 14:45 UTC (wheel sha256 `9c98c2610be03dce…`), registers `Kolibri1_1ForCausalLM`, targets vLLM 0.33, requires transformers ≥ 5.16.1, CUDA-only kernels (SM90/SM100), NVFP4 KV + FlashAttention 4, TP=1. NOT gateable on our CPU/aarch64 class — the v1 kolibri1 gateability path (vLLM 0.31+cpu) does not carry to v2. A pin advance needs a CUDA lease measurement; until then v1 stays the pinned record and kolibri1_1 has no gateable plugin oracle. |
| transformers | `.agents/oracles/transformers.md` | kolibri1_1 is NOT in transformers main at check time (2026-11-23, latest release v5.24.0): no `models/kolibri1_1` package, no `Kolibri1_1` symbols, no indexed-attention fields in `kolibri1`. The blog's "day-one transformers support" claim is not yet visible upstream — likely an unmerged PR or plugin-side reference. Record: NO transformers oracle for kolibri1_1; re-check before any port wave starts. |

Registry semantics unchanged: `aleph-alpha-inference` and `llama-cpp` oracle
rows already exist; kolibri1_1 stays inside their scopes only after their pins
advance. No `oracle-registry` edit needed.

## Refusal behaviour of the current tree (deliverable 6)

- **HF checkpoint path: correct.** The registry resolves by exact architecture
  string (`ModelRegistry::Resolve`, src/vllm/model_executor/models/model_registry.cpp:262)
  and kolibri1 registers exactly `Kolibri1ForCausalLM`
  (src/vllm/model_executor/models/kolibri1_registry.cpp:381). A config with
  `Kolibri1_1ForCausalLM` fails resolution and raises the unsupported-arch error
  that names the architecture and lists the supported set
  (model_registry.cpp `RaiseForUnsupported`). It never falls into kolibri1.
  Even if it resolved, `ParseKolibri1Params` refuses `indexed_attention`
  layer-type entries by name (kolibri1_registry.cpp:63, the
  `sliding_attention`/`full_attention` VT_CHECK).
- **GGUF path: TRAP — silently wrong claim, found this session.** llama.cpp's
  converter maps `Kolibri1_1ForCausalLM` to GGUF `general.architecture =
  "kolibri1"` (conversion/kolibri1.py, PR #32891). Our GGUF arm keys on that
  same string (`kKolibri1GgufArch`, kolibri1_gguf_weights.cpp:25, checked at
  :186/:222) and reads only the kolibri1 KVs — a Kolibri-1.1 GGUF would load as
  kolibri1: the shared-KV layers' missing `attn_k/v` tensors surface only as an
  incidental tensor-missing error (or worse, load wrong if a future quant
  carries them), the indexer KVs are ignored, and `layer_types` would call the
  indexed layers "full". This is a correctness refusal owed, not a cosmetic
  one: the arm must refuse a kolibri1 GGUF that carries the 1.1 indexer KVs
  (`kolibri1.attention.indexer_top_k` or the kv-source map) with a message
  naming the missing kolibri1_1 support. Filed as a row-level issue in this
  change; the guard belongs to the port row's first unit (U1).

## Proposed row and wave sequence

Row id `MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm` (parallel to
`MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm`). Waves:

1. W1 — U1 + the GGUF refusal guard (spec first; the guard is small enough to
   ride W1 but is independently landable).
2. W2 — U2 + U4 (sliding layers onto the shared-KV cache spec; dense prefix).
3. W3 — U3 (indexed_attention kernel + retrieval cache), gated on goldens
   captured from the official GGUF through a llama.cpp PR-head build.
4. W4 — U5 + U6 (paged NVFP4 KV, serving surface, quantized-arm gates, ported
   upstream tests, USAGE weights pin: exact HF repo + revision + sha256 per
   arm).

Gate plan: token-exact CPU gate vs the official Q4_K_M GGUF through llama.cpp
(post-merge revision, or the PR head recorded as scratch), plus the fp8 arm vs
plugin goldens only if a CUDA lease is available. Ground truth checkpoints:
HF `Aleph-Alpha/Kolibri-1.1` @ `e9515166e22131ef`; GGUF LFS oid `ee511452…`.

## Walk and evidence

- `walk-q4km.txt` — one-pass GGUF walk (metadata KVs, tensor census, offsets,
  type-id audit against gguf_reader.cpp) + the KV diff against
  `/mnt/models/Hob-forge-Kolibri-1-GGUF/Kolibri-1-Q4_K_M.gguf`. Written when the
  download completes (this commit carries the assessment; the walk file is
  amended in the same branch).

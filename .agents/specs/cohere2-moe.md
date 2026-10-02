# Cohere2MoeForCausalLM (North): the MoE text model with interleaved sliding window

**Row:** `MODEL-TEXT-cohere2-moe-cohere2-moe-for-causal-lm` (`INVENTORIED` ->
`SPIKE` with this spec).
**Issue:** `ISSUE-LOCAL-01M3S23H9B25EFPFJN75Y1XBWY`.
**Date:** 2026-09-30. **Base:** `b45a94273`.
**Branch:** `feat/cohere2-moe` (developer direction for this campaign: one
branch per item, local commits only, no pull request). The spec commit precedes
the implementation commits on the same branch.

## Now

`ACTIVE`. Registered, loaded from bf16 safetensors and forwarded through
`ModelRegistry::Forward` and `GPUModelRunner` on CPU. The CPU gates are green
(`## Evidence`). A fresh review, the token gate against vLLM, GGUF and the
quantized arms are open (`## Owed`).

## Scope

Register `Cohere2MoeForCausalLM` and make it forward on CPU through the shared
seams, for the released `CohereLabs/North-Mini-Code-1.0`
(@ `d11e61a842617a22dc328552fa5bb86231ee4f37`, bf16, 56.8 GiB, 49 layers,
hidden 2048, 32 q heads / 4 kv heads, head_dim 128, 128 experts top-8,
vocab 262144). The mechanisms, each from the pinned file:

1. Per-layer sliding window from `layer_types`: a `sliding_attention` layer has
   window `sliding_window + 1` (`cohere2_moe.py:205-211`); a full layer has none.
2. RoPE only on sliding layers, and on prefix-dense layers when
   `prefix_dense_sliding_window_pattern == 1` (`:213-222`, `:242-243`). GPT-J
   interleaved style (`is_neox_style=False`, `:202`). A full-attention MoE layer
   is NoPE.
3. `RMSNorm` (f32 statistics, weight applied in f32, `:75-94`) when
   `rms_norm_eps` is set, else the Cohere `LayerNorm` (`:97-103`).
4. The parallel block: one input norm feeds attention AND the MLP, and
   `hidden = residual + attn + mlp` (`:371-384`).
5. A dense prefix MLP from `mlp_layer_types`, normalized from
   `first_k_dense_replace` when absent (`:405-419`), sized
   `prefix_dense_intermediate_size` (`:351-360`).
6. The router: `sigmoid(gate(x))` then top-k, renormalized only when
   `norm_topk_prob` (`:56-72`, `:301-314`). North sets it False.
7. Shared experts when `num_shared_experts > 0`, combined `average` (the MoE sum
   halved, `:326-327`) or `sum` (`:287-303`). North ships none; the arm is
   ported and synthetic-gated.
8. Tied embeddings: `compute_logits` uses `embed_tokens` with `logit_scale`
   (`:499-502`, `:526-530`); `lm_head.*` is dropped at load (`:478`).
9. `use_qk_norm` is NOT in the pinned `cohere2_moe.py`. A checkpoint that sets
   it True is refused by name (the pin has no such mechanism to mirror).

Out of scope, each refused by name or owed: the GGUF arm (llama.cpp `b10451`
defines `cohere2moe`; the loader arm is owed), FP8 / W4A16 / NVFP4 sibling
checkpoints, the EAGLE drafter (`North-Mini-Code-1.0-eagle`), the gated
`North-Small-Translate-1.0` (406 GiB), GPU device arms beyond what the shared
ops provide, and speed.

## Upstream chain

The vLLM parity pin is `a7c23ac96d` (`.agents/upstream-sync.md`). The line
citations in this spec were taken at `e126687a9a`; between the two revisions
`cohere2_moe.py` only gained the EAGLE3 auxiliary-hidden-state plumbing
(`EagleModelMixin`, `SupportsEagle3`) and `commandr.py` a docstring, so the
forward semantics are identical. Lines up to `:439` are unchanged; the
`Cohere2MoeForCausalLM` class moved by +11 (`lm_head` drop `:489`,
`LogitsProcessor` `:510-513`, `compute_logits` `:537-541` at `a7c23ac96d`).
The code and the transcription cite `a7c23ac96d`.

vLLM `e126687a9a`: `vllm/model_executor/models/cohere2_moe.py` (534 lines),
`commandr.py::LayerNorm`, the `Attention` layer's `per_layer_sliding_window`,
`get_rope(..., is_neox_style=False)`, `FusedMoEFactory` with
`custom_routing_function`. The HF config class is transformers' `Cohere2MoeConfig`
(`model_type: cohere2_moe`).

## Our baseline

At `b45a94273`: `CohereForCausalLM` is registered (dense, LayerNorm, full-width
GPT-J RoPE, parallel block, tied, `logit_scale`) and refuses `use_qk_norm` and
`sliding_window` by name (`commandr_registry.cpp:129-137`). Laguna
(`laguna*.cpp`) carries the closest shared pieces: interleaved sliding window
over one paged cache, a dense layer 0 and a sigmoid MoE router. The
sliding-window switch is `ENG-ATTENTION-WINDOW`.

## Port map

| Ours (new files) | Upstream |
|---|---|
| `cohere2_moe.h`, `cohere2_moe.cpp` (forward) | `Cohere2MoeAttention`, `Cohere2Moe`, `Cohere2MoeDecoderLayer`, `Cohere2MoeModel` |
| `cohere2_moe_weights.cpp` (loader) | `load_weights` + `hf_to_vllm_mapper` |
| `cohere2_moe_registry.cpp` | the registration, config hook, KV spec |

## Dependencies

- Shared ops (`vt::MatmulBT`, `vt::RmsNorm`, `vt::RopeFromCache` non-neox, the
  paged attention with a window, the grouped MoE GEMM and SiLU-and-mul).
- `ENG-ATTENTION-WINDOW` for the per-layer window.
- The Command-R `LayerNorm` path for the non-RMS arm.

## Tests to port

vLLM `e126687a9a` ships no `cohere2_moe` model test (`tests/models/` has no
file naming it), and neither does `a7c23ac96d`. Its only test-side entry is
`tests/models/registry.py:232-237` @ `a7c23ac96d`
(`_HfExamplesInfo("CohereLabs/North-Mini-Code", trust_remote_code=True,
is_available_online=False, min_transformers_version="5.9.0")`),
so upstream never runs the model in CI. The registry example-config coverage
entry is ported by extending `test_model_registry`.

## Gates

- Config: the released `config.json` committed as a fixture and parsed; each
  mechanism flag read from it; `use_qk_norm: true` refused.
- Forward: a tiny synthetic checkpoint against a torch transcription of the
  pinned `cohere2_moe.py` (f32 arm tight, bf16 envelope), with one mutation-
  proven case per mechanism 1-8.
- Real tensors (env-gated): the first layers of North-Mini-Code-1.0, fetched by
  HTTP range, per-layer against the transcription.
- Reachability: the model loads through `ModelRegistry` and decodes through the
  runner.

## Work breakdown

1. This spec and the records.
2. Config parse and registration, with the refusals.
3. Loader (bf16 safetensors), enumeration accounted against the released index.
4. Forward (host reference and the runner path through the shared seams).
5. Synthetic gates, the real-tensor layer gate, reachability.
6. Records: FEATURES, USAGE weights, matrix row, `## Outcome`.

## Risks/decisions

- The window is `sliding_window + 1`, not `sliding_window`. Off by one keeps
  every shape.
- NoPE on full MoE layers but RoPE on the full prefix-dense layer. Getting
  either wrong keeps every shape.
- The router does NOT renormalize for North. Renormalizing keeps every shape.
- `shared_expert_combination_strategy == "average"` halves the WHOLE MoE output
  (routed + shared), not only the shared part (`:326-327`).
- A token gate on the real checkpoint needs 56.8 GiB over a ~5 MB/s link and a
  GPU oracle; it is owed.

Decisions taken while implementing (each argued here, per AGENTS.md "Changing
the rules"):

- **Config defaults are the HF config class's.** vLLM reads the config through
  transformers' `Cohere2MoeConfig`, so an absent key reaches the model file as
  the class default (logit_scale 0.0625, sliding_window 4096, head_dim 128,
  expert_selection_fn "softmax", norm_topk_prob True, strategy "average",
  sliding_window_pattern 4, and the class's layer_types derivation from
  first_k_dense_replace). The model file's `getattr` fallbacks are unreachable
  behind the class and are not mirrored. A sliding layer with `sliding_window`
  null or 0 is refused, because upstream computes `None + 1` there.
- **`--disable-sliding-window` is refused by name** when any layer is sliding:
  `config/model.py:857-860` @ `a7c23ac96d` sets `sliding_window = None` and the
  model file then fails at `sliding_window + 1`.
- **`sliding_window` as a float.** An integral float (`4096.0`) is the same
  window to Python's `+ 1` and is accepted; a fractional value is refused by
  name (`sliding_window must be an integer`).
- **One full-attention KV group.** Upstream gives a sliding layer a
  `SlidingWindowSpec`; that only lets the allocator free blocks behind the
  window. The attended keys are set by the per-layer window at the kernel
  (`ResolveAttentionWindow`, ENG-ATTENTION-WINDOW), which is the tree's
  convention for every interleaved-window model (Gemma-3, Laguna). Memory-only;
  recorded under `## Owed`.
- **Tracked exception: `dense_attn::AttnBlock` is not used.** It is the Qwen3
  block (qk-norm, NeoX RoPE on every layer) and cannot express a NoPE layer or
  GPT-J pairing. The attention block composes the same vt ops the Command-R block
  does (`MatmulBT` through `layers::UnquantizedLinearMethod`, `QkvSplit`,
  `RopeFromCache(is_neox_style=false)`, `ReshapeAndCache`, `PagedAttention`).
- **Seam extension: `layers::UnquantizedMlpGateUpMethod` takes an activation
  dtype** (default bf16, so every earlier caller runs the same op sequence).
  The dense prefix MLP, the shared expert and every routed expert go through
  it. The f32 value is used only by the arithmetic arm of the gates; the loader
  always sets bf16.
- **Sigmoid routing through `vt::MoeRouterTopK` with one expert group.** The
  op defines sigmoid scoring on its grouped path only; one group that survives
  keeps every expert, so it is exactly `token_choice_with_bias` (the form
  `kimi_linear.cpp` uses). A non-"sigmoid" `expert_selection_fn` takes the
  softmax top-k FusedMoE uses without a custom routing function.
- **Router ties.** bf16 router logits produce EXACT ties at the k-th boundary
  on the synthetic configs. `torch.topk` leaves their order unspecified; the
  tree's router takes the lowest expert index (ops.h "DETERMINISM DEVIATION").
  The transcription selects with a stable sort so both sides agree. The first
  bf16 run of config a was red by 0.156 relative for exactly this reason.
- **Shared-expert combine order.** FusedMoE returns `shared + routed`, each in
  the model dtype; the port rounds the routed sum, adds the shared output, then
  halves for "average". This is `vt::MoeCombine` without its `shared` operand
  followed by `vt::Add`, because the operand form rounds once instead of twice.
- **Logits are f32**, as for every dense registration here (the ForwardLogits
  carrier and the sampler read f32). Upstream rounds them to bf16 first.

## Stop conditions

- A mechanism the shared seams cannot represent: extend the seam or record the
  exact tracked exception, never a parallel path.

## Owed

Every item below is tracked by the row issue named in the header.

- The end-to-end token gate against pinned vLLM `a7c23ac96d` (GPU lease, the
  56.8 GiB checkpoint).
- The GGUF arm against llama.cpp `b10451` (`cohere2moe`). Refused by name
  today.
- The quantized siblings (FP8, W4A16, NVFP4). Refused by name
  (`quantization_config`, or a non-BF16 tensor).
- The EAGLE drafter (`North-Mini-Code-1.0-eagle`) and the EAGLE3
  auxiliary-hidden-state path the pin added to `Cohere2MoeModel.forward`
  (`EagleModelMixin`, `SupportsEagle3`, `cohere2_moe.py:454-470` @
  `a7c23ac96d`).
- The device MoE arm. Routed experts run one expert at a time through the
  shared gate-up seam (correct on every backend, slow). The grouped bf16 GEMM
  (`vt::MoeGroupedGemmBf16GateUpSilu` / `vt::MoeGroupedGemmBf16`) and a decode
  graph are the speed work, with no denominator yet.
- The `SlidingWindowSpec` KV group (memory only; see `## Risks/decisions`).
- A fresh review of the implementation head (AGENTS.md "How work gets done").

Found during this work and not part of it:

- `ISSUE-LOCAL-01M3S6GT9BQR7H0T3M2HRXR0A1`: `tests/scripts/test_q4exp_layerfp_diff.py`
  fails 2 of 38 cases at the base `46f7c27ca` already (its publisher no longer
  finds its `docs/USAGE.md` anchor). `scripts/agent-preflight.sh --staged`
  reports it.

## Evidence

CPU build `build-cpu` (Release, `-Werror`), 2026-09-30. Commands run from the
worktree root.

- `./build-cpu/tests/test_cohere2_moe`: 13/13 cases pass (the real-tensor and
  structural arms print SKIP without their env). Synthetic arms, relative max
  error against `scripts/cohere2-moe-ref.py`:

  | config | mechanisms switched | f32 prefill | f32 decode past W | bf16 |
  |---|---|---|---|---|
  | a | RMSNorm (both eps keys), sigmoid no renorm, forced-RoPE dense prefix, window 4, NoPE full MoE, logit_scale 0.5 | 5.68e-7 | 5.68e-7 | 1.36e-2 |
  | b | LayerNorm, renormalized sigmoid, 2 shared experts averaged, derived layer_types, NoPE dense prefix (pattern 2) | 3.74e-7 | 3.74e-7 | 6.40e-3 |
  | c | softmax router, class defaults (norm_topk_prob, logit_scale), 1 shared expert summed, no dense prefix | 3.46e-7 | 3.46e-7 | 2.48e-2 |
  | d | non-contiguous dense layer (dense, sparse, dense, sparse; pattern 1): layer 2 is dense, full attention and NoPE; `sliding_window` given as `3.0` | 3.84e-7 | 3.84e-7 | 6.96e-3 |

  Bounds: f32 2e-5, bf16 3e-2 (the reference's own bf16-vs-f32 distance on the
  configs whose bf16 routing agrees with f32 is 4.6e-3..2.6e-2).
- `VLLM_COHERE2_MOE_REAL_DIR=<dir> ./build-cpu/tests/test_cohere2_moe
  -tc="*REAL*"`: real North tensors (subset fetched by HTTP range, 2.57 GB,
  shards 1, 2, 4, 5 and 49 at `d11e61a842617a22dc328552fa5bb86231ee4f37`), 24
  tokens at positions 1000..1023, relative max error of the layer contribution
  (output minus residual):

  | layer | kind | f32 arm | bf16 arm |
  |---|---|---|---|
  | 0 | full attention, forced RoPE, dense MLP 3072 | 7.74e-6 | 5.19e-3 |
  | 1 | sliding (W 4097), RoPE, MoE 128 top-8 | 3.88e-6 | 1.21e-2 |
  | 4 | full attention, NoPE, MoE 128 top-8 | 5.09e-6 | 5.54e-3 |
  | final norm | RMSNorm | 3.34e-6 | 5.08e-3 |

  The window does not bite at 24 tokens against W = 4097; the synthetic
  configs gate it.
- `VLLM_COHERE2_MOE_INDEX=<index> ./build-cpu/tests/test_cohere2_moe
  -tc="*STRUCTURAL*"`: shipped 18730, enumerated 18730, unaccounted 0, missing 0.
- `./build-cpu/tests/test_cohere2_moe_registry`: 2/2. `ModelRegistry::Load`
  under "Cohere2MoeForCausalLM" from a temp checkpoint and config.json, then a
  greedy decode through `GPUModelRunner` over out-of-order blocks: `28 49 49 49
  49 49 49 42`, equal to the transcription's bf16 greedy continuation (smallest
  top-1/top-2 gap 0.42).

Mutation evidence (each applied to the tree, rebuilt, both binaries run,
restored byte-for-byte with a sha256 check):

| mutation | red cases |
|---|---|
| window `sliding_window` instead of `+ 1` | config fixture, defaults, forward a/b/c, runner |
| window not applied at the kernel | forward a/b/c, runner |
| RoPE on every layer (NoPE lost) | config fixture, defaults, forward a/b/c |
| force_rope dropped on the dense prefix | config fixture, defaults, forward a, runner |
| force_rope ignores prefix contiguity (a dense layer after a sparse one keeps RoPE) | forward d |
| NeoX pairing instead of GPT-J | forward a/b/c, runner |
| RMSNorm and LayerNorm arms swapped | config fixture, defaults, forward a/b/c, runner |
| layer_norm_eps on the RMS arm | forward a |
| MLP reads the residual, not the shared norm | forward a/b/c, runner |
| attention branch not added | forward a/b/c, runner |
| mlp_layer_types normalization off by one | config fixture, forward a/c, loader, runner |
| prefix width ignores prefix_dense_intermediate_size | config fixture, forward a/b, runner |
| router always renormalizes | forward a, runner |
| sigmoid scored as softmax | config fixture, forward a/b, runner |
| "average" not applied | forward b |
| "average" halves only the shared part | forward b |
| shared expert dropped | forward b/c |
| logit_scale not applied | forward a/b/c |
| lm_head not dropped | loader |
| use_qk_norm refusal removed | refusals |
| gate-up seam ignores the activation dtype | forward a/b/c (f32 arm) |
| REGISTER_VLLM_MODEL removed | config fixture, GGUF refusal, KV spec, registry load, runner |

Rerun with `VLLM_COHERE2_MOE_REAL_DIR` set, the real-tensor case is red as well
for: RoPE on every layer (layer 4 is NoPE), force_rope dropped (layer 0),
NeoX pairing, layer_norm_eps on the RMS arm, and the router always
renormalizing.

Red-first note: the tests were written with the implementation in one pass, so
the red for each mechanism is the mutation above (the mechanism absent from the
tree). Two reds were real first runs: the bf16 arm of config a (0.156 relative,
the router tie order, fixed in the transcription as recorded above) and the
runner gate at block size 4 (no CPU attention backend supports it; the gate
uses 16).


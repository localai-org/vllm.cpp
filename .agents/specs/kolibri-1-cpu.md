# Kolibri-1 — CPU-only port spec (`MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm`)

Row: `MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm`

## Scope

Port the `Kolibri1ForCausalLM` architecture (model_type `kolibri1`,
Aleph-Alpha Kolibri-1) to the **CPU path only**: config parse + validation,
the two-group KV-cache spec (full-attention + sliding-window groups),
the 32-shard safetensors weight loader, the sigmoid-logit-add MoE router,
the hybrid SWA/full-attention CPU forward, and tokenizer integration,
token-exact against a reference.

Not in this row (owed, see `## Owed`): GGUF arms, CUDA/Tenstorrent arms, the
GPU gateability measurement for the aleph-alpha-inference oracle.

## The checkpoint

- Source: `Aleph-Alpha/Kolibri-1`, downloaded and verified at
  `/mnt/models/Aleph-Alpha/Kolibri-1` (32 safetensors shards, index present,
  `metadata.total_size = 78,827,029,120` bytes).
- **The checkpoint is FP8 block-quantized, not bf16.** `config.json` carries
  `quantization_config: {quant_method: fp8, activation_scheme: dynamic,
  weight_block_size: [128, 128]}` and every linear weight has a
  `weight_scale_inv` sibling in the index. The task brief's "78.8 GB bf16"
  is wrong about dtype; 78.8 GB is the FP8 total size. The `mlp.gate`
  routers are in `modules_to_not_convert` (bf16).
- `config.json` also carries `head_dtype: float32` (attention-head compute
  dtype hint) and `dtype: bfloat16`.
- Weight-name inventory (from `model.safetensors.index.json`, experts and
  layer indices elided): `model.embed_tokens.weight`, `lm_head.weight`
  (untied), per layer `input_layernorm`, `post_attn_norm`,
  `post_attention_layernorm`, `post_ffn_norm`,
  `self_attn.{q,k,v,o}_proj(.weight_scale_inv)`,
  `self_attn.{q,k}_norm.weight`, `mlp.gate.weight`,
  `mlp.moe.router.expert_bias`, `mlp.moe.experts.N.{gate,up,down}_proj…`,
  `mlp.moe.shared_experts.{gate,up,down}_proj…`, plus `model.norm.weight`.
  Note the HF layout nests the MoE under `mlp.moe.*` while the router gate
  sits at `mlp.gate` — the loader must reconcile this with the plugin's
  mapper (`kolibri1.py:258-266`).

## Architecture (from the oracle plugin, pinned)

Primary oracle: `Aleph-Alpha/aleph-alpha-inference`, pinned at
`049a6a7bd2405b27d6d280d256bd3d585191c7ae` (release 1.0.0). The clone used
for this study is `/tmp/aai-reference`. The oracle file
`.agents/oracles/aleph-alpha-inference.md` does **not** exist on
`origin/main` yet (the task brief claimed it did); this spec records the pin
and the row that lands it must add the oracle file. Gateability is
`no` until measured on GPU — explicitly not this row's job.

Config (from the checkpoint, cross-checked against the plugin):

| Field | Value |
|---|---|
| hidden_size | 2560 |
| num_hidden_layers | 50 |
| num_attention_heads / num_key_value_heads | 48 / 4 (GQA 12:1) |
| head_dim | 128 (48×128 = 6144 q-size; no partial rotary) |
| layer_types | 4×`sliding_attention` then 1×`full_attention`, repeating |
| sliding_window | 513 (`use_sliding_window: true`) |
| rope_theta | 10000, full rotary (rotary_dim = head_dim = 128) |
| num_experts / num_experts_per_tok | 384 / 6 |
| moe_intermediate / shared_expert_intermediate | 512 / 512 |
| norm_topk_prob | false |
| rms_norm_eps | 1e-06, hidden_act silu |
| vocab_size | 128000 |
| max_position_embeddings | 262144 (1M via override) |
| bos_token_id | null (no BOS); eos 127906; pad 127901 |

### Attention math — `aleph_alpha_inference/kolibri1.py`

- `Kolibri1Attention` (`kolibri1.py:39-123`): GQA, per-head **qk-norm**
  (`RMSNorm(head_dim, eps)` applied per head before RoPE,
  `kolibri1.py:107-108`, `115-118`). `qkv_proj` and `o_proj`, no bias.
- **RNoPE**: sliding-attention layers apply standard RoPE
  (`get_rope(head_dim, max_position, rope_parameters)`, `kolibri1.py:91-95`,
  applied at `119-120`); full-attention layers have `rotary_emb = None`
  (`kolibri1.py:81-83`) — **no positional encoding at all** on the 10
  full-attention layers. This is the inverse of the Mistral-style
  SWA-per-layer convention and must not be folded into a single shared
  rope/attention path.
- Sliding window: 513 for sliding layers (`per_layer_sliding_window`,
  `kolibri1.py:85-106`). The upstream harness passes `sliding_window` to
  vLLM's `Attention` per layer; vLLM window semantics are *last
  `sliding_window` tokens inclusive* (window = 513 tokens, i.e. offset
  `pos - 512`). Local attention backend semantics must match the pinned
  vLLM's sliding-window mask.
- Scale: `head_dim**-0.5` (`kolibri1.py:100`).
- `head_dtype: float32` in config suggests head-space compute in f32; the
  plugin relies on vLLM defaults (`head_dtype` is not read in kolibri1.py).
  Treat it as a model-path annotation, resolve the actual compute dtype with
  the oracle at gate time.

### MoE routing math — `kolibri1.py:127-207`

- Router: `mlp.gate` linear, **fp32 output** (`out_dtype=torch.float32`,
  `kolibri1.py:171-177`), bias-free, plus a separate
  `e_score_correction_bias` parameter loaded from
  `mlp.moe.router.expert_bias` (mapper at `kolibri1.py:258-261`;
  checkpoint stores a per-expert fp32 vector).
- `sigmoid_logit_add_routing` (`kolibri1.py:126-142`):
  1. cast logits to fp32,
  2. `topk_ids = topk(logits + e_score_correction_bias, k=6)`,
  3. `topk_weights = sigmoid(logits[topk_ids])` — **sigmoid of the
     unbiased logits**, not softmax, and selection adds the bias while the
     weight does not (the docstring at `134-136` contrasts this with vLLM's
     built-in sigmoid scoring),
  4. `norm_topk_prob=false` ⇒ `renormalize=False` ⇒ **no weight
     normalization** (the `141` branch is skipped).
- Shared expert: ungated, always added, named `shared_experts`
  (`kolibri1.py:146-188`), SwiGLU at intermediate 512, routed experts at
  intermediate 512, all 50 layers are MoE (no dense layers).

### Layer structure — `kolibri1.py:210-253`

Sandwich (post-norm) layout per layer: `input_layernorm` → attention →
`post_attn_norm` → residual add via `post_attention_layernorm` → MoE →
`post_ffn_norm` → residual. Every layer is MoE.

### Tokenizer

`PreTrainedTokenizerFast` (`tokenizer.json`), `add_bos_token: false`, no
bos token, eos `<|im_end|>` (127906), pad `<|endoftext|>` (127901),
`model_max_length: 262144`. The plugin registers a kolibri1 reasoning
parser and reuses the Hermes tool parser (`tests/test_kolibri1.py:28-40`);
reasoning/tool parsing is serving-layer and **owed**, not part of the CPU
forward gate.

## Design — stage inventory and reuse

| Stage | Reuse from | Delta |
|---|---|---|
| Config parse/validate | `mimo_v2_registry.cpp` / `cohere2_moe_registry.cpp` pattern | new fields: `layer_types`, `use_sliding_window`, `head_dim`, `head_dtype`; RNoPE means rope only on sliding layers |
| Two-group KV cache | `mimo_v2_registry.cpp` (full+SWA groups, `head_size_v` may differ) | direct analog; kolibri has uniform head_dim 128 across both groups; `sliding_window=513` |
| Weight loader | `mimo_v2_weights.cpp` pattern | FP8 block-quant (128×128, `weight_scale_inv`) is the shipped arm; bf16 dequant for router/gate; `mlp.moe.*` vs `mlp.gate` name reconciliation; 32-shard index |
| MoE router | existing router `renormalize` flag plumbing (`cohere2_moe.cpp:151`) | new scoring mode: sigmoid-logit-add (selection on `logits+bias`, weights `sigmoid(logits)`), fp32 logits, no renorm |
| CPU forward | mimo/dots3 hybrid attention path | per-layer rope on/off (RNoPE), qk-norm, sandwich norms |
| Tokenizer | existing tokenizer integration | no-BOS encode path |

## Risks

- **R1 — checkpoint is FP8, not bf16.** The brief's dtype was wrong. The
  CPU row must either (a) dequantize FP8 blocks at load (memory: ~2× at
  dequant time) or (b) implement the FP8 CPU GEMM arm. Decision recorded in
  the implementation commit; (a) is the minimum-complete default, recorded
  as a scratch dequant in the porting inventory.
- **R2 — `rope_parameters`.** The checkpoint config has flat `rope_theta`;
  the plugin reads `config.rope_parameters["rope_theta"]` via the
  transformers `Qwen3MoeConfig` base (`tests/test_kolibri1.py:53`). The C++
  parser must accept the flat field and derive rope_parameters
  equivalently.
- **R3 — window semantics.** 513 must map to the same mask the oracle's
  pinned vLLM `Attention` builds; an off-by-one here passes short-prompt
  gates and fails long ones. Test with a prompt longer than 513 tokens.
- **R4 — no committed reference outputs.** The plugin's tests use a tiny
  synthetic checkpoint (`tests/checkpoints.py`), not golden token strings.
  The token-exact reference for this row is a transformers golden-generation
  run on this host (255 GB RAM, model fits at FP8/bf16) or the plugin's
  synthetic-checkpoint path if transformers cannot run the 78 GB artifact
  in budget.
- **R5 — `head_dtype: float32`** is unresolved against oracle defaults;
  name it beside any f32 buffer and reconcile at gate time.

- **R6 — the spec's weight-name inventory was wrong (2026-10-03, found by
  the W1 manifest capture against the real index).** The real layout is
  `mlp.experts.N.{gate,up,down}_proj...`, `mlp.shared_experts.{gate,up,
  down}_proj...`, `mlp.gate.weight`, and the router bias OUTSIDE mlp at
  `moe.router.expert_bias` — NOT `mlp.moe.*` / `mlp.moe.router.*` as
  recorded above. The `weight_scale_inv` grids are **F32** on disk, not
  BF16; the router bias is **BF16**, not F32 (the loader widens both per
  upstream's converting-copy semantics). The enumeration consumes the real
  names and `tests/vllm/models/kolibri1_manifest.inc` pins them against
  the shard headers.
- **R7 — the tokenizer engine refuses the Kolibri-1 tokenizer.json.** Its
  Qwen3Moe-style pre-tokenizer split regex uses a `(?i:...)` group form the
  engine's recognized set does not cover, so `Tokenizer::FromHfJson`
  throws "unrecognized pre-tokenizer split regex" (asserted by name in the
  W1 gate test). Extending the recognized set is owed before any encode
  path; nothing silently mis-tokenizes in the meantime.

## Tests

- Hermetic: small synthetic safetensors in the exact kolibri1 layout
  (fp8-block + scale tensors, `mlp.moe.*` naming, sandwich norms, q/k norm)
  driving the loader; a manifest-capture test against the real
  `model.safetensors.index.json` without loading tensor data (pattern:
  `tests/vllm/models/*manifest*`).
- Router unit test: sigmoid-logit-add math (selection on logits+bias,
  sigmoid weights, no renorm) against hand-computed values.
- Attention test: hybrid 4:1 pattern, window 513 boundary (>513-token
  prompt), RNoPE on full-attention layers.
- Token-exact gate: golden generation vs the reference (R4), same prompts,
  greedy decode.

## Gates

1. Loader: synthetic-layout checkpoint loads tensor-complete; real-index
   manifest test passes.
2. Router: numeric-exact vs the plugin formula.
3. Forward: token-exact greedy generation vs reference on the real
   checkpoint (CPU, dequantized), prompts crossing the 513 window boundary.
4. Registry: `Kolibri1ForCausalLM` resolves; refused arms (GGUF/CUDA/TT)
   error with a message naming the missing part.

## Evidence plan

Record build recipe, checkpoint sha256 of `config.json` +
`model.safetensors.index.json`, reference-run command and token output,
and the plugin `file:line` anchors above in the row evidence directory.

## Stop conditions

- If implementation reveals architecture elements the config does not
  describe beyond those already recorded here (qk-norm, sandwich norms,
  RNoPE, router expert_bias, fp8 scales — all discovered and recorded),
  STOP and record in Risks before continuing. Mirroring upstream exactly
  matters more than speed.
- The row is DONE when gates 1-4 pass and the records land; performance
  gates are out of scope.

## Now

W2 landed (2026-10-03, branch `row/kolibri-cpu`): the CPU hybrid forward
(`kolibri1_forward.cpp`) — sandwich norms with the residual accumulating the
POST-NORMED attention output (kolibri1.py:248-250), per-head qk-norm before
RoPE (:107-118), RNoPE on the 10 full-attention layers (:81-83), the 513
window on the sliding layers via `AttentionWindow{W-1, 0}`, the
sigmoid-logit-add router (fp32 logits, top-6 on logits + bias, weights =
sigmoid of the UNBIASED logits, no renormalisation, :126-142), the ungated
shared expert (:146-188), and the untied lm_head. The R1 disposition is
recorded in the forward header: DEQUANT-AT-LOAD (each fp8-block weight is
dequanted to bf16 per use against the f32 scale grid); a CPU fp8-block GEMM
arm stays owed. Gate: `tests/vllm/models/test_kolibri1_w2.cpp` — 6 cases /
1757 assertions green: the forward matches an in-test scalar transcription of
kolibri1.py (max abs logits gap 0.014 on the tiny 2-layer model, bf16
rounding only), a single-token prefill, a ragged two-sequence batch vs
per-sequence runs, the logits_indices gather, the full-depth 50-layer real
geometry run (hermetic weights), and the GPU-queue refusal by name. W1 stays
green (27/186). Next: the tokenizer engine gap (R7), then the token-exact
gate against the reference on the real checkpoint (R4).

## Git integration

ONE pull request (developer's standing choice for this row).

## Owed

- `.agents/oracles/aleph-alpha-inference.md`: create the oracle file with
  the pin `049a6a7bd2405b27d6d280d256bd3d585191c7ae`, `gateable = no`, and
  the issue that owns the GPU gateability measurement.
- GPU gateability measurement for the aleph-alpha-inference oracle.
- GGUF arms (k-quants), CUDA arm, Tenstorrent arm — later rows.
- Reasoning parser / tool parser serving integration (plugin registers
  `kolibri1` reasoning + Hermes tool parsers).
- Tokenizer engine: extend the recognized pre-tokenizer split-regex set to
  cover the Kolibri-1 `(?i:...)` form (R7), then gate the no-BOS encode
  contract (`add_bos_token: false`, eos 127906, pad 127901) on a real
  load.
- The CPU forward wave also owns the R1 disposition record: packed
  fp8-block + f32 scale is what the loader stores today; dequant-at-load
  vs a CPU fp8-block GEMM arm is decided when the forward consumes it.

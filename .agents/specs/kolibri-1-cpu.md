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
- **R7 — the tokenizer engine refuses the Kolibri-1 tokenizer.json.**
  `Tokenizer::FromHfJson` throws "unrecognized pre-tokenizer split regex"
  (asserted by name in the W1 gate test). RESOLVED 2026-10-08 — see
  `## R7 resolution` below: the refusal was caused by the number
  alternative's `\p{N}{1}` spelling (the recognized classic Qwen2 constant
  writes `\p{N}`), NOT by the `(?i:...)` group form, which the engine
  already implements. The earlier diagnosis recorded here was wrong and
  is corrected by measurement in that section.

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

W3 landed (2026-10-05, branch `row/kolibri-cpu`): the row's TOKEN GATE ran
against the REAL checkpoint. The bf16 reference (dequant verification) and
the golden generation are recorded in
docs/bench-evidence/kolibri1-goldens-20261004.md; the gate is
tests/vllm/models/test_kolibri1_w3.cpp against
tests/vllm/models/kolibri1_goldens.json (fingerprints generated by
scripts/gen-kolibri1-goldens.py, a pure-torch transcription of the pinned
plugin's math over the dequantized bf16 reference — transformers has no
native Kolibri1ForCausalLM and the plugin file imports vLLM, so the
transcription path is the documented adaptation). Measured result: the
dequant spot-checks match a direct fp8->f32 decode to 1.05e-4 (pure bf16
rounding); the gate replayed the 8 golden prompts greedy and matched
141/145 compared argmax positions, with 4 flips ALL adjudicated near-ties
(gaps 0.0093-0.382, inside the measured bf16-noise envelope of 2.19
absolute on teacher-forced final-step top-8 logits — the envelope the
2.5 band is derived from); 4 of 8 prompts replay token-identically end to
end. The HF token ids are fed directly (the engine tokenizer still refuses
the kolibri1 pre-tokenizer regex, R7 — the encode contract stays owed).
Remaining for the row: R7 tokenizer, the aleph-alpha-inference oracle
gateability measurement (GPU), GGUF/CUDA/Tenstorrent arms (later rows).

SERVING COMPLETION (2026-10-08, branch `row/kolibri-serve`; REVIEW-REPAIRED
2026-10-09, PR #3422): the OpenAI chat path now serves kolibri1 end to end
on CPU. The checkpoint's tokenizer_config.json chat template renders through
the minja adapter byte-identically to the PINNED transformers 5.14.1
renderer on 20 scenarios covering the plugin's thinking switch (tests/
fixtures/kolibri1_chat_template_references.json, regenerated through
`render_jinja_template` by tests/fixtures/
gen-kolibri1-chat-template-references.py). The 2026-10-08 version of this
paragraph recorded the opposite tojson decision — a child-scope sorted-dump
`tojson` overriding minja's insertion order, because the references had
been captured with plain jinja2 — and the PR #3422 review falsified it:
the pinned renderer installs its OWN tojson over Jinja's builtin with
sort_keys=False / ensure_ascii=False / no HTML escaping (transformers 5.14.1
utils/chat_template_utils.py:481), so plain Jinja's default is not the
serving behavior and the sorted override rendered prompt bytes no serving
reference produces. The repair replaced the override with a port of the
pinned filter's full signature (ensure_ascii/indent/separators/sort_keys,
CPython json.dumps semantics) in src/vllm/entrypoints/chat_template.cpp,
made `FunctionDefinition::parameters` order-preserving (ordered_json, with
RestoreToolSchemaOrder re-reading `tools` from an order-preserving body
parse at every chat entry point — api_server, the C ABI, run_batch), and
made BuildTools mirror pinned vLLM's measured `model_dump` tool shape
(description/parameters present as null when absent). The template input is
committed at tests/fixtures/kolibri1-chat-template-tokenizer_config.json so
the gate runs on a clean checkout (no /mnt path). The kolibri1 reasoning
parser ports the plugin's reasoning.py @ 049a6a7bd240:
the Qwen3 engine grammar with the starting state derived the way the
template switches thinking (reasoning_effort wins and only "none"
disables; else a literal enable_thinking false does), threaded from the
request's chat_template_kwargs per call. The kolibri1 tool parser is the
plugin's registration mirrored exactly: an alias to the Hermes
`<tool_call>` class (__init__.py:50-54). Both detection tables resolve
"kolibri1" off the template's no-reasoning sentence, ahead of the
generic `<think>` and hermes rows. Gates: test_reasoning_kolibri1,
test_tool_parser_kolibri1, test_kolibri1_chat_template green (red-first:
detection resolved think_auto/hermes and the registry names did not
exist before the change); the full host battery and the row's kolibri
gates stay green; W3 rerun in a verified quiet window. Evidence:
docs/bench-evidence/kolibri1-serve-20261008.md. REVIEW-REPAIR GATES
(2026-10-09): test_kolibri1_chat_template 61, test_chat_template 204
(196 pre-existing + 8 non-kolibri tojson guard assertions),
test_reasoning_parser_detect 75, test_tool_parser_detect 361,
test_reasoning_qwen3 164, test_openai_tool_parsers 64,
test_kolibri1 27/234, test_kolibri1_decode_bench anchor 109726 — all
green; serving/protocol suites and the parameters-reading tool-parser
suites green; W3 not rerun (no forward change). Remaining for the row:
the aleph-alpha-inference oracle gateability measurement (GPU),
GGUF/CUDA/Tenstorrent arms (later rows).

RE-REVIEW REPAIR (2026-10-09, PR #3422, branch row/kolibri-serve): the
operator's clean-head re-verification at d93d9d492 found the review repair
itself RED — test_openai_run_batch failed 3 of 7 cases, all throwing
json.exception.type_error.302, because DispatchChat called
RestoreToolSchemaOrder(body, request) with the parsed json OBJECT where the
seam takes the body TEXT (the repair's recorded 'run_batch 16/16' gate counted
assertions and missed the throwing cases — a stale binary).
ISSUE-LOCAL-01M4FR20MES4HQVBRWJBJ2AVCN. Fixed by threading the original
request text through: RunLine re-serializes the chat body from an
order-preserving ordered_json parse of the original line and DispatchChat
calls RestoreToolSchemaOrder(body_json, request) — a first body.dump()
attempt was reverted because the sorted dump no-ops the order restoration.
The scoped re-review's two findings also landed
(ISSUE-LOCAL-01M4FR1D7RH7CN5X2R2CAR6N61): F1, the ordered-parameters seam
is now detected at two entry points (a new test_run_batch case drives
RunLine with a raw non-alphabetical tools line and asserts the 200 row and
the document-ordered schema at the prompt seam; a new test_api_server case
posts an unsorted-schema request through the production dispatch — both
mutation-proven against a no-op seam and against the throwing call); F2,
the 'byte-stable for already-ordered schemas' guard's AlphabeticalTool
fixture is now genuinely alphabetical at every level and renders the schema
alone, so its before/after byte-identity claim is real (mutation-proven:
stays green under a sorted tojson while the insertion-order case goes red).
RE-REVIEW GATES (2026-10-09, fixed head): test_openai_run_batch 8/8 cases /
89 assertions (7 pre-existing + the new F1 case), test_openai_api_server
1521, test_openai_serving 1365, test_chat_template 204,
test_kolibri1_chat_template 61, test_reasoning_kolibri1 43,
test_tool_parser_kolibri1 19, test_reasoning_parser_detect 75,
test_tool_parser_detect 361, test_kolibri1 27/234,
test_kolibri1_decode_bench anchor 109726 — all green; W3 not rerun (no
forward change). Evidence: docs/bench-evidence/kolibri1-serve-20261008.md
"Re-review repair".

NEXT-LEVER UNIT (2026-10-08, branch `row/kolibri-perf-next`; review repair
2026-10-09): the TESTED candidates in the CPU decode lever space after the
NEON GEMM tier and the dequant cache are measured and closed — see
docs/bench-evidence/kolibri1-perf-next-lever-20261008.md (issue
ISSUE-LOCAL-01M4EEX40G7NH571CR5GQY4CA1). Production reproduces at 3.17-3.21
decode tok/s (8 threads, 16 GiB). Attribution is split by phase (2026-10-09
repair): the WHOLE-RUN profile (60 forwards, prefill included) is
linear_gemm ~50%, cold-miss whole-matrix decodes ~26% (prefill/warmup
concentrated — 95.8% of misses occur in the first 10 forwards), attention
~17%, lm_head 4%; the MEASURED DECODE phase (~302-315 ms/step) is
linear_gemm ~59%, attention ~30%, lm_head ~5%, steady-state cold misses ~3%.
The recorded "moe_glue residual" reading is corrected — the moe_glue scope
nests the routed experts' GEMM/dequant, so its separable residual is small.
Falsified by measurement (tested candidates only): fp8-direct GEMV (one
implementation, two shapes, single thread; bit-exact but 8× slower — the
order-preserving kernel is ALU-bound, not bandwidth-bound) and bit-exact
changes to the landed GEMM inner loop (the contract forbids fmla/dot). NOT
closed, and left open per the no-ceiling rule: attention (~30% of the
measured decode step, never adjudicated), GEMM scheduling/layout/threading
variants, decoder improvements, and prefill-path levers. The next DECISION
is the dequant-cache production budget — a measured policy knob, not the only
remaining lever; the unit delivers the budget-sensitivity table and the
8/16/32 GiB sensitivity numbers (knee still open at 32 GiB, +5-6% over 16
GiB) and lands no product code.

NEON paged-attention lane (2026-10-08, branch `row/kolibri-neon-attn`, issue
ISSUE-LOCAL-01M4EQ7TPR5Q7JQGNM8628X9HW): the shared CPU seam
`vt::PagedAttention` gains an aarch64 lane for the K dot-product and V
accumulation behind `VT_CPU_PAGED_ATTN_NEON` (default OFF; scalar stays the
reference and the fallback). attn_core halves (4.9->2.3 s at t32, 5.3->2.5 s
at t8 over the 64-step decode bench); decode tok/s +6% (t8) under load, idle
re-measure owed. Gates with the lane ON: W3 900/900, ARGMAX 141/145 with 4
near-tie flips and 0 hard flips (worst topk diff 2.057 vs the committed scalar
fingerprint 2.186); kolibri battery green; the unit sweep runs the scalar
kernel as oracle over the variant matrix (f32/f16/bf16/fp8 KV, GQA, window,
softcap, varlen) plus a replay of the real prefill operands (1 bf16 ulp).
OPEN: the decode-bench strict token anchor flips on a 0.197-nat near-tie with
the lane ON — the default flip is blocked on (1) the idle-host A/B and (2) a
near-tie adjudication of that anchor (gate-semantics change, own spec).
Evidence: docs/bench-evidence/kolibri1-neon-paged-attn-20261008.md.

N-GRAM SPEC DECODE LANDED ON THE ENGINE PATH (2026-10-12, branch
`row/kolibri-spec-engine`, issue ISSUE-LOCAL-01M4P22PX6BPRQAY1CK20FZQF6): the
spec-probe's one seam is fixed — `LoadedEngine::MakeKVCacheMaybeSpec`
(src/vllm/entrypoints/model_loader.cpp) builds the model's own KV spec and
takes the widened qwen3.5 branch only when `spec->use_eagle()` or the model's
own groups carry a recurrent/mamba group. A draft-free n-gram config on
kolibri keeps kolibri's own two groups (full + sliding attention): the drafts
are ordinary scheduled tokens, so the target's groups are the whole cache need
and no draft-proposal group is added (the `fa_draft` group serves only the MTP
draft head's paged KV, a forward n-gram does not have). The default (no spec
config) path is byte-identical. No new engine wiring was needed:
`EngineParams::speculative_config` (the C ABI v6 field / the server's
--speculative-config) already fed the generic scheduler slots, engine draft
pull, runner verify and n-gram proposer. `test_kolibri1_ngram_spec` is flipped
from the pinned blocker into a checkpoint-free seam unit (kolibri keeps its
own groups under an ngram spec; qwen3.5 still widens) plus the engine e2e
gate: spec-ON == spec-OFF BYTE-IDENTICAL on 9/9 prompts (bench prompt +
8 W3 goldens), acceptance exactly 53/53 on the bench prompt and 146/270 on
the goldens (the probe harness's numbers), and engine-level decode tok/s
4.53 -> 13.78 on the bench prompt (3.04x, CLEARS the 12 tok/s target;
harness reference 4.92 -> 14.51) and 6.27 -> 8.13 on the goldens (1.30x),
measured with the synchronous scheduler on BOTH arms (ngram forces sync;
VT_ASYNC_SCHED=0, t32 + 32 GiB cache, under the lock). Decode tok/s uses the
CORRECTED denominator (2026-10-12 review fix): the tokens generated inside
the decode interval — the full count less the tokens already present when
the first output arrived — over that interval; wall tok/s keeps the full
count (bench wall 3.04 -> 5.89; goldens wall 5.46 -> 6.91). The first report
divided the full count by the decode interval, overcounting decode by the
first step's tokens (the operator's overcounted verification: 14.06 bench /
8.60 goldens decode); the correction is reporting-only — acceptance
(53/53, 146/270) and byte-identity (9/9) are unchanged. W3 unchanged: 900/900
with spec off; with spec ON the chains are byte-identical to spec-OFF, so the
torch-golden relationship (4/8 prompts end-to-end identical, 4 adjudicated
near-tie flips) is unchanged. Owed: the engine's reject-heavy path costs more
than the harness's on the lowest-acceptance golden (verify T=1+kr for ~0.5
emitted tokens/pass; aggregate still wins); batch>1 acceptance; real-traffic
acceptance; serving-shaped k/window tuning (config, not code).

DEQUANT CACHE DEFAULT FLIPPED ON (2026-10-20, branch
`row/KOLIBRI1-DEQUANT-CACHE-DEFAULT`, issue
ISSUE-LOCAL-01M5DMH6KQJSP9F8PFCQAAS4WH): the CPU fp8 decode path keeps its
dequantised weights resident by default, because the shipped configuration was
the slow one. The cache was OFF by default (budget 0), and the configuration
is worth MORE THAN 2x on CPU decode: 4.02 tok/s with the cache off against
9.07 tok/s with a 16 GiB budget (2026-10-20, node 3, aarch64, 3.4 GHz, 32
threads, empty-prompt 32-token decode; prefill p512 43.33 tok/s with the
cache off). The DEFAULT budget is DERIVED from the host's own memory report,
read through the house helper `vllm::v1::host_available_memory_bytes()` (the
same probe the DeepSeek-V4 EXL3 host-residency budget uses) — superseded by
the correction below, which is where that policy landed.
`VT_KOLIBRI1_DEQUANT_CACHE_MB=0` still disables the cache (the original
threaded pool dequant, byte for byte) and an explicit value still sets the
budget exactly (16 GiB is the shape the measurement above was taken in); an
unparseable value falls back to the default policy rather than switching a
default-ON lever off. The budget is a BOUND, not a reservation, and an
allocation that cannot be satisfied degrades the process to the uncached path
instead of failing the forward: the cache drops its entries, reports itself
disabled, and counts the degradation. NO NUMERICS CHANGE: the cache stores
the DECODED weights and serves them unchanged, and the gate pins the cached
bytes against the PRODUCTION uncached dequant (`DequantFp8BlockUncached`) byte
for byte, cold and hot. The consequence to record: the routed-expert GEMV's
warm-bf16 residency precondition is now met by default, so the bf16-dot tier
(already the default on a FEAT_BF16 host since 2026-10-19) also covers the
routed experts without an environment variable.

DEQUANT CACHE DEFAULT BUDGET, CORRECTED (2026-10-21, branch
`row/KOLIBRI1-DEQUANT-CACHE-DEFAULT`, issue
ISSUE-LOCAL-01M5DVY9ZWF1XXAM2HSHH5BDRK): the first form of the default
budget, `clamp(MemAvailable / 8, 1 GiB, 32 GiB)` read at the plan time, was
wrong in TWO ways with ONE root cause, and the row's own review measured it.
The plan is taken on the FIRST USE of the process-wide cache, and the first
use happens DURING the 74 GB checkpoint load: the page cache is full by then,
so MemAvailable reads 66 GiB where the host's idle reading is ~212 GiB. The
divisor then yields 8.25 GiB, and a budget that low buys nothing: on node 3
(aarch64, 3.4 GHz, 32 threads, empty-prompt 32-token decode, 2026-10-21) the
default leg measured `budget_mib=8454 budget_source_mib=67638 hits=9097
misses=18349 evictions=16068` and 3.84 tok/s, against 6.38 tok/s for an
explicit 16384 MiB budget (`hits=21553 misses=18393 evictions=12940`) and
3.62 tok/s with the cache off. The hit rate at 8.25 GiB was 33%. So the
policy could land BELOW the knee, and the sampling moment is part of the
policy, not an implementation detail of it.

THE KNEE is 16 GiB: the budget above which the cache stops thrashing, from
the 2026-07-20 budget-sensitivity table (9.07 tok/s at 16 GiB) and the
2026-10-21 measurement above (6.38 tok/s at an explicit 16 GiB against
3.84 tok/s at 8.25 GiB and 3.62 tok/s with the cache off). A budget below
the knee is worth little for the memory it takes, and the default must not
serve one to a host that can afford the knee.

THE CORRECTED POLICY:

    honest_bytes = max(start_bytes, plan_bytes)
    honest_bytes == 0              -> budget = 16 GiB   (UNKNOWN)
    honest_bytes < 16 GiB + 8 GiB  -> budget = 0        (DISABLED)
    otherwise                      -> budget = clamp(honest_bytes / 8,
                                                        16 GiB, 32 GiB)

  * `start_bytes` is MemAvailable read by a NAMESPACE-SCOPE initialiser
    (a function-local static behind it), which runs before main and before
    any model load: the host's own availability, not what our own load left
    behind. `plan_bytes` is the reading at plan time. The MAXIMUM is the
    honest one, and the process-start reading is what the checkpoint load
    would otherwise have hidden.
  * The KNEE is the FLOOR for every host that can afford it: a budget under
    the knee is worth little for the memory it takes, so the policy never
    serves one to a host that can afford the knee.
  * The RESERVE is 8 GiB: the room the host must keep free BESIDE the knee
    (the OS's working set and the decode's own transient allocations). A
    host that reports less than the knee plus the reserve cannot afford the
    knee at all, so the policy DISABLES the cache (budget 0) and the forward
    runs the UNCACHED path, byte for byte. This is the explicit rule: a
    sub-knee cache is WORSE than useless — it costs the memory and buys
    nothing — so the cache either runs at or above the knee or it does not
    run. The reserve is a stated policy number, not a measured constant,
    and the gate pins it.
  * The 32 GiB CAP bounds the footprint (the measured budget-sensitivity
    table had its knee still open at 32 GiB, +5-6% over 16 GiB). An UNKNOWN
    report (MemAvailable unreadable) uses the KNEE: an unknown budget is
    not a reason to refuse (the house rule) and not a reason to hand out a
    sub-knee budget; a host that cannot hold the knee degrades to the
    uncached path through the allocation fallback.
  * The env contract is unchanged: 0 disables, N sets exactly N MiB, and an
    unparseable value falls back to this default policy, never to disabled.

The pure policy stays a function of an input reading
(`DefaultBudgetBytesFor(available_bytes)` for the honest reading,
`HonestAvailableBytes(start_bytes, plan_bytes)` for the maximum of the two
moments), so the gate pins it without reading the host and a policy change is
a deterministic red. The profile report carries all three numbers
(`budget_mib`, `budget_source_mib`, `budget_start_mib`, `budget_plan_mib`) so
a reader can check the derivation after the fact.

MEASURED AFTER THE REPAIR (2026-10-21, node 3, aarch64, 3.4 GHz, 32
threads, empty-prompt 32-token decode, `VT_KOLIBRI1_PROFILE=1`, the file
mutex held, the box quiet): the DEFAULT leg planned
`budget_mib=27157 budget_source_mib=217262 budget_start_mib=217262
budget_plan_mib=143920 enabled=1` and finished at hits=72190, misses=17756,
evictions=7993, an 80% hit rate, with 7.44833 tok/s; the second and third
runs planned 27096 MiB and 27199 MiB and delivered 7.39141 and 7.46827
tok/s. The explicit 16384 MiB leg, in the same window and the same binary,
delivered 7.05777, 7.05438 and 6.98659 tok/s, and the cache-off leg
delivered 3.62428, 3.78799 and 3.54865 tok/s. The plan-time reading
(143920 MiB) sat 72 GiB under the process-start reading (217262 MiB): that
gap IS the checkpoint load, and the honest reading closes it.

## R7 resolution — the tokenizer engine accepts the Kolibri-1 split regex

### Scope — what is actually in the file

Read of `/mnt/models/Aleph-Alpha/Kolibri-1/tokenizer.json` (2026-10-08),
every element the loader walks:

- `pre_tokenizer`: a `Sequence` of
  `Split(pattern={Regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"}, behavior="Isolated", invert=false)`
  followed by `ByteLevel(add_prefix_space=false, trim_offsets=true,
  use_regex=false)` — the same pipeline shape every recognized Qwen-family
  checkpoint ships (the walk at tokenizer.cpp:469-480 accepts it).
- `normalizer`: null. `post_processor` and `decoder`: `ByteLevel`
  (add_prefix_space=true, use_regex=true) — the standard Qwen shape, already
  accepted.
- `model`: BPE, 127900 vocab entries, no `continuing_subword_prefix`.
- `added_tokens`: 20 specials, incl. `<|endoftext|>` 127901 and
  `<|im_end|>` 127906.
- `tokenizer_config.json`: `add_bos_token: false`, no bos token, eos
  `<|im_end|>`, pad `<|endoftext|>`.
- The ONLY element the engine refuses is the Split regex STRING.

### The refusal site and the corrected diagnosis

- Refusal site: `DetectPattern` (src/vllm/tokenizer/tokenizer.cpp:457-527)
  recognizes a Split regex by WHOLE-PATTERN byte equality against verbatim
  constants (:504-513) plus one `\p{N}{1,3}` substring heuristic (:524);
  anything else hits
  `Fail("unrecognized pre-tokenizer split regex: " + re)` at :527.
- Measured diff (2026-10-08, character-level difflib over the decoded
  pattern): the Kolibri-1 regex is byte-identical to `kClassicQwen2Regex`
  (tokenizer.cpp:27-28) except ONE 3-byte insert — the number alternative
  is `\p{N}{1}` where the constant writes `\p{N}`. Nothing else differs:
  same `(?i:'s|'t|'re|'ve|'m|'ll|'d)` contraction group, same letter run,
  same punct run, same three whitespace rules.
- CORRECTION: the R7 text above (and the W1 test comment it quotes) blame
  the `(?i:...)` group form. That diagnosis was wrong. The engine already
  implements `(?i:...)`: `MatchContraction`
  (src/vllm/tokenizer/pretokenizer.cpp:81-98) matches the contraction
  case-insensitively with Unicode simple case folding (U+017F folds to
  `s`), pinned against the HF onig engine by
  tests/vllm/test_pretokenizer.cpp ("contractions are case-insensitive and
  unconditional") and tools/gen_pretok_goldens.py. The refusal is caused
  by the `\p{N}{1}` spelling alone.

### Design

- Add one verbatim constant `kClassicQwen2N1Regex` beside the existing
  four (tokenizer.cpp:25-51), transcribed from the checkpoint, and return
  `SplitPattern::kQwen2Classic` for it in `DetectPattern`, placed with the
  other exact matches BEFORE the `\p{N}{1,3}` heuristic.
- Why mapping onto `kQwen2Classic` is exact, not approximate: `{1}` is the
  identity quantifier — `\p{N}{1}` matches exactly what `\p{N}` matches.
  The kQwen2Classic scanner (pretokenizer.cpp:1044-1047: `max_digits=1`,
  `marks_in_run=false`, `marks_excluded=false`, contraction group on)
  implements that regex alternative-for-alternative: the case-insensitive
  contraction group, `[^\r\n\p{L}\p{N}]?\p{L}+`, single-codepoint
  `\p{N}`, ` ?[^\s\p{L}\p{N}]+[\r\n]*`, and the three whitespace rules.
  The extension is RECOGNITION-ONLY; no scanner code changes.
- Rejected: textual canonicalization of `\p{N}{1}` to `\p{N}` before the
  equality checks (the CR/LF canonicalization at tokenizer.cpp:481-503 is
  the precedent shape). A textual rewrite can corrupt a character class
  that contains a literal `{1}` (e.g. `[\p{N}{1}]`), and this file's own
  rule is "recognition is by whole pattern, never by a substring that a
  sibling shares" (tokenizer.cpp:514-522). An exact-match constant cannot
  change any other model's recognition.
- Rejected: a general regex compiler for the Split pattern. Out of scope
  for this unit; the engine deliberately hand-implements each recognized
  pattern so an unsupported feature fails loudly at load instead of
  mis-tokenizing.

### Risks

- Other models' tokenizers must not change behavior. The change is one
  additive exact-match arm placed after the existing exact matches; every
  previously recognized pattern still matches its own constant first. The
  `\p{N}{1,3}` heuristic still fires only for patterns containing that
  literal substring — the Kolibri-1 regex contains `\p{N}{1}|`, not
  `\p{N}{1,3}`. Non-regression gate: the full tokenizer test surface
  (below) stays green.
- The case-insensitive group must match case-insensitively. The Kolibri-1
  pattern maps onto the scanner whose contraction behavior is already
  pinned against the HF oracle (test_pretokenizer.cpp); the new test adds
  an equivalence probe over mixed-case contractions (below).

### Tests (red-first)

Rework the W1 refusal test case (tests/vllm/models/test_kolibri1.cpp:754-849,
live-gated on the real model dir) into the R7 encode test:

1. RED (before the fix): loading the real tokenizer.json throws
   `unrecognized pre-tokenizer split regex` — the named refusal, captured
   in the gate log.
2. GREEN: the load succeeds; `Encode(prompt)` equals the HF reference
   `input_ids` from tests/vllm/models/kolibri1_goldens.json for all 8
   golden prompts. Those ids were produced by the REAL HF `tokenizers`
   library (`scripts/gen-kolibri1-goldens.py:268-273`,
   `encode(prompt, add_special_tokens=False)`) — so this assertion is the
   HF-id-match evidence: the ids the W3 gate has been feeding directly are
   now reproduced by our engine, and the no-BOS encode contract
   (`add_bos_token: false`, Encode adds no BOS) is gated on a real load.
3. Equivalence probe (nothing else changes): load a rewritten copy of the
  same file whose regex is `\p{N}` where the checkpoint writes `\p{N}{1}`
  (parsed JSON, one string edited, re-serialized — recognized TODAY as
  kQwen2Classic) and assert byte-identical ids over a probe corpus: the 8
  golden prompts plus mixed-case contractions (`I'M I'll DON'T can'tt`),
  digit runs (`x123 1234567`), whitespace/newline mixes, German umlauts and
  CJK. Both sides run our engine; the classic spelling is the one already
  accepted, so identity proves the new arm adds recognition only.
4. The existing tokenizer_config.json special-token assertions (eos
  `<|im_end|>`, pad `<|endoftext|>`, no BOS) stay, plus a Decode round-trip
  of the 8 prompts.

### Gates

- The new R7 test green (red capture first, above).
- Full tokenizer test surface green: test_pretokenizer, test_bpe,
  test_bpe_equivalence, test_detokenizer, test_unicode_data,
  test_tokenizer_metaspace_split, test_tokenizer_parity,
  test_tokenizer_parity_deepseek, test_tokenizer_parity_deepseek_v3,
  test_tokenizer_parity_gpt4o, test_tokenizer_parity_mistral.
- test_kolibri1 green (W1 gate incl. the reworked tokenizer case).
- W3 gate unchanged and green: test_kolibri1_w3, 900/900 assertions
  (8 threads, quiet window).
- decode_bench gate unchanged and green: anchor last token 109726.
- scripts/check-agent-record.py rc=0; scripts/agent-preflight.sh rc=0.

### Stop conditions

- If the tokenizer.json needs more than whole-pattern recognition of the
  `\p{N}{1}` variant (a real regex feature the scanner cannot express),
  STOP and report what is missing; do not rewrite the engine in this unit.
- If our Encode disagrees with the HF reference ids on any golden prompt,
  STOP: that would mean the scanner semantics differ, not just the
  recognition — report as NEEDS_DECISION rather than widening anything.

## Outcome

MEASURED (2026-10-20/21; node 3, aarch64, 128 cores, 255 GB RAM, governor
`performance` at 3.4 GHz, 32 threads, one NUMA node, the file mutex held,
the box quiet; the bench and the spec probe rebuilt from the committed
head). The noise floor came from the repeated legs: two runs of the same
cell (same binary, weights, configuration) differed by 3.5% in the t32
cache-off leg (4.02 against 4.16) and by 4.7% in the decode table's
cache-off leg (3.62 against 3.79), so no cell below is quoted more finely
than that, and two legs that differ by less than 4% are a tie.

| leg | decode tok/s (the in-repo bench: 128 tokens in, 64 greedy out) |
|---|---|
| cache OFF — the configuration `main` ships today | 3.55 / 3.62 / 3.79 (median **3.62**) |
| default, first implementation (8.25 GiB budget, 33% hit rate, 16068 evictions) | 3.84 (one run) |
| **default, repaired** (26.5 GiB budget from a 212 GiB process-start reading; hits 72200, misses 17746, evictions 7967, 80% hit rate) | **7.39 / 7.45 / 7.47** |
| explicit `VT_KOLIBRI1_DEQUANT_CACHE_MB=16384` | 6.99 / 7.05 / 7.06 |

Prefill p512 with the cache off: t8 12.27-12.30, t16 23.78-23.89, t32
43.02-43.33 tok/s. Decode thread scaling with the cache off, empty prompt:
t8 1.38, t16 2.59, t32 4.02 tok/s, and the ratios that row derives are
1.88x, then 1.55x, and 2.91x over the whole range. The n-gram spec-decode
probe — the campaign's serving number — reads, cache OFF, plain 3.90-3.92
and spec 6.27-6.30; with the **default** (repaired), plain 8.63-9.36 and
spec **12.74-13.14**; with an explicit 16 GiB budget, plain 8.33-8.73 and
spec 10.42-10.55.

Every leg above is ONE measurement set. The evidence report names the file
and the date of each one, and it lists the other sets the session took for
the same cells instead of mixing them into the tables.

WHAT THE DEFAULT NOW DELIVERS. The configuration that ships with the FP8
dequant cache ON decodes at **7.45 tok/s** on the in-repo production shape
(128 tokens in, 64 greedy out), **2.06x** the cache-off 3.62 and
1.97x-2.10x the cache-off leg's own band (7.45 against 3.55-3.79), and its
n-gram spec-decode leg reads **12.74-13.14 tok/s**, so the campaign's
**12 tok/s target is MET**. The target is not met with the cache off
(6.27-6.30), and the default's spec leg is **2.03x** the cache-off leg's
(12.74 against 6.27): the number belongs to the default and to the default's
host, not to the model in the abstract. The engine-level reading clears it
as well: 4.53 -> 13.78 tok/s (3.04x, 2026-10-12, t32 with a 32 GiB cache,
in `## Now` above). The default was still **OPEN** (PR #3488) when this was
written, so the row's numbers describe the configuration `main` ships when
it lands; today `main` ships the cache-off leg.

REJECTED, each with the measurement that rejected it. The **sub-knee
budget** (the default's first implementation: 8.25 GiB, 33% hit rate, 16068
evictions) read 3.84 tok/s, which is +6.1% over the cache-off median
(3.84/3.62) and +1.3% over the cache-off leg's best run (3.84/3.79), so
under the 4% rule it is **not a tie** to the cache-off median. It is not a
win either: the leg pays 8.25 GiB of memory for a 33% hit rate and leaves
the 2x of the repaired default on the table, which is why the policy
refuses a sub-knee budget outright and why the sampling moment is part of
the policy rather than an implementation detail of it. The **explicit 16
GiB budget** as the recommended setting read 6.99-7.06 tok/s in the
published set against 7.39-7.47 for the derived default, a margin of 5.7%
(7.45/7.05) that is above the noise floor and small for 10.5 GiB more cache,
and 1.95x the cache-off median (7.05 against 3.62), so it is **not** claimed
as a reason to prefer one over the other, and pinning 16 GiB by hand takes
the host's own memory report out of a decision the report makes better.
**Spin tuning** (`VT_CPU_SPIN_ROUNDS`) moved the same cell from 4.019 to
4.193 tok/s at 32 threads (+4.3%) and from 2.587 to 2.581 at 16 (-0.2%), so
the two paired readings do not agree in sign, the pair does not separate the
arms, and no spin setting is recommended. One paired reading on one host is
not a settled lever question.

THE CAVEATS TRAVEL WITH EVERY NUMBER ABOVE. They are **host-specific**: 32
threads on one NUMA node at 3.4 GHz, governor `performance`, aarch64, node 3
of a shared box. They are **not** a vLLM comparison and not a parity claim:
no vLLM or plugin denominator exists for this model, and the author's
plugin is still unverified as a runnable oracle. They sit on a **~4% noise
floor**, so the 12.74 against a target of 12 is a **+6.2% margin, about 1.5
noise floors**: it clears the target as measured, and the margin is not
wide. And the **16 GiB leg carries its own trade-off**: it is the shape the
PR #3488 body quotes, and a reader who pins 16 GiB explicitly gets a number
5.7% below the derived default for 10.5 GiB less cache. The derived default
(26.5 GiB from a 212 GiB process-start reading) is the configuration this
row's numbers were taken at, and the policy refuses a host that cannot
afford the knee.

THE RECORDS. The public page is
`docs/benchmarks/kolibri1-cpu-dequant-cache-default.md`, with the index row
in `docs/BENCHMARKS.md` and the dated evidence in
`docs/bench-evidence/kolibri1-cpu-dequant-cache-default-20261021/report.md`.
The public recipe still pins the short revision `e52eb462` for the
checkpoint; the full revision of the artifact these numbers were measured
on is recorded on the public page, and the bf16 sibling artifact is named
there as **refused**, not shipped.

## Git integration

ONE pull request (developer's standing choice for this row).

## Owed

- $ID: the gate-semantics decision for the CPU paged-attention
  NEON lane — ratify a near-tie instrument for the decode-bench
  anchor (owner: Ettore), then re-decide the default under it.
- `.agents/oracles/aleph-alpha-inference.md`: create the oracle file with
  the pin `049a6a7bd2405b27d6d280d256bd3d585191c7ae`, `gateable = no`, and
  the issue that owns the GPU gateability measurement.
- GPU gateability measurement for the aleph-alpha-inference oracle.
- The decode and prefill benchmark rows for the DEFAULT dequant-cache
  configuration (the derived budget), pending the node-3 window; the exact
  commands are in the pull request body. The harness rows that mean
  "cache off" must now pass `VT_KOLIBRI1_DEQUANT_CACHE_MB=0` explicitly:
  an UNSET variable is the default-ON configuration, not the off switch.
- GGUF arms (k-quants), CUDA arm, Tenstorrent arm — later rows.
- Reasoning parser / tool parser serving integration (plugin registers
  `kolibri1` reasoning + Hermes tool parsers).
- The CPU forward wave also owns the R1 disposition record: packed
  fp8-block + f32 scale is what the loader stores today; dequant-at-load
  vs a CPU fp8-block GEMM arm is decided when the forward consumes it.
- NUMA interleave row (this spec, the in-process PolicyGuard change): the
  loader-level focused test now pins the `kolibri1_weights.cpp` guard site
  (deleting it turns the loader case of test_kolibri1_numa red); the
  dequant-cache guard site is pinned by the same file's cache case. Residue:
  the loader case is model-gated and single-threaded-load only — a loader
  that materializes on worker threads would need per-thread policy scope,
  which nothing today exercises.

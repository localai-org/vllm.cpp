# SPEC — `MODEL-CLM`: CLM System 1 decision model (Qwen3-8B + dual projection heads)

Port `Contrastive-LM/CLM-v0.1-8B` into vllm.cpp as the fifth SystemOne-class
model, reusing the `/v1/systemone` API (choice/score/noul question types)
already on `main` from the GLiNER2.5 / Laya / kev / cua-s1-forms work. CLM
is a frozen Qwen3-8B encoder (last-token pooling) with two small MLP
projection heads — a state head and an action head — trained with a
bidirectional InfoNCE loss. At inference, the state head projects the
(state + question) embedding and the action head projects each candidate
embedding into a 512-d space; the score is `exp(logit_scale) *
cos(state_proj, action_proj)`, softmaxed over a question's candidates. CPU +
GPU (CUDA), OpenAI-compatible serving.

## Now

`ACTIVE`. The fidelity repair (ISSUE-LOCAL-01M3T6ZZ2WBM4FSMTQZAWM8974) is
implemented and CPU-verified; see `## Fidelity repair`. On 2026-09-30 the
published checkpoint, converted by `scripts/convert-clm.py` and served by
`vllm-server` on CPU, matched the reference `Engine.answer` on 5 requests and
8 questions: argmax 8/8, max probability difference 0.029 against the
reference with a bf16 encoder (0.077 against fp32; the reference's own bf16
versus fp32 difference is 0.049). The first port scored 5/8 and 0.98 on the
same requests. The independent review of the repair, CUDA and GGUF are
`PENDING`.

## Scope

- **Row.** `MODEL-CLM` (this spec). New model-matrix row under
  `MODEL-TOKCLS` (CLM answers structured questions, same SystemOne class as
  kev, Laya, cua-s1-forms, and GLiNER2.5).
- **In.** Frozen Qwen3-8B encoder backbone (36 layers, hidden=4096,
  32 attn heads, 8 KV heads, head_dim=128, intermediate=12288, RoPE theta
  1e6, rms_norm_eps=1e-6, attention_bias=false, tie_word_embeddings=false)
  in feature-extraction mode (ForwardHidden, no LM head, last-token
  pooling); two MLP projection heads (state head + action head), each
  `Linear(4096, 1536) → GELU → Linear(1536, 1536) → LayerNorm(1536) → GELU
  → Linear(1536, 512)` (width=1536, depth=3, activation=gelu,
  layernorm=true, residual=false), L2-normalized after projection;
  logit_scale (scalar, `exp(logit_scale)` clamped to 100.0) learned
  InfoNCE temperature inverse; scaled-cosine scoring +
  temperature-scaled softmax; candidate text construction from choice/
  score/noul question schema (`build_pairs`); model registration via
  `REGISTER_VLLM_MODEL`; `/v1/systemone` dispatch for CLM; config loading
  from the HF `config.json` (non-standard: `model_type: "clm"`) +
  `CLM_v0.1-8B.pt` checkpoint (torch pickle); CPU build + tests; GPU
  forward (CUDA).
- **Out (owned by other rows).** ROCm kernel tuning (routes through
  existing `vt::` ops). GGUF k-quants (owed, named below). Action embedding
  caching / reuse optimization (CLM's reference caches action embeddings so
  repeated candidates across questions skip re-encoding; deferred — CLM
  re-encodes per question in the simplest correct approach). Runtime
  fine-tuning of projection heads (CLM ships a frozen head checkpoint;
  fine-tuning is a separate training step, not an inference concern).
  LocalAI backend is a separate PR in a separate repo.
- **Reuse.** The Qwen3 dense backbone forward path
  (`Qwen3DenseModel::ForwardHidden` in `qwen3.cpp:570-592`, the existing
  pooling forward that stops after final RMSNorm with no lm_head). The
  `/v1/systemone` API endpoints, request/response structs, and server
  dispatch from GLiNER2.5 / Laya. The `DecisionFn` callback mechanism from
  Laya (PR #3263). The `DecisionResult` struct and
  `BuildSystemOneAnswerDecision` from the shared `systemone.{h,cpp}`. The
  Qwen3 BPE tokenizer already supported in the tree. The
  `Qwen3DenseWeights` loader (`LoadQwen3ForCausalLMWeights`,
  `qwen3_weights.cpp`).

## Upstream chain

### Oracle: CLM reference implementation

Repository: `Contrastive-LM/CLM` @ `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`.

- `src/clm/heads.py`: `HIDDEN=4096`, `PROJ_DIM=512`, `make_head(width,
  depth, proj, activation, layernorm, residual, hidden)` — the MLP head
  architecture. `HeadPair` — loads `CLM_v0.1-8B.pt` (torch save dict with
  `state_head`, `action_head`, `logit_scale`, `cfg`), runs projection +
  L2-normalize. `scale = float(torch.as_tensor(ck["logit_scale"]).float()
  .exp().clamp(max=100.0))`.
- `src/clm/engine.py`: `Engine.answer()` — the inference pipeline. Calls
  `build_pairs`, encodes state texts and candidate texts through the embedder
  (Qwen3-8B `/v1/embeddings` with last-token pooling), projects through
  state head and action head, L2-normalizes, computes `scale * cos / temp`,
  applies `answer_from_logits` (softmax + answer assembly).
- `src/clm/schema.py`: `build_pairs`, `candidates`, `state_text`,
  `answer_from_logits`, `answer_from_probs`, `softmax`, `confidence`.
  Question types: `noul`, `choice`, `score`. Confidence formula:
  `max(0, min(1, p_max - mean(rest)))` — top probability minus the mean of
  the rest.
- `src/clm/embedder.py`: `Embedder` — wraps an OpenAI-compatible
  `/v1/embeddings` endpoint (vLLM pooling server for Qwen3-8B), L2-normalizes
  embeddings.
- `train/finetune.py`: training script. Head config defaults: `width=1536`,
  `depth=3`, `proj=512`, `activation="gelu"`, `layernorm=True`,
  `residual=False`, `hidden=4096`. `logit_scale` init: `log(1/0.07)`.
  Loss: bidirectional in-batch InfoNCE, `logits = logit_scale.exp().clamp(
  max=100.0) * state_z @ action_z.t()`.

### Base model: Qwen3-8B

- `Qwen/Qwen3-8B` — HuggingFace repo, safetensors format, bf16.
- `config.json`: `model_type: "qwen3"`, `architectures:
  ["Qwen3ForCausalLM"]`, `hidden_size=4096`, `num_hidden_layers=36`,
  `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=128`,
  `intermediate_size=12288`, `vocab_size=151936`, `rope_theta=1000000`,
  `rms_norm_eps=1e-6`, `tie_word_embeddings=false`,
  `attention_bias=false`, `sliding_window=null`, `max_position_embeddings=
  40960`.
- Already fully implemented in vllm.cpp: `qwen3.cpp`,
  `qwen3_dense.cpp`, `qwen3_weights.cpp`.
  `Qwen3DenseModel::ForwardHidden` (qwen3.cpp:570-592) extracts
  post-final-RMSNorm hidden states with no lm_head — the exact pooling
  forward CLM needs.

### Adapter: Contrastive-LM/CLM-v0.1-8B

- `config.json` — non-standard HF config: `model_type: "clm"`,
  `architecture: "state/action projection heads (InfoNCE)"`,
  `base_model: "Qwen/Qwen3-8B"`, `encoder_pooling: "last-token"`,
  `embedding_dim: 4096`, `checkpoints: ["CLM_v0.1-8B.pt"]`,
  `library: "contrastive-lm"`.
- `CLM_v0.1-8B.pt` — torch save dict: `state_head` (state dict for the
  state MLP head), `action_head` (state dict for the action MLP head),
  `logit_scale` (scalar tensor, the log of the InfoNCE temperature
  inverse), `cfg` (dict: `width`, `depth`, `projection_dim`,
  `activation`, `layernorm`, `residual`, `hidden_size`).
- `tokenizer.json` — standard Qwen3 BPE (already supported).
- License: Apache 2.0.

## Design

### Phase 1: checkpoint conversion + dual-head host forward + golden tests

- Script: `scripts/convert-clm.py` — loads `CLM_v0.1-8B.pt` via torch,
  saves `head.safetensors` (the state-head and action-head weight tensors
  as F32), and records the checkpoint `cfg` and the raw `logit_scale` as
  `clm_*` keys in `config.json` (see `## Fidelity repair`; the first plan
  here named a `meta.json` and a pre-clamped scale).
- Head tensor names in `head.safetensors` (following the PyTorch state-dict
  convention from `make_head`):
  - State head: `state_head.inp.weight` [1536, 4096],
    `state_head.inp.bias` [1536],
    `state_head.hidden.0.weight` [1536, 1536],
    `state_head.hidden.0.bias` [1536],
    `state_head.norms.0.weight` [1536],
    `state_head.norms.0.bias` [1536],
    `state_head.out.weight` [512, 1536],
    `state_head.out.bias` [512].
  - Action head: same layout with `action_head.` prefix.
- Dual-head forward: `z_state = L2normalize(state_head(h_state))`,
  `z_action[k] = L2normalize(action_head(h_action[k]))`,
  `logits[k] = scale * dot(z_state, z_action[k])`.
  `scale = exp(logit_scale) clamped to 100.0`.
- Golden tests: generate from reference implementation with known hidden
  states. Tests live in `tests/vllm/models/test_clm.cpp`.

### Phase 2: ForwardHidden for Qwen3 dense (already exists)

- `Qwen3DenseModel::ForwardHidden` (qwen3.cpp:570-592) is ALREADY
  IMPLEMENTED: it runs the embed + 36-layer stack, stops after final
  RMSNorm with NO lm_head, and returns `[n_out, hidden_size]` f32 rows.
  This is the exact last-token-pooled embedding CLM needs — no new
  backbone code is required.
- Precedent: `LlamaEmbeddingLoadedModel` uses `ForwardHidden` for pooling
  (`llama_embedding_registry.cpp:115`). kev uses
  `Qwen3_5DenseModel::ForwardDenseHidden` for the same purpose on the
  Qwen3.5 backbone.
- CLM's encoder_pooling is "last-token": the forward returns the hidden
  state at every position, and CLM takes the LAST token's hidden state
  as the pooled embedding (the reference embedder uses vLLM
  `--runner pooling` which defaults to LAST for decoder-only models).

### Phase 3: Candidate text construction + inference pipeline

- `build_pairs` / `state_text` / `candidates` (schema.py): construct the
  state text and candidate texts from a SystemOne question.
  - `state_text(state, instructions)`: `"{state}\n\n{instructions}"` —
    context first, question last. The state head sees this combined text.
  - `candidates(question)`:
    - choice: `keys = list(criteria)`, `texts = [criteria[k] or k for k in
      keys]` — the action head sees each option's description (or key if no
      description).
    - score: `keys = ["0", "1", ...]`, `texts = [to_text(c) for c in
      criteria]` — the action head sees each level's text.
    - noul: `keys = ["false", "true"]`, `texts = ["false: ..." / "true:
      ..."]` — two candidates, the action head sees each.
  - The action head sees each candidate verbatim (no prefix, no special
    tokens). The state head sees state + question as one combined text.
- **Simplest correct approach** (no action caching): for each question,
  encode the state text through `ForwardHidden`, take the last-token
  hidden state, project through the state head. Encode each candidate
  text through `ForwardHidden`, take the last-token hidden state, project
  through the action head. L2-normalize both. Score:
  `logits[k] = scale * dot(z_state, z_action[k]) / temperature`.
  Softmax over logits gives the probability distribution.
- This requires N+1 forward passes per question (1 state + N candidates),
  but each is a simple prefill with no decode loop. The reference
  implementation batches all states and all candidates through the
  embedder endpoint, but the simplest correct port encodes them one at a
  time.
- Action embedding caching (reusing candidate embeddings across questions)
  is deferred to future work.

### Phase 4: Registration, server dispatch, /v1/systemone endpoint

- Register via `REGISTER_VLLM_MODEL` (mirror `kev_registry.cpp` and
  `llama_embedding_registry.cpp`).
- `LoadedModel` subclass owning Qwen3-8B dense weights + dual projection
  head weights + logit_scale + head config.
- `is_pooling_model=true`, `is_text_generation_model=false`.
- `/v1/systemone` dispatch: reuse the `DecisionFn` callback from Laya
  (PR #3263). CLM plugs in as an alternative model behind the same API.
- Confidence formula DIFFERS from kev AND from Laya — CLM uses its own:
  - confidence: `max(0.0, min(1.0, p_max - mean(rest)))` — top
    probability minus the mean of the rest. This is a TypeSafe-style
    margin, NOT kev's `(max(p) - 1/K) / (1 - 1/K)` and NOT Laya's
    entropy-based `1 - H(p)/log(k)`.
  - choice: `argmax(logits)`, `confidence(probs)`, `probabilities` dict.
  - score: `sum(i * p[i])`, `confidence(probs)`, `legend` dict,
    `probabilities` dict.
  - noul: `probs["true"]` (the probability of the "true" candidate). No
    confidence.
  - temperature: divides logits before softmax
    (`scale * cos / temperature`).
- All probabilities are NOT rounded to 2 decimals in the reference (the
  reference returns full-precision floats). This differs from kev/laya
  which round to 2 decimals. The port matches the reference: no rounding.

### Phase 5: E2E parity test vs reference

- Run the CLM Python server (`clm-serve` with a vLLM Qwen3-8B pooling
  embedder) on identical inputs (choice/score/noul).
- Compare outputs. The choice winner should match exactly; probabilities
  within tolerance for bf16 paths.
- Server E2E test through `/v1/systemone` HTTP endpoint.

## Fidelity repair

Filed as ISSUE-LOCAL-01M3T6ZZ2WBM4FSMTQZAWM8974. Each point below was checked
against the reference source at `bb42c6c` and against `CLM_v0.1-8B.pt` @
`Contrastive-LM/CLM-v0.1-8B` `e939398d4556fcd9400c76fa8c5a513202f42b0a`
(sha256 `b2b4a8c9...4eda5`) before any change.

| # | Engine before | Reference | Where |
|---|---|---|---|
| 1 | head tensors `<head>.0/.2/.4/.6` | `<head>.inp`, `.hidden.0`, `.norms.0`, `.out` (`weight`, `bias`) | `heads.py:44-66`, the `.pt` state dicts |
| 2 | Linear, GELU, LayerNorm, Linear, GELU, Linear | `act(inp(x))`, then per hidden block `act(norm(lin(x)))` (plus `x` when residual), then `out` | `heads.py:60-64` |
| 3 | `exp(min(ls, 100))` = 100.81; a missing key falls back to 4.6 | `exp(ls).clamp(max=100.0)` in float32 = 100.0 | `heads.py:92` |
| 4 | state text is the state alone | `state_text`: `to_text(state).strip() + "\n\n" + to_text(instructions).strip()` | `schema.py:62-65` |
| 5 | head input is the raw last-token hidden state | L2-normalized: vLLM pooling normalizes, and `Embedder._fetch` applies `l2` again | `embedder.py:61`, vLLM `adapters.py:as_embedding_model` |
| 6 | candidates from the shared kev `OptionText`: `key: description`, `no`/`yes` | choice: the description alone, else the key; noul: `true: <d>` / `false: <d>` with `d` defaulting to `Yes. This is true: <instructions>`; score: `to_text(level)`; objects through `to_text`, not `RenderJson` | `schema.py:30-59, 76-100` |
| 7 | HTTP `/v1/systemone` answers through `BuildSystemOneAnswerDecision` (4-decimal rounding, kev confidence, `rl_agent`) | `answer_from_probs`: no rounding, margin confidence, noul carries no confidence | `schema.py:118-148`, `api_server.cpp` decision path |

Design of the repair:

- `scripts/convert-clm.py` writes a self-contained directory: the base
  Qwen3-8B shards and tokenizer copied, `head.safetensors` with the reference
  tensor names under `state_head.` / `action_head.` (F32), and a
  `config.json` that is the base config with `architectures: ["ClmModel"]`
  and the checkpoint's `cfg` and raw `logit_scale` as `clm_*` keys. The
  engine computes the scale itself, so the formula lives in one place.
- The loader reads those names, checks every shape against the config, and
  refuses a missing `clm_*` key by name instead of defaulting.
- The head forward follows `make_head` for any depth, LayerNorm on or off,
  residual on or off, and GELU, ReLU or SiLU.
- CLM gets a request-level seam, `ClmDecide(model, tokenizer, body)`, that
  ports `build_pairs`, `to_text`, `Engine.answer` and `answer_from_logits`,
  including the `temperature` field and the reference's refusals. The server
  registers it with `set_systemone_request` and `vllm_decide` calls it, like
  Nimble. The per-question `DecisionFn` path and `BuildSystemOneAnswerClm`
  go away, so HTTP and FFI cannot drift.
- Identical texts in one request are encoded once, as the reference
  embedder's `dict.fromkeys` does, and a text longer than 2048 tokens keeps
  its first 2048, as `truncate_prompt_tokens=2048` does with the Qwen3
  tokenizer's `truncation_side="right"`.

Tests (every golden from running the reference, `scripts/gen-clm-goldens.py`):
heads (both `make_head` branches) loaded through the real loader from a
safetensors file with the reference names, the scale including the published
`logit_scale`, `build_pairs`, `answer_from_logits`, refusals, the whole
`Engine.answer` pipeline with raw (unnormalized) embeddings, `ClmDecide` over a
synthetic ClmModel against an independent `ForwardHidden` last row, and
`vllm_decide` reachability. With `VLLM_CPP_CLM_MODEL_DIR` set, the real
converted checkpoint must reproduce the reference's answers on five requests.

Gate: on CPU, the real checkpoint served by `vllm-server` must pick the
reference's argmax on every question, with the maximum probability difference
recorded next to the reference's own bf16-versus-fp32 difference on the same
requests.

### Measured (2026-09-30, CPU)

- Reference: `Engine.answer` @ `bb42c6c` with its own heads, schema and
  answer code; encoder Qwen3-8B @ `b968826d` through `transformers` 5.3.0,
  last token of the post-norm hidden state, L2-normalized (the stand-in for
  `vllm serve --runner pooling`, which the reference's `Embedder` calls).
- Engine: `vllm-server` Release CPU build of this branch, the converted
  directory, `/v1/systemone`; the same numbers through `vllm_decide`
  (`VLLM_CPP_CLM_MODEL_DIR`).
- Result: argmax 8/8, max probability difference 0.0286 (bf16 reference),
  0.0767 (fp32 reference). Reference bf16 versus fp32: 0.0491, 8/8.
- Before the repair, with the reference tensors loaded under the names the
  first port read: 5/8, 0.9808.
- Mutation: reintroducing each first-port behavior (old tensor names, the
  LayerNorm order, clamp before exp, the state without instructions, no
  pooling normalization, `key: ` candidates, the first row pooled, and
  deleting the `clm::Answer` call in `ClmDecide`) turns at least one
  default-run case of `test_clm` red.

## Our baseline

Before this row: the Qwen3-8B dense backbone forward and `ForwardHidden`
already exist (the Qwen3 dense port and the ARCH-ONE-SURFACE pooling row).
The `/v1/systemone` API, the `DecisionFn` callback, the shared `systemone`
helpers, and the C ABI `vllm_decide` all exist (from Laya + kev +
ABI-DECIDE). No CLM model, no dual projection-head readout, and no
scaled-cosine scoring existed.

## Port map

- Qwen3-8B backbone ForwardHidden: vLLM `qwen3.py` @ pin (dense forward,
  no LM Head) → `src/vllm/model_executor/models/qwen3.cpp` (ForwardHidden
  already at lines 570-592, NO new code).
- Dual projection heads + scaled-cosine readout: CLM source
  `Contrastive-LM/CLM` `src/clm/heads.py` (not in vLLM) →
  `src/vllm/model_executor/models/clm_registry.cpp` (new file).
- Checkpoint conversion: `scripts/convert-clm.py` (new file, loads
  `CLM_v0.1-8B.pt` torch save, writes `head.safetensors` + `clm_*` config keys).
- SystemOne dispatch: shared `src/vllm/entrypoints/openai/systemone.{h,cpp}`
  (reused from Laya PR).
- C ABI: `vllm_decide` / `vllm_decide_free` in `include/vllm.h` /
  `src/capi/vllm_c.cpp` (ABI v29; already on main). The architecture
  allowlist in `vllm_decide` (`src/capi/vllm_c.cpp:1711-1718`) currently
  accepts `KevModel`, `LayaModel`, `CuaS1Forms`. Add `"ClmModel"` to
  this list and add a `ClmInference` branch (alongside the existing
  `KevInference` / `LayaInference` branches) so CLM is dispatched by
  engine architecture name — same pattern, same ABI, one more family.
  Similarly, `server_main.cpp` must route the `"ClmModel"` architecture
  to the CLM decision callback via `set_decision` (the `DecisionFn`
  callback and `set_decision` API already exist from the Laya/kev
  work; only the CLM wiring remains).
- Registration: `src/vllm/model_executor/models/clm_registry.cpp` (new file,
  self-registers via `REGISTER_VLLM_MODEL`).
- Tests: `tests/vllm/models/test_clm.cpp` (new file).

## Tests to port

No upstream vLLM tests exist for CLM (not in vLLM registry). Tests are
authored from the CLM reference implementation:

- Dual-head golden-vector tests: state head projection (inp → GELU →
  LayerNorm → hidden → GELU → out), action head projection (same),
  L2-normalization, scaled dot-product — 25+ cases, 100+ assertions.
- logit_scale: verify `exp(logit_scale).clamp(max=100.0)` is loaded and
  applied correctly.
- ForwardHidden correctness: verify Qwen3-8B backbone extracts last-token
  hidden states of the right shape ([1, 4096] for a single text).
- Candidate text construction: verify `state_text`, `candidates` for
  choice/score/noul question types.
- Confidence formula: `max(0, min(1, p_max - mean(rest)))`.
- E2E: choice/score/noul through `/v1/systemone` via LocalAI HTTP server.

## Dependencies

- The `/v1/systemone` API and C ABI (landed in the Laya row at
  `c0320715f`, unified into `vllm_decide` at ABI v29 by PR #3301). This
  row adds the CLM model that uses it.
- The Qwen3 dense forward infrastructure (`src/vllm/model_executor/models/
  qwen3.cpp`, `qwen3_weights.cpp`, `qwen3_dense.cpp`). The `ForwardHidden`
  pooling path is already implemented.
- The shared systemone helpers
  (`src/vllm/entrypoints/openai/systemone.{h,cpp}`).
- The `DecisionFn` callback mechanism (landed in the Laya PR #3263, used
  by kev).
- No new CUDA kernels — CLM routes through existing `vt::` ops.

## Work breakdown

- Phase 1: checkpoint conversion + dual-head host forward + golden tests.
- Phase 2: ForwardHidden for Qwen3 dense — ALREADY EXISTS (no work).
- Phase 3: Candidate text construction + inference pipeline.
- Phase 4: Registration, server dispatch, /v1/systemone endpoint.
- Phase 5: E2E parity test vs reference.

## Gates

- CPU-correct: all golden tests pass (dual-head, logit_scale, ForwardHidden,
  candidate construction, confidence, end-to-end).
- E2E parity: choice/score/noul outputs match reference within tolerance.
  Choice winner must match exactly.
- Reachability: `/v1/systemone` endpoint serves CLM model through
  `ModelRegistry::Forward`.
- Inertness: the new model is additive. No existing model's forward path
  changes. Text-generation SACRED gates (27B, 35B, Coder) must stay
  byte-identical. kev, Laya, GLiNER2.5, cua-s1-forms tests must stay green.
- GPU build verification: owed (not pre-PR gate; stop condition is correct
  on CPU).

## Risks

- **Head config from checkpoint**: the reference head config (`width`,
  `depth`, `activation`, `layernorm`, `residual`) is stored in the
  `cfg` dict inside `CLM_v0.1-8B.pt`, NOT in the HF `config.json`. The
  conversion script must extract and persist these values so the C++
  loader can reconstruct the head architecture. The defaults from
  `finetune.py` (width=1536, depth=3, gelu, layernorm=true, residual=false)
  are the expected values, but the checkpoint is authoritative.
- **`CLM_v0.1-8B.pt` is torch pickle**: conversion script needs torch.
  This is a build-time preprocessing step, not a runtime dependency.
- **Last-token pooling**: CLM uses last-token pooling (the reference
  embedder runs vLLM `--runner pooling` which defaults to LAST for
  decoder-only models). The `ForwardHidden` path returns ALL token
  positions' hidden states; CLM must take the last token's row. Getting
  the wrong row (e.g. first token, or mean pooling) produces plausible but
  wrong embeddings.
- **L2 normalization**: both the state projection and the action
  projection must be L2-normalized BEFORE the dot product. Omitting
  normalization or normalizing at the wrong stage produces silently wrong
  scores — the dot product is still finite and the softmax still produces
  a distribution.
- **logit_scale**: the scale factor is `exp(logit_scale).clamp(max=100.0)`.
  The checkpoint stores the log-scale; the port must exponentiate and
  clamp. Using the raw logit_scale or forgetting the clamp produces scores
  off by a constant factor.
- **Confidence formula mismatch**: CLM uses
  `max(0, min(1, p_max - mean(rest)))`, which differs from BOTH kev
  (`(max(p) - 1/K) / (1 - 1/K)`) and Laya (`1 - H(p)/log(k)`). Using the
  wrong formula produces a plausible confidence value that is silently
  wrong.
- **No probability rounding**: the reference returns full-precision
  floats (unlike kev/laya which round to 2 decimals). The port must match
  the reference.
- **The Qwen3-8B checkpoint (~16 GB bf16) must be available.** The
  projection heads are tiny (~20M params, <100 MB).

## Stop conditions

- CPU-correct + E2E parity + reachability = ready for PR.
- GPU build = owed.
- Action embedding caching = future optimization.
- GGUF k-quants = owed.

## Owed

- GPU (CUDA) build verification.
- GGUF k-quant arm.
- Action embedding caching optimization (reuse candidate embeddings across
  questions — the reference caches action embeddings so repeated
  candidates skip re-encoding).
- The reference's `model` field (named head checkpoints, the `clm-raw`
  ablation) and its `/v1/rank` route: the engine serves the one converted
  head and ignores `model`.

## Git integration

One pull request (repository default policy). Spec commit precedes
implementation commits in the same pull request.

## Weights

- Base: `Qwen/Qwen3-8B` — bf16, ~16 GB. Already loadable via
  `LoadQwen3ForCausalLMWeights`.
- Heads: `Contrastive-LM/CLM-v0.1-8B` @ `e939398d4556fcd9400c76fa8c5a513202f42b0a`
  : `CLM_v0.1-8B.pt` (torch save, 75557149 bytes, sha256
  `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`),
  converted by `scripts/convert-clm.py` to `head.safetensors` plus `clm_*`
  keys in `config.json`. The repo carries no tokenizer; the converter copies
  the base's.

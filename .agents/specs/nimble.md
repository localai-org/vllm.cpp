# SPEC: `MODEL-NIMBLE`: Bespoke Labs Nimble decision model (Qwen3.5-9B + LoRA, candidate-logit readout)

Row: `MODEL-NIMBLE`

Serve `bespokelabs/Bespoke-Nimble-9B` (Ollama's `nimble`) through the existing
`/v1/systemone` endpoint and the `vllm_decide` C ABI. Nimble is a LoRA adapter
on the Qwen3.5-9B dense backbone that vllm.cpp already runs. It answers each
schema field with ONE forward pass: the logits of the candidate letter tokens at
the last prompt position, then `softmax(logits / T)`. It does not generate.

## Now

`ACTIVE`. Implemented and CPU-verified: reference prompt and answer goldens,
real-tokenizer ids, and a served end-to-end run on Qwen3.5-0.8B-Base with a
synthetic LoRA. On 2026-09-30 the published 9B checkpoint was converted, served
and compared on CPU with the author's own `prepare_prompts`,
`candidate_logits` and `decision_result` over HF `transformers` 5.3.0 + PEFT
0.21.0 BF16 (adapter unmerged): 7 questions, argmax 7/7, input tokens equal,
max probability difference 0.0054. The CUDA run of the reference, which its
`ParallelScorer` requires, is still `PENDING`.

## Scope

- **Row.** `MODEL-NIMBLE` (this spec). A SystemOne-class decision model, the
  same serving lane as kev, Laya, CLM, xor and GLiNER2.5-Decide.
- **In.**
  - A convert-time LoRA merge (`scripts/convert-nimble.py`) that writes a
    self-contained BF16 Qwen3.5 checkpoint with `architectures: ["NimbleModel"]`
    and the checkpoint's own temperature. This is kev's precedent
    (`scripts/convert-kev.py`); the server path has no runtime LoRA.
  - A `NimbleModel` registration over the Qwen3.5 dense loader.
  - One request-level library seam, `vllm::NimbleDecide(model, tokenizer, body)`,
    that both `/v1/systemone` and `vllm_decide` call. Per-question callbacks
    cannot express Nimble, because every field's prompt embeds the whole schema.
  - A last-row logits entry on the dense forward
    (`Qwen3_5DenseModel::ForwardDenseLastLogits`), mirroring HF
    `logits_to_keep=1`. `ForwardDense` materializes `[T, 248320]` f32, which is
    about 1 GB at a 1,000-token prompt.
  - The openjev answer semantics that Nimble's own server returns (below).
  - Both published revisions: `Bespoke-Nimble-9B` (T=1.0) and
    `Bespoke-Nimble-9B-v2` (T=2.179078721266035). Their prompt code is
    byte-identical; only the weights and T differ.
- **Out.**
  - Fields with more than 26 choices. The reference switches to
    `extended_schema.py`, which scans the vocabulary for 2-3 letter uppercase
    codes. Refused by name here; owed.
  - Prefix caching of the shared schema prefix across fields (openjev warms the
    prefix once). Each field runs its own full forward here.
  - GGUF k-quant arms and CUDA serving: owed.
  - `/v1/systemone/permute` and `/separate`: they are defined over per-question
    callbacks and are not answered for Nimble.
  - Runtime (unmerged) LoRA.

## Upstream chain

vLLM does not implement this pipeline, and no oracle in the AGENTS.md registry
covers it. The model author's reference is the only definition:

- `bespokelabs/Bespoke-Nimble-9B` @ `bd792f44ec8e265be861bfcdf4e05967ffe0e858`:
  `parallel_schema.py` (sha256 `a0a0f94d...`, the prompt contract for at most
  26 choices), `inference.py` (`candidate_logits`: `logits_to_keep=1`, gather
  the candidate ids, `decision_result`: `softmax(logits / T)`), `schema_config.json`
  (`max_length` 8192, base `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`),
  `temperature_config.json` (T=1.0).
- `bespokelabsai/nimble` @ `62076b4f2d365b5879dafcf7f6dd072a1fe76df7`,
  `nimble/serving/compiler.py`: the Jev question to schema mapping.
- `ekzhang/openjev-sglang` @ `7f84bedc169439f03379c2fa8d00ada220af2295`
  (pinned by `bespokelabsai/nimble` `requirements/modal.txt`):
  `src/openjev/scoring.py` (answer, confidence), `src/openjev/models.py`
  (request and response types).

This is a secondary-oracle situation without a registered oracle. The row
records `gateable = no` for a model-author oracle until someone runs the
reference on the 9B checkpoint (it needs a CUDA BF16 GPU and about 20 GB).

### The prompt (parallel_schema.py, 26 choices or fewer)

For each field, in request order:

```
<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n
<|im_start|>user\n{safe_json({"context": C, "schema": FIELDS})}\n\nRequested field: {safe_json(name)}<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

(line breaks above are for reading; the rendered string has exactly the `\n`
shown). `safe_json` is Python `json.dumps(ensure_ascii=False)` with the default
`", "` and `": "` separators, then `<` and `>` replaced by `<` and
`>`. `FIELDS` lists every field as `{"name", "description", "choices":
[{"code": "A", "value": v, "description"?}]}`. The chat template string was
checked against `tokenizer.apply_chat_template(..., enable_thinking=False)` on the
checkpoint's own `chat_template.jinja`.

The candidate ids are the single tokens that `code` adds at the answer boundary.
The reference refuses a code that is not exactly one ordinary token there, and
refuses duplicate ids. This port runs the same check with the loaded tokenizer.

### The Jev mapping (compiler.py)

| question | schema field |
|---|---|
| `noul` | `boolean`, choices `[false, true]`, descriptions `criteria.false` / `criteria.true`, default `"No"` / `"Yes"` (openjev `NoulCriteria`) |
| `choice` | `enum`, choices = criteria keys, description = criteria value, or the key when the value is `null` |
| `score` | `enum`, choices `"0"`..`"L-1"`, description = level text |

`description` is `serialize(instructions)`, and the context is
`serialize(state)`: a string as-is, anything else `json.dumps(ensure_ascii=False)`.
This differs from kev's `RenderJson`, so the lane keeps the raw JSON.

### The answer (openjev scoring.py)

- `p = softmax(candidate_logits / T)`.
- `noul`: `{"type": "noul", "noul": p[true]}`.
- `choice`: `choice` = argmax key (first on ties), `probabilities` = every key,
  `confidence` = `clamp(1 - H(p) / ln(K), 0, 1)` (normalized negative entropy).
- `score`: `score` = sum(i * p_i), `legend`, `probabilities`, `confidence` as
  above.
- No rounding. `usage.input_tokens` = the sum of the per-field prompt lengths.
  `usage.output_tokens` = 0, because nothing is sampled. openjev counts one
  warm-up token plus one per field, because SGLang samples them; that is a
  documented difference.

## Design

1. `scripts/convert-nimble.py <adapter_dir> --base-model-dir <Qwen3.5-9B> -o <out>`:
   verify the adapter's `SHA256SUMS`, verify `schema_config.json` names the base
   and revision it is given, merge
   `W' = bf16(float(W) + (alpha / r) * (B @ A))` (PEFT `merge` on a BF16 base
   with F32 LoRA), map `base_model.model.model.language_model.layers.N.P` to
   `model.language_model.layers.N.P.weight`, require every one of the 248
   modules to be found with matching shapes, and write the base's shard layout,
   `config.json` (`architectures: ["NimbleModel"]`, `nimble_temperature`,
   `nimble_max_length`), and the adapter's tokenizer and chat template.
2. `nimble_registry.cpp`: `NimbleModel` loads through `LoadQwen3_5Dense` and
   keeps the queue from `prepare`, like kev.
3. `nimble_inference.{h,cpp}`: request parsing and validation (the reference's
   refusals), prompt construction, tokenization, the boundary check, the
   forward, the openjev answer.
4. `ForwardDenseLastLogits`: the `ForwardDense` layer stack, then a one-row
   gather of the final-norm output, then `DenseLogitsF32D`. The gather is the
   one `Forward` uses for `logits_indices`.
5. Dispatch: `vllm_decide` gains the `NimbleModel` arm; `ApiServer` gains one
   opt-in request-level hook for `/v1/systemone`; `server_main.cpp` wires it.

## Tests

- Prompt goldens: the exact rendered strings from the reference
  `prepare_prompts` (chat template, JSON escaping, `<`/`>`, non-ASCII, noul
  defaults, `null` choice descriptions, non-string state).
- Answer goldens: openjev `answer()` outputs for fixed logits and T.
- Request refusals: the reference's own error cases.
- Reachability: `NimbleDecide` over a synthetic Qwen3.5 dense model and a
  byte-level tokenizer. The candidate logits must equal the independent
  `ForwardDense` logits at the last row and the candidate ids.
- Converter: a Python test on a tiny synthetic base and adapter (merge values,
  bf16 rounding, key mapping, refusal of an unmatched module).
- Real weights (manual, CPU): `Qwen/Qwen3.5-0.8B-Base` with a synthetic rank-16
  LoRA on Nimble's 12 target modules, converted, served on `/v1/systemone`, and
  compared with HF `transformers` + PEFT running the reference `inference.py`
  math. This is not the 9B checkpoint; it proves the pipeline, not the model.

## Risks

- The 9B base is 19.3 GB BF16. The machine this was written on has 12 GB of
  free disk, so the published checkpoint is not run end to end here.
- `ForwardDenseLastLogits` is a new public entry on a shared model. It must be
  byte-identical to the matching row of `ForwardDense`; a test asserts that.

## Gates

- CPU: all new tests pass; `test_kev`, `test_tev1`, and both Qwen3.5 vision
  suites unchanged.
- Reachability: the synthetic test enters through `NimbleDecide`, which both
  production entries call.
- E2E on the real 9B checkpoint against the reference: CPU done (argmax 7/7,
  max dp 0.0054, 2026-09-30); the CUDA run is `PENDING`.

## Stop conditions

- A field wider than 26 choices is refused by name, not approximated.
- If the Qwen3.5-9B backbone does not load, stop and report; do not work
  around it here.

## Owed

- The E2E gate on `bespokelabs/Bespoke-Nimble-9B` @ `bd792f44` against the
  reference `inference.py` ON CUDA (the CPU comparison is done, see `## Now`),
  and `-v2` at T=2.179.
- The 27-255 choice arm (`extended_schema.py`).
- Shared-prefix caching across fields.
- CUDA serving and GGUF k-quants.

## Git integration

One branch, spec commit before the implementation commit (repository default).

## Weights

- Base: `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16,
  four shards.
- Adapter (Ollama `nimble`): `bespokelabs/Bespoke-Nimble-9B` @
  `bd792f44ec8e265be861bfcdf4e05967ffe0e858`, `adapter_model.safetensors`
  173,188,512 bytes, sha256
  `29ef39b072dee97287947455337879c1e916705c2f727287922a2d81f5e2f20a`, T=1.0.
- Adapter (earlier release): `bespokelabs/Bespoke-Nimble-9B-v2` @
  `4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe`, 173,188,512 bytes, sha256
  `1bd126be997be6d9a0c25ce483ccf858c31b3422d480c33f02ba47b614be68ae`,
  T=2.179078721266035.

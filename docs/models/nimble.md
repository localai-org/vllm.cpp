# Nimble

Nimble is a decision model from Bespoke Labs, and Ollama serves it as `nimble`.
It is a LoRA adapter on the Qwen3.5-9B dense model. For each question it runs
one forward pass and reads the logits of the answer letters (`A`, `B`, ...) at
the last prompt position. It does not generate text. It answers `choice`,
`noul`, and `score` questions through `/v1/systemone` and `vllm_decide`.

This page carries the checkpoint, the conversion, the exact command, and what
has not been measured. [The usage guide](../USAGE.md#system-1-decisions-with-v1systemone)
covers the `/v1/systemone` request shape.

## The checkpoint

| field | value |
|---|---|
| adapter (Ollama `nimble`) | [bespokelabs/Bespoke-Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B) @ `bd792f44ec8e265be861bfcdf4e05967ffe0e858` |
| adapter file | `adapter_model.safetensors`, 173,188,512 bytes, sha256 `29ef39b072dee97287947455337879c1e916705c2f727287922a2d81f5e2f20a` |
| temperature | 1.0 (the adapter's `temperature_config.json`) |
| base | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16, four shards |
| format | PEFT LoRA, r=16, alpha=32, on 248 modules: attention, MLP, and the Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` |
| architecture after conversion | `NimbleModel` |
| license | Apache-2.0 |

The older release, [bespokelabs/Bespoke-Nimble-9B-v2](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2)
@ `4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe` (sha256
`1bd126be997be6d9a0c25ce483ccf858c31b3422d480c33f02ba47b614be68ae`), converts
the same way. Its temperature is 2.179078721266035. The names are misleading:
the repository without a suffix is the newer checkpoint. Both use the same
prompt code (`parallel_schema.py`, sha256 `a0a0f94d...`).

## Convert it

This engine has no LoRA on the server path, so the adapter is merged into the
base once. The merge is PEFT's own: `W' = bf16(float32(W) + (B @ A) * 2)`.

```sh
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir Qwen3.5-9B
hf download bespokelabs/Bespoke-Nimble-9B \
  --revision bd792f44ec8e265be861bfcdf4e05967ffe0e858 --local-dir Bespoke-Nimble-9B
python3 scripts/convert-nimble.py Bespoke-Nimble-9B \
  --base-model-dir Qwen3.5-9B --output-dir nimble-9b
```

The script needs `torch` and `safetensors`, and about 19 GB free for the output.
It checks the adapter's `SHA256SUMS`, the prompt-contract hash, and that every
one of the 248 modules pairs with a base tensor of the right shape. It refuses
the conversion instead of skipping a module.

## Run it

```sh
build/examples/vllm-server --model nimble-9b --served-model-name nimble --port 8000
```

```sh
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "nimble",
  "state": "I was charged twice. Please refund the duplicate.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Routine", "Urgent", "Emergency"]}}}'
```

The response has one answer per question, keyed by the question ID. The
semantics are those of Nimble's own server (openjev):

- `noul` is the probability of `true`.
- `choice` is the most probable key, and `score` is the expected level index.
- `confidence` is `1 - H(p) / ln(K)`, the normalized negative entropy. kev and
  Laya use a margin formula instead, so the two lanes are not comparable.
- Probabilities are not rounded.
- `usage.input_tokens` is the sum of the per-question prompts.
  `usage.output_tokens` is 0, because nothing is sampled. openjev reports one
  token per question plus one, because its backend samples them.

The request follows openjev's strict schema. An unknown key, a `choice` or
`score` with fewer than 2 entries, or a prompt longer than 8,192 tokens is
refused with HTTP 400, as the reference refuses it. `model` is optional here.
`/v1/systemone/permute` and `/v1/systemone/separate` are not served (404).

## What has been measured

On 2026-09-30, CPU:

- The prompts match the reference byte for byte. `scripts/gen-nimble-goldens.py`
  runs the model author's `compiler.py` and `parallel_schema.py`, and
  `test_nimble` compares every rendered prompt against it.
- With the checkpoint's own tokenizer, every prompt tokenizes to the
  reference's exact ids, and the answer candidates are the reference's ids
  (`VLLM_CPP_NIMBLE_TOKENIZER_DIR`, opt-in case of `test_nimble`).
- The answers match openjev's `scoring.answer` to 1e-12 on fixed logits at both
  published temperatures.
- The full pipeline was served on `/v1/systemone` from a real Qwen3.5 backbone:
  `Qwen/Qwen3.5-0.8B-Base` @ `dc7cdfe2` with a synthetic rank-16 LoRA on
  Nimble's 12 module types, converted by `convert-nimble.py`. On two requests
  (5 questions), against HF `transformers` 5.3.0 BF16 running the reference
  compiler and openjev: input tokens are equal (913 and 638), the argmax is
  equal on 5 of 5, and the largest probability difference is 0.0225 against
  the merged LoRA. HF's own merged and unmerged runs differ by up to 0.0317.
- The published checkpoint, end to end: `bespokelabs/Bespoke-Nimble-9B` @
  `bd792f44` converted onto `Qwen/Qwen3.5-9B` @ `c2022362` by
  `convert-nimble.py`, served on `/v1/systemone`, against the model author's
  own `parallel_schema.prepare_prompts`, `inference.candidate_logits` and
  `inference.decision_result` over `transformers` 5.3.0 BF16 with the adapter
  UNMERGED through PEFT 0.21.0. Five requests with seven questions (choice,
  noul with and without criteria, score): argmax equal on 7 of 7, input tokens
  equal on 5 of 5, largest probability difference 0.0054.

## What has not been measured

The 9B comparison above ran on CPU on both sides. The reference runner
(`inference.ParallelScorer`) refuses to start without a CUDA BF16 GPU, so its
functions were called directly on CPU; a CUDA run of the reference, and CUDA
serving here, are owed. Seven questions are a check, not an accuracy
measurement, and `Bespoke-Nimble-9B-v2` (T=2.179) was not run.

Also owed: fields with more than 26 choices (refused by name, the reference
switches to a 255-code prompt), reuse of the shared prompt prefix across
questions (each question runs its own forward), CUDA serving, and GGUF k-quants.

Spec: [`.agents/specs/nimble.md`](../../.agents/specs/nimble.md)

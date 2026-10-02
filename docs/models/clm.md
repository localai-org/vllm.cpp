# CLM

CLM (Contrastive-LM) is a bi-encoder System-1 decision model. A frozen
Qwen3-8B backbone encodes the state and each candidate separately. The last
token's hidden state is L2-normalized and projected by one of two MLP heads,
a state head and an action head, into a 512-d space. A question's distribution
is `softmax(scale * cos(state, candidate) / temperature)`, where
`scale = min(exp(logit_scale), 100)`. It answers `choice`, `score`, and `noul`
questions through `/v1/systemone` and `vllm_decide`.

The engine follows the reference implementation, `Contrastive-LM/CLM` @
`bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` (`src/clm/heads.py`, `schema.py`,
`engine.py`). This page carries the checkpoint setup, the request and answer
format, and what is measured.

## The checkpoint

| field | value |
|---|---|
| heads | [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) @ `e939398d4556fcd9400c76fa8c5a513202f42b0a`, `CLM_v0.1-8B.pt` (75,557,149 bytes, sha256 `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`) |
| base | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` (bf16, 16 GB) |
| head config | width 1536, depth 3, projection 512, GELU, LayerNorm, no residual, `logit_scale` 4.6132 (scale 100.0) |
| architecture | `ClmModel` |

The CLM repository holds only the heads, as a torch pickle, and no tokenizer.
`scripts/convert-clm.py` writes one directory that the engine loads: the base
shards and tokenizer copied unchanged, `head.safetensors` with the
checkpoint's own tensor names (`state_head.inp.weight`,
`state_head.hidden.0.weight`, `state_head.norms.0.weight`,
`state_head.out.weight`, their biases, and the same for `action_head`), and a
`config.json` that names `ClmModel` and carries the head config and the raw
`logit_scale` as `clm_*` keys. The loader refuses a missing key or tensor by
name. The converter needs `torch` and `safetensors`, and it refuses a `.pt`
whose sha256 is not the one above unless you pass `--allow-other-checkpoint`.

## Run it

```sh
hf download Contrastive-LM/CLM-v0.1-8B --local-dir CLM-v0.1-8B
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --local-dir Qwen3-8B
python3 scripts/convert-clm.py CLM-v0.1-8B --base-model-dir Qwen3-8B \
    --output-dir clm-v0.1-8b
build/examples/vllm-server --model clm-v0.1-8b --port 8000
```

Then send a SystemOne request:

```sh
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"john works at google",
       "questions":{"pick":{"type":"choice","instructions":"entity type",
       "criteria":{"person":null,"organization":null}}}}'
```

## Requests and answers

The request and answer follow the reference server (`server.py`), not the
shared kev format:

- The state head sees `state + "\n\n" + instructions`. An object or array
  state or description is rendered as the reference renders it (`to_text`:
  `key: value` fields, `- item` lines). Put the question in `instructions`.
- The action head sees a `choice` option's description, or its key when the
  description is null or empty, with no prefix; a `score` level's text; and
  for `noul`, `true: <description>` and `false: <description>`, where a missing
  description is `Yes. This is true: <instructions>` or
  `No. This is false: <instructions>`.
- An optional `temperature` in (0, 100] divides the logits. The reference's
  `model` field is ignored: the directory holds one head.
- `choice` answers carry `choice`, `confidence` and `probabilities`; `score`
  answers carry `score`, `confidence`, `legend` and `probabilities`; `noul`
  answers carry only `noul`. Nothing is rounded. `confidence` is
  `max(0, min(1, p_max - mean(rest)))`.
- `usage` carries `billing_units` (the number of questions), `input_tokens`
  and `output_tokens` (0). Identical texts in one request are encoded once.
  `input_tokens` counts each distinct text once; the reference counts only
  texts its embedding cache has not seen, so it reports less on repeated
  requests.
- A text longer than 2048 tokens keeps its first 2048, as the reference
  embedder's `truncate_prompt_tokens=2048` does.
- A malformed request is refused with the reference's message, as HTTP 400
  (the reference server answers 422).

## What is measured

On 2026-09-30, on CPU, the converted checkpoint was served by `vllm-server`
and compared with the reference `Engine.answer` on 5 requests with 8
questions (choice, noul, score, an object state, and `temperature` 2). The
reference ran its own heads, `build_pairs` and answer code over Qwen3-8B
run by `transformers` 5.3.0 in bf16, with the last token's post-norm hidden
state L2-normalized, which is what `vllm serve --runner pooling` returns.

| run | max probability difference | argmax agreement |
|---|---|---|
| this engine vs the reference (bf16) | 0.029 | 8/8 |
| this engine vs the reference (fp32) | 0.077 | 8/8 |
| the reference in fp32 vs the reference in bf16 | 0.049 | 8/8 |
| the first port (before the repair) vs the reference (bf16) | 0.981 | 5/8 |

The first port's row loaded the reference tensors under the names its loader
expected, in its own layer order. Scale 100 makes the answers sensitive to
the encoder's rounding: the reference itself moves by up to 0.049 between
bf16 and fp32.

`tests/vllm/models/test_clm.cpp` holds goldens produced by running the
reference (`scripts/gen-clm-goldens.py`), and with `VLLM_CPP_CLM_MODEL_DIR`
set it repeats the comparison above through `vllm_decide` (0.029, the same as
the HTTP route, which calls the same function).

## Not measured

The reference's own path runs Qwen3-8B through vLLM's pooling server on a GPU;
this comparison used `transformers` on CPU in its place. CUDA, GGUF k-quants,
and the reference's action-embedding cache are owed.

## History

The first CLM port (`ba8571340`) did not match the reference: its loader read
head tensors named `.0/.2/.4/.6`, which no checkpoint carries, and no
converter existed; its head put the LayerNorm before the hidden Linear; it
clamped `logit_scale` before `exp` (scale 100.81); it encoded the state
without the question and the candidates with a `key: ` prefix; it fed the
head an unnormalized hidden state; and the HTTP route rounded answers and
used kev's confidence. Its tests ran on synthetic weights in its own layout,
so they passed. See ISSUE-LOCAL-01M3T6ZZ2WBM4FSMTQZAWM8974.

Spec: [`.agents/specs/clm.md`](../../.agents/specs/clm.md)

# Qwen3.8 TensorFold endpoint benchmark manifest

This directory defines the engine-neutral corpus used by
`tools/bench/qwen38_endpoint_bench.py`. The harness sends UTF-8 OpenAI-compatible
requests with seed 0, greedy decoding, `ignore_eos`, streaming enabled with
usage requested, and at least five closed-loop waves. Aggregate request and
token rates use the measured makespan of the complete workload, rather than an
individual request latency.

Run either endpoint with its runtime model identifier (model names are not
embedded in the harness):

```sh
python3 tools/bench/qwen38_endpoint_bench.py \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model "$MODEL" \
  --tokenizer-identity "$TOKENIZER_NAME_OR_DIGEST" \
  --adapter tensorfold \
  --corpus benchmarks/manifests/qwen38_tensorfold/corpus.json \
  --output raw.json --concurrency 1 --waves 5 --draft off
```

`--tokenizer-identity` must identify the common tokenizer by immutable revision
or digest; the harness fails closed when it is omitted, and comparisons refuse
different identities. Runtime model aliases and adapter-owned evidence-only
fields are excluded from canonical workload hashes, while semantic prompt,
decoding, and endpoint-extra parameters remain covered.

`--adapter generic` adds no endpoint-specific evidence seam and therefore
supports absolute profiling but fails closed for token-exact matching. With
`--adapter tensorfold`, draft mode is explicit on the wire (`draft:false` or
`draft:true`) and the harness sends `return_token_ids:true`. The terminal
`tensorfold` SSE block supplies token IDs/hash and cache, draft, prefill, and
decode telemetry. These IDs are generated/reply IDs only; TensorFold exposes
no prompt-token IDs or `/tokenize` endpoint at the pin. With `--adapter
vllm-cpp`, the harness POSTs the exact assembled generated text to the server's
`/tokenize` endpoint with `add_special_tokens:false`; this is the pinned
`api_server.cpp` reply-text retokenization seam because generation SSE does
not expose
token IDs. For plain completions it also tokenizes the exact input prompt after
the measured workload. Chat prompt evidence fails closed unless an endpoint can
apply the identical chat template; raw message JSON is not treated as token
evidence. `--tokenize-url` may override the safely derived endpoint-origin URL
with another absolute HTTP(S) URL. All evidence-only calls happen after measured
generation, outside E2E, makespan, and closed-loop pacing.

`--extra-json` fields are semantic by default and are included in canonical
per-sample and run identity (including `top_k`, `stop`, and
`chat_template_kwargs`). They cannot contradict canonical or adapter semantic
fields. Only an adapter-owned, documented transport-only allowlist is omitted;
currently that is TensorFold's evidence request `return_token_ids`. The
effective draft policy remains in run identity.

Keep every raw repetition. `comparison_verdict` reports `MATCHED_INPUT` only
when prompt-token fingerprints are present and equal. TensorFold's actual reply
IDs are stored as provenance-qualified `engine_generated_token_*` evidence.
vllm.cpp decode→retokenize evidence is stored separately as
`reply_text_token_*`. `reply_text_token_equality` is reported only when both
runs provide that same evidence class; actual engine generation IDs are never
compared with retokenized reply text. The pinned TensorFold endpoint cannot
establish a prompt fingerprint or equivalent reply-text retokenization seam, so
current TensorFold/vllm.cpp runs are `PROFILE_COMPARISON` and reply-text equality
is unavailable. Prompt token counts alone do
not prove matched input. Missing generated IDs do not invalidate absolute
timing or throughput measurements when usage counts are present. E2E ends at
the terminal SSE event. ITL is reported only when every timed delta is proven
to contain exactly one token; chunk spacing is never presented as token ITL.
The HTTP adapter requests streaming usage; if usage or reliable generated-token
accounting is absent, the sample and enclosing run are refused and token
throughput/TPOT are not reported as valid. Comparisons also require an equal
run identity covering draft mode, concurrency, waves, and canonical semantic
requests. Corpus changes alter the canonical payload hash and must start a new
comparison series.

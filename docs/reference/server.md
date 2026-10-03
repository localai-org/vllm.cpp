# Server reference

Use this page to look up server behavior, flags, and defaults.

`vllm-server` is a small HTTP server speaking the OpenAI API. Source:
[`examples/server/main.cpp`](../../examples/server/main.cpp) and
[`src/vllm/entrypoints/openai/`](../../src/vllm/entrypoints/openai/).

```sh
build/examples/vllm-server --model /path/to/Qwen3.6-27B --port 8000 --max-num-seqs 32
```


For build and packaging commands, see [Building vllm.cpp](../BUILD.md). For
published artifacts and verification steps, see [Binary releases](../RELEASES.md).

## Selecting an x86 CPU ISA tier

The x86_64 CPU library is one adaptive binary: portable, SSE2,
SSE2+F16C, AVX2, and AVX-512 elementwise matmul kernels are isolated in their
own translation units and selected only after CPUID plus the required XCR0 OS
state are checked. Leave `VT_CPU_MATMUL_TIER` unset for automatic selection, or
set it to `portable`, `sse2`, `sse2+f16c`, `avx2`, or `avx512` for a same-binary
correctness/performance check. A forced tier that the current CPU or OS cannot
execute fails closed instead of silently narrowing or risking an illegal
instruction. Release builds never use `-march=native`.

On arm64, leave the same variable unset to select between portable and NEON
elementwise matmul, or force `portable`/`neon`. DotProd and i8mm kernels are
independently selectable with `VT_CPU_Q8_DOT`, `VT_CPU_QUANT_MMLA`, and
`VT_CPU_QUANT_REPACK`; `auto` uses Linux HWCAP/HWCAP2 or Darwin feature sysctls,
while an unavailable forced tier fails closed. The exact accepted values are
listed in [the environment reference](../ENVIRONMENT.md).


NVFP4 optimization defaults and current measurements are listed in
[the environment reference](../ENVIRONMENT.md) and
[the benchmark report](../BENCHMARKS.md).
## Endpoints

Registered in
[`src/vllm/entrypoints/openai/api_server.cpp`](../../src/vllm/entrypoints/openai/api_server.cpp).

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/completions` | Text completion (JSON or `text/event-stream`) |
| POST | `/v1/chat/completions` | Chat completion (JSON or streaming SSE) |
| GET | `/v1/models` | List the served model |
| GET | `/health` | Process liveness (200) |
| GET, POST | `/ping` | Liveness probe (200, mirrors `/health`) |
| GET | `/version` | Engine version |
| GET | `/metrics` | Prometheus metrics (`vllm:*` names, text format 0.0.4), recorded per engine step by the engine that serves your requests. Series and families keep stable addresses as new ones register (#330), so a long-lived scrape target does not read through a reallocated registry |
| POST | `/tokenize` | Tokenize a `prompt` to token ids (optional `token_strs`) |
| POST | `/detokenize` | Detokenize token ids back to text |
| GET | `/server_info` | Server info (`vllm_config`, `vllm_env`, `system_env`) |
| POST | `/reset_prefix_cache` | Reset the prefix cache; returns `{"success": bool}` |
| POST | `/v1/ner` | Extract named entities. Requires an NER callback |
| POST | `/v1/systemone` | Answer typed questions. Requires an NER, decision, or request-level decision callback |
| POST | `/v1/systemone/permute` | Repeat a choice question with reordered options. Requires an NER or per-question decision callback |
| POST | `/v1/systemone/separate` | Answer each question in a separate call. Requires an NER or per-question decision callback |
| POST | `/v1/score` | Score candidate options. Requires a scoring callback |
| POST | `/v1/embeddings` | Embeddings. Registered **only** when an embedder is attached, so a text server answers 404 at the route table |
| POST | `/v1/audio/transcriptions` | Speech to text (multipart: audio as `file`, `response_format` as a form field). Registered **only** when a transcriber is attached |
| POST | `/v1/videos` | Start a video generation job, returns `{id, status}` (MiniMax-H3) |
| POST | `/v1/videos/sync` | Same, but runs to completion before answering |
| GET | `/v1/videos/{id}` | Job status |
| GET | `/v1/videos/{id}/content` | The finished MP4 (`video/mp4`) |
| POST | `/v1/audio/speech` | Text (or lyrics + a music description) to audio; responds with `audio/wav` bytes. Registered **only** when a synthesizer is attached (`--speech-model`) |

The server registers decision routes according to the loaded architecture.
Nimble, CLM, and Tev1 use request-level callbacks, so their `/permute` and
`/separate` routes are absent and return HTTP 404. Tev1 requires
`architectures: ["Tev1Model"]` in `config.json` and keeps the chat routes.
See [System 1 requests](../USAGE.md#system-1-decisions-with-v1systemone),
[the decision and scoring C API](c-api.md#decisions-and-option-scoring), and the
[Tev1 recipe](../models/tev1.md).

`/v1/audio/speech` is registered only when you start the server with
`--speech-model`. Without that flag, the route returns 404. MiniMax-Music3
returns a 44.1 kHz stereo WAV. See the
[MiniMax-Music3 recipe](../models/minimax-music3.md).

IndexTTS-2.5 loads from `--speech-model` and returns a 22.05 kHz mono WAV. Each
request must include `reference_audio` as a data URL for a 16-bit PCM mono WAV.
The clip must have a 16 kHz sample rate. Its CAMPPlus speaker vector conditions
the talker and S2Mel stages. Different clips change the output, but this does
not establish voice-cloning parity. The inferred emotion path is not
implemented, and the vLLM-Omni quality comparison is pending. See the
[owning model specification](../../.agents/specs/indextts-2-5.md) for the owned
limitations and verification evidence.

`/v1/chat/completions` accepts `chat_template_kwargs`, an object of extra Jinja
variables handed to the model's chat template, exactly as vLLM does. It is how a
client selects a reasoning mode on a template that gates one:

```sh
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "qwen38-27b",
    "messages": [{"role": "user", "content": "hi"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

The request keys win over anything `--enable-thinking` / `--no-enable-thinking`
set at startup. **A key nobody supplies is not a template variable at all**, so
`{% if enable_thinking is undefined %}` answers true and the model's own default
applies; that is what vLLM does and what the Qwen3.8 family's template expects.
`/tokenize`'s chat form accepts the same field and renders through the same
template, so its token ids match what `/v1/chat/completions` would send.

A key valued `null` or `"auto"` means "not set", so it leaves the server-wide
default standing rather than clearing it, and a key that names something the
renderer supplies is **refused** with HTTP 400 rather than honoured:
`messages`, `tools`, `chat_template` and `tokenize`. Without that refusal a
request could hand the model a conversation its own `messages` field never
carried, which the request log, `usage` and any policy layer would then
describe wrongly. vLLM refuses the same four.

Two further groups of keys are accepted and **ignored**, again as vLLM ignores
them. `add_generation_prompt` and `continue_final_message` are request fields of
their own, and the field always wins over the kwarg. The second group is any key
that names a Jinja built-in: a global such as `namespace` or `range`, a filter
such as `tojson`, `upper` or `join`, or a test such as `equalto`. Such a key is
dropped, because the template needs the built-in and CPython Jinja2 never lets a
render variable replace one. `raise_exception` is the single name in that group
that does bind, which is also what vLLM does with it.

A chat template can refuse the request itself, through an unknown message role
or a kwarg value the template rejects. That answers **HTTP 400**, not 500, on
both `/v1/chat/completions` and `/tokenize`.

The four `/v1/videos` routes are registered **only** when the server was started
with `--video-dit`; without it they are absent (404) and the server is identical
to one built without video support. Use the
[MiniMax-H3 recipe](../models/minimax-h3.md) for the combined video and audio
workflow.

## Prompt log probabilities

For non-streaming requests, both completion endpoints return `prompt_logprobs`:

| Endpoint | Response field |
|---|---|
| `/v1/completions` | `choices[i].prompt_logprobs` on every choice |
| `/v1/chat/completions` | Top-level `prompt_logprobs` |

Set `prompt_logprobs` to a positive integer for that many top alternatives plus
the actual prompt token. Use `0` for the actual token alone, or `-1` for the
full vocabulary. Counts above the model's vocabulary size are refused.

The response array follows prompt-token order. The first entry is `null`
because the first token has no predecessor. Later entries map token IDs to
objects with `logprob`, `rank`, and `decoded_token`. Without requested prompt
log probabilities, the response field is `null`.

With `stream: true`, positive values and `-1` return HTTP 400. The validator
accepts `0` with streaming, but streaming responses do not carry this payload.
Other negative values return HTTP 400. The separate `echo` behavior, which
prepends prompt text and its token log probabilities to generated output,
remains unavailable.

The recorded computation and HTTP tests use CPU fixtures. GPU correctness
remains unverified. See the [prompt-logprobs specification](../../.agents/specs/prompt-logprobs.md)
for the pending accelerator gate.

## `max_tokens`: what a non-positive value means

Some clients (Hermes among them) send `max_tokens: -1` to mean "no client-side
limit". A non-positive `max_tokens`, or `max_completion_tokens` on
`/v1/chat/completions`, which takes precedence, is treated as **unset**, not as
an error and not as a clamp to some constant. Unset then generates up to
`max_model_len` minus the prompt length, mirroring vLLM.

That distinction is load-bearing for long-context requests: substituting a
constant would cap exactly the request that asked to be left unlimited, and the
client would see `finish_reason: length` with no way to tell it apart from a
limit it set itself. Use `VT_SERVER_MAX_NEW_TOKENS` when you want a serving-side
ceiling.

## Which token ids stop a generation

The engine resolves the primary end-of-sequence (EOS) token in this order:

1. The top-level `eos_token_id` in `config.json`, using the first integer when
   the value is a list.
2. The tokenizer's EOS ID, if available.
3. `eos_token` from `tokenizer_config.json`, only if it encodes to exactly one token.

The last fallback accepts a string or an object with a `content` string. It
does not replace an EOS ID resolved earlier. This lets checkpoints such as
[Tev1](../models/tev1.md#chat-completions) stop on their tokenizer's turn-ending token.

Additional EOS IDs come from `config.json` and the sibling
`generation_config.json`. For example, Gemma-4-26B uses:

```text
config.json             eos_token_id: [1, 106]
generation_config.json  eos_token_id: [1, 106, 50]
```

The engine merges secondary IDs into the request's `stop_token_ids`. When
`generation_config.json` is absent, model-config EOS IDs supply the fallback.
The top-level value wins over a nested text-config value. The nested value
applies only when the top-level value is absent or `null`.
A present but malformed `generation_config.json` supplies no additional IDs.

`ignore_eos: true` suppresses **all** of them, primary and secondary alike, and
generation then runs to the token budget. The ids still count toward
`min_tokens` masking either way, so `min_tokens` cannot be satisfied by emitting
a stop token early.

## Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir\|file.gguf\|org/repo\|org/repo:QUANT>` | Required except when `--speech-model` selects a speech/music-only server | A local directory or `.gguf` file, opened as before, or a Hugging Face repository, which is fetched into the cache. The local forms are probed first, so a network call can never shadow a path on disk. See [Access Hugging Face checkpoints](../guides/hugging-face-access.md) |
| `--revision <ref>` | repository default branch | A branch, a tag, or a 40 character commit for a `--model org/repo`. vLLM's own flag, and there is no inline `org/repo@rev` syntax |
| `--download-dir <path>` | the resolved Hugging Face cache root | The directory that holds the `models--org--repo` folders. vLLM's own flag. Fetching a repository needs a build that carries transport layer security. See [Access Hugging Face checkpoints](../guides/hugging-face-access.md) for the three build options, the environment variables, and which release lanes carry it |
| `--host H` | `0.0.0.0` | Bind host |
| `--port P` | `8000` | Bind port |
| `--served-model-name N` | model dir basename, or the `org/repo` you typed | Model id in `/v1/models` and responses |
| `--tokenizer-config F` | `<dir>/tokenizer_config.json` | Chat template / tokenizer config |
| `--block-size N` | `32` | KV block size. **Must be a multiple of 16**, the attention backends' `get_kv_cache_shape` refuses anything else, and the server now rejects it at startup rather than throwing during engine init |
| `--num-blocks N` | `0` (auto, resolves to `256`) | KV block count, and vLLM's `num_gpu_blocks_override`. It wins over every other sizing knob. `0` means auto, which uses `--kv-cache-memory` when that is set and otherwise falls back to `256` blocks |
| `--kv-cache-memory BYTES` | `0` (unset) | Absolute KV-pool size in bytes, vLLM's `kv_cache_memory_bytes`. The block count is this budget divided by the model's own bytes per block, summed across its KV groups, so it is correct on MLA and heterogeneous-KV architectures too. It ignores `--gpu-memory-utilization`, as vLLM does. A budget smaller than one KV block is refused at startup |
| `--gpu-memory-utilization F` | `0.92` | **Accepted, and it does not size anything yet.** See [What `--gpu-memory-utilization` does not do yet](#what---gpu-memory-utilization-does-not-do-yet) |
| `--max-model-len N` | `0` (config default) | Max sequence length |
| `--max-num-seqs N` | `32` | Maximum concurrent sequences and HTTP workers. For GDN or Mamba speculative decoding, it also sizes recurrent state as `N x (k+1)`. The loader refuses an unsupported state budget. |
| `--max-num-batched-tokens N` | `0` (per-arch default) | Per-step token budget |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | model default | Override automatic prefix caching |
| `--disable-sliding-window` / `--enable-sliding-window` | enabled | Disable the model-level sliding window for every model that has one, mirroring vLLM's `ModelConfig.disable_sliding_window` (`vllm/config/model.py:248`). A model without a window ignores the flag, which is upstream's own stated behaviour. A per-layer window still wins, matching upstream precedence. Covers Gemma-2, Gemma-3, Gemma-4, OLMo-2 and Muse-Glimmer; it replaced `VT_GEMMA2_SLIDING` and `VT_GEMMA3_SLIDING`, which reached only the first two. |
| `--scheduling-policy fcfs\|priority\|lpm` | `fcfs` | Scheduler policy (`lpm` is the SGLang cache-aware policy, see [the SGLang compatibility guide](../SGLANG-COMPAT.md)) |
| `--enable-radix-attention` / `--disable-radix-attention` | model default | SGLang-named alias for the prefix-cache toggle |
| `--enable-jump-forward` / `--disable-jump-forward` | off | **Accepted and validated, and nothing decodes with it yet.** See [What `--enable-jump-forward` does not do yet](#what---enable-jump-forward-does-not-do-yet) |
| `--enable-force-include-usage` | off | Force the usage block in responses |
| `--generation-config auto\|vllm\|<dir>` | `auto` | Where the server's default sampling parameters come from. `auto` takes `temperature`, `top_k`, `top_p`, `min_p` and `repetition_penalty` from the checkpoint's own `generation_config.json`, and a request that OMITS one of those knobs then gets the checkpoint's value; a request that sends one still wins. `vllm` loads no file and keeps the neutral OpenAI defaults. A directory path reads a `generation_config.json` from there instead. `Qwen/Qwen3.8-27B` ships `top_k: 20` and `top_p: 0.95`, so under `auto` an unparameterised request samples from 20 candidates rather than all 248,320. The server prints what it resolved at startup. `--override-generation-config` is unavailable ([#1985](https://github.com/mudler/vllm.cpp/issues/1985)). |
| `--tool-call-parser <name>` | `hermes` | Select one of 42 registered names across 38 dialect families. `auto` detects the dialect from the chat template, and `none` disables parsing. Gemma-4 accepts wrapped or bare text calls. Inkling requests require `"skip_special_tokens": false`. The `inkling` reasoning parser is unavailable. |
| `--reasoning-parser <name>` | `none` | Select `think_auto`, `deepseek_r1`, `deepseek_v3`, `holo2`, `mistral`, `minimax_m2`, `minimax_m2_append_think`, `step3`, `olmo3`, `muse_glimmer`, `qwen3`, or `mimo`. `auto` detects from the template. The `qwen3` and `mimo` names share an adapter. |
| `--kv-transfer-config '<json>'` | (unset) | External KV connector, same JSON as vLLM's flag. See [the KV offload guide](../KV-OFFLOAD.md) |
| `--offload-config '<json>'` | (unset) | Configure vLLM weight-offload fields and the `vllm_cpp` disk-residency and hybrid-placement tiers. vLLM offload backends currently refuse at startup because loaders do not use them. A `vllm_cpp`-only config works. Transcription refuses this flag. See [Weight offload and placement](../WEIGHT-OFFLOAD.md) and [Expert streaming](../guides/expert-streaming.md). |
| `--speculative-config '<json>'` | (unset) | Configure `mtp`, `dflash`, `ngram`, or `dspark`. Unknown fields, unsupported methods, incompatible targets, and invalid depths refuse at startup. Sampling defaults to `greedy` with `standard` rejection. The document also accepts a `vllm_cpp` extension object whose only key is `drafter_chain`, a preference-ordered list of speculators; it is validated but refused at startup, because nothing resolves a chain yet. See [Speculative decoding](../SPECULATIVE-DECODING.md). |
| `--language-model-only` / `--no-language-model-only` | off | Set every multimodal limit to zero. Multimodal requests then return HTTP 400. The Qwen3-VL, MuseGlimmer, and `clip` projector loaders also skip a tower when all its modalities have zero limits; the server names what it skipped. The dots3-note loader still loads its supported vision and audio towers at zero limits. Measured on **Qwen3-VL-4B-Instruct only**: **0.770 GiB of host RSS at load**, `--device cpu`, `dgx:gpu0`, 2026-08-28 ([#607](https://github.com/mudler/vllm.cpp/issues/607)). Read that as one model's tower rather than a general saving, and note it is load-time host RAM, not VRAM. An earlier 2026-08-24 run read 1.542 GiB; about half of that was our own bf16→f32 widening ([#1359](https://github.com/mudler/vllm.cpp/issues/1359)), whose Qwen3-VL half has since landed, and the 2026-08-28 rerun measured the resulting 0.499x fall — correct, not a regression. Other models are unmeasured. See [Multimodal input](../guides/multimodal-input.md) and [Memory benchmarks](../benchmarks/memory.md). |
| `--limit-mm-per-prompt '<json>'` | `999` per modality | Set lower per-prompt limits with a JSON object such as `'{"image": 2, "video": 0}'`. Effective limits cannot exceed the architecture's ceiling: Qwen3-VL accepts one image; dots3-note accepts up to 512 images. Dots3-note also accepts up to 128 audio items when `audio_config` exists. Both refuse video. Malformed JSON, negative counts, and unknown image, video, or audio options refuse at startup. Dotted flag syntax is unavailable. See [Multimodal input](../guides/multimodal-input.md). |
| `--mmproj <mmproj-*.gguf>` | (unset) | Load and validate a `clip` GGUF projector for a GGUF model. The server refuses incompatible model types, architectures, projector types, and incomplete temporal patch weights. HTTP multimodal inference for GGUF is unavailable, so this option does not produce image answers. See [Multimodal input](../guides/multimodal-input.md). |
| `--enable-log-requests` / `--disable-log-requests` | on | Log each incoming request. Mirrors vLLM's flag of the same name |
| `--enable-log-outputs` | off | Also log the generated output, not just the request |
| `--max-log-len N` | `256` | Truncate logged prompts and outputs to N characters |
| `--enable-metrics` / `--disable-metrics` | on | Serve the metrics endpoint |
| `--enable-thinking` / `--no-enable-thinking` | neither | Set the `enable_thinking` chat-template variable for templates that gate a reasoning block on it. Our spelling of vLLM's `--default-chat-template-kwargs enable_thinking`, whose default is also to set nothing. **Passing neither is not the same as `--no-enable-thinking`:** it leaves the variable UNSET, so a template asking `{% if enable_thinking is undefined %}` gets its own default (the Qwen3.8 family reasons; Gemma-4 does not). `--no-enable-thinking` forces it off for every request |
| `--verbose`, `-v` | off | Verbose server logging |
| `--cuda-profile-graph-replays N` | `0` (off) | Trace-only diagnostic: arm the CUDA-graph-replay profiler and stop after N replays, printing a pid to signal with `SIGUSR2`. Requires a build with `VT_BENCH_PROFILE_CONTROL` |
| `--cuda-profile-graph-batch N` | `16` when replays are armed | Batch size the profiler traces. Must not exceed `--max-num-seqs` |
| `-h`, `--help` | | Print usage and exit |

### Accepted for recipe compatibility, these flags have NO effect

A published `vllm serve` line has to reach model load. The flags below appear in
most official [vllm-project/recipes](https://github.com/vllm-project/recipes)
commands, mean nothing to this engine, and are therefore **accepted and ignored**
rather than rejected. Each one prints a notice on startup naming itself and the
reason it does nothing, so a log never implies it took effect.

| Flag | Effect here | Why it is inert |
|---|---|---|
| `--enable-auto-tool-choice` | **none** | Tool parsing is already unconditional once `--tool-call-parser` resolves; there is no second gate to open. Note `--tool-call-parser` defaults to `hermes` here, where upstream's defaults to unset, so the two flags do not line up when the parser is omitted. Upstream's validation is still mirrored: combining it with `--tool-call-parser none` is refused, as in `vllm/entrypoints/openai/cli_args.py:395` |
| `--trust-remote-code` | **none** | It authorizes executing Python from the checkpoint. This engine has no Python runtime, so there is nothing to authorize, N/A by construction, not unimplemented |

The notice is on stderr at startup, one line per flag actually passed, so what
you see in a log matches this table:

```text
server: accepted '--trust-remote-code' for published-recipe compatibility; it has no effect here: no Python runtime, so there is no remote code to trust
```

The mirrored validation is reported before the parser dialect is checked, so a
contradiction is named as a contradiction rather than passing silently (`none` is
itself a valid selection):

```text
server: Error: --enable-auto-tool-choice requires --tool-call-parser
server: (--tool-call-parser none selects NO parser; name a parser, or drop --tool-call-parser to keep the hermes default)
```

This list is **enumerated, not a catch-all**. Any other unrecognized flag still
aborts with `server: unknown argument '<flag>'`, including flags that are inert
only because the capability is missing (`--tensor-parallel-size` and the other
parallelism flags), silently accepting those would let you believe you got
tensor parallelism when you did not.

### What `--gpu-memory-utilization` does not do yet

The flag is accepted, keeps vLLM's exact name and fraction semantics, and is
then discarded. It does not size the KV pool. Passing
`--gpu-memory-utilization 0.85` gives the same 256-block pool as passing
nothing.

Turning a free-memory fraction into a block count needs a profile run that
measures what the weights and activations cost on the device first. That run is
not implemented. It is `ROAD-V1-MEM` M3, tracked by
[issue #83](https://github.com/mudler/vllm.cpp/issues/83), and it needs a GPU to
gate.

The flag is accepted rather than refused so that a published `vllm serve`
command line runs here unchanged. Setting it prints this warning at startup, so
a log never implies it took effect:

```text
vllm.cpp: WARNING --gpu-memory-utilization 0.85 was accepted but did NOT size the KV cache.
vllm.cpp:   The profile run that turns a free-memory fraction into a block count is not
vllm.cpp:   implemented yet (ROAD-V1-MEM M3, https://github.com/mudler/vllm.cpp/issues/83).
vllm.cpp:   The pool fell back to 256 blocks. To size it today, pass
vllm.cpp:   --kv-cache-memory <bytes> for an absolute KV budget, or --num-blocks <n> for an
vllm.cpp:   exact block count.
```

To size the pool today, use `--kv-cache-memory` for an absolute byte budget or
`--num-blocks` for an exact count. A run that never sets the flag prints
nothing.

**Warning.** On a unified-memory board such as NVIDIA GB10, a fraction of
"device" memory is a fraction of the one pool the host shares, so it reserves
host RAM as well. A value of 0.85 has hard-rebooted a GB10 box. When M3 lands
and this flag starts to bind, choose the fraction on such a board against the
whole 119 GiB pool and leave the host its headroom. Until then the flag reserves
nothing, on any board.

### What `--enable-jump-forward` does not do yet

The server accepts the flag, refuses it when you pass it twice, and resolves it
into the engine as `LoadedEngine::jump_forward_enabled()`. Nothing then reads the
resolved value. Passing `--enable-jump-forward` therefore leaves the decode path
exactly as passing nothing leaves it. `--disable-jump-forward` turns off
something that was never on.

Everything up to that latch exists, and each part has unit tests:

- `NativeGrammar::forced_token`, the grammar hook that reports the single valid
  next token at a non-accepting state.
- `vllm::v1::DrainForcedTokens`, the driver that drains a run of forced tokens
  and advances the grammar over them.
- The tri-state resolution of the flag against the `VT_ENABLE_JUMP_FORWARD`
  environment override.
- The C ABI field `vllm_model_params.enable_jump_forward`, added in ABI v10.

One call site is missing. Draining forced tokens between model steps means
recomputing KV for the tokens the model never ran, and that scheduler splice is
not implemented. Until it lands, `DrainForcedTokens` is called only from
`tests/vllm/v1/structured_output/test_jump_forward.cpp`, and
`jump_forward_enabled()` is read only from `tests/capi/test_capi.cpp`.

The work is `ENG-STRUCTURED-OUTPUT`, tracked by
[issue #2387](https://github.com/mudler/vllm.cpp/issues/2387).

The server accepts the flag rather than refusing it because the knob is already
published: ABI v10 added the field, and the [SGLang compatibility
guide](../SGLANG-COMPAT.md) documents it. Refusing it now would break a caller
that already sets it, and setting it cannot give a wrong answer, because it
changes no answer at all.

**Note.** Unlike `--gpu-memory-utilization`, this flag prints no startup warning,
so a log tells you nothing either way. This page is the only signal.

### Context length vs the KV pool

The KV pool holds `--num-blocks × --block-size` tokens, `256 × 32 = 8192` by
default. A request longer than that can never be scheduled, so the engine
refuses it early rather than leaving it in the waiting queue forever. Two checks
do that, mirroring vLLM:

- **At startup.** If `--max-model-len` is given and the pool cannot hold one
  sequence that long, the server exits with the sizes and the flags that close
  the gap (vLLM's `_check_enough_kv_cache_memory`). If it is **not** given, the
  serving length is auto-fitted down to what the pool holds and logged
  (vLLM's `_auto_fit_max_model_len`), so raising `--num-blocks` is what buys a
  longer context.
- **At admission.** A prompt at or past the resolved `max_model_len` is
  rejected with **HTTP 400** (`BadRequestError`) naming both lengths, exactly as
  vLLM's `_validate_prompt_len` does. It is never a finish reason and never a
  500.

Set `VT_ENGINE_STEP_LOG=1` to print a per-step engine heartbeat if you need to
confirm that a quiet engine is idle rather than stalled.

For a production deployment, use [LocalAI](https://localai.io), which can embed
engines like this behind a model gallery, multi-model serving, the full OpenAI
API surface, auth, and metrics.

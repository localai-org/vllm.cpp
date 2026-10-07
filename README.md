<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.png">
    <img alt="vllm.cpp" src="assets/logo-dark.png" width="420">
  </picture>
</p>

<p align="center">
  <b>Same tokens as vLLM. Same throughput. 140x less to install.</b><br>
  <sub>Continuous batching, paged KV, 44 registered architectures, CUDA / CPU / Metal / Vulkan. No Python anywhere.</sub>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue"></a>
  <a href="docs/BENCHMARKS.md"><img alt="vs vLLM" src="https://img.shields.io/badge/Qwen3.6--27B_vs_vLLM-token--exact_%2B_same_throughput-3ec8e0"></a>
  <a href="#project-status"><img alt="Architectures" src="https://img.shields.io/badge/architectures-25%2B_gated-7ee787"></a>
  <a href="#performance"><img alt="Binary size" src="https://img.shields.io/badge/one_binary-66_MiB,_no_Python-6e7681"></a>
  <a href="https://github.com/mudler/LocalAI"><img alt="LocalAI" src="https://img.shields.io/badge/LocalAI-Run_Locally-orange"></a>
</p>

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team**, the folks behind LocalAI, the open-source AI engine that runs any model (LLMs, vision, voice, image, video) on any hardware, no GPU required.

> **Independent and unofficial.** vllm.cpp is a community port. It is not affiliated with, endorsed
> by, or sponsored by the vLLM project, the PyTorch Foundation, or the Linux Foundation. "vLLM" is
> used here only to name the upstream project this port mirrors and is measured against. See
> [Trademarks](#trademarks).

> **Why this exists:** [MANIFESTO.md](MANIFESTO.md).

> ⚠️ **Under heavy development.** This project moves fast right now: internals, CLI flags, and
> server behavior can change between commits, so expect breakage if you track `main`.
> The one thing we keep disciplined is the **C ABI** in [`include/vllm.h`](include/vllm.h):
> it is versioned (`VLLM_ABI_VERSION`, checkable at runtime with `vllm_abi_version()`), grows by
> appending fields whose zero value keeps existing behavior byte-identical, and only bumps on an
> incompatible change. If you embed us, embed through that header.

## News

- **2026-09** **Qwen3.8 gains an endpoint measurement tool.** The TensorFold comparison remains
  blocked by missing artifacts, with no speed result. See the
  [measurement instructions and limits](docs/benchmarks/qwen38-tensorfold-gap.md#measurement-tools).
- **2026-09** **Vulkan gains TQ1_0 ternary kernels.** Matrix multiplication and fused MoE
  kernels operate on compressed weights. TQ1_0 and TQ2_0 GGUF loading remain unsupported.
  See [kernel coverage and limits](docs/FEATURES.md#ternary-kernels).
- **2026-09** **Tev1 serves typed decisions and chat from one engine.**
  The decision route returns probabilities for each question. See the
  [Tev1 recipe](docs/models/tev1.md) for activation, CPU comparisons, and remaining validation gaps.
- **2026-09** **Multimodal chat reaches the model through HTTP.** Qwen3-VL accepts images;
  dots3-note also accepts audio and multiple media items. CPU tests use synthetic weights;
  real-checkpoint token parity remains unverified. See the [input guide](docs/guides/multimodal-input.md)
  for formats and limits.
- **2026-09** **C ABI 26 exposes more engine controls.** Applications can select the KV cache
  dtype, read speculative acceptance counters, and disable the model-level sliding window.
  See the [C API reference](docs/reference/c-api.md#recent-abi-additions) for defaults and limits.
- **2026-09** **EXL3 gains a native ROCm path.** Llama-3.2-1B-Instruct EXL3 generated on
  gfx1151 with zero CPU fallbacks. Discrete AMD validation and competitive performance remain
  unmeasured. See the [ROCm build guide](docs/BUILD.md#rocm-build-amd-gpus).
- **2026-09** **CUDA EXL3 now supports Qwen3.8-27B and its DFlash2 draft.**
  The CUDA path handles their `mul1` codebooks and uses reconstruction plus cuBLASLt for long
  prefills. See [supported artifacts and limits](docs/USAGE.md#checkpoint-registry) and
  [benchmark conditions](docs/benchmarks/qwen38-27b-exl3-gb10.md).
- **2026-08** **Qwen3.8-27B block-wise FP8 passes its GB10 text gate.** Six of seven
  prompts matched vLLM token for token. One passed the ratified near-tie test. This is a
  correctness result with no speed claim. See [the model guide](docs/models/qwen3-8-27b.md#the-token-gate-against-vllm).
- **2026-08** **EXL3 checkpoints now generate on CPU and CUDA.** A stock
  Llama-3.2-1B-Instruct EXL3 checkpoint loads through the shared dense model path and emits text.
  The current CUDA path supports its 3-bit body and 6-bit output head. No speed claim is available.
- **2026-08** **GGUF gains IQ2_XS and IQ4_XS.** Both formats decode and run directly on their
  compressed blocks on CPU. This lets the 101.25 GiB GLM-5.3-Flash GGUF weight tower load without
  expanding to 426.72 GiB. Its model forward is still incomplete.
- **2026-08** **GLM-5.3-Flash now generates on CPU from a 101.25 GiB GGUF.** The shipped
  `UD-Q2_K_XL` artifact emits coherent text while keeping IQ2_XS and IQ4_XS blocks compressed.
  Both formats also have CUDA keep-quant kernels, but this model's CUDA forward and every speed
  gate remain pending.
- **2026-08** **GLM-5.3 joins the model registry.** Its GGUF loader and first-token forward run
  through the shared expert-streaming path. The real 201.83 GiB artifact has not completed a load,
  and resumed sparse decoding still needs the indexer side cache, so no real-checkpoint token or
  speed claim is available.
- **2026-08** **Hybrid CPU/GPU expert placement reaches five architecture families.** Qwen3-MoE,
  Qwen3.5/3.6, Nemotron-H, DeepSeek-V2, and Kimi-Linear can run routed experts on the CPU while the
  rest of the model stays on the selected accelerator. The end-to-end token and speed gates are
  still pending.
- **2026-08** **IndexTTS 2.5 now uses its reference audio.** The clip conditions both the talker
  and S2Mel stages. Repeated runs with one clip are bit-identical, while different clips change
  the output. Voice-cloning parity with vLLM-Omni remains unmeasured.
- **2026-08** **FP8 KV cache storage reaches the server.** `--kv-cache-dtype fp8` stores K/V in
  1-byte E4M3 pages. A fixed memory budget holds twice as many blocks, but the throughput trade-off
  is not measured yet.
- **2026-08** **The server returns prompt log probabilities.** Both completion endpoints accept
  `prompt_logprobs`, including `-1` for the full vocabulary, and reject unsupported request shapes
  with `400`.
- **2026-08** **MiniMax-Music3 generates music through the public API.** Every pipeline stage is
  implemented and gated. The server exposes it through `POST /v1/audio/speech`; no reference speed
  number is available yet.
- **2026-08** **LTX-2.5 gains its video and audio pipeline surface.** Text-to-audio, retake,
  audio-to-video, keyframe interpolation, and one-stage and two-stage video recipes are wired and
  fixture-gated.
- **2026-08** **v0.0.2 ships eight server archives.** Download CPU, CUDA, Vulkan, Metal, and MLX
  builds from [GitHub Releases](https://github.com/mudler/vllm.cpp/releases/tag/v0.0.2).
- **2026-08** **MiniMax-H3 generates video with audio.** All tasks run through `POST /v1/videos`;
  use Q4_K_M.
- **2026-08** **MXFP4 holds parity with vLLM.** Qwen3-8B MXFP4 runs W4A16 Marlin by default,
  matches the vLLM oracle token for token, and decodes **45.45 vs 41.94 tok/s**.

vllm.cpp is a from-scratch C++20 inference engine chasing three things at once: be the
**smallest** thing you can deploy, be the **fastest** on the hardware you already own, and still
carry **every feature people actually want**. No Python and no PyTorch at inference time.

It gets there by taking the best of each engine rather than reimplementing one of them.
[vLLM](https://github.com/vllm-project/vllm)'s serving core (continuous batching, block-paged KV,
automatic prefix caching, speculative decoding). [SGLang](https://github.com/sgl-project/sglang)'s
scheduling ideas (RadixAttention, LPM cache-aware admission, jump-forward decoding).
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s deployment story (one library behind a flat C
ABI, GGUF straight off the shelf, and compute directly on the quantized blocks). MLX's GEMM where it
wins on Apple Silicon. Safetensors and GGUF, CUDA and CPU and Metal and Vulkan, from one source tree.

## Project status

vllm.cpp is under heavy development. The README carries the stable public
overview; [Features](docs/FEATURES.md) lists shipped surfaces and
[Benchmarks](docs/BENCHMARKS.md) indexes published measurements. Contributors
derive detailed lifecycle state and history from Git, row specs, and matrices.

Completed architecture gates compare **token output against vLLM** on the same workload. Rows that
have only a fixture or a blocked gate say so. Speed claims use the reference engine's production
configuration.

![vllm.cpp vs vLLM on Qwen3.6-27B: identical output at every concurrency](benchmarks/media/concurrency_race.gif)

> The same model, the same prompts, side by side: **token-for-token identical output**, finishing a
> hair ahead. That margin is inside our noise band at five of the six concurrencies
> ([full clip](benchmarks/media/concurrency_race.mp4)).

Where that stands today:

- **Small.** **66 MiB** of binary against a **9.1 GiB** vLLM install, both measured on the same GB10:
  about **140x less to deploy**, serving the same model in **24.88 GiB of peak host memory against
  vLLM's 28.18**. No interpreter in the process, and 0 bytes of bundled CUDA userspace.
- **Fast.** On Qwen3.6-27B we **match vLLM's throughput** against its graphed production config,
  ahead at all six concurrencies but only c1 outside our noise band. Also **1.18x llama.cpp's
  prefill** on the same GGUF file (denominator SUPERSEDED, see below), and **ahead of MLX-LM on
  prefill** on Apple Silicon. Most other architectures are speed-pending, and say so.
- **Everything.** 44 registered architectures, 38 tool-parser families, structured output including
  GBNF, three speculative decoders, image, video, and audio input, music generation, external KV
  offload, Prometheus metrics, and the SGLang knobs, all in a library you can `dlopen`. Multimodal
  HTTP input has CPU tests with synthetic weights on Qwen3-VL and dots3-note. Real-checkpoint
  token parity remains unverified ([formats and limits](docs/guides/multimodal-input.md)).

## Performance

Qwen3.6-27B (NVFP4) on NVIDIA GB10, greedy, closed loop, against the vLLM oracle in its
**production graphed config** (not `--enforce-eager`). Output is token-for-token identical at every
point on this curve:

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| **vllm.cpp** (tok/s) | **86.05** | **159.68** | **292.34** | **508.77** | **801.76** | **1095.01** |
| vLLM (tok/s) | 82.32 | 158.03 | 290.31 | 505.46 | 789.16 | 1076.25 |
| **Ratio** | **1.045x** | **1.011x** | **1.007x** | **1.007x** | **1.016x** | **1.017x** |

We are ahead at all six, but only c1 at 4.5% is clearly outside our 0.5% run-to-run noise band, so
treat c2 through c32 as ties. The tokens are identical either way, and the install is 66 MiB against
9.1 GiB.

Cold start to first `/health`: **53 s vs vLLM's 780 s (14.7x)**, medians of three, and
deliberately NOT like-for-like -- we answer `/health` on process liveness, so the first
request then pays 91.6 s of real first-inference cost that vLLM front-loads
([measurement details](docs/benchmarks/vllm-online-serving.md)).

Peak host memory is a clean win at **24.88 GiB vs vLLM's 28.18 GiB**, with no Python stack behind it:

![What you install: a 9.1 GiB venv, or one 66 MiB binary](benchmarks/media/footprint.png)

And we hold every other engine to the same treatment: same model, same workload, same box.


### vs llama.cpp, on CPU, from the same GGUF file

| | vllm.cpp | llama.cpp | ratio |
|---|--:|--:|--:|
| prefill | **223.8 tok/s** | 177.3 | **1.18x** |
| decode | 24.7 tok/s | 25.4 | 0.97x (tie) |
| peak memory | 2.83 GiB | 2.80 GiB | 1.01x |

Decode lands inside llama.cpp's own spread, and the memory gap is 30 MiB. Tokens are
**byte-identical to llama.cpp's greedy decode**. Single-stream only: concurrent serving against
llama.cpp's server is unmeasured.

> **Every llama.cpp denominator here is SUPERSEDED**: they came from `237ad9b96`, our own local-only
> fork, 65 performance commits deep. The pin is now stock `b10451` and each figure is owed a re-take
> that can move it either way ([#1003](https://github.com/mudler/vllm.cpp/issues/1003)).

### vs MLX-LM, on Apple M4, warm b=1

| | vllm.cpp | MLX-LM | ratio |
|---|--:|--:|--:|
| prefill TTFT | **524.5 ms** | 532.6 ms | **1.5% ahead** |
| decode | 27.23 tok/s | 27.85 | 97.8% |
| warm total | 24.37 tok/s | 24.96 | **97.6%** |

That 2.4% is a real gap, not run-to-run noise: across 6 interleaved runs our spread was 0.12% and
MLX-LM's 0.34%. All of it sits in decode, 0.81 ms per token, and we know where it goes. Prefill is
ahead.

Under the hood on Metal: our GEMM runs at **97% of MLX's own** (3.91 TFLOP/s mma issue rate), the
decode GEMV streams weights at **83% of the part's memory-bandwidth peak**, and moving prefill
attention onto the matrix units was worth **4.3x**. Indicative rather than binding: two models
(OPT-125m, Qwen3-0.6B), 18 of 75 ops native, and the 97.6% needs the optional MLX GEMM provider
shape-gated to prefill (95.9% on the default build).

### Speculative decoding

MTP is **token-identical to vLLM's MTP and about 4% faster** at c1 on Qwen3.6-27B-NVFP4, on both gate
models end to end. Block-diffusion DFlash runs **2.9x over spec-off** and lands at vLLM's own
DFlash (1.003x); its successor DFlash2 is the one still **behind, at 0.8017x**, on an open bf16
acceptance floor. Every speculator's current position is in
[the speculative-decoding measurements](docs/benchmarks/speculative-decoding.md).

Full per-axis grids, memory tables, the nine residual axes, and exact reproduction recipes:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). The two figures above are rendered from these measured
numbers by [`benchmarks/demo/`](benchmarks/demo/), which reads its values from a committed spec, so
every figure traces back to the run that produced it.

> **Pre-release, under heavy development.** Correctness is gated token-for-token against a pinned
> vLLM oracle across 27 gated architectures. Speed is proven on one GPU (GB10, sm_121a) plus a CPU
> path that matches or beats llama.cpp on GGUF, against a SUPERSEDED fork denominator (#1003).
> The current public overview is kept in [Project status](#project-status).

## Quickstart

Start at **[docs/QUICKSTART.md](docs/QUICKSTART.md)**. It runs one model from one command, with
no build and no checkpoint of your own: a container lane, the `curl` that answers it, the release
archive for a host without Docker, and the cache, token, and offline notes. `--model` accepts a
directory, a `.gguf` file, `org/repo`, or `org/repo:TAG`, and fetches what your cache lacks.

Full CLI, server, and library reference: [docs/USAGE.md](docs/USAGE.md) and
[docs/reference/](docs/reference/README.md). CUDA, Metal, Vulkan, and every CMake option:
[docs/BUILD.md](docs/BUILD.md).

## Features: vLLM parity, then everything else

vllm.cpp **mirrors vLLM by default**: same scheduler, same sampling order, same flags, same JSON
configs, token-for-token the same output. Switching to it should be boring. Everything below is what
you get on top, most of it borrowed from whichever engine does it best:

- **One 66 MiB binary instead of a 9.1 GiB install.** A flat, exception-free, llama.cpp-style C ABI
  ([`include/vllm.h`](include/vllm.h), ABI v26) for C, C++, Go, or Rust. No Python
  interpreter in the process.
- **GGUF as a first-class citizen.** Load the same quantized files llama.cpp uses, and on CPU
  **compute directly on the compressed blocks** (Q4_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XS,
  and IQ4_XS) with no BF16 expansion. Byte-identical greedy output to llama.cpp on the gated paths.
- **EXL3 trellis weights.** A stock 3.0 bpw Llama-3.2-1B-Instruct checkpoint generates on CPU and
  CUDA through the shared dense model path. Native ROCm generation is measured on gfx1151.
  The DeepSeek-V4 rank-sliced layout remains partial.
- **SGLang's good ideas, as documented toggles.** RadixAttention / prefix caching, LPM cache-aware
  scheduling, jump-forward decoding, and custom logits processors, opt-in from the library, the C
  ABI, or server flags. Each defaults to today's behavior, so an engine that sets none of them is
  byte-identical to one built without them ([docs/SGLANG-COMPAT.md](docs/SGLANG-COMPAT.md)).
- **Runs on hardware people actually have.** CUDA, CPU, Metal, and Vulkan ship from one tree. ROCm
  and Tenstorrent are growing; Apple MLX and Arm i8mm providers cover their useful shapes.
- **Speculative decoding beyond ngram.** MTP, block-diffusion DFlash, and draft-free ngram, through
  the same `--speculative-config` JSON vLLM takes
  ([docs/SPECULATIVE-DECODING.md](docs/SPECULATIVE-DECODING.md)).
- **Smaller KV pages when capacity matters.** `--kv-cache-dtype fp8` stores K/V in 1-byte E4M3
  pages on the routed model families. The memory win and current kernel trade-offs are documented
  in [the usage guide](docs/USAGE.md#halve-the-kv-cache-with---kv-cache-dtype-fp8).
- **Additive by design.** New architectures and new GPU targets land as additive files mirroring
  vLLM's own structure, so upstream changes port mechanically and a contribution stays a small diff.
- **Honest numbers.** Every capability is labelled correctness-complete, speed-pending, build-only,
  or hardware-blocked, and the evidence is in the record. Nothing is called fast that was not
  measured against a reference.

<details>
<summary><b>The full feature surface</b> (serving core, sampling, structured output, tool calling, multimodal, external KV)</summary>

- **vLLM's serving core.** Continuous batching, block-paged KV cache, automatic prefix caching (on
  by default for dense models), and the V1 / Model Runner V2 scheduler and engine step loop.
- **Sampling.** Greedy, temperature, top-k/p, min-p, presence/frequency/repetition penalties, seed,
  stop sequences, `logit_bias`, `allowed_token_ids`, and `bad_words`, in vLLM's exact order, plus
  sample logprobs.
- **Structured output.** JSON schema, JSON object, regex, choice, and GBNF grammar, enforced in the
  engine with a per-step logits bitmask.
- **Tool calling and reasoning.** 38 tool-parser families (42 accepted names) and 12 reasoning
  parser names, streaming, selectable with `--tool-call-parser` / `--reasoning-parser`. Chat templates
  render through the vendored google/minja engine, the same renderer llama.cpp ships.
- **Multimodal.** Image, video, and audio to text have committed goldens on the single-sequence
  drivers. HTTP input reaches the model on Qwen3-VL and dots3-note, with CPU tests using synthetic
  weights. See [formats, model limits, and remaining gates](docs/guides/multimodal-input.md).
- **Quantization.** NVFP4 W4A4/W4A16, compressed-tensors NVFP4A16, EXL3 trellis, GGUF F32/F16,
  Q4_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XS, IQ4_XS, and an FP8 W8A8 slice.
- **External KV.** KV offload to CPU/disk and an `lm://` LMCache client, plus KV-cache events for
  external routers. Opt-in, off by default ([docs/KV-OFFLOAD.md](docs/KV-OFFLOAD.md)).
- **Observability.** Prometheus `/metrics` with vLLM's metric names, `/server_info`, `/tokenize`,
  `/detokenize`, and `/reset_prefix_cache`.

Per-capability lifecycle state, active gaps, and the next gate for each:
[Project status](#project-status).

</details>

## Supported models

Each row states its current correctness gate. Most completed gates compare tokens against the pinned
vLLM oracle on GB10. Where vLLM's greedy output is deterministic, the bar is strict token identity.
Where bf16 near-ties make vLLM self-inconsistent, the bar uses the ratified near-tie check. "Speed"
is a separate bar (match or beat vLLM on every axis).

**Gate models:** Qwen3.6-27B and Qwen3.6-35B-A3B (hybrid GDN + MoE, NVFP4), both token-exact, the
27B at or above vLLM throughput on every axis. **Also running:** Llama-3.x, Mistral, Qwen3/Qwen2
dense and MoE, DeepSeek-V2 and V4-Flash (MLA), GLM-4 and GLM-4.7-Flash, Laguna-S/XS-2.1,
Kimi-Linear-48B, Gemma-1 through Gemma-4, Phi-1 through Phi-4, OLMo-2, Granite-3, StableLM,
InternLM2/3, MiniCPM and MiniCPM3, Yi, OPT, plus Qwen3-VL and Qwen3.6-27B vision (image + video)
and Voxtral (audio).

<details>
<summary><b>The full architecture matrix</b> (44 registered architectures grouped by family)</summary>

| Architecture | Example checkpoint | GGUF | Correctness | Speed |
|---|---|:---:|---|---|
| Qwen3.5/3.6 hybrid (GDN + MoE) | Qwen3.6-27B, Qwen3.6-35B-A3B | 35B only | Token-exact | 27B at/above vLLM; 35B prefill-pending |
| Qwen3 / Qwen2 dense | Qwen3-4B, Qwen3-32B | dense qwen35 | Token-exact (near-tie-robust) | Speed-pending |
| Qwen3-MoE | Qwen3-Coder-30B-A3B | - | Token-exact (near-tie-robust) | Speed-pending |
| Llama-3.x dense | Llama-3.2-1B | - | Token-exact (near-tie-robust) | Speed-pending |
| Mistral dense | Mistral-7B-v0.3 | - | Token-exact | Speed-pending |
| OPT | OPT-125m | - | Strict token-exact | Speed-pending |
| DeepSeek-V2 (MLA) | DeepSeek-V2-Lite | - | Token-exact | Speed-pending |
| DeepSeek-V4-Flash (MLA + MHC + DSA) | DeepSeek-V4-Flash-GGUF (80.7 GB, single GB10) | keep-quant | Coherent (near-tie-robust) | Decode beats ds4 1.144x by default (byte-exact) |
| GLM-4 dense | GLM-4-9B-0414 | - | Token-exact | Speed-pending |
| GLM-4.7-Flash (MLA MoE) | zai-org/GLM-4.7-Flash | - | Token-exact (near-tie-robust) | Speed-pending |
| Laguna-S / Laguna-XS 2.1 (MoE) | poolside/Laguna-S-2.1-NVFP4 | NVFP4 + Q4_K | Near-tie (byte-exact) | vLLM parity+ 1.03x by default |
| Kimi-Linear-48B-A3B (KDA + MLA + MoE) | Kimi-Linear-48B-A3B | - | Near-tie (106/128) | 1.59 tok/s, default off |
| Nemotron-H hybrid (Mamba2 + GQA + MoE) | Nemotron-3.5-Lightning-30B-A3B-NVFP4 | NVFP4 | Host gate strict 96/96; GB10 rerun pending | Speed-pending |
| Muse Glimmer text + vision | Muse Glimmer 30B | GGUF text only | Text partial-depth gated; vision ungated | Speed-pending |
| Gemma-3 / Gemma-2 / Gemma-1 dense | gemma-3-1b-it, gemma-2-2b-it, gemma-2b | - | Token-exact (48/48 each) | Speed-pending |
| Gemma-4 text (Gemma4ForConditionalGeneration) | unsloth/gemma-4-E4B-it | - | Strict token-exact 32/32 (text path) | Speed-pending |
| OLMo-2 dense | OLMo-2-0425-1B | - | Token-exact (near-tie-robust) | Speed-pending |
| OLMo-3 dense | OLMo-3-1025-7B | - | Implemented, oracle-blocked | Speed-pending |
| Granite-3 dense | granite-3.3-2b-instruct | - | Token-exact (16/16) | Speed-pending |
| Phi-3 / Phi-4 dense | Phi-4-mini-instruct, phi-4 (14B) | - | Token-exact (near-tie 16/16) | Speed-pending |
| Phi-1 / Phi-2 dense | microsoft/phi-2 | - | Token-exact (16/16) | Speed-pending |
| StableLM dense | stablelm-2-1_6b | - | Token-exact (16/16) | Speed-pending |
| InternLM2 / InternLM3 dense | internlm2-chat-1_8b, internlm3-8b-instruct | - | Token-exact (16/16 each) | Speed-pending |
| MiniCPM dense | openbmb/MiniCPM-2B-sft-bf16 | - | Token-exact (16/16) | Speed-pending |
| MiniCPM3 (MLA) | openbmb/MiniCPM3-4B | - | Token-exact (16/16) | Speed-pending |
| Yi (Llama architecture) | 01-ai/Yi-Coder-1.5B-Chat | - | Token-exact (16/16) | Speed-pending |
| Command-R / Cohere dense | (HF-gated) | - | Implemented, gate-blocked | Speed-pending |
| Qwen3-VL (image + video) | Qwen3-VL-4B-Instruct | - | Strict token-exact 32/32 (image) | Speed-pending |
| Qwen3.6-27B vision (image + video) | Qwen3.6-27B | - | Strict token-exact 32/32 | Speed-pending |
| Voxtral (audio) | Voxtral-Mini-3B-2507 | - | Near-tie-robust (decoder 48/48 exact) | Speed-pending |
| Pooling / embeddings | synthetic `LlamaModel` fixture | - | Fixture-gated; real checkpoint pending | Not applicable |
| Parakeet ASR (CTC, RNNT, TDT) | Parakeet 0.6B / 1.1B families | - | Token IDs exact vs HF on retained runs | Speed-pending |
| **MiniMax-H3 (video + audio GENERATION)** | MiniMaxAI/MiniMax-H3 | Q4_K_M / NVFP4 | Renders 864x480 / 124f with audio | **34.6 s/step, one Jetson Thor** |

**Video + audio GENERATION is supported**, not just video *input*. MiniMax-H3 renders end to
end: prompt -> Qwen3-VL-32B encoder -> DiT denoise -> ViT3D video VAE + DAC/BigVGAN audio VAE
-> MP4 with a stereo track. The project's first DIFFUSION architecture (no KV cache, no
sampler, no logits); upstream is `vllm-project/vllm-omni`. Five conditioning modes and
`POST /v1/videos`. Use **Q4_K_M**. See [Project status](#project-status).

Compressed-tensors NVFP4A16 (W4A16) dense weights also load and compute natively
(RedHatAI/Qwen3-32B-NVFP4A16). Long-context RoPE (YaRN, Llama-3, LongRoPE, dynamic-NTK) and
sliding-window attention are gated feature-positive. The authoritative per-architecture list, bound
to the C++ registry (all 44 registered architectures with their tested checkpoint and gate, plus the
standalone audio/diffusion lanes and the inventoried-but-blocked archs), is in
[docs/FEATURES.md](docs/FEATURES.md); family-by-family lifecycle detail, including what is
hardware-blocked and why, is linked from [Project status](#project-status).

</details>

## Hardware

| Backend | Hardware | State |
|---|---|---|
| **CUDA** | GB10 / DGX Spark (sm_121a) | Runtime-gated. 27B at/above vLLM throughput, 35B prefill-pending |
| **CUDA** | Blackwell, Hopper, Ampere, Ada (sm_80 to sm_121a) | Per-arch builds pass; ten-SM archive candidate awaits hosted cubin audit; no runtime proof here |
| **CPU** | x86-64, arm64 | Correctness / CI reference. At or ahead of llama.cpp on every GGUF axis (SUPERSEDED, #1003), Arm i8mm tier |
| **Metal** | Apple Silicon | Two models end to end, 18 of 75 ops native. Prefill ahead of MLX-LM, warm total 97.6% with the MLX provider |
| **Vulkan** | Portable GPU | `opt-125m` STRICT token-exact; Qwen3.6-27B decode **matches llama.cpp Vulkan** (4.36 vs 4.35, denominator SUPERSEDED, #1003) |
| **ROCm** | AMD GPUs | Native EXL3 generation on gfx1151, matching the CPU reference. Discrete GPU correctness and competitive performance remain unverified ([evidence](.agents/specs/backend-rocm-exl3.md)) |
| **Tenstorrent** | Blackhole | OPT-125m strict 6/6; Qwen3 gate wired, full rerun pending |
| **Intel XPU / ANE** | Intel, Apple NPU | Spiked or roadmap |

Per-arch build flags, per-op coverage, and the quantization format table:
[docs/BUILD.md](docs/BUILD.md).

## Build

CMake (>= 3.24) and a C++20 compiler. The core has no ML dependencies, and the tree builds
-Werror-clean on gcc 14.2.

```sh
cmake -S . -B build && cmake --build build -j   # CPU: the correctness / CI reference
ctest --test-dir build
```

```sh
cmake -S . -B build-cuda -DVLLM_CPP_CUDA=ON   # NVIDIA GB10
cmake --build build-cuda -j
```

Triton-AOT cubins for the fast GDN path are vendored, so Python and Triton are needed only to
regenerate them, never to build or run them; a CUDA build ships them. Metal is auto-detected on
Apple hosts; Vulkan is opt-in with `-DVLLM_CPP_VULKAN=ON`. Every CMake option, per-backend recipe,
and the quantization format table: [docs/BUILD.md](docs/BUILD.md).

## Running inference (CLI)

```sh
build/examples/vllm-cli --model /path/to/Qwen3.6-27B --prompt "The capital of France is" --max-tokens 64
```

`vllm-bench` (throughput/latency harness) and `tokenize` (tokenizer smoke tool) ship alongside it.
All flags, including `--speculative-config`: [docs/USAGE.md](docs/USAGE.md).

### Multimodal INPUT and video GENERATION

Send supported image and audio content parts to `/v1/chat/completions`.
The [multimodal input guide](docs/guides/multimodal-input.md) lists each model's limits and provides
an image request example. HTTP paths have CPU tests with synthetic weights; real-checkpoint token
parity remains unverified. PNG/JPEG and video input are unavailable over HTTP. Video GENERATION:

```sh
build/examples/minimax-h3-gen --dit MiniMax-H3-FL2VA-Q4_K_M.gguf --dequant-bf16 \
  --encoder qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf --tokenizer tokenizer.json \
  --video-vae video_vae.safetensors --audio-vae audio_vae.safetensors \
  --prompt "A golden retriever runs across a sunlit beach" \
  --frames 124 --height 480 --width 864 --steps 50 --device cuda --out out.mp4
```

Weights: [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs)
(DiT + encoder) and [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (VAEs,
tokenizer). For SPEECH put the spoken line in the prompt. Recipe: [docs/USAGE.md](docs/USAGE.md).

## OpenAI-compatible server

```sh
build/examples/vllm-server --model /path/to/Qwen3.6-27B --port 8000 --max-num-seqs 32
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.completions.create(model="Qwen3.6-27B",
                                prompt="The capital of France is",
                                max_tokens=64).choices[0].text)
```

`/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`, `/ping`, `/version`,
Prometheus `/metrics` with vLLM's metric names, `/tokenize`, `/detokenize`, `/server_info`, and
`/reset_prefix_cache`. The full endpoint and flag tables are in [docs/USAGE.md](docs/USAGE.md).

For a production deployment, use [LocalAI](https://localai.io), which can embed engines like this
behind a model gallery, multi-model serving, the full OpenAI API surface, auth, and metrics.

## Use it as a library (C API)

Link `libvllm` and include [`include/vllm.h`](include/vllm.h): a flat, exception-free,
llama.cpp-style C ABI (currently `VLLM_ABI_VERSION 26`) suitable for `dlopen` / FFI. Check the
header for the version that your build provides.

```c
vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/path/to/model";

vllm_engine *engine = NULL;
if (vllm_engine_load(&mp, &engine) != VLLM_OK) { fprintf(stderr, "%s\n", vllm_last_error()); return 1; }

vllm_sampling_params sp = vllm_sampling_params_default();
sp.max_tokens = 64;               /* sp.temperature = 0.0 means greedy */

vllm_completion out;
if (vllm_complete(engine, "The capital of France is", &sp, &out) == VLLM_OK) {
    printf("%s\n", out.text);
    vllm_completion_free(&out);
}
vllm_engine_free(engine);
```

Structured output, chat with tools, tool and reasoning parser selection, speculative decoding,
prefix caching, custom logits processors, and engine sizing all have ABI surface. C++ consumers get
`LoadedEngine::FromModelDir(...)` handing back the same `LLMEngine` / `AsyncLLM` the server uses.
Both surfaces: [docs/USAGE.md](docs/USAGE.md).

## Why vllm.cpp

Every engine today makes you give something up. vLLM has the best serving core in the business, but
running it drags in a heavy Python / PyTorch / CUDA stack and assumes a datacenter. llama.cpp is
wonderfully light and runs anywhere, but it is built around one user at a time. SGLang has scheduling
ideas nobody else ships. MLX is excellent, and only on a Mac.

We do not think you should have to choose. vllm.cpp is the attempt to have all of it in one place:
**real continuous batching and paged KV, in a 66 MiB library, on the hardware you already own, with
the features the community keeps asking for, and none of the Python.** Where another engine has
solved something better, we port from it and cite the file we ported from rather than inventing our
own version.

What makes that checkable is the gate. A new architecture is not "done" here when it produces
plausible text; it is done when it emits **the same tokens as vLLM**, and it is not "fast" until it is
measured against the reference in that reference's own production configuration. When we fall short,
the number stays in the README and the label says *speed-pending*.

## Documentation

| Doc | What is in it |
|---|---|
| [Contribute](CONTRIBUTING.md) | Agent contribution guide |
| [docs/USAGE.md](docs/USAGE.md) | CLI, server endpoints/flags, C ABI and C++ API |
| [docs/BUILD.md](docs/BUILD.md) | Backend recipes, CMake options, hardware and quantization |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Measured grids, memory and repro recipes |
| [docs/FEATURES.md](docs/FEATURES.md) | Feature comparison: vLLM, SGLang and llama.cpp |
| [Project status](#project-status) | Stable public overview and contributor sources of truth |
| [docs/SGLANG-COMPAT.md](docs/SGLANG-COMPAT.md) | SGLang knobs and when to use them |
| [docs/SPECULATIVE-DECODING.md](docs/SPECULATIVE-DECODING.md) | MTP, DFlash, ngram |
| [docs/KV-OFFLOAD.md](docs/KV-OFFLOAD.md) | CPU/disk KV offload, LMCache and events |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Runtime environment variables |

The canonical project record is indexed by [AGENTS.md](AGENTS.md) and lives
under [`.agents/`](.agents/). See [current state](.agents/NOW.md),
[parity evidence](.agents/parity-ledger.md), the
[model inventory](.agents/model-matrix.md), and the
[portfolio roadmap](.agents/specs/roadmap-v1-completion.md).

## Credits, and what we borrow

vllm.cpp exists because other people built excellent engines first. Every implementation here is
grounded in upstream source and cites the file it was ported from, so this is where the work comes
from:

- **[vLLM](https://github.com/vllm-project/vllm)** is the reference this project is measured against
  and the origin of the serving core: continuous batching, block-paged KV, the V1 scheduler, sampling
  order, speculative decoding. It is also the oracle: every architecture here has to emit the same
  tokens vLLM does before it counts as working.
- **[SGLang](https://github.com/sgl-project/sglang)** contributed serving ideas nobody else ships,
  which we carry as documented toggles: RadixAttention, LPM cache-aware scheduling, and jump-forward
  decoding ([docs/SGLANG-COMPAT.md](docs/SGLANG-COMPAT.md)).
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** set the deployment standard we copied
  outright: one library, a flat C ABI, quantized weights straight off the shelf, and the vendored
  google/minja chat-template renderer it also ships. It is our CPU speed bar, and the reason GGUF is a
  first-class input here.
- **[MLX](https://github.com/ml-explore/mlx)** is the Apple Silicon bar, and optionally the GEMM
  provider on that path.
- Kernel work is ported from **CUTLASS**, **FlashInfer**, **Marlin**, and **TRT-LLM** rather than
  reinvented, cited per kernel in the porting inventory.

**A note on ggml:** vllm.cpp does not use ggml. It reads GGUF and follows llama.cpp's C ABI style, but
tensors, kernels, and dispatch are its own portable `vt::` runtime
([`include/vt/`](include/vt/)) with no ggml or PyTorch dependency at any point. ggml is a superb piece of engineering and llama.cpp
built the ecosystem this project plugs into; we needed a different tensor layer, that is all.

## Citation

If you use vllm.cpp, please cite this repository and the upstream vLLM project:

```bibtex
@software{vllm_cpp,
  title  = {vllm.cpp: a C++ inference engine porting vLLM},
  author = {Di Giacinto, Ettore},
  url    = {https://github.com/mudler/vllm.cpp},
  year   = {2026}
}
```

vLLM is by the vLLM project ([vllm-project/vllm](https://github.com/vllm-project/vllm)). Model
weights are governed by their own licenses, so check each model card.

## Author

Ettore Di Giacinto ([@mudler](https://github.com/mudler)).

## Trademarks

vllm.cpp is an independent community project. It is **not affiliated with, endorsed by, or sponsored
by** the vLLM project, the [PyTorch Foundation](https://pytorch.org/projects/vllm/), or the Linux
Foundation.

"vLLM" is a mark of its respective owners. It is used throughout this repository in its descriptive
sense only: to identify the upstream project that vllm.cpp ports, mirrors, and is gated against.
Nothing here should be read as a claim of ownership of that mark, or as a suggestion that the vLLM
project has reviewed, approved, or endorsed this software. The Apache License, Version 2.0 under
which vLLM is distributed grants no trademark rights (section 6), and none are claimed.

Other names and marks (llama.cpp, SGLang, MLX, CUTLASS, and the rest) belong to their respective
owners and are likewise used descriptively.

## License

vllm.cpp is released under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for
third-party attributions. The model weights keep their own licenses.

---

Built by the [LocalAI](https://github.com/mudler/LocalAI) team. If you want to run LLMs (and vision, voice, image, and video models) locally on any hardware with an OpenAI-compatible API, [give LocalAI a star](https://github.com/mudler/LocalAI).

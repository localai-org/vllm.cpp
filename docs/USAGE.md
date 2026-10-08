# Using vllm.cpp

The complete surface: the CLI, the OpenAI-compatible server, and the library
(C ABI and C++). The [README](../README.md) carries the quickstart; this page is
the reference behind it. Per-capability lifecycle state is
[README](../README.md); measured numbers are
[docs/BENCHMARKS.md](BENCHMARKS.md).

## Building

Full recipes are in [docs/BUILD.md](BUILD.md); the one rule worth stating here
is that the build must be **out-of-source**. Every command on this page assumes
a separate build directory:

```sh
cmake -S . -B build
cmake --build build -j
```

`cmake .` in the checkout is refused at configure time. It cannot work: the
example targets are named after the directories they are built from, so an
in-source build makes the linker write each executable over its own source
directory (issue #85).

### One ROCm-specific behaviour

ROCm builds register the full V1 sampler surface (temperature, top-k/top-p, min-p,
penalties, allowed-token masks, logprobs, random sample) so EngineCore does not
fatal with `no kernel for op` after prefill on AMD. Non-positive chat
`max_tokens` is treated as unset on all backends (Hermes `max_tokens=-1`).

Worth knowing before you read a hang as a bug in the tests: a build that sets no
`CMAKE_BUILD_TYPE` floors **HIP device code** at `-O1` and prints a configure
line saying so. At `-O0` the ROCm runtime starts a hostcall listener the kernels
never use, and its teardown can deadlock at process exit — every test passes,
`Status: SUCCESS!` prints, and the process never returns
([#132](https://github.com/mudler/vllm.cpp/issues/132)). Setting a build type,
or putting your own `-O` in `CMAKE_HIP_FLAGS`, overrides it.

### ROCm op coverage is incremental (and throws are by design)

The ROCm backend registers native ops family by family
([#41](https://github.com/mudler/vllm.cpp/issues/41)); landed GDN slices so far:
the indexed state I/O pair (`kGdnStateGather`/`kGdnStateScatter`), the causal
conv1d pair (`kCausalConv1dFwd`/`kCausalConv1dUpdate`, incl. the exact-chunks
descriptor form Qwen3.5 prefill passes), the fused post-conv glue
(`kGdnPostConv`), the gated-delta recurrence (`kGdnPrefill`/`kGdnDecode`,
portable scan), and the norm-gate/preamble ops (`kRmsNormGated`,
`kSigmoidGateBf16`, `kAttnQkNormRopeGate`) — the full set Qwen3.5-class
GDN-hybrid models call. Compressed conv/SSM state (bf16, the vLLM
`mamba_cache_dtype` default) is advertised via the
`SupportsCompressedConvState`/`SupportsCompressedGdnState` backend probes.
MoE-path coverage is partial: `MoeRouterTopK` (f32/bf16 logits, ungrouped
softmax, no bias) and `MoeSiluMul` are native; the remaining chain
(`kSharedExpertGate`, `kMoeCombine`/`kMoeCombineGate`, and the grouped quant
expert GEMM) is not registered yet, so MoE-bearing models still throw on
those ops. On a
discrete card there is no CPU fallback tier, so a model whose layers call an op
that is not registered yet fails loudly with `vt: no kernel for op N on device
type 5` — that is the memory-safety design working, not a crash. Run with
`VT_OP_PROVIDER_STATS=1` to see which ops resolve native.

### CUTLASS is fetched as headers only

`-DVLLM_CPP_CUTLASS_FETCH=ON` downloads CUTLASS v4.5.0 and stops there: the
sources are populated, but CUTLASS's own CMake project is never configured. Every
consumer in this tree `-isystem`s `${VLLM_CPP_CUTLASS_DIR}/include`, and nothing
links a CUTLASS CMake target, so its `tools/`, `library/`, `examples/` and
`tests/` targets are never built.

This is why no `-DCUTLASS_ENABLE_TOOLS=OFF` is needed. Configuring those targets
used to be required and could fail on its own — building for `sm_80` under CUDA
13 dies inside CUTLASS `tools/library` with duplicate `sm_100f` flags
([#193](https://github.com/mudler/vllm.cpp/issues/193)) — for a build product we
never used.

## Confirming which CUDA architecture a build targets

`CMakeCache.txt` is now a reliable answer. Configuring with
`-DVLLM_CPP_CUDA_ARCHITECTURES=<arch>` writes that value into
`CMAKE_CUDA_ARCHITECTURES` in the cache, so the two agree:

```sh
grep '^CMAKE_CUDA_ARCHITECTURES' build-cuda/CMakeCache.txt
```

Which fast paths a given architecture compiles is decided by the CUDA feature
table, not by the arch string alone. `110` (Jetson Thor) builds the portable
kernels plus the vendored Marlin NVFP4 W4A16 GEMM; the CUTLASS FP4/FP8 paths and
`fp4-mma` stay off there because no kernel body exists for it. `cmake -P
cmake/CudaArchFeaturesTest.cmake` prints the resolution for any target list
without a GPU or a CUDA toolkit.

It previously reported the toolkit's detected default (typically `75`) no matter
what was requested, because the project set the variable without writing it back
to the cache. Only the report was wrong — the emitted gencode always followed the
requested value — but it sent a contributor looking in the wrong place
([#168](https://github.com/mudler/vllm.cpp/issues/168)). The `build.ninja`
gencode line remains the ground truth if you want to double-check.

## Using more than one engine in a process

Constructing a `LoadedEngine`, destroying it, and constructing another in the
same process is supported, including on CUDA. Each engine's device-resident MoE
and Marlin constants are owned by the weights they describe and are released
with them.

Before, that state lived in process-lifetime caches keyed on the *address* of a
weights block, so a second engine could land on a freed block's address and
reuse device pointers that had already been freed. Nothing crashed — the CUDA
context is never torn down, so the pointers stayed mapped — it simply produced
corrupted or zeroed output tokens, intermittently
([#237](https://github.com/mudler/vllm.cpp/issues/237)).

## Starting an agent-assisted contribution

Run `scripts/agent-start.py` first. It reports an inherited worktree role or,
for a new contributor with no declared role or explicit intent, prints the
welcome that the agent should relay. An explicit request can use
`--intent operator|helper|read-only` and a helper `--row ID`. Follow its printed
claim action, rerun it after declaration, then run `scripts/agent-preflight.sh`.
The entrypoint is non-interactive and does not mutate the checkout.

The operator role is a coordinator, and **several may run at once**:
`scripts/agent-role.py claim operator` records this worktree and is never
refused, `scripts/agent-role.py show` lists the other live coordinators, and
`scripts/agent-role.py release` removes only this worktree's record. What keeps
concurrent coordinators safe is that `main` is never force-pushed, so a plain
`git push` refuses any non-fast-forward.

## Running inference (CLI)

`vllm-cli` runs a one-shot completion through the C ABI. Source:
[`examples/cli/main.cpp`](../examples/cli/main.cpp).

```sh
build/examples/vllm-cli \
  --model /path/to/Qwen3.6-27B \
  --prompt "The capital of France is" \
  --max-tokens 64
```

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (config.json + tokenizer.json + safetensors) |
| `--prompt "<text>"` | (required) | Prompt text |
| `--tokenizer-config <path>` | (none) | Override `tokenizer_config.json` |
| `--max-tokens N` | `16` | Max tokens to generate |
| `--temperature T` | `0.0` | Sampling temperature (`<= 0` means greedy) |
| `--top-p P` | `1.0` | Nucleus cutoff |
| `--top-k K` | `0` | Top-k (`0` means all) |
| `--seed S` | (unset) | RNG seed (enables seeded sampling) |
| `--stream` | off | Stream token deltas to stdout |
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's flag. See [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) |
| `--repeat N` | `1` | Load once, then run N blocking completions. Use it to read a warm decode tok/s without paying model load each time. Not supported with `--stream`, which falls back to 1 |
| `-h`, `--help` | | Print usage and exit |

Two more example binaries ship alongside it:

- `vllm-bench` ([`examples/bench/main.cpp`](../examples/bench/main.cpp)), a
  throughput/latency harness taking `--model`, `--dataset-path`,
  `--num-prompts`, `--input-len`, `--output-len`, `--concurrency`,
  `--max-num-batched-tokens`, and `--num-blocks`. It pretokenizes before timing
  and atomically publishes each concurrency wave. Set
  `VT_BENCH_PRETOKENIZE=0` for the timed-string rollback; the report names the
  resolved mode. Its `--output-wait` mode
  defaults to `poll`; the explicit `blocking-c1` diagnostic is valid only with
  `--concurrency 1` and is not a general multi-request policy.
- `tokenize` ([`examples/tokenize/main.cpp`](../examples/tokenize/main.cpp)), a
  tokenizer smoke tool taking `<tokenizer.json | model.gguf> <corpus.txt>`.
  GGUF `tokenizer.ggml.pre` names accepted: `qwen35`, `qwen2`, `llama-bpe`,
  `gpt-4o` / `llama4` / `kanana2` / `talkie` (the GPT-4o / o200k family),
  `joyai-llm`, `deepseek-llm`, `deepseek-v3`, `laguna`. Any other name is
  refused by name rather than aliased onto a near-miss regex.

### Which HF tokenizers load

The loader reads `<model_dir>/tokenizer.json`. When that file is absent, it reads
`<model_dir>/tokenizer/tokenizer.json` instead, and the same applies to
`tokenizer_config.json`. The root file always wins. This lets a snapshot such as
`convaiinnovations/laya`, which keeps its tokenizer in `tokenizer/`, load as
downloaded, without a symlink.

A checkpoint's `tokenizer.json` is accepted when its `pre_tokenizer` is one this
build recognises. Recognition is by exact regex or pipeline shape, not by model
name, so a checkpoint from any vendor loads if it carries one of these:

| family | shape | examples |
|---|---|---|
| Qwen3.6 | one `Split` regex, single-codepoint `\p{N}`, `\p{M}` folded into letter runs | Qwen3.6-27B |
| Qwen2/Qwen3 classic | as above without `\p{M}` awareness | Qwen3-0.6B, Qwen3-Coder |
| Llama-3 | `\p{N}{1,3}` digit groups, no `\p{M}` awareness | Llama-3 family |
| Tekken (Mistral) | case-aware letter runs, single-codepoint `\p{N}`, `/` in the punct tail | Mistral-Nemo-Instruct-2407 |
| GPT-4o / o200k | the same case-aware letter runs, plus o200k's contraction SUFFIX and `\p{N}{1,3}` | Muse Glimmer (pre `llama4`), GPT-4o |
| GPT-2 byte-level | `ByteLevel(use_regex=true)` with no explicit `Split` | OPT, GPT-2 |
| DeepSeek | a seven-stage `Sequence` pipeline, not one alternation | DeepSeek-V2/V3 |
| SentencePiece | `Metaspace` + byte-fallback vocab | Mistral-7B-v0.3 |

An unrecognised one fails loudly at load with `tokenizer: unrecognized
pre-tokenizer split regex: <regex>`, rather than tokenizing incorrectly. If you
hit that, the printed regex is what a new pattern would have to match.

Note that Mistral ships **two** unrelated tokenizer families: Mistral-7B-v0.3 is
SentencePiece, while Mistral-Nemo is Tekken, a byte-level BPE whose regex is
tiktoken's `o200k_base` with the contraction group removed and `\p{N}{1,3}`
reduced to `\p{N}`. Support for one says nothing about the other. Putting those
two edits back gives the GPT-4o row above, so the two share one scanner's
character classes but stay separate patterns: they disagree on `don't` and on
every digit run longer than one.

### How much memory a Vulkan load needs

On a unified-memory device (a DGX Spark) the Vulkan heap and system RAM are the
same bytes, so budget roughly **the checkpoint size plus about 5%**, plus your KV
pool. Measured on GB10: Qwen3.6-27B bf16 (50.89 GiB on disk) peaks at 53.4 GiB of
process RSS. Reading the checkpoint also fills the page cache with about the file
size; that is reclaimable and does not need to be budgeted, but it does make
`MemFree` look alarming during a load. Use `MemAvailable`, not `MemFree`, to
decide whether a model fits. `VT_VULKAN_ALLOC_STATS=1` prints the running device
total and the `/proc` context if you need to see where it goes.
A Tenstorrent build (`-DVLLM_CPP_TENSTORRENT=ON`) needs TT-Metalium and TT-NN
on `CMAKE_PREFIX_PATH`. Blackhole currently runs OPT-125m through the shared
engine and has the Qwen3-0.6B correctness gate wired with device-specific
goldens. The full Qwen3 16x16 gate remains pending because paged attention is
still host-bound. This is an active correctness backend, not a performance
backend. See [README](../README.md) and the
[Tenstorrent backend spec](../.agents/specs/tenstorrent-backend.md).

A Vulkan build (`-DVLLM_CPP_VULKAN=ON`) adds three kernel-measurement binaries.
They exist so a Vulkan tuning knob can be A/B'd in ONE binary, which is this
project's benchmark protocol, and each one prints WHICH kernel variant it ran so
a silent fallback cannot post a plausible number:

- `vulkan-gemm-ab`, cooperative-matrix versus the portable scalar GEMM
  (`VT_VULKAN_COOPMAT=0` picks the arm). Takes `M K N [reps]`.
- `vulkan-dispatch-floor`, one op swept across a 65,536x range of element counts,
  to separate per-dispatch overhead from real kernel cost.
- `vulkan-gemv-ab`, the decode GEMV swept over the (k, n) shapes a 27B model
  actually dispatches, with `VT_VULKAN_GEMV_ROWS` / `VT_VULKAN_GEMV_PACK` /
  `VT_VULKAN_GEMV_UNROLL` selecting the arm. Takes `[reps] [warmup] [GB/s roof]`
  and reports GB/s against that roof. Set `VT_VULKAN_DISPATCH_STATS=1` so it
  reports GPU-timestamp time rather than wall clock; see
  [ENVIRONMENT.md](ENVIRONMENT.md) for what each knob does and what it measured.

### Quantized checkpoints: which weight forms load

**TQ1_0 and TQ2_0 GGUF files cannot load yet.** The
[GGUF reader](../src/vllm/model_executor/model_loader/gguf_reader.cpp) rejects their
tensor types as unknown, including in Vulkan builds. The
[ternary kernels](FEATURES.md#ternary-kernels) do not remove this loader limitation.

### How long a load takes, and how to see where it goes

`VT_LOAD_STATS=1` prints one line per load phase with its wall time, plus the
bytes the load actually MOVED: `host_copy` (materialized into a host buffer),
`borrowed` (read in place from the file mapping) and `device_upload`. The byte
line is printed twice, once when the weights are loaded and once at exit, because
the device uploads are lazy and happen at first use.

```
$ VT_LOAD_STATS=1 build/examples/vllm-cli --model /path/to/Qwen3.6-27B --prompt hi --max-tokens 1
[vt load] mmap+header       0.027 s
[vt load] weights          12.268 s
[vt load] bytes@load-end  host_copy=31.162 GiB borrowed=18.936 GiB device_upload=0.000 GiB
[vt load] bytes@exit      host_copy=31.162 GiB borrowed=18.936 GiB device_upload=50.098 GiB
```

A weight the device consumes verbatim is READ FROM the checkpoint mapping rather
than copied into a host buffer first, so it is moved once instead of twice; that
is `borrowed` above, and on this 27B it is 37.8% of the model and worth 1.54x on
the load phase warm, 1.61x cold. Tensors that are merged (`qkv`, `gate_up`),
transposed (`lm_head`) or dequantized at load are not verbatim and still copy.
`VT_LOAD_DIRECT_UPLOAD=0` turns the direct path off in the same binary; the
loaded bytes, and therefore the tokens, are identical either way.

Safetensors payloads are byte-addressed and do not promise natural scalar
alignment. Borrowed BF16/F16/F32 inputs therefore use defined byte-copy loads;
an odd payload offset neither forces a host copy nor changes the loaded bits.

`device_upload` counts every single-source weight upload: the bf16/fp8 weights
through `ResidentWeight` and the compressed-tensors NVFP4/MXFP4 `packed`/`scale`
residents through `ResidentNvfp4`. It does NOT yet count the merged fp4 operands
(`qkv`, `gate_up`) or the Marlin repack residents, which build one device buffer
out of several host tensors; on a bf16 checkpoint like the one above there are
none, so the line is the whole model. Once a weight has been uploaded its source
pages are released, and that release is independent of `VT_ADOPT_DEVICE_BYTES` --
switching the adoption off leaves the release on.

### Quantized checkpoints: which `lm_head` forms load

Publishers do not agree on how weights are stored, and a single repo can change
it between revisions (one 27B "NVFP4" repo silently became FP8 throughout).
The table below is about `lm_head`; the same three forms are accepted for the
attention, MLP and `linear_attn` projections, in both compressed-tensors
(`weight_packed` + `weight_global_scale`) and ModelOpt (`weight` +
`weight_scale_2`) naming. For the Qwen3.6 dense family we accept all three
forms in use, so pick a checkpoint by its quality, not by its head:

| `lm_head.weight` | Companion tensors | Seen in |
|---|---|---|
| `BF16` | none | `unsloth/Qwen3.6-27B-NVFP4` @`890bdef7` |
| `F8_E4M3` | `lm_head.weight_scale` (per-output-channel or per-tensor) | `unsloth/Qwen3.6-27B-NVFP4` @`ccdaab7e` |
| `U8` NVFP4 | `lm_head.weight_scale` + `weight_scale_2` (ModelOpt) or `weight_global_scale` (compressed-tensors) | `nvidia/Qwen3.6-27B-NVFP4` |

The head is dequantized to BF16 at load, so all three cost the same memory once
running. Any other dtype fails at load with a message naming what it saw.

A `modelopt_mixed` checkpoint (`nvidia/Qwen3.6-27B-NVFP4`, and the 35B-A3B that
shares the tower) keeps its `linear_attn` input projections in FP8 W8A8, and
those two per-layer projections are packed into ONE merged `in_proj_qkvz` GEMM,
mirroring vLLM's `MergedColumnParallelLinear`. The merge only fires when the two
shards carry a bitwise-identical per-tensor `input_scale`, since one GEMM
quantizes the activation once; a checkpoint whose scales differ keeps the two
separate GEMMs automatically. `VT_GDN_MERGED_QKVZ_FP8=0` restores the two GEMMs
in the same binary.

### Architectures that resolve but refuse to run

A few architectures are registered so their config and weight layout are
accounted for, while their forward is deliberately not implemented. Pointing the
CLI or server at one of these loads far enough to resolve the architecture and
then fails with a message naming the missing piece, rather than emitting wrong
tokens quietly.

| Architecture | Why it refuses |
|---|---|
| `KimiK3ForConditionalGeneration` | Needs ~1.56 TB (MXFP4); no host here can run it |

This is a deliberate state, not a bug: registering the architecture is what lets
the config parse and weight-name mapping be tested before the forward exists.

### Muse Glimmer: exactly what has been checked

`MuseGlimmerForCausalLM` / `MuseGlimmerForConditionalGeneration` are not in that
table: both towers forward and the perception encoder is wired, so an image or
video prompt runs instead of refusing. What has been *measured* is much narrower
than "it works", so it is worth stating precisely.

- The text tower ran on real tensors from the released 30B checkpoint at
  **reduced depth — 4 of its 52 layers.** Its **5 prefill argmax positions** are
  identical to a standalone torch transcription of the upstream source and to
  HF's own `muse_glimmer` implementation. The full-depth 52-layer bf16 arm has
  generated once, 4 ungated tokens (2026-08-11); no bf16 token gate exists.
- Those are argmax positions from a single prefill, not generated tokens.
  **bf16 multi-step decode is ungated**, and the sliding window across steps is
  untested on either arm. The GGUF Q4_K_M arm IS token-gated at full depth
  against llama.cpp `b10451` (next section).
- Even at reduced depth this is agreement with independent transcriptions of the
  same upstream source, not agreement with the model's own runtime. The vLLM
  parity pin `a7c23ac96d` registers `muse_glimmer`, but no bf16 gate against it
  has run (it needs a GPU lease).
- The perception encoder has **one reference run** on the released tensors
  (2026-09-30): f32 soft tokens within 4.07e-5 relative of a torch
  transcription of the pinned formulas; bf16 within the gate's tolerance of the
  reference's own bf16 arm (worst row cosine 0.967 vs 0.970). The image processor is not ported, and no image-to-text result
  exists.
- Nothing has run end to end through the server, and **no speed number exists
  against vLLM**; a secondary llama.cpp bar exists (#333).
- The ATEM reasoning and tool parsers are ported and unit-gated, but at the
  server's default `skip_special_tokens: true` the framing tokens they key on
  (`<|start|>`, `<|message|>`, `<|eom|>`, `<|eot|>`) are stripped before the
  parser sees the text. Channel scoping is therefore an **open gap at server
  defaults** — see [FEATURES.md](FEATURES.md) and
  [the spec](../.agents/specs/muse-glimmer.md) §6.7.

### Nemotron 3 Nano Omni: which weights, and what has been checked

`NemotronH_Nano_Omni_Reasoning_V3` serves IMAGE input over the OpenAI API. Audio
and video are refused by name, and no end-to-end token gate against vLLM exists
yet. See [the model page](models/nemotron-nano-omni.md) for the measured bounds.

| Arm | Artifact | Size | Status |
|---|---|---|---|
| bf16 | [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) @ `e5e9932441de940c9a62185c870ea5bcd4cd24e2`, 17 shards (shard 1 sha256 `de952574c9189925ad15f8cf164184117b6e5eec2d8b7f092e1268c1f0872244`) | 66 032 308 536 B (61.5 GiB) | loads; image path gated per stage on the real tensors |
| NVFP4 | [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4) @ `16993199e436da4ba75ddc410855f87e0d996ee6` | 22 409 034 192 B (20.9 GiB) | **refused**: per-module ModelOpt scheme (FP8 `o_proj` and shared experts, bf16 `lm_head`) not resolved |
| GGUF | none | | **refused**, as for `NemotronHForCausalLM` |
| 12B VL V2 | [nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16) @ `ca9543b126e8bf3176916d3d305ccc415f89fd4d` | | **refused**: static InternVL tiling not ported |

The image processor sizes each image from the engine's `max_model_len`, as
upstream does, so a short `--max-model-len` shrinks the patch grid.

### North (`Cohere2MoeForCausalLM`): which weights, and what has been checked

The port was built and gated against the first-party bf16 release. There is
one arm; every other arm is refused at load with a message that names it.

| artifact | size | source |
|---|---|---|
| `model-00001-of-00049.safetensors` ... `model-00049-of-00049.safetensors`, with `config.json`, `model.safetensors.index.json` and the tokenizer files | 60,968,607,744 bytes (56.8 GiB) of bf16 weights, 18,730 tensors | [CohereLabs/North-Mini-Code-1.0](https://huggingface.co/CohereLabs/North-Mini-Code-1.0) at revision `d11e61a842617a22dc328552fa5bb86231ee4f37` |

The resident size is the weight size: 56.8 GiB of bf16, plus a 128 MiB RoPE
table for the 500,000-position context. The loader copies each expert's gate
and up rows into one owner, so plan for the whole checkpoint in host memory.

Refused arms:

- **GGUF.** llama.cpp `b10451` defines `cohere2moe`; the loader arm is owed.
- **Quantized siblings** (FP8, W4A16, NVFP4, anything with a
  `quantization_config`). Load the bf16 checkpoint.
- **`use_qk_norm: true`.** The pinned vLLM `cohere2_moe.py` has no q/k norm.
- **The EAGLE drafter** (`North-Mini-Code-1.0-eagle`) and the EAGLE3
  auxiliary hidden states. Not ported.
- **`--disable-sliding-window`.** The pinned model file computes
  `sliding_window + 1` for every sliding layer and cannot run without it.

What has been measured, on CPU only:

- Three tiny checkpoints that switch every mechanism on and off, loaded through
  the production loader, match a torch transcription of the pinned
  `cohere2_moe.py` within 6e-7 relative in f32 and within the bf16 envelope in
  bf16, for prefill and for decode through the paged cache past the window.
- Layers 0 (dense, RoPE), 1 (sliding window, MoE) and 4 (full attention, NoPE,
  MoE) of the real checkpoint, fetched by HTTP range, match the same
  transcription within 8e-6 relative in f32.
- The model loads through the engine registry and decodes through the runner.
- **No token gate against vLLM exists yet**, and no speed number exists on any
  axis. Both need a GPU lease and the full checkpoint
  ([the spec](../.agents/specs/cohere2-moe.md) `## Owed`).

## OpenAI-compatible server

`vllm-server` is a small HTTP server speaking the OpenAI API. Source:
[`examples/server/main.cpp`](../examples/server/main.cpp) and
[`src/vllm/entrypoints/openai/`](../src/vllm/entrypoints/openai/).

```sh
build/examples/vllm-server --model /path/to/Qwen3.6-27B --port 8000 --max-num-seqs 32
```

The install component and deterministic archive target both stage from install
rules rather than copying the build tree:

```sh
cmake --build build --target vllm-server-stage
cmake --build build --target vllm-server-archive
build/release/stage/bin/vllm-server --help
```

The basic CMake archive under `build/release/` includes the version, configured
backend, OS, and host architecture in its name. It is a developer package. The
release workflow separately produces host-ABI-specific archives with a
manifest, `VERSION`, SPDX SBOM, notices, licenses, and detached checksum and
provenance sidecars; no release download is claimed until that workflow has
completed on a release tag.

To reproduce the W1 heterogeneous CUDA archive candidate, configure the exact
release architecture set. Portable translation units compile for all ten SMs;
architecture-specific kernels compile only for their supported intersection.
`VLLM_CPP_TRITON` is left to its default, which is `ON` here — a fat CUDA build
embeds every vendored per-arch cubin tree and selects one by exact SM at
runtime, which is what the released archive contains:

```sh
cmake -S . -B build-cuda-fat -G Ninja \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES='80;86;87;89;90a;100a;103a;110;120a;121a' \
  -DVLLM_CPP_CUTLASS_FETCH=ON
cmake --build build-cuda-fat --target vllm
python3 scripts/check-cuda-fat-gencode.py \
  --compile-commands build-cuda-fat/compile_commands.json \
  --library build-cuda-fat/libvllm.a
```

The release workflow applies this audit to independently linked x86_64 and
arm64 host executables, packages each as a preview `cuda-fat` archive, and then
runs the extracted-archive validator. Each archive must contain all ten SM
images and the six available exact-SM Triton AOT namespaces; the manifest keeps
runtime evidence separate per SM. These build-only preview candidates are not
a downloadable release claim until the tagged workflow publishes them.

The complete primary download matrix and its runtime boundaries are documented
in [RELEASES.md](RELEASES.md). A manual workflow dispatch runs all eight tuples
without publication. An exact version tag runs the same build, produces
`release-index.json` and `RELEASE_INDEX.md` from the verified archive manifests,
attests the archive bytes, and publishes every archive/checksum/provenance
triplet through the protected release environment.

Inside the workflow, generated archives live under `release-assets` (and then
`unverified/release-assets` / `verified/release-assets`). This transient root is
deliberately separate from the checkout's tracked `assets/` directory, so exact
handoff validation sees only the planned archive/checksum/provenance triplets.
The release filenames and published eight-tuple inventory are unchanged.

### Selecting an x86 CPU ISA tier

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
listed in [ENVIRONMENT.md](ENVIRONMENT.md).

### NVFP4 dense sinks

The `E=1` dense NVFP4 projections run on vLLM's own dense Marlin GEMM rather
than the single-expert grouped-MoE route, which pays `moe_align` bookkeeping and
row padding for a problem that has neither. `VT_MARLIN_DENSE` covers the single
projections and `VT_MARLIN_DENSE_PAIR` the fused shared-expert gate_up sink;
both default ON, opt out with `=0`. The pair sink was the last one still on the
MoE route: enabling it measured **+1.31% at c8 and +1.38% at c4** on
`nvidia/Qwen3.6-35B-A3B-NVFP4` with both SACRED gates unmoved. Only the
throughput changes; the routed experts still use the grouped MoE kernel, which
is where they belong.

The shared expert's `down_proj` keeps its bf16 output rather than upcasting to
f32 (`VT_SHARED_DOWN_BF16`, default ON, opt out with `=0`). Both consumers widen
bf16 in-kernel — which is exact — and re-round through bf16 on store, so the
f32 form was writing and re-reading a whole `[T,H]` buffer for a value it
already had. The change is bit-identical and worth **+2.05% at c8**.

### The NVFP4 output head

On a Qwen3.6 dense checkpoint whose `lm_head` is stored NVFP4 (ModelOpt
`weight`/`weight_scale`/`weight_scale_2`, or compressed-tensors
`weight_packed`/`weight_global_scale`) the head is kept **packed** and the logits
GEMM runs on it directly, as vLLM does. Nothing is dequantized at load, so the
head costs `K*N/2 + K*N/16` bytes instead of `2*K*N`, about 0.715 GB instead of
2.543 GB on `nvidia/Qwen3.6-27B-NVFP4` (measured peak host RSS 21.06 to 19.36
GiB, a 1.70 GiB saving on CUDA; the figure is owed a re-measurement after
`ENG-LOAD-DIRECT-UPLOAD` changed the RSS accounting).

That accounting is CUDA's. A backend with no fp4 GEMM (CPU, Vulkan, Metal, HIP,
Tenstorrent) has to multiply against a dequantized bf16 copy, so on those the
head costs the packed bytes **plus** one `2*K*N` operand, built once when the
model is prepared rather than per call — 0.666 + 2.368 = 3.034 GiB on the same
checkpoint. The sign of the change therefore depends on the backend: on Vulkan,
which used to stage a host bf16 head *and* a device copy of it, the head goes
4.736 to 3.034 GiB, the same **-1.70 GiB**; on plain CPU it goes 2.368 to 3.034,
a **+0.67 GiB** regression, paid once instead of rebuilding 2.368 GiB on every
decode step as that backend did before. Only the head is kept that way; every
other NVFP4 projection dequantizes per call, so a quantized tower is never
expanded in memory. The head runs W4A16 under both namings: the on-disk
activation divisor next to it (`input_scale`, or `input_global_scale` in the
compressed-tensors spelling) is NOT consumed unless `VT_MODELOPT_W4A4=1`,
matching vLLM, which deletes it on this path. Set `VT_LMHEAD_FP4=0` for a
same-binary A/B that restores the old dequantize-at-load owner. BF16, FP8, GGUF
and `tie_word_embeddings` heads are unaffected by either setting.

### Validating a staged release archive

Release verification reads only a freshly extracted archive, never files from
the build tree. Pass the archive together with its final-byte SHA256 and SLSA
provenance sidecars:

```sh
python3 scripts/validate-release-archive.py \
  --archive vllm.cpp-0.0.2-linux-x86_64-glibc-cpu.tar.gz \
  --checksum vllm.cpp-0.0.2-linux-x86_64-glibc-cpu.tar.gz.sha256 \
  --provenance vllm.cpp-0.0.2-linux-x86_64-glibc-cpu.tar.gz.provenance.json \
  --forbid-path "$PWD/build"
```

The validator checks the content allowlist, executable and host ABI, manifest,
`VERSION`, SPDX SBOM, licenses, ELF dependencies and RPATH/RUNPATH, extracted
`--help`/`--version` smokes, and backend-specific CUDA or adaptive-CPU claims.
The digest and provenance are sidecars because both describe the final archive
bytes; placing either inside those bytes would create a self-reference.

The CPU release helper is the reproducible entry point used by CI. It requires
an explicit artifact tuple, architecture, channel, build directory, libc ABI,
a feature-poor QEMU userspace emulator, and a feature-rich runner. x86_64 uses
the SHA256-pinned Intel SDE installed by `scripts/install-intel-sde.sh` so the
AVX-512 tier is really executed even when the host lacks AVX-512. The gate then
executes the baseline and proves rich-tier refusal under the feature-poor QEMU
model before metadata can be generated:

```sh
SOURCE_SHA=$(git rev-parse HEAD) \
VERSION=0.0.2 \
SOURCE_DATE_EPOCH=$(git show -s --format=%ct HEAD) \
EVIDENCE_URL=https://github.com/mudler/vllm.cpp/actions/runs/EXAMPLE \
scripts/build-cpu-release.sh \
  linux-x86_64-glibc-cpu x86_64 stable build-release-cpu-x86 \
  2.39 /usr/bin/qemu-x86_64 /tmp/intel-sde/sde64
```

The corresponding arm64 tuple is `linux-aarch64-glibc-cpu`. The only literal
static tuple is the CPU-only `linux-x86_64-musl-cpu-static` experiment; normal
CPU and accelerator archives are static-core bundles with audited host runtime
dependencies.

## Container images

Published to one GHCR package with the lane in the tag:
`ghcr.io/mudler/vllm.cpp:<version>-cuda`, `-vulkan`, `-cpu`, plus the moving
`:latest-<lane>`. The bare `:latest` is the **cpu** lane, so pulling it on a
machine with no accelerator gets a working server rather than a library-load
failure. Every lane is a `linux/amd64` + `linux/arm64` manifest.

The entrypoint is `vllm-server`, so flags go straight after the image name and
the server keeps its own default of `0.0.0.0:8000`:

```sh
docker run --rm -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/mudler/vllm.cpp:latest \
  --model /models/Qwen3.6-35B-A3B
```

For the CUDA lane, the GPU driver comes from the host through the container
runtime; the image carries only the CUDA *runtime* libraries it links:

```sh
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/mudler/vllm.cpp:latest-cuda \
  --model /models/Qwen3.6-35B-A3B
```

`/models` is the weights mount and `/cache` is the tokenizer/HF cache; the
container runs as uid 1000, so `/cache` must be writable by it if you bind-mount
one. `ffmpeg` is installed in every lane, so `/v1/videos` works out of the box —
a deliberate difference from the tarballs, which never vendor it because they
are extracted onto a host that already has a `PATH`.

macOS Metal and MLX have no image and never will: there is no macOS container
runtime and no Metal passthrough. ROCm has no image until its backend compiles.

### Building and validating an image locally

One Dockerfile, one target per lane. The builder stage runs the same
`scripts/build-*-release.sh` the release workflow runs, so there is no second
build definition to drift:

```sh
docker build -f docker/Dockerfile --target cpu \
  --build-arg VERSION=0.0.1 \
  --build-arg SOURCE_SHA=$(git rev-parse HEAD) \
  --build-arg JOBS=$(nproc) \
  -t vllm-cpp:local-cpu .
```

Then gate it. Without `--model` the validator checks configuration and layout
and says plainly that the image has no runtime evidence; with one it also boots
the server, requires `/health` and `/version`, runs the image's own declared
healthcheck, and requires a clean SIGTERM shutdown:

```sh
python3 scripts/validate-container-image.py \
  --image vllm-cpp:local-cpu --lane cpu --version 0.0.1 \
  --model /path/to/opt-125m
```

`scripts/check-container-matrix.py` keeps `release/container-matrix.json` and
the Dockerfile agreeing about lanes, tags and digest-pinned bases;
`scripts/check-container-workflow.py` holds the publish workflow to its
least-privilege stages. Both run in preflight and CI.

To exercise the release pipeline without publishing anything, trigger its
manual entry point:

```sh
gh workflow run release.yml --ref main
```

Manual runs are always dry runs. Publication additionally requires an exact
`v<project-version>` tag, a release matrix whose required lanes are all marked
ready, successful verification and attestation jobs, and approval of the
protected `release` environment. Build and verification jobs have read-only
repository permissions; only attestation receives OIDC authority, and only the
final protected job receives `contents: write`.

Any OpenAI client works by pointing its `base_url` at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.completions.create(model="Qwen3.6-35B-A3B",
                                prompt="The capital of France is",
                                max_tokens=64).choices[0].text)
```

### Endpoints

Registered in
[`src/vllm/entrypoints/openai/api_server.cpp`](../src/vllm/entrypoints/openai/api_server.cpp).

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
| POST | `/v1/videos` | Start a video generation job, returns `{id, status}` (MiniMax-H3) |
| POST | `/v1/videos/sync` | Same, but runs to completion before answering |
| GET | `/v1/videos/{id}` | Job status |
| GET | `/v1/videos/{id}/content` | The finished MP4 (`video/mp4`) |

Both completion endpoints return prompt log probabilities in non-streaming
responses. See [Prompt log probabilities](reference/server.md#prompt-log-probabilities)
for request values, response fields, streaming restrictions, and verification limits.

The four `/v1/videos` routes are registered **only** when the server was started
with `--video-dit`; without it they are absent (404) and the server is identical
to one built without video support. See
[MiniMax-H3: video + audio generation](#minimax-h3-video--audio-generation).

### `max_tokens`: what a non-positive value means

Some clients (Hermes among them) send `max_tokens: -1` to mean "no client-side
limit". A non-positive `max_tokens` — or `max_completion_tokens` on
`/v1/chat/completions`, which takes precedence — is treated as **unset**, not as
an error and not as a clamp to some constant. Unset then generates up to
`max_model_len` minus the prompt length, mirroring vLLM.

That distinction is load-bearing for long-context requests: substituting a
constant would cap exactly the request that asked to be left unlimited, and the
client would see `finish_reason: length` with no way to tell it apart from a
limit it set itself. Use `VT_SERVER_MAX_NEW_TOKENS` when you want a serving-side
ceiling.

### Which token ids stop a generation

The engine resolves a primary end-of-sequence (EOS) ID and merges secondary
EOS IDs into `stop_token_ids`. `ignore_eos: true` suppresses both primary and
secondary EOS IDs. See the [server reference's EOS rules](reference/server.md#which-token-ids-stop-a-generation)
for config precedence, tokenizer fallback, and `min_tokens` masking.

### Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (safetensors or `.gguf`) |
| `--host H` | `0.0.0.0` | Bind host |
| `--port P` | `8000` | Bind port |
| `--served-model-name N` | model dir basename | Model id in `/v1/models` and responses |
| `--tokenizer-config F` | `<dir>/tokenizer_config.json` | Chat template / tokenizer config |
| `--block-size N` | `32` | KV block size |
| `--num-blocks N` | `256` | KV blocks |
| `--max-model-len N` | `0` (config default) | Max sequence length |
| `--max-num-seqs N` | `32` | Max concurrent sequences (also sizes the HTTP worker pool). Was `8`, which put a c8 client exactly on the batch ceiling; vLLM's own default is 1024, which we do not mirror because this also caps the padded decode-graph set. On a GDN/Mamba model under speculative decoding this also multiplies the recurrent state, which is sized `max-num-seqs x (k+1)`; an unservable budget is refused at load with the arithmetic |
| `--max-num-batched-tokens N` | `0` (per-arch default) | Per-step token budget |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | model default | Override automatic prefix caching |
| `--scheduling-policy fcfs\|priority\|lpm` | `fcfs` | Scheduler policy (`lpm` is the SGLang cache-aware policy, see [docs/SGLANG-COMPAT.md](SGLANG-COMPAT.md)) |
| `--enable-radix-attention` / `--disable-radix-attention` | model default | SGLang-named alias for the prefix-cache toggle |
| `--enable-jump-forward` | off | Jump-forward decoding for structured output (token-unique subset) |
| `--enable-force-include-usage` | off | Force the usage block in responses |
| `--tool-call-parser <name>` | `hermes` | Tool-call dialect (41 names over 37 families). `auto` detects from the chat template, `none` disables. For `gemma4`, OpenAI chat uses the text-seam parser (wrapped `<\|tool_call>` **or** bare `call:NAME{ARGS}`) so free-form / detokenized tool bodies still become `tool_calls` |
| `--reasoning-parser <name>` | `none` | Reasoning parser (`think_auto`, `deepseek_r1`, `deepseek_v3`, `holo2`, `mistral`, `minimax_m2`, `minimax_m2_append_think`, `step3`, `olmo3`, `muse_glimmer`). `auto` detects, `none` disables |
| `--kv-transfer-config '<json>'` | (unset) | External KV connector, same JSON as vLLM's flag. See [docs/KV-OFFLOAD.md](KV-OFFLOAD.md) |
| `--speculative-config '<json>'` | (unset) | Speculative decoding (`mtp`, `dflash`, `ngram`), same JSON as vLLM's flag. `dspark` speculates on the Qwen3.6 gate models (native + Speculators drafts), token-identically to speculative-off, but is not gated on speed (currently ~2% behind at c1). A GGUF target, or a target with no aux multi-tap, is refused by name (`SPEC-DSPARK`). See [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) |
| `--enable-log-requests` / `--disable-log-requests` | on | Log each incoming request. Mirrors vLLM's flag of the same name |
| `--enable-log-outputs` | off | Also log the generated output, not just the request |
| `--max-log-len N` | `256` | Truncate logged prompts and outputs to N characters |
| `--enable-metrics` / `--disable-metrics` | on | Serve the metrics endpoint |
| `--enable-thinking` / `--no-enable-thinking` | off | Set the `enable_thinking` chat-template variable for templates that gate a reasoning block on it (Gemma-4 and friends). Our spelling of vLLM's `--default-chat-template-kwargs enable_thinking` |
| `--verbose`, `-v` | off | Verbose server logging |
| `--cuda-profile-graph-replays N` | `0` (off) | Trace-only diagnostic: arm the CUDA-graph-replay profiler and stop after N replays, printing a pid to signal with `SIGUSR2`. Requires a build with `VT_BENCH_PROFILE_CONTROL` |
| `--cuda-profile-graph-batch N` | `16` when replays are armed | Batch size the profiler traces. Must not exceed `--max-num-seqs` |
| `-h`, `--help` | | Print usage and exit |

#### Context length vs the KV pool

The KV pool holds `--num-blocks × --block-size` tokens — `256 × 32 = 8192` by
default. A request longer than that can never be scheduled, so the engine
refuses it early rather than leaving it in the waiting queue forever. Two checks
do that, mirroring vLLM:

- **At startup.** If `--max-model-len` is given and the pool cannot hold one
  sequence that long, the server exits with the sizes and the flags that close
  the gap (vLLM's `_check_enough_kv_cache_memory`). If it is **not** given, the
  serving length is auto-fitted down to what the pool holds and logged
  (vLLM's `_auto_fit_max_model_len`) — so raising `--num-blocks` is what buys a
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

## System 1 decisions with `/v1/systemone`

A decision model answers typed questions about a `state`. Depending on the
model, it scores options directly or generates an answer token.
The server registers `POST /v1/systemone` when the model directory resolves
to a decision architecture. The `vllm_decide` C API accepts the same body.
See the [C API reference](reference/c-api.md#decisions-and-option-scoring).

```sh
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice. Please refund the duplicate.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
    "department": {"type": "choice", "instructions": "Which department?",
                   "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Routine", "Urgent", "Emergency"]}}}'
```

Each model page carries its checkpoint, its answer semantics, and what is
measured: [CLM](models/clm.md), [GLiNER2.5-Decide](models/gliner25-decide.md),
[xor](models/xor.md), [Nimble](models/nimble.md), and [Tev1](models/tev1.md).
Tev1 is a generation model: a directory whose `config.json` names `Tev1Model`
serves `/v1/systemone` beside the chat routes, and scores each question's answer
letters through the engine's own scheduler.

### Nimble: the exact weights

Nimble is an adapter, so it runs only after `scripts/convert-nimble.py` merges
it into its base. The [Nimble page](models/nimble.md) has the commands.

| arm | repo @ revision | file | sha256 |
|---|---|---|---|
| base, BF16 | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | four `model.safetensors-0000N-of-00004` shards | as published at that revision |
| adapter (Ollama `nimble`), T=1.0 | `bespokelabs/Bespoke-Nimble-9B` @ `bd792f44ec8e265be861bfcdf4e05967ffe0e858` | `adapter_model.safetensors`, 173,188,512 bytes | `29ef39b072dee97287947455337879c1e916705c2f727287922a2d81f5e2f20a` |
| adapter (earlier release), T=2.179078721266035 | `bespokelabs/Bespoke-Nimble-9B-v2` @ `4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe` | `adapter_model.safetensors`, 173,188,512 bytes | `1bd126be997be6d9a0c25ce483ccf858c31b3422d480c33f02ba47b614be68ae` |

GGUF k-quant arms are not implemented; `NimbleModel` refuses a GGUF source by
name.

### CLM: the exact weights

CLM ships its two projection heads as a torch pickle beside an unmodified
base, so it runs only after `scripts/convert-clm.py` writes a `ClmModel`
directory. The [CLM page](models/clm.md) has the commands.

| arm | repo @ revision | file | sha256 |
|---|---|---|---|
| base, BF16 | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` | five `model-0000N-of-00005.safetensors` shards | as published at that revision |
| heads, F32 | `Contrastive-LM/CLM-v0.1-8B` @ `e939398d4556fcd9400c76fa8c5a513202f42b0a` | `CLM_v0.1-8B.pt`, 75,557,149 bytes | `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5` |

GGUF k-quant arms are not implemented; `ClmModel` refuses a GGUF source by
name.

## Muse Glimmer 30B from a GGUF k-quant

The text tower loads from a `muse-glimmer`-architecture GGUF, so the 30B model
runs from a ~17 GB k-quant instead of a ~60 GB bf16 checkpoint. Point `--model`
straight at the file; the config comes from the GGUF's own metadata, so no
`config.json` is needed:

```sh
./build/vllm-server --model /path/to/muse-glimmer-30B-kquant-17gb.gguf
```

Both published k-quants load (`muse-glimmer-30B-kquant-17gb.gguf` and the mixed
per-tensor `muse-glimmer-30B-kquant-dynamic.gguf`). Standard GGUF residency
knobs apply (`VT_GGUF_KEEP_QUANT`, `VT_GGUF_MMAP`, `VT_CPU_REF`); `o_proj`, the
attention output gate, `down_proj` and the merged `gate_up` stay quantized, while
the merged QKV, `lm_head` and the embedding table expand to bf16 because the
shared forward consumes them in a form a block encoding cannot take.

Three caveats:

- **The k-quant is token-gated against llama.cpp `b10451`** on the same file
  (`test_muse_glimmer_gguf_paged_engine`, `VLLM_MUSE_GGUF_PARITY=<file>`,
  2026-09-30): 16/16 battery prompts inside the 500 mnat near-tie band over 32
  greedy tokens, 10/16 token-exact. Quantization-matched agreement, not a bf16
  or vLLM claim. Two defects had to be fixed first: the GGUF tokenizer gap
  ([#347](https://github.com/mudler/vllm.cpp/issues/347), pre `llama4` = the
  GPT-4o / o200k family) and the converter's Q/K RoPE row permutation
  ([#359](https://github.com/mudler/vllm.cpp/issues/359), which produced
  `" is is is ..."`).
- **Image and video need the bf16 safetensors.** The released
  `mmproj-kquant.gguf` ships its patch embedding without the `patch_temporal`
  axis, so half the weight is not in the file; loading it is refused by name.
- **No speed number exists against vLLM in any weight format.** The parity pin
  `a7c23ac96d` registers `muse_glimmer`, but no vLLM run exists on this fleet;
  the secondary llama.cpp bar is #333.

Set `VLLM_MUSE_GGUF=<file>` (or `VLLM_MUSE_GGUF_LOAD=<file>` for the full
materialization) to run `test_muse_glimmer_gguf` against a real checkpoint;
without them the gate runs off committed header-only manifests.

## MiniMax-H3: video + audio generation

### The exact weights (so a render is reproducible)

Five files. The DiT and encoder are community GGUF quantisations; the two VAEs and
the tokenizer come from the official checkpoint.

| file | size | source |
|---|---|---|
| `MiniMax-H3-FL2VA-Q4_K_M.gguf` | 19.9 GB | [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) |
| `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | 14.6 GB | [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) |
| `vae/diffusion_pytorch_model.safetensors` | 5.2 GB | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/video_vae/` |
| `audio_vae/model.safetensors` | 0.6 GB | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/audio_vae/` |
| `tokenizer.json` | 7 MB | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) `FL2VA/tokenizer/` |

Take each VAE's `config.json` from the same directory as its weights: they carry the
per-channel `latents_mean` / `latents_std` and the temporal `clip_length` /
`token_drop`, and the decode is wrong without them.

**Use Q4_K_M, not Q3_K_M.** H3's split-half RoPE produces channel-wise magnitude
outliers that 3-bit cannot hold. In a controlled A/B (same prompt, seed, code and
VAEs, only the DiT quantisation changed) Q3_K_M gave a murky silhouette under a
visible lattice and Q4_K_M gave a photoreal close-up. The full bf16 release is
66.3 GB across 13 shards if you want to go further.

Higher-precision arms that exist but are not the default: NVFP4
([lilcheaty/MiniMax-H3-NVFP4](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4))
and the original bf16 weights under `FL2VA/transformer/`.

### The PRUNED checkpoints — more precision for the same footprint

The community `pruned` variants **are supported and are drop-in**: pass one to
`--dit` exactly as you would an unpruned file. Nothing else about the command
changes.

They are not lossily pruned. AdaLN modulation dominates the unpruned parameter
count — `adaln_proj` alone is 13.04B of 33.12B (39.4%) — because the model
projects a 5376-wide conditioning vector into modulation parameters in every one
of the 50 blocks. But modulation depends only on the timestep, so that projection
is almost entirely redundant, and the pruned form replaces it with a `[1025, 8]`
timestep table feeding an 8-wide `adaln_proj.linear`. 13.04B parameters become
0.04B and the DiT drops from 33.12B to 20.11B, with the modulation path kept at
full precision.

The practical consequence: **a pruned Q8_0 costs about what our unpruned Q4_K_M
costs.**

| file | size | source |
|---|---|---|
| `minimax_h3_fl2va_pruned-Q8_0.gguf` | 21.4 GB | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) |
| `minimax_h3_ref2va_pruned-Q8_0.gguf` | 21.4 GB | same repo — the `ref2va` partition |
| `minimax_h3_{fl2va,ref2va}_pruned-{Q2_K,Q3_K,Q4_K,Q5_0,Q6_K}.gguf` | 6.7-16.6 GB | same repo |
| `minimax_h3_{fl2va,ref2va}_pruned_nvfp4.safetensors` | 12.5 GB | [lilcheaty/MiniMax-H3-NVFP4](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4) |

The partition rule below still applies: a `fl2va` file serves `t2va` and `fl2va`,
a `ref2va` file serves `ref2va`.

**What is actually verified, and what merely exists.** The distinction matters
because a render takes hours before it tells you anything:

| arm | status |
|---|---|
| **Q4_K_M** | **VERIFIED end to end** — every render in this doc, on BOTH partitions (t2va + fl2va on FL2VA, ref2va on REF2VA). Use this. |
| Q3_K_M | verified BAD (the A/B above): murky silhouette under a lattice |
| bf16 (66.3 GB, 13 shards) | loader + device streamer implemented and gated, but **CPU-only** verification — no end-to-end GPU render has been done |
| NVFP4 | exists; loads (unpruned and pruned) |
| **pruned Q8_0** | **loads and renders** — the A/B is in `.agents/specs/minimax-h3.md` section 8.21 |
| pruned Q6_K / Q5_0 / Q4_K / Q3_K / Q2_K ([unsloth](https://huggingface.co/unsloth/MiniMax-H3-GGUF)) | load through the same path; only Q8_0 has been rendered |

### The trap: this checkpoint does not serve every task

**`MiniMax-H3-FL2VA-Q4_K_M.gguf` is the FL2VA partition. It serves `t2va` and
`fl2va` — NOT `ref2va`.** H3 ships two independently-served DiT partitions and
the task must match the one you loaded; upstream's `_resolve_task` raises on the
mismatch.

Pass a reference image against this file and you get a task/partition mismatch.
It does not fail loudly — it renders, and the render is *wrong*: a coloured
diagonal lattice over the whole frame, worse the larger the canvas. Measured on
one prompt and canvas (1344x768 / 124f), as a period-16 seam ratio where 1.15 is
clean:

| configuration | seam ratio |
|---|---|
| ref2va against FL2VA (the mismatch) | 2.28 |
| t2va against FL2VA (correct) | **1.19** |

The small-canvas case is what makes this expensive to spot: at 864x480 the same
mismatch measures 1.15 and looks acceptable, so the bug only becomes obvious at
the resolution you actually want.

Pass `--partition fl2va` explicitly. The driver mirrors upstream's raise, so a
mismatch is rejected at the CLI rather than silently rendered.

For a reference-image render you need the **Ref2VA** partition instead, and the
one to use is **`MiniMax-H3-REF2VA-Q4_K_M.gguf`** (19.9 GB,
[realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs)) —
the same quantisation as the FL2VA file above, and verified coherent:

```sh
build/examples/minimax-h3-gen \
  --dit MiniMax-H3-REF2VA-Q4_K_M.gguf --dequant-bf16 --partition ref2va \
  --encoder qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf --tokenizer tokenizer.json \
  --prompt "..." --ref-image subject.ppm \
  --video-vae video_vae.safetensors --video-vae-config video_vae_config.json \
  --audio-vae audio_vae.safetensors --audio-vae-config audio_vae_config.json \
  --frames 124 --height 512 --width 512 --steps 50 \
  --device cuda --out out.mp4 --workdir /tmp/h3
```

**Do NOT use the NVFP4 Ref2VA weights.** `minimax_h3_ref2va_nvfp4_full` renders the
multicolour patch grid, and it took three investigations to establish that this is the
QUANTISATION and not the ref2va path: the identical reference-row assembly, packed-block
layout and denoise loop render coherently on Q4_K_M (period-16 seam **1.13**, VAE-input
latent adjacent-cell cosine **0.8526**). Ref2VA on Q4_K_M is a working mode; Ref2VA on
NVFP4 is not.

### Writing the prompt (read this first)

Two things decide whether you get what you asked for, and neither is obvious.

**To get SPEECH, ask for it and supply the line.** The model generates video and
audio jointly, so a prompt describing a silent performance produces room tone and
ambience, which is correct but not what most people expect. Say that the character
talks, describe the voice, and put the words in the prompt:

```
It is TALKING to the camera: its mouth moves clearly in sync with its speech,
in a dry, deadpan tone.

It says, clearly and audibly: "Michael scheduled another all-hands.
It is about the printer. Again."

Audio: a single clear voice, close-miked, with quiet room tone underneath.
```

That prompt produced audio an ASR pass transcribed back word for word. A prompt
that only described expressions and sighs produced ambience at about 13 dB lower
level and no speech at all.

**Refer to references BY TAG in the prompt text.** A reference is bound by naming
it, not merely by being passed on the command line. Use `<Picture i>`, `<Video k>`
and `<Audio j>`, numbered from 1 per type, matching the order you pass them:

```
<Picture 1> is a cyan llama mascot wearing white sunglasses.

A talking-head interview. The subject is the llama from <Picture 1>, sitting in a
grey office chair ...
```

Other prompt notes: frame count runs on the 17n+5 grid at 24 fps, and the trained
range is roughly 124 to 362 frames (about 5 to 15 seconds). Text rendered *inside*
the video (signage, wordmarks) is the model's weakest area and will often come out
malformed; composite real logos in afterwards.

`/v1/videos` generates video with sound through the MiniMax-H3 diffusion model.
It speaks **OpenAI's Sora video shape**, so an OpenAI client works against it
unmodified, and it keeps the richer native knobs alongside.

```sh
build/examples/vllm-server --model /path/to/Qwen3.6-27B \
  --video-dit /path/to/h3-dit.gguf --video-vae /path/to/video-vae.safetensors \
  --audio-vae /path/to/audio-vae.safetensors \
  --video-vae-config video_vae/config.json --audio-vae-config audio_vae/config.json \
  --video-encoder /path/to/h3-encoder.gguf
```

```python
video = client.videos.create(model="sora-2-pro", prompt="a cat on a skateboard",
                             size="1280x720", seconds="8")
while client.videos.retrieve(video.id).status not in ("succeeded", "failed"):
    time.sleep(5)
open("out.mp4", "wb").write(client.videos.download_content(video.id).read())
```

### Request fields

| Field | Spelling | Meaning |
|---|---|---|
| `prompt` | both | Required. The text conditioning |
| `model` | OpenAI | Recorded and echoed back. A name this server does not serve is a `warning` on the job, never a rejection: the video model is chosen at startup |
| `size` | OpenAI | `"<width>x<height>"`, e.g. `"1280x720"`. Whole pixels, both positive |
| `seconds` | OpenAI | Duration, as a number or a numeric string (`8` and `"8"` both work) |
| `input_reference` | OpenAI | The image the video starts from. A filesystem path or a `data:` URL |
| `metadata` | OpenAI | Free-form string map, passed through untouched. Two keys are acted on: `input_reference_video` and `input_reference_audio` (see below) |
| `width`, `height` | native | Output geometry in pixels |
| `duration` | native | Duration in seconds |
| `task` | native | `t2va`, `fl2va`, `ref2va`; resolved from the inputs when omitted |
| `num_frames`, `num_inference_steps`, `flow_shift`, `audio_flow_shift`, `seed` | native | The H3 generation knobs. Accepted at the top level or nested under `extra_params` |

**Precedence.** When a body carries both spellings of one value, the **native
field wins**: `width`/`height` beat `size`, `duration` beats `seconds`. That
direction keeps every request that parses today meaning exactly what it meant
before. Both spellings are validated either way, so a malformed `size` is a 400
even when explicit `width`/`height` would have overridden it.

**`input_reference` maps to fl2va first-frame conditioning.** OpenAI documents
it as the image the generated video starts from, which is what fl2va expresses:
the supplied image is pinned as frame 0 of the output. H3's other image mode,
ref2va, prepends whole reference images as their own blocks (subject or style
guidance that never becomes a frame), so it stays reachable only through the
native `task` field and the `minimax-h3-gen` CLI. Two limits: the image must be
a **binary PPM (P6)** (no PNG or JPEG codec is vendored, the same residual the
chat multimodal path carries), and it must already be at the output resolution
(no image resampler is vendored). A mismatch is refused with the resolved
geometry in the message.

### Video and audio references (`metadata`)

H3 supports three reference modalities and OpenAI's schema has a slot for one,
so the other two enter through `metadata`, the standard OpenAI free-form string
map. Strict clients tolerate it, and no invented top-level field breaks their
schema validation. Unknown metadata keys are passed through untouched.

```jsonc
{
  "prompt": "the same scene, at dusk",
  "metadata": {
    "input_reference_video": "/tmp/vllm_h3_videos/job0",  // DIR of frame_%06d.ppm
    "input_reference_audio": "/tmp/voice.wav"             // 16-bit PCM WAV, or a data: URL
  }
}
```

`input_reference_video` is a **directory of `frame_%06d.ppm`**, which is exactly
what this server and `minimax-h3-gen` write, so one run's frames chain straight
into the next request. It is not a container file: no demuxer is vendored.

**A video reference is SILENT.** `MiniMaxH3EncodeReferenceVideo` emits a
`kVideoAudio` block with `ref_audio_t == 0`, so the clip contributes no sound of
its own. Supplying `input_reference_audio` alongside it attaches the audio to
that same block (one block carrying both, the layout upstream builds); without
it the reference is picture only. That is a real limitation, not an omission.

**Legal combinations.** fl2va keyframes and ref2va reference blocks are
exclusive in the pipeline itself
([`minimax_h3_pipeline.cpp`](../src/vllm/model_executor/models/minimax_h3_pipeline.cpp)),
so the request parser enforces the same rule and returns a 400 naming the
offending pair rather than dropping a reference you supplied.

| `input_reference` | `metadata.input_reference_video` | `metadata.input_reference_audio` | |
|---|---|---|---|
| (none) | (none) | (none) | t2va, prompt only |
| image | (none) | (none) | fl2va, the image is frame 0 |
| (none) | clip | (none) | ref2va, silent video reference |
| (none) | (none) | WAV | ref2va, audio reference |
| (none) | clip | WAV | ref2va, one block carrying both |
| image | clip and/or WAV | | **400**: keyframe and reference conditioning are exclusive |

The video reference needs `--video-vae` (the encoder half of the same file) and
the audio reference needs `--audio-vae`; both load lazily, once, on the first
request that asks for them.

### The job lifecycle

`POST /v1/videos` returns immediately with `{"id": "vid_1", "status": "queued"}`;
generation is minutes long, so the synchronous twin `POST /v1/videos/sync` exists
for scripts that would rather block. `GET /v1/videos/{id}` reports `queued`,
`running`, `succeeded` (with `output_path`) or `failed` (with `error`).

`GET /v1/videos/{id}/content` returns the finished MP4 with
`Content-Type: video/mp4`. An unknown id is a 404; a job that has not finished is
a **409** naming its current status rather than a truncated file; a failed job is
a 500 carrying its failure; an output that has since vanished from disk is a 500
rather than a 200 with zero bytes.

The library never spawns a process, so generation and muxing enter through a
caller-supplied `VideoRunner` callback (`examples/server/main.cpp` supplies one
that invokes `ffmpeg`, path configurable with `--video-ffmpeg`).

## Consuming it as a library (C ABI)

Link `libvllm` (static or shared) and include [`include/vllm.h`](../include/vllm.h).
It exposes a flat, exception-free, llama.cpp-style C ABI (`VLLM_ABI_VERSION 30`)
for `dlopen`, FFI, and LocalAI integration. Call `vllm_abi_version()` at runtime
to detect a header and library mismatch.

```c
#include "vllm.h"

vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/path/to/model";

vllm_engine *engine = NULL;
if (vllm_engine_load(&mp, &engine) != VLLM_OK) {
    fprintf(stderr, "%s\n", vllm_last_error());
    return 1;
}

vllm_sampling_params sp = vllm_sampling_params_default();
sp.max_tokens = 64;               /* sp.temperature = 0.0 means greedy */

vllm_completion out;
if (vllm_complete(engine, "The capital of France is", &sp, &out) == VLLM_OK) {
    printf("%s\n", out.text);
    vllm_completion_free(&out);
}
vllm_engine_free(engine);
```

The ABI covers lifecycle, blocking and streaming completion, non-blocking
concurrent requests, memory helpers, and diagnostics. The table lists early ABI
additions. See the [C API reference](reference/c-api.md#recent-abi-additions)
for newer interfaces and their limits:

| ABI | Adds |
|---:|---|
| v2 | Structured output (JSON schema, JSON object, regex, choice, GBNF) |
| v3 | Chat with tools and chat templates |
| v4 | Tool-parser selection |
| v5 | Reasoning-parser selection |
| v6 | Speculative decoding |
| v7 | Prefix caching (tri-state) |
| v8 | Custom logits processors |
| v9 | Engine sizing: chunked-prefill token budget, scheduling policy, external KV connector / LMCache |
| v10 | Jump-forward decoding (tri-state, default off) |
| v11 | Audio transcription through `vllm_transcribe` |
| v12 | Video and audio generation through `vllm_video_*` |
| v13 | Pre-tokenized completion through `vllm_complete_tokens` |
| v14 | Explicit device selection (`auto`, CPU, or CUDA) |
| v15 | Embeddings through `vllm_embed` |
| v16 | Absolute KV-cache memory sizing |
| v17 | The OpenAI server as a thin ABI client through `vllm_server_main` |

Chat templates render through the vendored google/minja engine, the same
renderer llama.cpp ships.

### Speaker diarization

ABI 30 adds `vllm_diarization_load()` for a separate Nemotron-3-Diarization
GGUF model. The API accepts WAV paths or mono float32 PCM at 16 kHz. Use
`vllm_diarization_free()` for results and `vllm_engine_free()` for the model.

The combined transcription API needs an additional ASR context that an ordinary
transcription load does not guarantee. No verified combined-ASR loading recipe
is provided here. Read the [audio API limits](reference/c-api.md#speaker-diarization-and-attributed-transcription)
before using these functions. The bundled server has no diarization startup option.

## Consuming it from C++

The higher-level surface lives under [`include/vllm/`](../include/vllm/).
`LoadedEngine::FromModelDir(...)`
([`entrypoints/model_loader.h`](../include/vllm/entrypoints/model_loader.h))
hands back either the synchronous `LLMEngine`
([`v1/engine/llm_engine.h`](../include/vllm/v1/engine/llm_engine.h)) or the async
`AsyncLLM` ([`v1/engine/async_llm.h`](../include/vllm/v1/engine/async_llm.h)) that
the server itself uses.

```cpp
vllm::entrypoints::EngineParams ep;
ep.enable_prefix_caching = true;
ep.policy = vllm::SchedulerPolicy::kLPM;
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, ep);
```

The underlying portable tensor runtime is `vt::` ([`include/vt/`](../include/vt/)),
which carries no ggml or PyTorch dependency.

`Sampler`'s `logprobs_mode` selects which tensor the returned logprobs are read
from, and all four of vLLM's values now work: `raw_logprobs` (the default) and
`raw_logits` are snapshotted before any logits processor runs, so they describe
the MODEL's distribution; `processed_logprobs` and `processed_logits` are taken
after temperature and top-k/top-p, so they describe the distribution actually
SAMPLED from — a token top-k masked away reads `-inf` there and its true value
under the raw pair. It is selectable by constructing a `Sampler` directly; there
is no config, CLI or request field for it yet.

`SamplingParams::logprobs` accepts `-1` for "every vocab entry", as vLLM's does;
it returns the same gathered shape a finite count returns, one entry per vocab id
per position.

Over HTTP the same `-1` reaches the chat surface: `{"logprobs": true,
"top_logprobs": -1}` is accepted, as in vLLM, and returns every vocab entry for
each generated token. No numeric range is enforced on either surface — vLLM's
`check_logprobs` request validation and its `max_logprobs` model cap are not
ported yet. Two consequences: `{"logprobs": -1}` on the **completion** surface
returns empty `top_logprobs` maps where vLLM answers `400`, and an out-of-range
count is not rejected. Both are tracked by
[issue #249](https://github.com/mudler/vllm.cpp/issues/249).

## Multimodal input (image, video, audio to text)

Multimodal input is served over the **OpenAI API**, not the CLI. `vllm-cli` is text-only:
`--model --prompt --max-tokens --temperature --top-k --top-p --seed --stream
--speculative-config --tokenizer-config`.

Start the server with a multimodal model, then send content parts on
`/v1/chat/completions`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

client.chat.completions.create(model="Qwen3.6-27B", messages=[{"role": "user", "content": [
    {"type": "text",      "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<...>"}},
]}])
```

Accepted part types (`src/vllm/entrypoints/openai/chat_mm.cpp`):

| part type | modality |
|---|---|
| `image_url` | image |
| `video_url` | video |
| `input_audio` / `audio_url` | audio |

## MiniMax-H3: video + audio generation


Renders an MP4 with a stereo track. Weights: a GGUF DiT (use **Q4_K_M**), the Qwen3-VL-32B
encoder, and both VAEs.

```sh
build/examples/minimax-h3-gen \
  --dit MiniMax-H3-FL2VA-Q4_K_M.gguf --dequant-bf16 --partition fl2va \
  --encoder qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf --tokenizer tokenizer.json \
  --prompt "A golden retriever runs across a sunlit beach, waves crashing behind it" \
  --video-vae video_vae.safetensors --video-vae-config video_vae_config.json \
  --audio-vae audio_vae.safetensors --audio-vae-config audio_vae_config.json \
  --frames 124 --height 768 --width 1344 --steps 50 \
  --device cuda --out out.mp4 --workdir /tmp/h3
```

`--partition` is REQUIRED and names the partition the checkpoint you passed
actually serves — see the trap above. This is the command every render in this
document was produced with: Q4_K_M DiT and encoder, `--dequant-bf16`, task
**t2va** (no reference image), the 1344x768 default canvas, 124 frames, 50 steps.

Cost, so you can plan: **~176 s per step** at 1344x768 / 124f on a 20-SM sm_110
device, so a 50-step render is about **2.5 hours** plus roughly 30 minutes of
weight loading. Dropping to 512x512 costs ~15 s/step (~13 minutes end to end),
which is the right canvas for iterating on a prompt before committing to a full
render. `--dequant-bf16` holds the DiT as bf16 (~66 GB resident); `--keep-quant`
is the low-memory arm.

Conditioning modes, all optional and mutually exclusive where noted:

```sh
--first-frame start.ppm --last-frame end.ppm   # pin the first and/or last frame (fl2va)
--ref-image subject.ppm                        # reference image, repeatable (ref2va)
                                               # NOT served by the FL2VA checkpoint above --
                                               # needs a Ref2VA partition (see the trap)
--ref-video prev_workdir/                      # reference clip, reads frame_%06d.ppm
--ref-audio voice.wav                          # reference audio
--noise-aug 0.9                                # how hard a keyframe is pinned (1.0 = exact)
```

Reference frames are binary PPM, which is what this tool also **writes**, so one run's `--workdir`
feeds straight back in as `--ref-video` and clips chain. Convert anything else with
`ffmpeg -i in.png -pix_fmt rgb24 out.ppm`.

Worked reference renders, all on the **Ref2VA** checkpoint (`--partition ref2va`); the flags
below replace `--ref-image` in the command above:

```sh
# a SUBJECT carried into a new scene, from one still
--ref-image subject.ppm

# a reference CLIP: a directory of frame_%06d.ppm. A previous run's --workdir already
# has that layout, so clips chain without converting anything:
--ref-video /tmp/h3/            # reads /tmp/h3/frame_000000.ppm, frame_000001.ppm, ...

# reference AUDIO: 16-bit PCM WAV. Resample first -- the audio VAE is 32 kHz:
#   ffmpeg -i voice.mp3 -ac 1 -ar 32000 -c:a pcm_s16le voice.wav
--ref-audio voice.wav
```

To build a `--ref-video` directory from an arbitrary clip:

```sh
mkdir -p /tmp/refclip && ffmpeg -i source.mp4 -pix_fmt rgb24 /tmp/refclip/frame_%06d.ppm
```

Reference conditioning is **ref2va only**. On the FL2VA checkpoint these flags are refused
rather than silently ignored, which is the guard from the task/partition mirror.

Useful for measurement: `--prompt-embeds` replays text conditioning saved earlier, so two
checkpoints can be compared on byte-identical conditioning. `VT_H3_DUMP_DIR=<dir>` writes the
latents that enter each VAE (`vae_input_video_latent.f32`, `vae_input_audio_latent.f32`) plus
the pre-denormalize audio rows — that is how a render is checked numerically rather than by
eye, and it is byte-inert when unset.

(`--denoise-only`, `--dump-params` and `--save-embeds` belonged to the pre-fold driver and
were removed when the example became a thin ABI client; see the header comment in
`examples/minimax_h3_gen/main.cpp`.)

Served over HTTP too: pass `--video-dit` (plus the VAEs and configs) to `examples/server` and
`POST /v1/videos`, `POST /v1/videos/sync` and `GET /v1/videos/{id}` register. Without it the
routes stay unregistered.

## SSE keepalives on long prefill

Async chat/completion streams may emit SSE **comment** frames (`:\n\n`) while
waiting on the engine (long prefill / TTFT). Interval is `VT_SERVER_SSE_PING_S`
(default 15s; `0` disables). Comment frames are not `data:` events and do not
carry tokens. Token streaming still uses a timed wait on the request collector
so deltas are not collapsed by a poll loop.

## Diagnosing a CPU/CUDA divergence (Qwen4-EXP LayerFp)

When the CPU and CUDA arms of a Qwen4-EXP decode disagree on tokens, arm the
fingerprint tap with `VT_Q4EXP_LAYER_FP=9` on both arms and diff the server
logs:

```sh
python3 scripts/q4exp-layerfp-diff.py A-CPU/server.log B-CUDA/server.log --top 50
```

The differ reports `rel(sumabs)` per tap. `rel(sumabs)` is a DIFFERENCE OF
NORMS, not a norm of differences: its zero means "equal L1 norm", not "equal",
and a zero-mean perturbation cancels in it at `O(sqrt(n))`. At the tap's real
size (`n = 12800`), over the 400 seeds of the `MetricSpread` control in
`tests/scripts/test_q4exp_layerfp_diff.py`, the under-report is MEDIAN 75x-140x
with a p05..p95 of 34..1500. Hold the true divergence fixed: two readings differ by
a median **2.1x** and by **24x** at p95. A ratio between two `rel(sumabs)`
numbers is worth what that says and no more: NO CHANGE AT ALL produces a ratio at
least as large as 16.7x or 19.9x in 7% and 6% of draws. Those two ratios sit at
6% and 7% of the metric. No change at all produces a ratio at least as large as
1.80x, 2.02x, 2.34x and 3.15x in 59%, 52%, 45% and 33% of draws, and 3.15x sits
at the **67th** percentile — it is the LEAST ordinary of the four, and still an
ordinary reading. So read any single `rel(sumabs)` ratio as an order of
magnitude, and confirm a real divergence with `head_dmax` or the sign-sensitive
`rel_proj` column. Worked example and evidence:
[`docs/bench-evidence/qwen4exp-layerfp-2999-20260923.md`](bench-evidence/qwen4exp-layerfp-2999-20260923.md).

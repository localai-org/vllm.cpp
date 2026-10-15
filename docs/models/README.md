# Model recipes

Read a model recipe when a checkpoint needs extra files, model-specific flags,
or when it has a known limitation. Read [the quickstart](../QUICKSTART.md) to
run your first model, and [the usage guide](../USAGE.md) for the shared CLI,
server, C ABI, and source-tree workflows.

Each page tells you the same four things in the same order: what the model is,
which checkpoint was used, the exact command, and what has not been measured.

## Text generation

| Model | What it is | Read the page for |
|---|---|---|
| [Qwen3.8 2.4T](qwen3-8-2-4t.md) | A 2.4-trillion-parameter mixture of experts, 370 GiB at `UD-Q1_0` | Serving a checkpoint three times larger than the machine's memory, at 11.05 s/token |
| [Qwen3.8 27B](qwen3-8-27b.md) | A 27B dense model | Checkpoint limits and correctness gates, with block-wise FP8 on CPU and CUDA `sm_120a` and `sm_121a` |
| [Qwen3.6](qwen3-6.md) | The Qwen3.6 dense and MoE family | Which `lm_head` forms load, and the merged FP8 `in_proj_qkvz` GEMM |
| [Qwen3.5](qwen3-5.md) | The Qwen3.5 Gated DeltaNet family | The `output_gate_type` key, and one load refusal that is about this code |
| [Qwen3-Next](qwen3-next.md) | The Qwen3-Next Gated DeltaNet family | The `output_gate_type` key and its refusals |
| [Nemotron 3.5 Lightning](nemotron-3-5-lightning.md) | A 30B-A3B hybrid of attention, Mamba2, and MoE | Which arms run on the device, which run on the host, and what that costs per token |
| [Nemotron 3 Nano Omni](nemotron-nano-omni.md) | The Nemotron-H hybrid with a RADIO image encoder | Which checkpoint loads, what the image path was checked against, and what is refused |
| [Muse Glimmer](muse-glimmer.md) | A 30B multimodal model, text with image and video input | Running the text tower from a 17 GB GGUF, and how narrow the verified surface is |
| [Gemma 4](gemma-4.md) | The Gemma 4 family | The ROCm RDNA4 dual-GPU FP8 recipe |

## Decision models

| Model | What it is | Read the page for |
|---|---|---|
| [CLM](clm.md) | A bi-encoder (Qwen3-8B + dual MLP heads) decision model | The projection-head conversion, the confidence formula, and what has not been measured |
| [GLiNER2.5-Decide](gliner25-decide.md) | A DeBERTa-v3-large + classification head decision model | The checkpoint, the sequence layout, and what has not been measured |
| [xor](xor.md) | A 35B MoE (Qwen3.6-35B-A3B) decision model with forward+reverse calibration | The checkpoint, the double-pass cost, and what has not been measured |
| [Nimble](nimble.md) | A Qwen3.5-9B LoRA decision model that reads answer-letter logits (Ollama's `nimble`) | Converting the adapter, the openjev answer semantics, and why the 9B checkpoint is not yet measured end to end |
| [Tev1](tev1.md) | A decision model, 4B and 0.8B (Qwen3.5 SFTs), on `/v1/systemone`, `vllm_decide` and chat completions | Enabling the decision route, the prompt and where it differs from Ollama, and the CPU check against `transformers` |

## Speech, music, and video

| Model | What it is | Read the page for |
|---|---|---|
| [IndexTTS 2.5](indextts-2-5.md) | Text to speech, 22.05 kHz mono | Converting the checkpoint and starting a speech-only server |
| [MiniMax-Music3](minimax-music3.md) | Music generation | The checkpoint set and the render command |
| [MiniMax-H3](minimax-h3.md) | Video with audio | The five-file checkpoint set and the render workflow |
| [LTX 2.5](ltx-2-5.md) | Video generation | The checkpoint set, the render command, and what a render costs |

## Speculative decoding

| Model | What it is | Read the page for |
|---|---|---|
| [DSpark](dspark.md) | Draft models for speculative decoding | Which draft layouts run, and which are refused by name |

## What these pages do not carry

A model page records what was measured and what was not. It does not carry the
design rationale for a default or the history of a fixed defect. Those live in
the row's spec under [`.agents/specs/`](../../.agents/specs/), and each page
links to its own.

| You want | Read |
|---|---|
| What is proven and what is not | [Project status](../../README.md#project-status) |
| The complete measurement record | [`docs/BENCHMARKS.md`](../BENCHMARKS.md) |
| Which features, backends, and quantizations exist | [`docs/FEATURES.md`](../FEATURES.md) |
| Every server flag and endpoint | [server reference](../reference/server.md) |
| Every environment knob | [`docs/ENVIRONMENT.md`](../ENVIRONMENT.md) |

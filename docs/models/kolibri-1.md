# Kolibri-1

Kolibri-1 is Aleph Alpha's mixture-of-experts text model with sliding-window
and full attention. The released FP8 checkpoint has CPU and Tenstorrent forwards
through the shared model registry. The engine accepts its tokenizer.

## Checkpoint

Use [Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1/tree/e52eb462)
at recorded revision `e52eb462`.
Keep `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`model.safetensors.index.json`, and all 32 shards named by the index together.
The index records 78,827,029,120 bytes of tensors, approximately 78.8 GB.
This is the checkpoint size, not a bound on runtime memory.

The checkpoint contains FP8 block-quantized weights, F32 scale grids, and BF16 tensors.
The loader requires `quantization_config` with FP8 blocks of 128 by 128.
A plain BF16 checkpoint, one whose `config.json` carries no
`quantization_config`, loads through the same CPU path; its weights are read
as they are, with no dequant and no scale.
A `quantization_config` that names any other scheme is refused by name, and
the refusal names the arms that exist.
GGUF files are refused.

The existing record supplies a short revision and a tensor manifest, but no shard SHA-256 checksums.
A complete artifact checksum record remains unavailable.
See the [checkpoint and loader specification](../../.agents/specs/kolibri-1-cpu.md#the-checkpoint)
and [manifest](../../tests/vllm/models/kolibri1_manifest.inc).

## Run on CPU

Build the [CPU target](../BUILD.md#cpu-build-the-correctness--ci-reference).
Set `--model` to the local checkpoint directory:

```sh
build/examples/vllm-cli \
  --model /path/to/Kolibri-1 \
  --device cpu \
  --prompt "The capital of Australia is" \
  --max-tokens 32 \
  --temperature 0
```

The CLI takes a local path. It does not download the checkpoint.
The [CLI parser](../../examples/cli/main.cpp) defines these options.
The CPU forward caches dequantized weights by default when memory permits.
See the [memory policy and overrides](../ENVIRONMENT.md#kolibri-1-cpu-dequant-cache)
and [host-specific measurements](../benchmarks/kolibri1-cpu-dequant-cache-default.md).

## Validation and limits

The recorded CPU comparison uses a PyTorch transcription of the author's plugin math over a dequantized BF16 reference.
It is not a served vLLM comparison.
It matches 141 of 145 compared argmax positions, with four documented near-tie differences.
Four of eight prompts match token-for-token through the complete generated sequence.
The separate tokenizer check matches reference input IDs for all eight prompts.
See the [CPU comparison evidence](../bench-evidence/kolibri1-goldens-20261004.md)
and [tokenizer validation](../../.agents/specs/kolibri-1-cpu.md#r7-resolution--the-tokenizer-engine-accepts-the-kolibri-1-split-regex).

CUDA and other non-Tenstorrent accelerator forwards are refused.
The Tenstorrent forward uses resident dense weights and streams routed experts by default.
Its recorded correctness gate still fails, and captured decode is not wired.
See the [current Tenstorrent gate](../../.agents/specs/kolibri-tt.md#now)
and [captured-decode gaps](../gap-kolibri1-tt-captured-decode.md).

The author's vLLM plugin runs on CPU with a BF16-converted reference checkpoint,
as recorded in the [oracle record](../../.agents/oracles/aleph-alpha-inference.md).
That backend cannot run the released FP8 checkpoint directly.
The recorded transcription comparison does not establish parity with the plugin;
their reference outputs differ. Full correctness and speed parity remain unverified.

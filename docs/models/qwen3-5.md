# Qwen3.5

Qwen3.5 is a Gated DeltaNet (GDN) architecture. Its MoE members are also the
family that the expert streaming lane serves, which is how
[Qwen3.8 2.4T](qwen3-8-2-4t.md) runs from a file mapping.

Qwen3.5 runs through the shared paths, so [the quickstart](../QUICKSTART.md) and
[the usage guide](../USAGE.md) cover starting a server and sending a request.
This page carries the one config key that changes the arithmetic, and one load
refusal that is worth recognizing.

## The `output_gate_type` key

A Gated DeltaNet checkpoint, meaning the Qwen3.5 and Qwen3-Next family, chooses
its output-gate activation in `config.json`:

| `output_gate_type` | Gate applied |
|---|---|
| absent | `silu`, the upstream default |
| `"silu"` or `"swish"` | `silu`. `swish` is an alias, collapsed at load |
| `"sigmoid"` | `sigmoid` |
| present but `null`, `""`, or not a string | refused |

The key is read from the **resolved text config**, so a flat text-only
`config.json` and a multimodal wrapper that nests the text model under
`text_config` behave identically.

Any other value is **refused at load**, with a message naming the key and the
accepted set. It is never silently defaulted, because the wrong gate is a
numerics change that still emits plausible tokens
([#489](https://github.com/mudler/vllm.cpp/issues/489)).

Only an **absent** key takes the default. A key that is present but `null` or
empty is a value, not an absence. Upstream hands it straight to its
`assert output_gate_type in ["silu", "swish", "sigmoid"]` and errors, so this
loader refuses it as well rather than quietly reading it as `silu`.

## One load refusal that is about this code, not your checkpoint

Almost every load refusal names something your `config.json` or your tensors
actually declare. Exactly one does not:

```text
dense loader: LoadQwen3_5DenseLayer was given a tensor-presence probe that
answered YES for '__vllm_cpp__a_tensor_no_checkpoint_carries__', a name no
checkpoint carries.
```

That name is not in your checkpoint and is not supposed to be. The loader asks
about it to find out whether its own "is this tensor present?" predicate can
answer `no`, and this message means it cannot. Your checkpoint is fine. Please
report it with the model you were loading
([#1258](https://github.com/mudler/vllm.cpp/issues/1258)).

The check exists because a predicate that only ever said yes shipped twice in
one file, and what a reader saw was the opposite of the truth: a refusal naming
a block-wise FP8 scale tensor the checkpoint had never contained
([#1256](https://github.com/mudler/vllm.cpp/issues/1256)). A message that blames
the wrong side costs more than the failure does.

## EXL3 checkpoints: integer and half-integer rates

The EXL3 trellis arm reads each tensor's bit width off the tensor itself, so a
per-layer or per-tensor mixed rate needs no flag. Integer `K` (2..8) and
half-integer `K+0.5` rates both decode on ROCm; the fractional form is
`mul1`-codebook only, alternating `KA` and `KA+1` bits per position
(`mask 0xAAAA`), which is what the `3.5bpw`-class checkpoints publish. A frac
tensor without a `mul1` marker, or at `KA` outside 1..7, refuses by name.
CUDA and Vulkan have no frac arm yet and refuse the same way; on ROCm the
frac tensors run on the dot/`Exl3GemmK` path (the GEMV and reconstruct arms
are integer-width only, and the router keeps frac off them).

Gated against [`orcarouter/OrcaSAQ-2-27B-EXL3-3.21bpw`](https://huggingface.co/orcarouter/OrcaSAQ-2-27B-EXL3-3.21bpw):

| file | bytes | sha256 |
|---|---|---|
| `model.safetensors` | 12,270,437,068 | `0b733783fd9e4dc053455955a9acff0317a9735cea51c62721a281a2a0025f34` |
| `model-dequant-embed.safetensors` | 2,542,796,920 | `c00a917770d131bd33a73f8dba11d392a7625a10e1ef2f643bb10ffe68c95558` |
| `config.json` | 6,151 | `6392a2cf7ee8a5dddf615ee067b9831dbcbb47a1f127d84e1f4e6f9e30728568` |

(The HF repo requires auth, so the revision sha could not be read; the file
hashes above pin the exact bytes.) Per-tensor rates in that checkpoint:
32/48/56/64/96 uint16 words per tile = 2/3/3.5/4/6 bpw, codebook `mul1`,
`bits_per_weight: 3.21`.

Measured on ROCm gfx1100 (2026-10-05): the checkpoint loads and greedy-decodes
coherent output; generation matches the exllamav3 oracle's greedy trace up to
the first FP-order argmax tie (char ~160 of 256-token captures), the same
drift point the integer-rate sibling checkpoint shows against the same
oracle — i.e. parity is distributional, not token-exact, and is recorded that
way in `.agents/specs/quant-exl3-frac-rates.md`.

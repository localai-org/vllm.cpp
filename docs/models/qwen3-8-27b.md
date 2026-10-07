# Qwen3.8 27B

Qwen3.8 27B is a 27B dense model. Use [the quickstart](../QUICKSTART.md)
to start a server and [the usage guide](../USAGE.md) for CLI options.
This page explains which quantized checkpoints load and which have recorded
correctness results. Loading a checkpoint does not establish token parity.

## Which arm to use on a GPU today

| Checkpoint or format | What the evidence establishes |
|---|---|
| BF16 | Recorded correctness gate in [the model's quantization spec](../../.agents/specs/qwen38-27b-quant-arms.md) |
| `Qwen/Qwen3.8-27B-FP8`, block-wise FP8 | CPU reference and CUDA on `sm_120a` and `sm_121a`. [GB10 text gate passed with one near-tie](#the-token-gate-against-vllm) |
| `unsloth/Qwen3.8-27B-NVFP4`, mixed FP8 and NVFP4 | NVFP4 modules load. The FP8 group and quantized KV-cache configuration are refused |
| `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` | Per-tensor static FP8 and NVFP4 W4A16 load. [The artifact's token gate remains owed](../../.agents/specs/qwen38-27b-quant-arms.md#now) |
| `RadixArk/Qwen3.8-27B-NVFP4` | Loads as W4A16 with a warning that the artifact declares W4A4. [The token gate remains owed](../../.agents/specs/qwen38-27b-quant-arms.md#now) |
| `unsloth/Qwen3.8-27B-GGUF`, Q4_K_M | Generates on CPU. [The recorded token gate failed on 5 of 6 prompts](../bench-evidence/qwen38-27b-q4km-token-gate-20260823.md) |
| EXL3 | CUDA generation and its measured limits are recorded in [the EXL3 benchmark](../benchmarks/qwen38-27b-exl3-gb10.md) |

Generic FP8 or NVFP4 kernel coverage does not establish correctness for every
checkpoint of this model. The [quantization spec](../../.agents/specs/qwen38-27b-quant-arms.md)
records artifact revisions, hashes, and remaining gates.

## The Unsloth mixed FP8 and NVFP4 checkpoint

`unsloth/Qwen3.8-27B-NVFP4` is a mixed-precision checkpoint despite its name.
The inspected revision is `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`.
Its backbone is 22,568,192,096 bytes. The BF16 MTP drafter is 849,400,392 bytes.
The [artifact inventory](../../.agents/specs/qwen38-27b-quant-arms.md#what-i-inspected-and-what-i-took-on-trust)
records the tensor accounting and provenance.

The 168 NVFP4 modules load. The engine refuses the 233-module FP8 group,
which needs per-output-channel weight scales and dynamic per-token activation
quantization. This loader does not implement those requirements. It also
refuses `kv_cache_scheme`, because it does not consume this checkpoint's K and
V scales. These are artifact-specific limits, not a refusal of all FP8 weights.

Run the real-checkpoint manifest check with:

```sh
VLLM_CPP_QWEN38_27B_NVFP4_DIR=/path/to/qwen3.8-27b-nvfp4 \
  ./build/tests/test_qwen38_27b_nvfp4_arm
```

This checks the manifest and named refusals. It does not run a generation gate.
The [quantization spec](../../.agents/specs/qwen38-27b-quant-arms.md#now)
tracks the missing FP8 path, KV-scale handling, and token gates.

## Block-wise FP8

Block-wise FP8, also called fine-grained FP8, stores one scale per 128x128
weight block. Per-tensor FP8 stores one scale for the whole weight.
The block-wise format declares `quantization_config.weight_block_size` and
stores `weight_scale_inv` tensors instead of `weight_scale` tensors.

`Qwen/Qwen3.8-27B-FP8` at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` uses `[128, 128]` blocks and
`activation_scheme: dynamic`. Its FP8 projections satisfy the CUDA shape
requirements described below.

### What runs on CPU

The CPU path is a correctness reference with no speed claim. It quantizes
activations per token in groups of 128, then applies the block scales during
matrix multiplication. The accumulator is F32. Each projection emits BF16.

The shared dense forward merges gate and up projections. It also merges Q, K,
and V when the fused attention preamble is enabled, as it is by default.
`VT_FUSE_ATTN_PREAMBLE=0` uses separate Q, K, and V multiplications.
Every shard except the last in a merged projection must have a row count
divisible by 128. The loader names a nonconforming shard in its refusal.

### The token gate against vLLM

On 23 August 2026, the first-party block-wise FP8 checkpoint passed its text
correctness gate on NVIDIA GB10. The historical oracle was vLLM revision
`5559679229bc961848b121ccdeaa8fa5d79bec98`.

Both engines used the same checkpoint bytes and prompt token IDs: seven
prompts, 16 generated tokens each, greedy sampling, batch 1, and concurrency 1.
vLLM used its production configuration. Six prompts matched at all 16
positions. The seventh passed the previously ratified near-tie test at its
first divergence. This is not a seven-prompt token-exact result.

The run recorded zero portable-host fallbacks. All 2,736 block-scaled GEMMs
used the `sm_121a` CUTLASS kernel. Across 400 resident FP8 tensors, weights
occupied one byte per element. These checks exclude silent dequantization to
a wider weight format on the measured path.

The [gate record](../../.agents/specs/gate-qwen38-27b-fp8-block.md#evidence)
contains the commands, raw evidence references, and near-tie adjudication.
**The run establishes correctness only. It supplies no throughput, latency,
or memory benchmark.**

### On a device with no block-scaled GEMM

If the build has no block-scaled GEMM for the selected device, model preparation
refuses the checkpoint before the first forward or CUDA graph capture.
The error identifies the projection and device. Use the CPU reference or a
CUDA build targeting `sm_120a` or `sm_121a` with the CUTLASS kernel enabled.

### The CUDA kernel, and the shapes it refuses

The CUDA path uses a block-scaled CUTLASS GEMM on `sm_120a` and `sm_121a`.
The recorded component test matched the CPU reference on seven GB10 shapes.
That component result does not establish model correctness on `sm_120a`.

The current kernel requires both N and K to be multiples of 128. Shapes that
violate the FP8 operand alignment of 16 receive an alignment refusal first.
The engine rejects unsupported shapes before allocating or launching the GEMM,
with a message that names the dimension and required granularity.

For example, N=576 leaves 64 rows beyond a complete 128-row scale block.
DeepSeek-V3's `kv_a_proj_with_mqa` has that width and cannot use this CUDA path.
The missing fallback is an open implementation gap. The CPU reference accepts
these shapes. `Qwen/Qwen3.8-27B-FP8` does not need them.

The [CUDA component spec](../../.agents/specs/vt-matmul-fp8-block-cuda.md)
records the tested shapes, dispatch constraints, and upstream sources.

### Two configurations refused at load

The loader refuses:

- An `activation_scheme` other than `dynamic`.
- A `weight_block_size` other than `[128, 128]`.

Each error names the configuration key and value.

### One lever that is incompatible

Leave `VT_KV_CACHE_F32` unset. Setting it to `1` selects an F32 paged KV cache,
but `v_proj` emits BF16. The KV write requires matching dtypes and refuses
this combination. [Issue #1249](https://github.com/mudler/vllm.cpp/issues/1249)
tracks the gap.

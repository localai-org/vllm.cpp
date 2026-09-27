ID: ISSUE-GH-1736
Title: **On `Qwen3.8-2.4T-A95B UD-Q1_0` the `--device cuda` arm and the `--device cpu` arm emit different tokens, and nothing measures whether CUDA is WORSE or only DIFFERENT.** W0g excluded the router gate weights, the embedding table and the W0f host alias, and named no cause. Three things keep the question open: the CUDA continuation degenerates into a mechanical recursion after the 8 tokens the arms share, which a coin flip between two equally good tokens does not produce; every comparison so far is arm-against-arm with no oracle, so "they differ" cannot say which arm is wrong; and the growth-rate argument for "partly systematic" does not survive a proper fit -- a least-squares fit of `log(divergence)` on `log(block + 1)` over all eight recorded points gives an exponent of 0.651 +/- 0.066, interval [0.489, 0.813], which INCLUDES the 0.5 a random walk predicts and EXCLUDES the 1.0 a systematic error predicts, so the two-point 24x-against-9.5x reading overstated it. Scoped as wave **W0h** of `ENG-EXPERT-STREAM-DEVICE`: feed BOTH arms the identical token sequence through the ABI logits processor (`include/vllm.h` v8, applied at `src/vllm/v1/sample/sampler.cpp:441`) and measure the negative log likelihood each assigns to held-out text, which is a quality statement needing no oracle. The decision rule is PRE-REGISTERED before any measurement, the oracle arm is `llama-cpp-unsloth` at `36fe8e1cc` (`gateable = no`, owed by [#933](https://github.com/mudler/vllm.cpp/issues/933)), and the in-tree `VT_CPU_REF=1` switch is excluded on arithmetic (every tensor to `kExpandBf16` at `gguf_keep_quant.cpp:157` is 4.37 to 4.87 TiB against a 119.631 GiB box). No product code, no speed claim, G0-SPEED stays VOID. Spec [`cuda-arm-degradation-experiment.md`](../../specs/cuda-arm-degradation-experiment.md)
Row: ENG-EXPERT-STREAM-DEVICE
State: UNKNOWN
Kind: verification
GitHub: 1736
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:653`

### Frozen archive evidence

> | [#1736](https://github.com/mudler/vllm.cpp/issues/1736) | `ENG-EXPERT-STREAM-DEVICE` | **On `Qwen3.8-2.4T-A95B UD-Q1_0` the `--device cuda` arm and the `--device cpu` arm emit different tokens, and nothing measures whether CUDA is WORSE or only DIFFERENT.** W0g excluded the router gate weights, the embedding table and the W0f host alias, and named no cause. Three things keep the question open: the CUDA continuation degenerates into a mechanical recursion after the 8 tokens the arms share, which a coin flip between two equally good tokens does not produce; every comparison so far is arm-against-arm with no oracle, so "they differ" cannot say which arm is wrong; and the growth-rate argument for "partly systematic" does not survive a proper fit -- a least-squares fit of `log(divergence)` on `log(block + 1)` over all eight recorded points gives an exponent of 0.651 +/- 0.066, interval [0.489, 0.813], which INCLUDES the 0.5 a random walk predicts and EXCLUDES the 1.0 a systematic error predicts, so the two-point 24x-against-9.5x reading overstated it. Scoped as wave **W0h** of `ENG-EXPERT-STREAM-DEVICE`: feed BOTH arms the identical token sequence through the ABI logits processor (`include/vllm.h` v8, applied at `src/vllm/v1/sample/sampler.cpp:441`) and measure the negative log likelihood each assigns to held-out text, which is a quality statement needing no oracle. The decision rule is PRE-REGISTERED before any measurement, the oracle arm is `llama-cpp-unsloth` at `36fe8e1cc` (`gateable = no`, owed by [#933](https://github.com/mudler/vllm.cpp/issues/933)), and the in-tree `VT_CPU_REF=1` switch is excluded on arithmetic (every tensor to `kExpandBf16` at `gguf_keep_quant.cpp:157` is 4.37 to 4.87 TiB against a 119.631 GiB box). No product code, no speed claim, G0-SPEED stays VOID. Spec [`cuda-arm-degradation-experiment.md`](../../specs/cuda-arm-degradation-experiment.md) | verification |

## Resolution

-

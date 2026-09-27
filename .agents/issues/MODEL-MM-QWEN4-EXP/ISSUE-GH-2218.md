ID: ISSUE-GH-2218
Title: **The `hc_norm` gamma polarity disagrees between the loader and the device op, and a layer loop wiring them together scales by ~0.** `LoadNormBf16(..., unshift=true)` at `qwen4_exp_weights.cpp:264` stores the RAW HuggingFace gamma, centred on 0, by inverting the GGUF converter's baked `+1`. `vt::Qwen4ExpGatedResidual` documents the OPPOSITE convention — "hc_norm_w is vLLM's parameterization, i.e. ALREADY `1 + w_hf` … This op never adds 1" — so a layer loop that hands the loader's tensor straight to that op applies a near-zero scale, and the result reads as a corrupt checkpoint rather than as a wiring bug. The contradiction is visible AT THE LOAD SITE: the comment at `qwen4_exp_weights.cpp:258-263` argues FOR the fold, elementwise-corroborated on three published artifacts, immediately above the line that strips it. Nothing is broken today because `Qwen4ExpTextModel::Forward` does not exist; the moment the layer loop lands it must fold `hc_norm`, `norm_key`, `norm_query` and `norm_conv` through `vllm::qwen4_exp::HcNormWeightFromHf` first. NOT repaired in W5b-5, which hit the same shape and got it right by accident of scope: the QSA block's norms take the raw gamma under `RmsNormArgs::gemma = true`, which mutations M9/M10/M11 red. Owned by `MODEL-MM-QWEN4-EXP` and listed under `## Owed` in [`specs/qwen4-exp-flash-next.md`](../../specs/qwen4-exp-flash-next.md); the layer-loop wave is where it gets fixed and gated.
Row: MODEL-MM-QWEN4-EXP
State: UNKNOWN
Kind: bug
GitHub: 2218
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:864`

### Frozen archive evidence

> | [#2218](https://github.com/mudler/vllm.cpp/issues/2218) | `MODEL-MM-QWEN4-EXP` | **The `hc_norm` gamma polarity disagrees between the loader and the device op, and a layer loop wiring them together scales by ~0.** `LoadNormBf16(..., unshift=true)` at `qwen4_exp_weights.cpp:264` stores the RAW HuggingFace gamma, centred on 0, by inverting the GGUF converter's baked `+1`. `vt::Qwen4ExpGatedResidual` documents the OPPOSITE convention — "hc_norm_w is vLLM's parameterization, i.e. ALREADY `1 + w_hf` … This op never adds 1" — so a layer loop that hands the loader's tensor straight to that op applies a near-zero scale, and the result reads as a corrupt checkpoint rather than as a wiring bug. The contradiction is visible AT THE LOAD SITE: the comment at `qwen4_exp_weights.cpp:258-263` argues FOR the fold, elementwise-corroborated on three published artifacts, immediately above the line that strips it. Nothing is broken today because `Qwen4ExpTextModel::Forward` does not exist; the moment the layer loop lands it must fold `hc_norm`, `norm_key`, `norm_query` and `norm_conv` through `vllm::qwen4_exp::HcNormWeightFromHf` first. NOT repaired in W5b-5, which hit the same shape and got it right by accident of scope: the QSA block's norms take the raw gamma under `RmsNormArgs::gemma = true`, which mutations M9/M10/M11 red. Owned by `MODEL-MM-QWEN4-EXP` and listed under `## Owed` in [`specs/qwen4-exp-flash-next.md`](../../specs/qwen4-exp-flash-next.md); the layer-loop wave is where it gets fixed and gated. | bug |

## Resolution

-

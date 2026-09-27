ID: ISSUE-GH-1875
Title: MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark runs a REAP-pruned K216 DeepSeek-V4-Flash at a claimed 44-47 tok/s decode on one GB10 -- EXL3 3.0bpw trellis quant (~99.5 GiB, the first V4-Flash quant that fits one Spark), SparkInfer (NVIDIA-vLLM-26.02 fork) with K5 speculative decoding. Developer direction: load the same quants and match or beat the speed. vLLM has no EXL3 at the pin, so the row proposes `exllamav3` @ `2398c056` as a pinned secondary oracle (its HEAD carries DSV4 support). Spike 2026-08-24 on the issue pins the full format (MCG `0xCBAC1FED` 3-instruction decode, 16x16 tail-biting trellis tiles, H128+sign vectors, no scales, lossless TP4-to-TP1 coalescing, K216 physical compaction our config-driven loader accepts unchanged). Caveats recorded: their number includes spec decode with no bare-AR figure, and the checkpoint's own README says end-to-end generation is runtime_pending. Row [spec](../../specs/model-dsv4-exl3.md)
Row: MODEL-DSV4-EXL3
State: UNKNOWN
Kind: feature
GitHub: 1875
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:710`

### Frozen archive evidence

> | [#1875](https://github.com/mudler/vllm.cpp/issues/1875) | `MODEL-DSV4-EXL3` | MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark runs a REAP-pruned K216 DeepSeek-V4-Flash at a claimed 44-47 tok/s decode on one GB10 -- EXL3 3.0bpw trellis quant (~99.5 GiB, the first V4-Flash quant that fits one Spark), SparkInfer (NVIDIA-vLLM-26.02 fork) with K5 speculative decoding. Developer direction: load the same quants and match or beat the speed. vLLM has no EXL3 at the pin, so the row proposes `exllamav3` @ `2398c056` as a pinned secondary oracle (its HEAD carries DSV4 support). Spike 2026-08-24 on the issue pins the full format (MCG `0xCBAC1FED` 3-instruction decode, 16x16 tail-biting trellis tiles, H128+sign vectors, no scales, lossless TP4-to-TP1 coalescing, K216 physical compaction our config-driven loader accepts unchanged). Caveats recorded: their number includes spec decode with no bare-AR figure, and the checkpoint's own README says end-to-end generation is runtime_pending. Row [spec](../../specs/model-dsv4-exl3.md) | feature |

## Resolution

-

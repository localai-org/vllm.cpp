ID: ISSUE-GH-2190
Title: **dots3-note's quantization refusal keys on `weight_block_size` alone, so a per-tensor or per-row fp8/gptq/awq config loads and silently dequantizes to bf16.** `Dots3NoteDeviceRefusal` (`dots3_note_device.cpp:855`) branches on `has_blockwise_quant()`, which is `!weight_block_size.empty()` (`dots3_note.h:208`). `quant_method` IS parsed (`dots3_note.cpp:264-268`) and stored (`dots3_note.h:206`), and is read for nothing but the text of the blockwise message (`:862-863`). A `config.json` with `quantization_config.quant_method = "fp8"` (or gptq/awq/mxfp4/compressed-tensors) and NO `weight_block_size` therefore passes, and `dense_loaders::MaterializeBf16Source` silently dequantizes a per-tensor or per-output-ROW `_scale` into a bf16 GEMM — which is precisely the case the refusal's own comment names as the worse one, five lines above the branch that does not cover it (`:849-854`). This row has NO oracle on any hardware we own (spec §6.4), so nothing downstream catches the plausible wrong answer. No released checkpoint is affected: the bf16 repo carries no `quantization_config` and the `-fp8` sibling carries `weight_block_size [128, 128]` and is refused correctly. Owed: refuse a non-empty `quant_method` this port cannot read, naming the method and W9, with the config-fixture gate the blockwise case already has. Found by the fresh review of [#2187](https://github.com/mudler/vllm.cpp/pull/2187) as F5 and deliberately not fixed there — a refusal-semantics change needs its own red-before fixture. Under `## Owed` in [specs/dots3-note.md](../../specs/dots3-note.md)
Row: MODEL-MM-dots3-note-dots3-note-for-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 2190
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:848`

### Frozen archive evidence

> | [#2190](https://github.com/mudler/vllm.cpp/issues/2190) | `MODEL-MM-dots3-note` | **dots3-note's quantization refusal keys on `weight_block_size` alone, so a per-tensor or per-row fp8/gptq/awq config loads and silently dequantizes to bf16.** `Dots3NoteDeviceRefusal` (`dots3_note_device.cpp:855`) branches on `has_blockwise_quant()`, which is `!weight_block_size.empty()` (`dots3_note.h:208`). `quant_method` IS parsed (`dots3_note.cpp:264-268`) and stored (`dots3_note.h:206`), and is read for nothing but the text of the blockwise message (`:862-863`). A `config.json` with `quantization_config.quant_method = "fp8"` (or gptq/awq/mxfp4/compressed-tensors) and NO `weight_block_size` therefore passes, and `dense_loaders::MaterializeBf16Source` silently dequantizes a per-tensor or per-output-ROW `_scale` into a bf16 GEMM — which is precisely the case the refusal's own comment names as the worse one, five lines above the branch that does not cover it (`:849-854`). This row has NO oracle on any hardware we own (spec §6.4), so nothing downstream catches the plausible wrong answer. No released checkpoint is affected: the bf16 repo carries no `quantization_config` and the `-fp8` sibling carries `weight_block_size [128, 128]` and is refused correctly. Owed: refuse a non-empty `quant_method` this port cannot read, naming the method and W9, with the config-fixture gate the blockwise case already has. Found by the fresh review of [#2187](https://github.com/mudler/vllm.cpp/pull/2187) as F5 and deliberately not fixed there — a refusal-semantics change needs its own red-before fixture. Under `## Owed` in [specs/dots3-note.md](../../specs/dots3-note.md) | bug |

## Resolution

-

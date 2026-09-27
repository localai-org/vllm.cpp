ID: ISSUE-GH-2140
Title: **The LTX-2.5 caption projections are NVFP4-only, so the bf16 text tower loads and the render still refuses.** `LoadProjection` (`src/vllm/model_executor/models/ltx2_loader.cpp`) computes `in_features = w->shape[1] * 2` unconditionally, with the comment "NVFP4 packs TWO values per byte", then requires `.weight_scale` and `.weight_scale_2` and dequantizes. On the bf16 checkpoint the stored width is already logical, so the doubling turns a correct 188160 into 376320 and the geometry check fires on the loader's own arithmetic. Measured on `dgx:gpu0` (`rc` job `001c36e9-76b1-432c-9536-2d24c0e613d0`, 2026-08-27) and confirmed by reading both safetensors headers: the bf16 file stores `text_embedding_projection.video_aggregate_embed.weight` as `BF16 [4096, 188160]` with **zero** `.weight_scale` tensors and **zero** `torchao_nvfp4` markers in the whole file, while the torchao file stores it as `U8 [4096, 94080]` with 334 of each. The fix resolves the storage format from the file the way upstream does — `_discover_nvfp4_layers` (`packages/ltx-core/src/ltx_core/quantization/nvfp4/prequant.py:30-50` at pin `fd4ded7f`) selects a layer only when `.weight_scale` and `.weight_scale_2` are BOTH present and the dtype triple is `U8`/`F8_E4M3`/`F32`, treats exactly one of the pair as an error, and leaves everything else the plain `nn.Linear(flat_dim, ...)` of `encoder_configurator.py:206-208`, whose stored width IS its logical width. Blocks [#1854](https://github.com/mudler/vllm.cpp/issues/1854)'s absolute gate, because [#1864](https://github.com/mudler/vllm.cpp/issues/1864)'s reference render was taken with the bf16 tower and an arm-matched comparison cannot substitute the NVFP4 one. Spec [`ltx25-text-proj-dtype.md`](../../specs/ltx25-text-proj-dtype.md)
Row: LTX25-TEXT-PROJ-DTYPE
State: UNKNOWN
Kind: bug
GitHub: 2140
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:853`

### Frozen archive evidence

> | [#2140](https://github.com/mudler/vllm.cpp/issues/2140) | `LTX25-TEXT-PROJ-DTYPE` | **The LTX-2.5 caption projections are NVFP4-only, so the bf16 text tower loads and the render still refuses.** `LoadProjection` (`src/vllm/model_executor/models/ltx2_loader.cpp`) computes `in_features = w->shape[1] * 2` unconditionally, with the comment "NVFP4 packs TWO values per byte", then requires `.weight_scale` and `.weight_scale_2` and dequantizes. On the bf16 checkpoint the stored width is already logical, so the doubling turns a correct 188160 into 376320 and the geometry check fires on the loader's own arithmetic. Measured on `dgx:gpu0` (`rc` job `001c36e9-76b1-432c-9536-2d24c0e613d0`, 2026-08-27) and confirmed by reading both safetensors headers: the bf16 file stores `text_embedding_projection.video_aggregate_embed.weight` as `BF16 [4096, 188160]` with **zero** `.weight_scale` tensors and **zero** `torchao_nvfp4` markers in the whole file, while the torchao file stores it as `U8 [4096, 94080]` with 334 of each. The fix resolves the storage format from the file the way upstream does — `_discover_nvfp4_layers` (`packages/ltx-core/src/ltx_core/quantization/nvfp4/prequant.py:30-50` at pin `fd4ded7f`) selects a layer only when `.weight_scale` and `.weight_scale_2` are BOTH present and the dtype triple is `U8`/`F8_E4M3`/`F32`, treats exactly one of the pair as an error, and leaves everything else the plain `nn.Linear(flat_dim, ...)` of `encoder_configurator.py:206-208`, whose stored width IS its logical width. Blocks [#1854](https://github.com/mudler/vllm.cpp/issues/1854)'s absolute gate, because [#1864](https://github.com/mudler/vllm.cpp/issues/1864)'s reference render was taken with the bf16 tower and an arm-matched comparison cannot substitute the NVFP4 one. Spec [`ltx25-text-proj-dtype.md`](../../specs/ltx25-text-proj-dtype.md) | bug |

## Resolution

-

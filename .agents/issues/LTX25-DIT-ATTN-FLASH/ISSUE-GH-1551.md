ID: ISSUE-GH-1551
Title: **`vt::AttentionDenseFa2` refuses anything but head_dim 64, so LTX-2.5's head_dim-128 DiT cannot reach tensor cores.** The guard is `src/vt/cuda/cuda_flash_attn_fa2.cu:557-560` plus the dispatch test `query.shape[2] == 64` at `src/vt/cuda/cuda_ops.cu:3396-3399`; everything else falls through to `AttentionDenseFlash`, which is a scalar warp-per-query online-softmax recurrence with shared-memory K/V tiling -- correct, tiled, and still not `mma.sync`. LTX's video stream is 32 heads x head_dim 128 (`include/vllm/model_executor/models/ltx2.h:124-125`). Reaching the vendored FA-2 path needs an extra `run_mha_fwd_<bfloat16_t, 128, false>` instantiation and the guard widened; `cuda_ops.cu:3375-3377` records the current narrowness as deliberate, so this is deferred cost and not oversight. Prize is BOUNDED, not measured: `.agents/specs/multimodal-speed.md` §16 puts warp-to-flash at 1.04x at 784 tokens and §14 at 1.82x at 1500, both the same scalar recurrence; what tensor cores buy over it at 2352 tokens is unmeasured, and this issue owes the measurement before it owes the port. Numerics caveat that must not be lost: FA-2 is NOT bit-identical (`include/vt/ops.h:2995-2997`, `mma.sync` reassociates both QK^T and P.V), and a diffusion model has no token gate, so a pixel-level comparison has to be designed first. NOT fixed in flow -- explicitly out of scope for #1549. Owner: row `LTX25-DIT-ATTN-FLASH`, under `## Owed` in [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md)
Row: LTX25-DIT-ATTN-FLASH
State: UNKNOWN
Kind: feature
GitHub: 1551
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:559`

### Frozen archive evidence

> | [#1551](https://github.com/mudler/vllm.cpp/issues/1551) | `LTX25-DIT-ATTN-FLASH` | **`vt::AttentionDenseFa2` refuses anything but head_dim 64, so LTX-2.5's head_dim-128 DiT cannot reach tensor cores.** The guard is `src/vt/cuda/cuda_flash_attn_fa2.cu:557-560` plus the dispatch test `query.shape[2] == 64` at `src/vt/cuda/cuda_ops.cu:3396-3399`; everything else falls through to `AttentionDenseFlash`, which is a scalar warp-per-query online-softmax recurrence with shared-memory K/V tiling -- correct, tiled, and still not `mma.sync`. LTX's video stream is 32 heads x head_dim 128 (`include/vllm/model_executor/models/ltx2.h:124-125`). Reaching the vendored FA-2 path needs an extra `run_mha_fwd_<bfloat16_t, 128, false>` instantiation and the guard widened; `cuda_ops.cu:3375-3377` records the current narrowness as deliberate, so this is deferred cost and not oversight. Prize is BOUNDED, not measured: `.agents/specs/multimodal-speed.md` §16 puts warp-to-flash at 1.04x at 784 tokens and §14 at 1.82x at 1500, both the same scalar recurrence; what tensor cores buy over it at 2352 tokens is unmeasured, and this issue owes the measurement before it owes the port. Numerics caveat that must not be lost: FA-2 is NOT bit-identical (`include/vt/ops.h:2995-2997`, `mma.sync` reassociates both QK^T and P.V), and a diffusion model has no token gate, so a pixel-level comparison has to be designed first. NOT fixed in flow -- explicitly out of scope for #1549. Owner: row `LTX25-DIT-ATTN-FLASH`, under `## Owed` in [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md) | feature |

## Resolution

-

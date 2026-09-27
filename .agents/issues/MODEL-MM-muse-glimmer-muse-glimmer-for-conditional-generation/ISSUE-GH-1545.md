ID: ISSUE-GH-1545
Title: **Muse Glimmer's perception encoder named the correctness-grade attention op for all 50 of its layers.** `src/vllm/model_executor/models/muse_glimmer_vision.cpp:639` called `vt::Attention`, which `src/vt/ops.cpp:2680` resolves to the kernel whose own header at `src/vt/cuda/cuda_ops.cu:1456-1460` calls itself "Correctness-grade (M0.9)": one 256-thread block per (query, head), a 256-wide shared-memory tree reduction for EVERY key, no K/V tiling. Sole attention path in the tower, non-causal, no knob, 50 layers, H=16, head_dim 96. An instance of the class issue [#1544](https://github.com/mudler/vllm.cpp/issues/1544), and the worst-shaped one in the tree. FIXED HERE by naming `vt::AttentionDenseFlash`, the rung `whisper_audio.cpp:310-322` and `qwen3_vl_vision.cpp:462-480` already default to; `AttentionDenseFa2` is not usable at head_dim 96. The geometry is now PINNED from the released checkpoint rather than inferred (`/mnt/nas_share/checkpoints/muse-glimmer-30b/`): `layer_types` ships 13 `full_attention` + 37 `window_attention`, the window is 32x32 = 1024 patches, and `max_image_tokens` is 4096, which puts the naive cost at 34 s or 375 s per image depending on whether that counts patch or post-merge tokens -- not the 4.8 s the issue illustrated with. Lands UNREACHED: the tower has no production caller, tracked by [#1566](https://github.com/mudler/vllm.cpp/issues/1566). Spec [muse-glimmer-vision-attn-flash.md](../../specs/muse-glimmer-vision-attn-flash.md)
Row: MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation
State: UNKNOWN
Kind: perf
GitHub: 1545
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:548`

### Frozen archive evidence

> | [#1545](https://github.com/mudler/vllm.cpp/issues/1545) | `MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation` | **Muse Glimmer's perception encoder named the correctness-grade attention op for all 50 of its layers.** `src/vllm/model_executor/models/muse_glimmer_vision.cpp:639` called `vt::Attention`, which `src/vt/ops.cpp:2680` resolves to the kernel whose own header at `src/vt/cuda/cuda_ops.cu:1456-1460` calls itself "Correctness-grade (M0.9)": one 256-thread block per (query, head), a 256-wide shared-memory tree reduction for EVERY key, no K/V tiling. Sole attention path in the tower, non-causal, no knob, 50 layers, H=16, head_dim 96. An instance of the class issue [#1544](https://github.com/mudler/vllm.cpp/issues/1544), and the worst-shaped one in the tree. FIXED HERE by naming `vt::AttentionDenseFlash`, the rung `whisper_audio.cpp:310-322` and `qwen3_vl_vision.cpp:462-480` already default to; `AttentionDenseFa2` is not usable at head_dim 96. The geometry is now PINNED from the released checkpoint rather than inferred (`/mnt/nas_share/checkpoints/muse-glimmer-30b/`): `layer_types` ships 13 `full_attention` + 37 `window_attention`, the window is 32x32 = 1024 patches, and `max_image_tokens` is 4096, which puts the naive cost at 34 s or 375 s per image depending on whether that counts patch or post-merge tokens -- not the 4.8 s the issue illustrated with. Lands UNREACHED: the tower has no production caller, tracked by [#1566](https://github.com/mudler/vllm.cpp/issues/1566). Spec [muse-glimmer-vision-attn-flash.md](../../specs/muse-glimmer-vision-attn-flash.md) | perf |

## Resolution

-

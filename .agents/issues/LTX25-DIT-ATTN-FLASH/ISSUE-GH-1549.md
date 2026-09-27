ID: ISSUE-GH-1549
Title: **One LTX-2.5 DiT forward costs 47.84 s on GB10 because the DiT self-attention never opted into a fast attention op.** `src/vllm/model_executor/models/ltx2_device.cpp:421` calls `vt::Attention`, which on CUDA is `AttentionKernel` (`src/vt/cuda/cuda_ops.cu:1463`) -- the kernel whose own header at `:1456-1459` calls itself "Correctness-grade (M0.9)": one 256-thread block per (query, head), a 256-wide shared-memory tree reduction per key, no K/V tiling. At `768x448/49f` (2352 tokens) that is 75,264 blocks x 2352 keys = 1.77e8 block-key iterations per call x 48 layers. MEASURED 47.84 s mean / 47.91 s median, n=119, spread 5.8%, from the engine's own `last=` lines (`render_phase_log.cpp:388` via `ltx2_video.cpp:4048-4054`), never from the governor, which has reported 1.00 s, 69.1 s, 162 s and 396.9 s for this one quantity. Attribution is arithmetic, not assertion: `.agents/specs/multimodal-speed.md:24-26` measures this same kernel on this same box at 5.70 ns per block-key iteration, and 1.77e8 x 5.70 ns x 48 = **48.4 s against the measured 47.84 s, a 1% match**. WHY IT WAS MISSED, which is the reusable part: `kAttention` is deliberately frozen on the naive kernel so text decode stays byte-identical (`cuda_ops.cu:3120-3122`), and the fast kernels are SEPARATE OPS each caller must opt into BY NAME. No automatic selection, no fallback notice. A model that never opts in gets correct output at ~500x the cost with no warning anywhere -- goldens pass, no refusal fires, and `GetOpProviderStats` counts the naive selection as the success it is. FIXED IN FLOW by routing the self-attention to `vt::AttentionDenseFlash`. Two things found doing it and filed rather than folded in: [#1551](https://github.com/mudler/vllm.cpp/issues/1551) (FA-2 refuses head_dim 128) and [#1552](https://github.com/mudler/vllm.cpp/issues/1552) (the same defect shape at every other `vt::Attention` caller). Spec [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md)
Row: LTX25-DIT-ATTN-FLASH
State: UNKNOWN
Kind: bug
GitHub: 1549
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:558`

### Frozen archive evidence

> | [#1549](https://github.com/mudler/vllm.cpp/issues/1549) | `LTX25-DIT-ATTN-FLASH` | **One LTX-2.5 DiT forward costs 47.84 s on GB10 because the DiT self-attention never opted into a fast attention op.** `src/vllm/model_executor/models/ltx2_device.cpp:421` calls `vt::Attention`, which on CUDA is `AttentionKernel` (`src/vt/cuda/cuda_ops.cu:1463`) -- the kernel whose own header at `:1456-1459` calls itself "Correctness-grade (M0.9)": one 256-thread block per (query, head), a 256-wide shared-memory tree reduction per key, no K/V tiling. At `768x448/49f` (2352 tokens) that is 75,264 blocks x 2352 keys = 1.77e8 block-key iterations per call x 48 layers. MEASURED 47.84 s mean / 47.91 s median, n=119, spread 5.8%, from the engine's own `last=` lines (`render_phase_log.cpp:388` via `ltx2_video.cpp:4048-4054`), never from the governor, which has reported 1.00 s, 69.1 s, 162 s and 396.9 s for this one quantity. Attribution is arithmetic, not assertion: `.agents/specs/multimodal-speed.md:24-26` measures this same kernel on this same box at 5.70 ns per block-key iteration, and 1.77e8 x 5.70 ns x 48 = **48.4 s against the measured 47.84 s, a 1% match**. WHY IT WAS MISSED, which is the reusable part: `kAttention` is deliberately frozen on the naive kernel so text decode stays byte-identical (`cuda_ops.cu:3120-3122`), and the fast kernels are SEPARATE OPS each caller must opt into BY NAME. No automatic selection, no fallback notice. A model that never opts in gets correct output at ~500x the cost with no warning anywhere -- goldens pass, no refusal fires, and `GetOpProviderStats` counts the naive selection as the success it is. FIXED IN FLOW by routing the self-attention to `vt::AttentionDenseFlash`. Two things found doing it and filed rather than folded in: [#1551](https://github.com/mudler/vllm.cpp/issues/1551) (FA-2 refuses head_dim 128) and [#1552](https://github.com/mudler/vllm.cpp/issues/1552) (the same defect shape at every other `vt::Attention` caller). Spec [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md) | bug |

## Resolution

-

ID: ISSUE-GH-1009
Title: The LTX-2.5 video VAE decode is **single-threaded on a 20-core box**. `ParallelForRows` (`src/vt/cpu/cpu_threadpool.cpp:413 @ 332aed738`) is synchronous and used by 10+ CPU kernels (`cpu_conv2d.cpp:78`, `cpu_layernorm.cpp:53`, `cpu_paged_attn.cpp:185`, `cpu_quant_gemm.cpp:191`, `cpu_ops.cpp:28 @ 332aed738`); zero are in the decode, whose loop nest at `ltx2_video_vae.cpp:161-164 @ 332aed738` is perfectly nested and parallel over `oc`. MEASURED from pre-existing evidence: the completed 320x192/49f render of 2026-08-15 (`~/work/ltx25-e2e/render8-console.log` on `dgx.casa`) holds a 1-minute load average of 1.0-1.3 for ~2h07m of its 2h23m wall — about 89%. **That log is NOT retrievable**: `dgx.casa` has not answered since, so this measurement has no committed evidence path, which is the whole of [#1040](https://github.com/mudler/vllm.cpp/issues/1040). A local seam, not an upstream mirror; no oracle has a host decode. Lever 3. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md)
Row: -
State: UNKNOWN
Kind: feature
GitHub: 1009
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:286`

### Frozen archive evidence

> | [#1009](https://github.com/mudler/vllm.cpp/issues/1009) | — | The LTX-2.5 video VAE decode is **single-threaded on a 20-core box**. `ParallelForRows` (`src/vt/cpu/cpu_threadpool.cpp:413 @ 332aed738`) is synchronous and used by 10+ CPU kernels (`cpu_conv2d.cpp:78`, `cpu_layernorm.cpp:53`, `cpu_paged_attn.cpp:185`, `cpu_quant_gemm.cpp:191`, `cpu_ops.cpp:28 @ 332aed738`); zero are in the decode, whose loop nest at `ltx2_video_vae.cpp:161-164 @ 332aed738` is perfectly nested and parallel over `oc`. MEASURED from pre-existing evidence: the completed 320x192/49f render of 2026-08-15 (`~/work/ltx25-e2e/render8-console.log` on `dgx.casa`) holds a 1-minute load average of 1.0-1.3 for ~2h07m of its 2h23m wall — about 89%. **That log is NOT retrievable**: `dgx.casa` has not answered since, so this measurement has no committed evidence path, which is the whole of [#1040](https://github.com/mudler/vllm.cpp/issues/1040). A local seam, not an upstream mirror; no oracle has a host decode. Lever 3. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) | feature |

## Resolution

-

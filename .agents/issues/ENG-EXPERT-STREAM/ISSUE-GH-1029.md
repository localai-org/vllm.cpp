ID: ISSUE-GH-1029
Title: `IsCudaKeepQuantSupported` (`src/vt/cuda/cuda_quant_dot.cu`) returns true for `kIQ1_S` and `kIQ1_XXXS` since [#967](https://github.com/mudler/vllm.cpp/pull/967), and THREE dispatch switches consume that predicate while #967 extended only the dense one. `MatmulBTQuantGroupedKernelCuda` uses the predicate to SKIP its CPU fallback and then dispatches through a `switch (w)` with no case for either dtype and no `default:`: it quantizes the activation, launches NOTHING, returns, and `CheckCuda(cudaGetLastError())` reports success because there was no launch to fail, so the output tensor is never written. Measured on GB10 by an independent review through a poisoned output buffer: both IQ1 encodings leave `-12345` in place at NMSE `4.58e6` / `9.96e6` against the CPU oracle, with `iq2_s` passing as the control. Reachable by DEFAULT (`qwen3_5_gguf_weights.cpp` -> `qwen3_5.cpp` `KqGrouped` -> `vt::MatmulBTQuantGrouped`, `VT_QWEN35_GROUPED_MOE` on) on the 96.92 % of `Qwen3.8-2.4T` those two encodings make up, so #967 converted correct-but-slow into silently wrong. Same omission in the fused `MoeGateUpSwiGLUGroupedCuda` seam, where it turned a NAMED refusal into silence. It landed green because the CUDA grouped dispatch had NO test: `grep -rl MatmulBTQuantGrouped tests/` found two files and neither mentioned `kCUDA`. Repaired by adding both arms to both grouped switches, a `default:` that THROWS and names the dtype on all three (the general fix: past that predicate there is no fallback left), and a grouped + fused CUDA gate over the dense gate's case table driven through a POISONED output buffer. Also seals the device codebooks against the CPU tables, which `cuda_quant_iq_tables.cuh` claimed from the day it landed and no test did: 266 of 2048 `d_iq1s_grid` entries (13.0 %) are never addressed by the gate's own `mt19937(0x5EED)` stream. Device evidence is OWED and the issue stays open for it: `dgx.casa` was unreachable throughout and this box has no CUDA device, so only the CPU arm ran. Spec [`expert-streaming.md`](../../specs/expert-streaming.md)
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: bug
GitHub: 1029
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:297`

### Frozen archive evidence

> | [#1029](https://github.com/mudler/vllm.cpp/issues/1029) | `ENG-EXPERT-STREAM` | `IsCudaKeepQuantSupported` (`src/vt/cuda/cuda_quant_dot.cu`) returns true for `kIQ1_S` and `kIQ1_XXXS` since [#967](https://github.com/mudler/vllm.cpp/pull/967), and THREE dispatch switches consume that predicate while #967 extended only the dense one. `MatmulBTQuantGroupedKernelCuda` uses the predicate to SKIP its CPU fallback and then dispatches through a `switch (w)` with no case for either dtype and no `default:`: it quantizes the activation, launches NOTHING, returns, and `CheckCuda(cudaGetLastError())` reports success because there was no launch to fail, so the output tensor is never written. Measured on GB10 by an independent review through a poisoned output buffer: both IQ1 encodings leave `-12345` in place at NMSE `4.58e6` / `9.96e6` against the CPU oracle, with `iq2_s` passing as the control. Reachable by DEFAULT (`qwen3_5_gguf_weights.cpp` -> `qwen3_5.cpp` `KqGrouped` -> `vt::MatmulBTQuantGrouped`, `VT_QWEN35_GROUPED_MOE` on) on the 96.92 % of `Qwen3.8-2.4T` those two encodings make up, so #967 converted correct-but-slow into silently wrong. Same omission in the fused `MoeGateUpSwiGLUGroupedCuda` seam, where it turned a NAMED refusal into silence. It landed green because the CUDA grouped dispatch had NO test: `grep -rl MatmulBTQuantGrouped tests/` found two files and neither mentioned `kCUDA`. Repaired by adding both arms to both grouped switches, a `default:` that THROWS and names the dtype on all three (the general fix: past that predicate there is no fallback left), and a grouped + fused CUDA gate over the dense gate's case table driven through a POISONED output buffer. Also seals the device codebooks against the CPU tables, which `cuda_quant_iq_tables.cuh` claimed from the day it landed and no test did: 266 of 2048 `d_iq1s_grid` entries (13.0 %) are never addressed by the gate's own `mt19937(0x5EED)` stream. Device evidence is OWED and the issue stays open for it: `dgx.casa` was unreachable throughout and this box has no CUDA device, so only the CPU arm ran. Spec [`expert-streaming.md`](../../specs/expert-streaming.md) | bug |

## Resolution

-

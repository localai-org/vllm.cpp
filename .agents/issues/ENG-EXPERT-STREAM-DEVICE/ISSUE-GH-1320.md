ID: ISSUE-GH-1320
Title: `VT_CPU_QUANT_REPACK` rewrites a Q8_0 weight into the `block_q8_0x4` i8mm interleave at load, and only the CPU `MatmulBTKernel` understands that layout — the CUDA quant dot reads plain `block_q8_0` and never consults `Tensor::repacked`, so a repacked weight uploaded to a device produces WRONG TOKENS rather than a crash. Its sibling `elem_kn_repack` has BOTH a CPU-platform gate in the loader policy (`gguf_keep_quant.cpp`, `CurrentPlatform().device_type() == kCPU`) and a refusal in `ResidentWeight`; `quant_repack` had NEITHER, because it rides `vt::cpu::QuantRepackActive()`, a HOST-CPU Arm i8mm probe that says nothing about where the weight will execute. An aarch64 box doing `--device cuda` — which is exactly `dgx:gpu0`, a Grace-Blackwell GB10 — satisfies it. Currently silent rather than firing, and that is measured: eligibility is Q8_0 with `n % 4 == 0 && k % 32 == 0`, `Qwen3.8-2.4T-A95B UD-Q1_0` has exactly one Q8_0 tensor at 0.01% of parameters, and the instrumented load of that checkpoint recorded `quant_repack = 0`. Found while implementing W0f ([#1299](https://github.com/mudler/vllm.cpp/issues/1299)), which changes the same function and whose G0-CORRECT gate compares CUDA tokens against CPU tokens — precisely the instrument that would have reported this as a W0f defect. FIXED IN FLOW as the tripwire its sibling already has, placed beside it so it covers the staging branch AND W0f's aliasing branch (where the bytes live was never the question; the kernel that reads them is), gated red-first on both arms and mutation-proven. `VT_CPU_QUANT_REPACK=0` is the operator's way past it. NOT fixed here, deliberately: adding the missing CPU-platform gate to `p.quant_repack` itself is a loader-policy semantics change owned by `QUANT-GGUF-KEEPQ-LOADER`, carried under `## Owed` in [`expert-stream-device-slots.md`](../../specs/expert-stream-device-slots.md)
Row: ENG-EXPERT-STREAM-DEVICE
State: UNKNOWN
Kind: bug
GitHub: 1320
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:486`

### Frozen archive evidence

> | [#1320](https://github.com/mudler/vllm.cpp/issues/1320) | `ENG-EXPERT-STREAM-DEVICE` | `VT_CPU_QUANT_REPACK` rewrites a Q8_0 weight into the `block_q8_0x4` i8mm interleave at load, and only the CPU `MatmulBTKernel` understands that layout — the CUDA quant dot reads plain `block_q8_0` and never consults `Tensor::repacked`, so a repacked weight uploaded to a device produces WRONG TOKENS rather than a crash. Its sibling `elem_kn_repack` has BOTH a CPU-platform gate in the loader policy (`gguf_keep_quant.cpp`, `CurrentPlatform().device_type() == kCPU`) and a refusal in `ResidentWeight`; `quant_repack` had NEITHER, because it rides `vt::cpu::QuantRepackActive()`, a HOST-CPU Arm i8mm probe that says nothing about where the weight will execute. An aarch64 box doing `--device cuda` — which is exactly `dgx:gpu0`, a Grace-Blackwell GB10 — satisfies it. Currently silent rather than firing, and that is measured: eligibility is Q8_0 with `n % 4 == 0 && k % 32 == 0`, `Qwen3.8-2.4T-A95B UD-Q1_0` has exactly one Q8_0 tensor at 0.01% of parameters, and the instrumented load of that checkpoint recorded `quant_repack = 0`. Found while implementing W0f ([#1299](https://github.com/mudler/vllm.cpp/issues/1299)), which changes the same function and whose G0-CORRECT gate compares CUDA tokens against CPU tokens — precisely the instrument that would have reported this as a W0f defect. FIXED IN FLOW as the tripwire its sibling already has, placed beside it so it covers the staging branch AND W0f's aliasing branch (where the bytes live was never the question; the kernel that reads them is), gated red-first on both arms and mutation-proven. `VT_CPU_QUANT_REPACK=0` is the operator's way past it. NOT fixed here, deliberately: adding the missing CPU-platform gate to `p.quant_repack` itself is a loader-policy semantics change owned by `QUANT-GGUF-KEEPQ-LOADER`, carried under `## Owed` in [`expert-stream-device-slots.md`](../../specs/expert-stream-device-slots.md) | bug |

## Resolution

-

ID: ISSUE-GH-1342
Title: `src/vt/vulkan/shaders/vt_silu_and_mul.comp` and `src/vt/metal/metal_msl.h` both carry a comment claiming the silu math is "ported 1:1 from `src/vt/cpu/cpu_ops.cpp` SiluAndMulKernel, including its `gate / (1 + exp(-gate))` spelling", and both emit `gate * vt_sigmoid(gate) * up` instead, where `vt_sigmoid(x) = 1.0/(1.0+exp(-x))` (`vt_common.glsl`). The reciprocal is rounded before the multiply, so it is one extra f32 rounding step that the CPU and CUDA kernels do not take; upstream `activation_kernels.cu::silu_kernel` spells it as the division too. `vt_rms_norm_gated.comp` warns against precisely this form in its own header, so the convention is written down and these two files are on the wrong side of it. Invisible to every gate we own, because both forms are single-rounding-on-store and a bf16-eps parity test passes either way — the false part is the citation. Found during the #1322 inventory. NOT fixed in flow: it is a separate defect from the polarity gap, survives that fix untouched, and belongs to whichever row repairs the two shaders. Owed under `## Owed` of [vt-act-round-polarity.md](../../specs/vt-act-round-polarity.md)
Row: VT-ACT-ROUND-POLARITY
State: UNKNOWN
Kind: bug
GitHub: 1342
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:439`

### Frozen archive evidence

> | [#1342](https://github.com/mudler/vllm.cpp/issues/1342) | `VT-ACT-ROUND-POLARITY` | `src/vt/vulkan/shaders/vt_silu_and_mul.comp` and `src/vt/metal/metal_msl.h` both carry a comment claiming the silu math is "ported 1:1 from `src/vt/cpu/cpu_ops.cpp` SiluAndMulKernel, including its `gate / (1 + exp(-gate))` spelling", and both emit `gate * vt_sigmoid(gate) * up` instead, where `vt_sigmoid(x) = 1.0/(1.0+exp(-x))` (`vt_common.glsl`). The reciprocal is rounded before the multiply, so it is one extra f32 rounding step that the CPU and CUDA kernels do not take; upstream `activation_kernels.cu::silu_kernel` spells it as the division too. `vt_rms_norm_gated.comp` warns against precisely this form in its own header, so the convention is written down and these two files are on the wrong side of it. Invisible to every gate we own, because both forms are single-rounding-on-store and a bf16-eps parity test passes either way — the false part is the citation. Found during the #1322 inventory. NOT fixed in flow: it is a separate defect from the polarity gap, survives that fix untouched, and belongs to whichever row repairs the two shaders. Owed under `## Owed` of [vt-act-round-polarity.md](../../specs/vt-act-round-polarity.md) | bug |

## Resolution

-

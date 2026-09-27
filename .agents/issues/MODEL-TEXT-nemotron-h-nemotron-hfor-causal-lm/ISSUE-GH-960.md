ID: ISSUE-GH-960
Title: `vt::QuantFp8Static`'s ONLY CUDA registration lived at `src/vt/cuda/cuda_matmul_fp8_cutlass.cu:376` (@ `0e1bee42f`), and `CMakeLists.txt:1668` compiles that translation unit only when `VT_CUTLASS_FP8_ARCHS` is non-empty — yet the kernel body has ZERO cutlass tokens (`:353-370`): it is `out[i] = e4m3(x[i] * (1/input_scale))`, a grid-stride elementwise convert. So on every CUDA arch outside the cutlass-fp8 cell — sm_110/Thor is the measured one, and `cutlass-fp8: DISABLED for [110]` is that arch's DOCUMENTED NORMAL PROFILE, not a misconfiguration — `OpId::kQuantFp8Static` was not registered for `DeviceType::kCUDA` at all. Nothing refused first: the GEMM partner `kMatmulFp8CublasLt` IS registered unconditionally (`src/vt/cuda/cuda_matmul.cu:920`), so `MatmulFp8CutlassD`'s guard passed, and the missing quant then resolved through `src/vt/op_provider.cpp:501` to the portable CPU reference tier — eligible because `CudaBackend::UnifiedMemory()` is true — which dereferenced DEVICE pointers on the host and SIGSEGV'd one call later under a banner reading "correct but slow". Fixed by relocating the registration to a new unconditionally-compiled TU `src/vt/cuda/cuda_quant_fp8.cu`, which restores upstream's own partition (vLLM builds `static_scaled_fp8_quant` from the unconditional `VLLM_EXT_SRC` list and gates only its cutlass `scaled_mm` sources). This removes one live INSTANCE of [#844](https://github.com/mudler/vllm.cpp/issues/844) and does not address its class, which stays open. Unblocks the FP8 W8A8 arm on every non-cutlass CUDA arch — the base [#810](https://github.com/mudler/vllm.cpp/issues/810)/[#517](https://github.com/mudler/vllm.cpp/issues/517) A2-Q1 needs, where 46 FP8 mamba projections are 36.6% of decode bytes. Spec [`vt-fp8-quant-arch-gate.md`](../../specs/vt-fp8-quant-arch-gate.md)
Row: MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 960
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:308`

### Frozen archive evidence

> | [#960](https://github.com/mudler/vllm.cpp/issues/960) | `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` | `vt::QuantFp8Static`'s ONLY CUDA registration lived at `src/vt/cuda/cuda_matmul_fp8_cutlass.cu:376` (@ `0e1bee42f`), and `CMakeLists.txt:1668` compiles that translation unit only when `VT_CUTLASS_FP8_ARCHS` is non-empty — yet the kernel body has ZERO cutlass tokens (`:353-370`): it is `out[i] = e4m3(x[i] * (1/input_scale))`, a grid-stride elementwise convert. So on every CUDA arch outside the cutlass-fp8 cell — sm_110/Thor is the measured one, and `cutlass-fp8: DISABLED for [110]` is that arch's DOCUMENTED NORMAL PROFILE, not a misconfiguration — `OpId::kQuantFp8Static` was not registered for `DeviceType::kCUDA` at all. Nothing refused first: the GEMM partner `kMatmulFp8CublasLt` IS registered unconditionally (`src/vt/cuda/cuda_matmul.cu:920`), so `MatmulFp8CutlassD`'s guard passed, and the missing quant then resolved through `src/vt/op_provider.cpp:501` to the portable CPU reference tier — eligible because `CudaBackend::UnifiedMemory()` is true — which dereferenced DEVICE pointers on the host and SIGSEGV'd one call later under a banner reading "correct but slow". Fixed by relocating the registration to a new unconditionally-compiled TU `src/vt/cuda/cuda_quant_fp8.cu`, which restores upstream's own partition (vLLM builds `static_scaled_fp8_quant` from the unconditional `VLLM_EXT_SRC` list and gates only its cutlass `scaled_mm` sources). This removes one live INSTANCE of [#844](https://github.com/mudler/vllm.cpp/issues/844) and does not address its class, which stays open. Unblocks the FP8 W8A8 arm on every non-cutlass CUDA arch — the base [#810](https://github.com/mudler/vllm.cpp/issues/810)/[#517](https://github.com/mudler/vllm.cpp/issues/517) A2-Q1 needs, where 46 FP8 mamba projections are 36.6% of decode bytes. Spec [`vt-fp8-quant-arch-gate.md`](../../specs/vt-fp8-quant-arch-gate.md) | bug |

## Resolution

-

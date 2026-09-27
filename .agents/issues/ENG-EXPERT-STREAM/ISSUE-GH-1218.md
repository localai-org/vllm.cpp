ID: ISSUE-GH-1218
Title: `EnsureGemma4Fp8NativeOnDevice` has no arm-existence guard, and it is the DEFAULT Gemma-4 expert arm, so [#1126](https://github.com/mudler/vllm.cpp/issues/1126) step 1 would still throw mid-decode after [#1205](https://github.com/mudler/vllm.cpp/issues/1205)'s guard lands. That guard covers the BF16 device-expert arm (`src/vllm/model_executor/models/gemma4_moe.cpp:571`, inside `:548-608`). The FP8-native twin at `:611` has none, and `VT_GEMMA4_FP8_NATIVE` returns true when unset (`:969-974`), so on a per-expert FP8 checkpoint the expert loop reaches the twin at `:1359` and `:1484` BEFORE the guarded arm. A `true` from it routes into `ExpertGeGLUFp8Native` (`:95-130`), which needs `vt::ExpertGeGLUFp8TopKM1` (`:105`, which merely returns false off ROCm), `vt::DequantFp8ChannelBf16` (`:117`, `:119`, refusing at `src/vt/fused_ops.cpp:194`) and `vt::MatmulBTAlphaBeta` (`gemma4_moe.cpp:128`, refusing at `src/vt/fused_ops.cpp:152`) — so T>1 throws at the first dequant and T==1 falls through the fused kernel and throws too, in both cases outside the upload's own `try`/`catch`. Latent for the same reason and for exactly as long: the twin's `MakeRoom` also needs `vt::Backend::DeviceMemoryInfo`, which only ROCm overrides (`src/vt/rocm/rocm_backend.hip:358-365`). FILED, NOT FIXED, and not for effort: the BF16 guard keys on ONE predicate that is the same condition its own dispatch uses, which is what makes it honest, whereas the twin depends on three different ops and needs a predicate per op — reusing `HasMatmulBTAlphaBeta` there would be a guard naming the wrong arm, the exact defect this row's review had just corrected in a refusal message. Found while repairing [#1200](https://github.com/mudler/vllm.cpp/pull/1200). Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md)
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: bug
GitHub: 1218
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:380`

### Frozen archive evidence

> | [#1218](https://github.com/mudler/vllm.cpp/issues/1218) | `ENG-EXPERT-STREAM` | `EnsureGemma4Fp8NativeOnDevice` has no arm-existence guard, and it is the DEFAULT Gemma-4 expert arm, so [#1126](https://github.com/mudler/vllm.cpp/issues/1126) step 1 would still throw mid-decode after [#1205](https://github.com/mudler/vllm.cpp/issues/1205)'s guard lands. That guard covers the BF16 device-expert arm (`src/vllm/model_executor/models/gemma4_moe.cpp:571`, inside `:548-608`). The FP8-native twin at `:611` has none, and `VT_GEMMA4_FP8_NATIVE` returns true when unset (`:969-974`), so on a per-expert FP8 checkpoint the expert loop reaches the twin at `:1359` and `:1484` BEFORE the guarded arm. A `true` from it routes into `ExpertGeGLUFp8Native` (`:95-130`), which needs `vt::ExpertGeGLUFp8TopKM1` (`:105`, which merely returns false off ROCm), `vt::DequantFp8ChannelBf16` (`:117`, `:119`, refusing at `src/vt/fused_ops.cpp:194`) and `vt::MatmulBTAlphaBeta` (`gemma4_moe.cpp:128`, refusing at `src/vt/fused_ops.cpp:152`) — so T>1 throws at the first dequant and T==1 falls through the fused kernel and throws too, in both cases outside the upload's own `try`/`catch`. Latent for the same reason and for exactly as long: the twin's `MakeRoom` also needs `vt::Backend::DeviceMemoryInfo`, which only ROCm overrides (`src/vt/rocm/rocm_backend.hip:358-365`). FILED, NOT FIXED, and not for effort: the BF16 guard keys on ONE predicate that is the same condition its own dispatch uses, which is what makes it honest, whereas the twin depends on three different ops and needs a predicate per op — reusing `HasMatmulBTAlphaBeta` there would be a guard naming the wrong arm, the exact defect this row's review had just corrected in a refusal message. Found while repairing [#1200](https://github.com/mudler/vllm.cpp/pull/1200). Listed under `## Owed` in [`expert-streaming.md`](../../specs/expert-streaming.md) | bug |

## Resolution

-

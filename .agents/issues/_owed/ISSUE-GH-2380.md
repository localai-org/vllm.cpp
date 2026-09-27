ID: ISSUE-GH-2380
Title: qwen4_exp has no CUDA arm for four vt:: ops, so all seven of its CUDA kernels are unreachable
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 2380
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-MM-QWEN4-EXP`
>
> Spec: [`.agents/specs/qwen4-exp-flash-next.md`](../../specs/qwen4-exp-flash-next.md), section `### W6-CUDA-B`.
>
> ## What this tracks
>
> Four `qwen4_exp`-path `vt::` ops had no `kCUDA` registration, and `GetOp` throws
> on an unregistered (op, device) rather than falling back — the portable CPU
> reference tier is gated on `Backend::DeviceMemoryIsHostAddressable()`, which
> `CudaBackend` leaves at the base `false`. So none of the seven `qwen4_exp` CUDA
> kernels could run, including the three W6-CUDA landed.
>
> | op | CPU arm mirrored |
> |---|---|
> | `vt::RmsNormGroup` | `src/vt/cpu/cpu_ops.cpp::RmsNormGroupKernel` |
> | `vt::Qwen4ExpGatedResidual` | `src/vt/cpu/cpu_qwen4_exp.cpp::Qwen4ExpGatedResidualKernel` |
> | `vt::Qwen4ExpQsaCompress` | `src/vt/cpu/cpu_qwen4_exp_qsa.cpp` |
> | `vt::Qwen4ExpQsaGatherAttention` | `src/vt/cpu/cpu_qwen4_exp_qsa.cpp` |
>
> W6-CUDA's own split table names each of these four under "the decision a device
> arm must make": the reduction WIDTH for the mixer and the grouped norm, the width
> plus the `round_intermediates_to_bf16` arm for the compressor, and the VISIT
> ORDER plus a device-side `keys_visited` for the gather. The spec section makes
> all four decisions and records which arm is byte-identical to its CPU sibling and
> which is not.
>
> ## What is still owed after it
>
> These arms land **UNREACHED** under AGENTS.md "Nothing lands dead".
> `ModelRegistry::Forward` is all-or-nothing and `EmbeddingKernelCuda` refuses a
> block-quantized table by name, so the block-decoding n-gram gather has no CUDA
> arm and no `qwen4_exp` step can reach a CUDA queue. That is now the ONLY
> remaining blocker where it was four, and it is owned by row
> `MODEL-MM-QWEN4-EXP`; the spec lists it under `## Owed`.
>
> Separately, `IsCudaKeepQuantSupported`
> (`src/vt/cuda/cuda_quant_dot.cu:1711-1739`) returns false for IQ4_NL, Q4_0, Q5_0
> and Q8_0, so the mixer's quantized projection route through `vt::MatmulBT`
> refuses or drains to the host on a CUDA queue. Also owed, also listed.
>
> Two SPEED items are declined here with their conditions recorded in `## Owed`:
> the gather's sequential pass-2 dot, and the mixer's per-call `cudaMalloc`.
>
> ## A note on the count
>
> The wave was dispatched as THREE ops. `kQwen4ExpGatedResidualWriteBack` contains
> `kQwen4ExpGatedResidual` as a substring, so an audit that matched on substrings
> read the mixer as already covered. It was not, and
> `src/vllm/model_executor/models/qwen4_exp_forward.cpp:418`, `:476` and `:535`
> call it on the production path. An op-name audit in this tree must anchor its
> match, because op ids here are deliberately built as extensions of one another.
>

## Resolution

GitHub links pull request #2391 as closing issue #2380 on 2026-08-31.

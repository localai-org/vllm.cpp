ID: ISSUE-GH-1170
Title: All four GDN Triton AOT fast paths reject any geometry whose linear V-head count is not 48 or 32 — `TryTritonPackedDecode` (`src/vt/cuda/cuda_gdn.cu:5207` @ `dd8a3b0e1`), `TryTritonDeltaH` (`:5264`), `TryTritonChunkO` (`:5298`) and `TryTritonWU` (`:5361`), each reading `if (hv_n != 48 && hv_n != 32) return false;` on top of `dk == 128 && dv == 128 && hk_n == 16`. Those two are the only vendored specializations (`src/vt/cuda/triton_aot_vendored/*/gdn_{decode,deltah,chunko,wu}_h{48,32}.*`): 48 is the dense 27B and 32 is `Qwen3.6-35B-A3B`. `Qwen/Qwen3.8-2.4T-A95B` has 128 linear V-heads ([`qwen38-text-only.md`](../../specs/qwen38-text-only.md)) and clears every other term, so it is rejected on `hv_n` alone and runs the hand CUDA kernels on all four legs — the ones `.agents/kernel-matrix.md` measured by cuobjdump at REG:255 + STACK:48 (spilling) against the vLLM FLA cubin's REG:205 / 0 spill, which is the whole reason the vendored cubins exist and are default-on. `Qwen3.8-27B` is NOT affected: it is the `Qwen3.6-27B` geometry retrained, 48 V-heads, and hits every AOT arm. Neither reference restricts the head count — SGLang's `TritonGDNKernel` sets `supports_packed_decode` from the platform alone and takes `num_v_heads` as a runtime argument (`python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py:43` @ `f63458b5be`), and `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` has no shape term (`vllm/envs.py:124` @ `5559679`); both JIT-compile per shape, which is the property the AOT vendoring trades away for a Python-free runtime. Closing it needs `h128` specializations vendored across the supported architectures, or a stated rule for which head counts get an AOT arm plus a visible fallback cost at the call site. Filed, not fixed: either close needs the checkpoint that motivates it, and this hardware cannot run the 2.4T (~4.8 TB bf16 against 128 GB unified memory), so the fallback cannot be measured here today. Listed under `## Owed` in [`gdn-moe-bf16-out.md`](../../specs/gdn-moe-bf16-out.md)
Row: -
State: UNKNOWN
Kind: perf
GitHub: 1170
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:367`

### Frozen archive evidence

> | [#1170](https://github.com/mudler/vllm.cpp/issues/1170) | — | All four GDN Triton AOT fast paths reject any geometry whose linear V-head count is not 48 or 32 — `TryTritonPackedDecode` (`src/vt/cuda/cuda_gdn.cu:5207` @ `dd8a3b0e1`), `TryTritonDeltaH` (`:5264`), `TryTritonChunkO` (`:5298`) and `TryTritonWU` (`:5361`), each reading `if (hv_n != 48 && hv_n != 32) return false;` on top of `dk == 128 && dv == 128 && hk_n == 16`. Those two are the only vendored specializations (`src/vt/cuda/triton_aot_vendored/*/gdn_{decode,deltah,chunko,wu}_h{48,32}.*`): 48 is the dense 27B and 32 is `Qwen3.6-35B-A3B`. `Qwen/Qwen3.8-2.4T-A95B` has 128 linear V-heads ([`qwen38-text-only.md`](../../specs/qwen38-text-only.md)) and clears every other term, so it is rejected on `hv_n` alone and runs the hand CUDA kernels on all four legs — the ones `.agents/kernel-matrix.md` measured by cuobjdump at REG:255 + STACK:48 (spilling) against the vLLM FLA cubin's REG:205 / 0 spill, which is the whole reason the vendored cubins exist and are default-on. `Qwen3.8-27B` is NOT affected: it is the `Qwen3.6-27B` geometry retrained, 48 V-heads, and hits every AOT arm. Neither reference restricts the head count — SGLang's `TritonGDNKernel` sets `supports_packed_decode` from the platform alone and takes `num_v_heads` as a runtime argument (`python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py:43` @ `f63458b5be`), and `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` has no shape term (`vllm/envs.py:124` @ `5559679`); both JIT-compile per shape, which is the property the AOT vendoring trades away for a Python-free runtime. Closing it needs `h128` specializations vendored across the supported architectures, or a stated rule for which head counts get an AOT arm plus a visible fallback cost at the call site. Filed, not fixed: either close needs the checkpoint that motivates it, and this hardware cannot run the 2.4T (~4.8 TB bf16 against 128 GB unified memory), so the fallback cannot be measured here today. Listed under `## Owed` in [`gdn-moe-bf16-out.md`](../../specs/gdn-moe-bf16-out.md) | perf |

## Resolution

-

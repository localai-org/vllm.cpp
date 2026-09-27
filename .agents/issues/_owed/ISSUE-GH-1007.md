ID: ISSUE-GH-1007
Title: The LTX-2.5 video VAE decode has **no device arm at all**: `vt::OpId::kLtx2` (`include/vllm/model_executor/models/ltx2_kernels.h @ 332aed738`, `src/vt/cuda/cuda_ltx2.cu @ 332aed738`) is the DiT device-forward glue — seven ops, no convolution — and nothing the decode reaches. Not an unwired path; the arm does not exist. Every oracle runs this decode GPU-resident and decides placement at build time: Lightricks `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:1139` + `packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:273 @ fd4ded7f2`, SGLang `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/decoding_av.py:71 @ f63458b5b`, vLLM-Omni `vllm_omni/diffusion/models/interface.py:92 @ a4ea67a21` ("VAE(s) (always on GPU)"), diffusers `src/diffusers/models/autoencoders/ltx2_diffusion_decoder.py:208-209 @ c6da9936e` ("No CPU path", which is scoped to ONE of that file's two processors, `LTX2VideoVaeNeighborhoodNattenProcessor` at `:203`, in the DIFFUSION decoder this port refuses by name; the sibling `LTX2VideoVaeNeighborhoodAttnProcessor` at `:153` is "Portable ... Runs anywhere the flex attention path runs", so this citation narrows to NATTEN and the other three carry the claim) — cited at the revision [`oracles/diffusers.md`](../../oracles/diffusers.md) PINS, not at the SHA the local checkout sits on. Lever 1, ranked first on magnitude and last on cost. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md)
Row: -
State: UNKNOWN
Kind: feature
GitHub: 1007
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:284`

### Frozen archive evidence

> | [#1007](https://github.com/mudler/vllm.cpp/issues/1007) | — | The LTX-2.5 video VAE decode has **no device arm at all**: `vt::OpId::kLtx2` (`include/vllm/model_executor/models/ltx2_kernels.h @ 332aed738`, `src/vt/cuda/cuda_ltx2.cu @ 332aed738`) is the DiT device-forward glue — seven ops, no convolution — and nothing the decode reaches. Not an unwired path; the arm does not exist. Every oracle runs this decode GPU-resident and decides placement at build time: Lightricks `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:1139` + `packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:273 @ fd4ded7f2`, SGLang `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/decoding_av.py:71 @ f63458b5b`, vLLM-Omni `vllm_omni/diffusion/models/interface.py:92 @ a4ea67a21` ("VAE(s) (always on GPU)"), diffusers `src/diffusers/models/autoencoders/ltx2_diffusion_decoder.py:208-209 @ c6da9936e` ("No CPU path", which is scoped to ONE of that file's two processors, `LTX2VideoVaeNeighborhoodNattenProcessor` at `:203`, in the DIFFUSION decoder this port refuses by name; the sibling `LTX2VideoVaeNeighborhoodAttnProcessor` at `:153` is "Portable ... Runs anywhere the flex attention path runs", so this citation narrows to NATTEN and the other three carry the claim) — cited at the revision [`oracles/diffusers.md`](../../oracles/diffusers.md) PINS, not at the SHA the local checkout sits on. Lever 1, ranked first on magnitude and last on cost. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) | feature |

## Resolution

-

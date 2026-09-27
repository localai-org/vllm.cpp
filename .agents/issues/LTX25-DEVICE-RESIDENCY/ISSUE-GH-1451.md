ID: ISSUE-GH-1451
Title: **W5 (#1007) put the LTX-2.5 video VAE decode's CONVOLUTION on the device; everything between the convolutions still runs on the host.** `Ltx2ConvVideoDecode` now dispatches every `nn.Conv3d` call — the whole of the decode's arithmetic, Lightricks/LTX-2 @ `fd4ded7f2` `packages/ltx-core/src/ltx_core/model/video_vae/convolution.py:312` — through `vt::Conv3d` on the queue `Ltx2VideoEngine::Load` resolved, uploading the padded volume, the weight and the bias and downloading the result per call. `PixelNorm`, the GroupNorm arm, `Silu`, `ApplyAdaLn`, `FeedSpatialNoise`, `DepthToSpaceUpsample`, `AttnBlock3d`, `Linear3d` and the `unpatchify` tail are still host `std::vector<float>` loops, and there is no device `Ltx2VaeWeights`, so a weight is re-uploaded per call. That is a DIVERGENCE and not only a cost: upstream never moves the tensor back — the decoder is built onto a device once (`single_gpu_model_builder.py:267-288`, CUDA by default at `:273`) with the latent following the weights (`conv_video_decoder.py:283-286`), and vLLM-Omni @ `a4ea67a21` states it as "VAE(s) (always on GPU)" (`vllm_omni/diffusion/models/interface.py:92`). NO NUMBER is attached: W5 took no lease and made no speed claim, so the round-trip cost is unmeasured and measuring it needs the same hardware [#1452](https://github.com/mudler/vllm.cpp/issues/1452) needs. [#1011](https://github.com/mudler/vllm.cpp/issues/1011)'s memory format is the rider that decides which kernel family a resident arm can reach. Listed under `## Owed` in [`ltx25-device-residency.md`](../../specs/ltx25-device-residency.md)
Row: LTX25-DEVICE-RESIDENCY
State: UNKNOWN
Kind: enhancement
GitHub: 1451
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:512`

### Frozen archive evidence

> | [#1451](https://github.com/mudler/vllm.cpp/issues/1451) | `LTX25-DEVICE-RESIDENCY` | **W5 (#1007) put the LTX-2.5 video VAE decode's CONVOLUTION on the device; everything between the convolutions still runs on the host.** `Ltx2ConvVideoDecode` now dispatches every `nn.Conv3d` call — the whole of the decode's arithmetic, Lightricks/LTX-2 @ `fd4ded7f2` `packages/ltx-core/src/ltx_core/model/video_vae/convolution.py:312` — through `vt::Conv3d` on the queue `Ltx2VideoEngine::Load` resolved, uploading the padded volume, the weight and the bias and downloading the result per call. `PixelNorm`, the GroupNorm arm, `Silu`, `ApplyAdaLn`, `FeedSpatialNoise`, `DepthToSpaceUpsample`, `AttnBlock3d`, `Linear3d` and the `unpatchify` tail are still host `std::vector<float>` loops, and there is no device `Ltx2VaeWeights`, so a weight is re-uploaded per call. That is a DIVERGENCE and not only a cost: upstream never moves the tensor back — the decoder is built onto a device once (`single_gpu_model_builder.py:267-288`, CUDA by default at `:273`) with the latent following the weights (`conv_video_decoder.py:283-286`), and vLLM-Omni @ `a4ea67a21` states it as "VAE(s) (always on GPU)" (`vllm_omni/diffusion/models/interface.py:92`). NO NUMBER is attached: W5 took no lease and made no speed claim, so the round-trip cost is unmeasured and measuring it needs the same hardware [#1452](https://github.com/mudler/vllm.cpp/issues/1452) needs. [#1011](https://github.com/mudler/vllm.cpp/issues/1011)'s memory format is the rider that decides which kernel family a resident arm can reach. Listed under `## Owed` in [`ltx25-device-residency.md`](../../specs/ltx25-device-residency.md) | enhancement |

## Resolution

-

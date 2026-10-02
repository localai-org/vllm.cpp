# Nemotron 3 Nano Omni

Nemotron-3-Nano-Omni-30B-A3B-Reasoning is the Nemotron-H hybrid language model
(the same `NemotronHForCausalLM` tower as [Nemotron 3.5 Lightning](nemotron-3-5-lightning.md))
with a RADIO ViT-H/16 image encoder, a Parakeet audio encoder and a video path.
The architecture names are `NemotronH_Nano_Omni_Reasoning_V3` and, for the older
12B release, `NemotronH_Nano_VL_V2`.

**Only the image path is ported, and no end-to-end token gate exists.** Audio
and video requests are refused by name. The image path is checked stage by stage
against the pinned vLLM formulas on the released tensors; the whole model has not
generated a token against the vLLM oracle.

## Run it

Serve the BF16 checkpoint and send an image on `/v1/chat/completions` as the
[usage guide](../USAGE.md#multimodal-input-image-video-audio-to-text) shows. The
server's image codec decodes raw RGB (`data:image/x-raw-rgb;base64,...`) only;
PNG and JPEG are a named residual of the codec, not of this model.

## The checkpoint

| field | value |
|---|---|
| repo | [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16), first party |
| revision | `e5e9932441de940c9a62185c870ea5bcd4cd24e2` |
| on-disk total | 66 032 308 536 bytes (61.5 GiB), 17 shards |
| checked | the index (7349 tensors: 6243 language, 390 vision and `mlp1`, 716 deferred by name) and the vision and `mlp1` tensors of shard 1 |
| sha256 (shard 1) | `de952574c9189925ad15f8cf164184117b6e5eec2d8b7f092e1268c1f0872244` (3 996 912 760 B), the revision's LFS record |

The NVFP4 sibling,
[nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)
@ `16993199e436da4ba75ddc410855f87e0d996ee6` (22 409 034 192 B, 20.9 GiB), is
**refused at load**. Its language tower quantizes `o_proj` and the shared experts
to FP8, ships an `input_scale` on every NVFP4 expert, and keeps `lm_head` in bf16.
The NemotronH loader enumerates the Lightning scheme and does not resolve the
scheme per module from `quantized_layers`, so 5946 of its tensors are
unclaimed. GGUF is refused, as it is for `NemotronHForCausalLM`.

The 12B `NemotronH_Nano_VL_V2` release
([nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)
@ `ca9543b126e8bf3176916d3d305ccc415f89fd4d`) is refused by name: it selects the
static InternVL tiling arm, which is not ported. Its `config.json` also carries
the Python JSON literal `Infinity`, which the config loader cannot parse today.

## What has been measured

- **Processor.** The tiling decision, the antialiased bicubic resize (torch's,
  not Pillow's) and the normalization match the transcription on a 333x517
  image: 344 of 838 656 bf16 pixel values differ, each by one bf16 unit.
- **RADIO tower.** On the released tensors, f32 arithmetic: max relative error
  2.4e-5 over the 1092-patch feature map. In bf16, the production dtype: 4.7e-2,
  which is the bf16 envelope of 32 blocks.
- **Projector.** f32: 4.4e-6. bf16: 5.8e-3.
- **End to end through the image path** (our pixels, tower and projector against
  the reference chain): the worst of the 273 embedding rows has cosine similarity
  0.9978.
- **Reachability.** A tiny checkpoint of the same layout loads under the
  released name, and an image request decodes through the runner token-equal to
  an independent host reference, and differently from the same prompt without
  the image.

## What has not been measured

- No token gate against pinned vLLM on the real checkpoint, and no GPU run.
- No speed number on any axis.
- Audio and video: refused by name.

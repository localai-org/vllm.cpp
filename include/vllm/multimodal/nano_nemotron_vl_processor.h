// The Nemotron Nano VL / Omni image processor: the dynamic-resolution tiler,
// the antialiased bicubic resize, the normalization, the patchify, and the
// `<img><image>*N</img>` prompt expansion.
//
// Ported from vLLM `e126687a9a`,
// vllm/transformers_utils/processors/nano_nemotron_vl.py:
//   NanoNemotronVLTokensAvailable      <- :330-331 (max_num_tokens_available)
//   NanoNemotronVLProcessMedia         <- :377-462 (process_media)
//   NanoNemotronVLComputeParams        <- :464-550 (compute_params)
//   NanoNemotronVLResizeNormalize      <- :61-92 (_bicubic_resize_and_normalize,
//                                         _pil_to_nhwc_tensor), :358-375
//   NanoNemotronVLPatchify             <- :552-570 (stack.rearrange_img)
//   NanoNemotronVLImageReplacement     <- :1086-1098 (get_image_repl)
//   NanoNemotronVLImageMmContent       <- the released chat template's image
//                                         prefix (chat_template.jinja:207-238 @
//                                         e5e9932441de940c9a62185c870ea5bcd4cd24e2)
//
// THE RESIZE IS TORCH'S, NOT PILLOW'S. Upstream calls
// `F.interpolate(mode="bicubic", align_corners=False, antialias=True)` on an
// f32 NCHW tensor. That is the separable antialiased kernel of
// aten/src/ATen/native/cpu/UpSampleKernel.cpp (torch v2.11.0,
// `_compute_indices_min_size_weights_aa` :746-783,
// `HelperInterpCubic::aa_filter` :1333-1349 with a = -0.5, the horizontal pass
// first :1601-1622): the support widens by the downscale factor and the
// weights are normalized per output pixel, all in f32. `pil_resize.h` is a
// different algorithm (a uint8 intermediate between the passes) and is not
// used here.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "vllm/multimodal/inputs.h"
#include "vllm/multimodal/processing/processor.h"

namespace vllm::multimodal {

struct NanoNemotronVLProcessorConfig {
  int64_t patch_size = 16;
  // 1 / downsample_ratio; the tiler asserts it is 2 (:275-283).
  int64_t reduction = 2;
  int64_t min_num_patches = 1024;
  // <= 0 means unbounded (:272).
  int64_t max_num_patches = 13312;
  // The ENGINE's max_model_len (models/nano_nemotron_vl.py:209), not the
  // preprocessor_config.json value.
  int64_t max_model_len = 0;
  double factor_max = 1.0;
  std::array<float, 3> norm_mean{0.48145466f, 0.4578275f, 0.40821073f};
  std::array<float, 3> norm_std{0.26862954f, 0.26130258f, 0.27577711f};
};

// One image's tiling decision: the target patch grid and the number of
// embedding rows it produces after the pixel shuffle.
struct NanoNemotronVLTileParams {
  int64_t grid_w = 0;
  int64_t grid_h = 0;
  int64_t num_embeddings = 0;
};

// `max_num_tokens_available` (:330-331): max_model_len - text_len - 4.
int64_t NanoNemotronVLTokensAvailable(const NanoNemotronVLProcessorConfig& cfg,
                                      int64_t text_prompt_length);

// `process_media` for one image of `width` x `height` pixels under a budget of
// `num_tokens_available` PATCHES. Returns the params and the patch count.
std::pair<NanoNemotronVLTileParams, int64_t> NanoNemotronVLProcessMedia(
    int64_t width, int64_t height, int64_t num_tokens_available,
    const NanoNemotronVLProcessorConfig& cfg);

// `compute_params` over every image of one prompt. `sizes_wh` is (width,
// height) per image; `num_tokens_available` is in POST-shuffle tokens, as
// `NanoNemotronVLTokensAvailable` returns it.
std::vector<NanoNemotronVLTileParams> NanoNemotronVLComputeParams(
    const std::vector<std::pair<int64_t, int64_t>>& sizes_wh,
    int64_t num_tokens_available, const NanoNemotronVLProcessorConfig& cfg);

// torch `F.interpolate(x, size=(out_h, out_w), mode="bicubic",
// align_corners=False, antialias=True)` of a CHW f32 image.
std::vector<float> TorchResizeBicubicAntialias(const std::vector<float>& chw,
                                               int64_t channels, int64_t in_h,
                                               int64_t in_w, int64_t out_h,
                                               int64_t out_w);

// `apply_params` + `_bicubic_resize_and_normalize`: HWC uint8 RGB in, CHW f32
// `(x/255 - mean)/std` out at (grid_h * p, grid_w * p). When `round_to_bf16`
// is set every value is rounded to bf16, the processor's `.to(dtype)` for the
// released bf16 config.
std::vector<float> NanoNemotronVLResizeNormalize(const uint8_t* rgb, int64_t height,
                                                 int64_t width,
                                                 const NanoNemotronVLTileParams& p,
                                                 const NanoNemotronVLProcessorConfig& cfg,
                                                 bool round_to_bf16);

// "c (py yy) (px xx) -> (py px) (c yy xx)": [grid_h * grid_w, 3 * p * p].
std::vector<float> NanoNemotronVLPatchify(const std::vector<float>& chw,
                                          int64_t height, int64_t width,
                                          int64_t patch_size);

// `get_image_repl`: each `<image>` (the target) becomes
// `<img>` + `<image>` x N + `</img>`, with the N `<image>` rows selected as the
// embedding rows (`PromptUpdateDetails.select_token_ids`).
PromptReplacement NanoNemotronVLImageReplacement(int32_t img_start_id,
                                                 int32_t img_context_id,
                                                 int32_t img_end_id,
                                                 const std::vector<int>& num_embeddings);

// The released chat template's image prefix for one message carrying
// `num_images` image parts and `text`: "<image>\n" for one image,
// "<image 1><image> <image 2><image>\n" for several, nothing when the text
// already carries "<image>". Returns the message content the template would
// build (mm prefix + text with leading newlines stripped).
std::string NanoNemotronVLImageMmContent(int num_images, const std::string& text);

struct NanoNemotronVLImageRgb {
  const uint8_t* rgb = nullptr;  // HWC uint8, height * width * 3
  int64_t height = 0;
  int64_t width = 0;
};

// The three special-token ids, resolved from the tokenizer BY STRING.
struct NanoNemotronVLTokenIds {
  int32_t img_start = -1;
  int32_t img_context = -1;
  int32_t img_end = -1;
};

// The whole image arm of the processor for one prompt
// (`_preprocess_image`, processors/...:676-741, and the prompt update,
// models/nano_nemotron_vl.py:427-480): budget every image together, resize,
// normalize, round to the model dtype (bf16), patchify, expand each `<image>`
// in `prompt_ids`, and emit one feature per image whose `data` carries the
// patch rows and `image_grid_thw = {1, grid_h, grid_w}`.
//
// `text_prompt_length` is the token count of the rendered prompt with every
// `<image>` removed (:689-692). The number of `<image>` targets in
// `prompt_ids` must equal the number of images; a mismatch throws
// std::invalid_argument, as upstream's assert does (:726-729).
MultiModalInputs NanoNemotronVLPrepareInputs(const std::vector<int32_t>& prompt_ids,
                                             int64_t text_prompt_length,
                                             const std::vector<NanoNemotronVLImageRgb>& images,
                                             const NanoNemotronVLProcessorConfig& cfg,
                                             const NanoNemotronVLTokenIds& ids,
                                             const std::string& model_id);

}  // namespace vllm::multimodal

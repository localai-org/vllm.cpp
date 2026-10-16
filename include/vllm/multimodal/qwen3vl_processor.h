// Qwen3-VL multimodal input processor — C++ mirror of the vLLM/HF processing
// pipeline for Qwen3VL / Qwen3.6 (they share Qwen3_VisionTransformer).
//
// Ported from:
//   - transformers Qwen2VLImageProcessor(Fast): smart_resize (image_processing_
//     qwen2_vl.py:62), _preprocess patchify (:148-229), fused rescale+normalize
//     (image_processing_backends.py:327 -> (x-127.5)/127.5 for mean=std=0.5,
//     rescale=1/255).
//   - vllm/model_executor/models/qwen3_vl.py: _get_prompt_updates :1400
//     (single <|image_pad|> -> prod(grid_thw)//merge^2 copies of image_token_id);
//     Qwen3VLProcessingInfo :848 (grid sizing).
//   - vllm/multimodal/processing/context.py::call_hf_processor :244
//     (_postprocess_output casts mm_kwargs to model dtype = bf16).
//
// The M1 correctness gate (processor parity) checks pixel_values + image_grid_thw
// + placeholder-expanded ids + mm-hash BIT-identical to the vLLM oracle fixture.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/multimodal/inputs.h"

namespace vllm::multimodal {

// Shared native Qwen3.5 admission/processor envelope. Encoder rows are the
// spatially merged patches, not the decoder's chunked-prefill token budget.
inline constexpr int64_t kNativeQwen3_5MaxImagePixels = 4194304;
inline constexpr int64_t kNativeQwen3_5MaxImagePatches = 16384;

// The subset of preprocessor_config.json + config.vision_config the image path
// needs. mean/std/rescale are fused into the (x - shift)/scale normalize.
struct Qwen3VLProcessorConfig {
  int patch_size = 16;
  int temporal_patch_size = 2;
  int merge_size = 2;  // == vision_config.spatial_merge_size
  double image_mean = 0.5;
  double image_std = 0.5;
  double rescale_factor = 1.0 / 255.0;
  int64_t min_pixels = 65536;      // size.shortest_edge
  int64_t max_pixels = 16777216;   // size.longest_edge
  // token ids (config.json top-level)
  int32_t image_token_id = 151655;
  int32_t vision_start_token_id = 151652;
  int32_t vision_end_token_id = 151653;

  std::string model_id = "Qwen/Qwen3-VL-4B-Instruct";  // for the mm-hash
  // Additive selection: existing BF16/aligned-image callers keep their contract.
  bool torchvision_bicubic_resize = false;
  ImagePixelDType pixel_dtype = ImagePixelDType::kBF16;
  bool retain_pixel_values_f32 = true;
  // Empty means broadcast the scalar fields above (legacy callers).
  std::optional<std::array<double, 3>> channel_mean;
  std::optional<std::array<double, 3>> channel_std;
};

// Load from the two HF json files (preprocessor_config.json + config.json).
Qwen3VLProcessorConfig LoadQwen3VLProcessorConfig(
    const std::string& preprocessor_config_json_path,
    const std::string& config_json_path, const std::string& model_id);

// smart_resize (transformers image_processing_qwen2_vl.py:62). Returns the
// (height, width) that are both divisible by `factor` and whose product lies in
// [min_pixels, max_pixels], preserving aspect ratio.
std::array<int64_t, 2> SmartResize(int64_t height, int64_t width, int64_t factor,
                                   int64_t min_pixels, int64_t max_pixels);

// Video smart_resize (transformers qwen3_vl/video_processing_qwen3_vl.py:35).
// Unlike the image variant the pixel budget includes the temporal dim: t_bar =
// ceil(num_frames/temporal_factor)*temporal_factor and the bound compares
// t_bar*h_bar*w_bar to [min_pixels, max_pixels]. Returns (height, width) only
// (t_bar is handled by the temporal patchify). No aspect-ratio-preserving branch
// when the identity holds (the fixture case); a genuine bicubic resize is deferred.
std::array<int64_t, 2> VideoSmartResize(int64_t num_frames, int64_t height,
                                        int64_t width, int64_t temporal_factor,
                                        int64_t factor, int64_t min_pixels,
                                        int64_t max_pixels);

// Per-temporal-group timestamps (qwen3_vl.py::_calculate_timestamps:975). For
// sampled frame `indices`, ts[i]=indices[i]/video_fps, then averaged in groups of
// `merge_size` (== temporal_patch_size) -> one timestamp per grid_t group.
std::vector<double> ComputeVideoTimestamps(const std::vector<int64_t>& frame_indices,
                                           double video_fps, int merge_size);

class Qwen3VLImageProcessor {
 public:
  explicit Qwen3VLImageProcessor(Qwen3VLProcessorConfig cfg)
      : cfg_(std::move(cfg)) {}

  const Qwen3VLProcessorConfig& config() const { return cfg_; }

  // Preprocess ONE RGB image (HWC uint8, height*width*3) into ImageKwargs:
  // pixel_values [num_patches, channel*temporal*patch*patch] + image_grid_thw,
  // plus the mm-hash. Native EXL3 selects torchvision_bicubic_resize and FP16;
  // legacy callers retain the aligned-image guard and BF16 output by default.
  ImageKwargs ProcessImage(const uint8_t* rgb, int64_t height,
                           int64_t width) const;

  std::string HashImage(const uint8_t* rgb, int64_t height,
                        int64_t width) const;

  // Preprocess ONE video (THWC uint8, num_frames*height*width*3) into VideoKwargs:
  // pixel_values_videos [num_patches, channel*temporal*patch*patch] +
  // video_grid_thw + per-group timestamps. num_frames is padded up to a multiple
  // of temporal_patch_size by repeating the last frame (transformers _preprocess
  // :228-232); each patch-row fuses `temporal_patch_size` REAL consecutive frames.
  // `frame_indices`/`video_fps` drive ComputeVideoTimestamps. Assumes conformant
  // frame dimensions (VideoSmartResize identity); a genuine resize is deferred.
  VideoKwargs ProcessVideo(const uint8_t* thwc, int64_t num_frames, int64_t height,
                           int64_t width,
                           const std::vector<int64_t>& frame_indices,
                           double video_fps) const;

 private:
  Qwen3VLProcessorConfig cfg_;
};

// Placeholder expansion (qwen3_vl.py::_get_prompt_updates:1400). Replaces each
// image_token_id in `prompt_ids` with N = prod(grid_thw)/merge^2 copies of
// image_token_id, consuming `grids` in order. Returns the expanded ids and fills
// `placeholders` with the [offset,length] span of every expanded item.
std::vector<int32_t> ExpandImagePlaceholders(
    const std::vector<int32_t>& prompt_ids, int32_t image_token_id,
    int merge_size, const std::vector<std::array<int64_t, 3>>& grids,
    std::vector<std::array<int, 2>>* placeholders);

// Build the per-video replacement sequence (qwen3_vl.py::get_video_repl:1479).
// For each of grid_t frames the structure is:
//   timestamp_token_ids[f] + vision_start + video_token*tokens_per_frame[f] + vision_end
// `timestamp_token_ids` are the BPE ids of the string f"<{ts:.1f} seconds>" for
// each group's timestamp (produced by the serving tokenizer); this function owns
// the genuinely-new INTERLEAVE structure, not the tokenizer. Returns the full
// expanded sequence for one video item.
std::vector<int32_t> BuildVideoRepl(
    const std::vector<int64_t>& tokens_per_frame,
    const std::vector<std::vector<int32_t>>& timestamp_token_ids,
    int32_t vision_start_token_id, int32_t video_token_id,
    int32_t vision_end_token_id);

}  // namespace vllm::multimodal

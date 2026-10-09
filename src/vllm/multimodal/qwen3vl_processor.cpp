// Ported from: transformers image_processing_qwen2_vl.py (smart_resize :62,
// _preprocess :148-229), image_processing_backends.py:327 (fused normalize),
// vllm/model_executor/models/qwen3_vl.py::_get_prompt_updates:1400. @ vLLM
// e24d1b24. See qwen3vl_processor.h for the full provenance.
#include "vllm/multimodal/qwen3vl_processor.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "vllm/multimodal/hasher.h"
#include "vllm/multimodal/torchvision_resize.h"
#include "vt/dtype.h"

namespace vllm::multimodal {

namespace {
nlohmann::json LoadJson(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open json: " + path);
  nlohmann::json j;
  f >> j;
  return j;
}
}  // namespace

Qwen3VLProcessorConfig LoadQwen3VLProcessorConfig(
    const std::string& preprocessor_config_json_path,
    const std::string& config_json_path, const std::string& model_id) {
  Qwen3VLProcessorConfig cfg;
  cfg.model_id = model_id;

  const nlohmann::json pp = LoadJson(preprocessor_config_json_path);
  cfg.patch_size = pp.value("patch_size", cfg.patch_size);
  cfg.temporal_patch_size =
      pp.value("temporal_patch_size", cfg.temporal_patch_size);
  cfg.merge_size = pp.value("merge_size", cfg.merge_size);
  auto channels = [](const nlohmann::json& value) -> std::array<double, 3> {
    if (value.is_number()) {
      const double scalar = value.get<double>();
      return {scalar, scalar, scalar};
    }
    if (!value.is_array() || value.size() != 3)
      throw std::runtime_error("Qwen image processor: normalization must be scalar or three RGB channels");
    return value.get<std::array<double, 3>>();
  };
  if (pp.contains("image_mean")) {
    cfg.channel_mean = channels(pp["image_mean"]);
    cfg.image_mean = (*cfg.channel_mean)[0];
  }
  if (pp.contains("image_std")) {
    cfg.channel_std = channels(pp["image_std"]);
    cfg.image_std = (*cfg.channel_std)[0];
  }
  cfg.rescale_factor = pp.value("rescale_factor", cfg.rescale_factor);
  if (pp.contains("size")) {
    const auto& sz = pp["size"];
    cfg.min_pixels = sz.value("shortest_edge", cfg.min_pixels);
    cfg.max_pixels = sz.value("longest_edge", cfg.max_pixels);
  }

  const nlohmann::json cj = LoadJson(config_json_path);
  cfg.image_token_id = cj.value("image_token_id", cfg.image_token_id);
  cfg.vision_start_token_id =
      cj.value("vision_start_token_id", cfg.vision_start_token_id);
  cfg.vision_end_token_id =
      cj.value("vision_end_token_id", cfg.vision_end_token_id);
  if (cj.contains("vision_config")) {
    const auto& vc = cj["vision_config"];
    cfg.merge_size = vc.value("spatial_merge_size", cfg.merge_size);
    cfg.patch_size = vc.value("patch_size", cfg.patch_size);
    cfg.temporal_patch_size =
        vc.value("temporal_patch_size", cfg.temporal_patch_size);
  }
  return cfg;
}

std::array<int64_t, 2> SmartResize(int64_t height, int64_t width, int64_t factor,
                                   int64_t min_pixels, int64_t max_pixels) {
  // Input geometry errors are distinct from a misconfigured processor, so
  // the image chat route can report them as client errors before scheduling.
  if (height <= 0 || width <= 0 || height > 32768 || width > 32768 ||
      height * width > 16777216)
    throw std::invalid_argument("smart_resize: invalid image geometry");
  if (factor <= 0 || factor > 32768 || min_pixels <= 0 ||
      max_pixels < min_pixels || max_pixels > 16777216)
    throw std::runtime_error("smart_resize: invalid pixel budget");
  const int64_t hi = std::max(height, width);
  const int64_t lo = std::min(height, width);
  if (lo <= 0 || static_cast<double>(hi) / static_cast<double>(lo) > 200.0) {
    throw std::invalid_argument("smart_resize: aspect ratio must be < 200");
  }
  auto round_by = [factor](int64_t v) -> int64_t {
    // python round(v/factor)*factor with round-half-to-even (banker's rounding).
    return static_cast<int64_t>(std::nearbyint(static_cast<double>(v) / factor)) *
           factor;
  };
  int64_t h_bar = round_by(height);
  int64_t w_bar = round_by(width);
  if (h_bar * w_bar > max_pixels) {
    const double beta = std::sqrt(static_cast<double>(height) *
                                  static_cast<double>(width) / max_pixels);
    h_bar = std::max<int64_t>(
        factor, static_cast<int64_t>(std::floor(height / beta / factor)) * factor);
    w_bar = std::max<int64_t>(
        factor, static_cast<int64_t>(std::floor(width / beta / factor)) * factor);
  } else if (h_bar * w_bar < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) /
                                  (static_cast<double>(height) * width));
    h_bar = static_cast<int64_t>(std::ceil(height * beta / factor)) * factor;
    w_bar = static_cast<int64_t>(std::ceil(width * beta / factor)) * factor;
  }
  return {h_bar, w_bar};
}

std::string Qwen3VLImageProcessor::HashImage(const uint8_t* rgb, int64_t height,
                                             int64_t width) const {
  return MultiModalHasher::HashImageRGB(cfg_.model_id, rgb, height, width);
}

ImageKwargs Qwen3VLImageProcessor::ProcessImage(const uint8_t* rgb,
                                                int64_t height,
                                                int64_t width) const {
  const int patch = cfg_.patch_size;
  const int merge = cfg_.merge_size;
  const int tp = cfg_.temporal_patch_size;
  if (!rgb || patch <= 0 || patch > 256 || merge <= 0 || merge > 32 ||
      tp <= 0 || tp > 16 ||
      (cfg_.pixel_dtype != ImagePixelDType::kBF16 && cfg_.pixel_dtype != ImagePixelDType::kF16))
    throw std::runtime_error("Qwen image processor: invalid RGB/patch geometry/dtype");
  const int64_t factor = static_cast<int64_t>(patch) * merge;

  const auto rs = SmartResize(height, width, factor, cfg_.min_pixels,
                              cfg_.max_pixels);
  const int64_t rh = rs[0], rw = rs[1];
  if (rh <= 0 || rw <= 0 || rh > 32768 || rw > 32768 ||
      rh * rw > cfg_.max_pixels || rh * rw > 16777216)
    throw std::runtime_error("Qwen image processor: resized image exceeds pixel budget");
  std::vector<uint8_t> resized;
  if (rh != height || rw != width) {
    // Native vision explicitly selects the pinned torchvision v2 CPU path.
    // Existing aligned-image callers retain their previous refusal contract.
    if (!cfg_.torchvision_bicubic_resize) throw std::runtime_error(
        "Qwen3VLImageProcessor: image requires resize (" +
        std::to_string(width) + "x" + std::to_string(height) + " -> " +
        std::to_string(rw) + "x" + std::to_string(rh) +
        "); bicubic resize path is deferred (M1 uses conformant images)");
    resized = TorchvisionResizeBicubicRgb(rgb, height, width, rh, rw);
    rgb = resized.data();
  }

  const int64_t grid_h = rh / patch;
  const int64_t grid_w = rw / patch;
  const int64_t grid_t = 1;  // single image
  const int64_t Gh = grid_h / merge;  // merged grid height
  const int64_t Gw = grid_w / merge;
  const int64_t num_patches = grid_t * grid_h * grid_w;
  const int64_t feat = static_cast<int64_t>(3) * tp * patch * patch;
  if (num_patches > 100663296 / feat)
    throw std::runtime_error("Qwen image processor: patch allocation budget exceeded");

  // Fused rescale+normalize shift/scale: (x - mean/rescale) / (std/rescale).
  // For mean=std=0.5, rescale=1/255 this is exactly (x - 127.5f) / 127.5f (f32).
  if (!std::isfinite(cfg_.rescale_factor) || cfg_.rescale_factor <= 0.0)
    throw std::runtime_error("Qwen image processor: invalid rescale factor");
  std::array<float, 3> shifts{}, scales{};
  const float inverse_rescale = static_cast<float>(1.0 / cfg_.rescale_factor);
  for (size_t c = 0; c < 3; ++c) {
    // The pinned Torch backend first creates float32 channel tensors, then
    // multiplies by inverse rescale; keep that rounding boundary explicitly.
    shifts[c] = cfg_.channel_mean
        ? static_cast<float>((*cfg_.channel_mean)[c]) * inverse_rescale
        : static_cast<float>(cfg_.image_mean / cfg_.rescale_factor);
    scales[c] = cfg_.channel_std
        ? static_cast<float>((*cfg_.channel_std)[c]) * inverse_rescale
        : static_cast<float>(cfg_.image_std / cfg_.rescale_factor);
    if (!std::isfinite(shifts[c]) || !std::isfinite(scales[c]) || scales[c] <= 0.0f)
      throw std::runtime_error("Qwen image processor: invalid normalization channels");
  }

  ImageKwargs out;
  out.num_patches = num_patches;
  out.patch_feature_dim = feat;
  out.image_grid_thw = {grid_t, grid_h, grid_w};
  out.pixel_dtype = cfg_.pixel_dtype;
  const size_t values = static_cast<size_t>(num_patches * feat);
  if (cfg_.retain_pixel_values_f32) out.pixel_values_f32.resize(values);
  auto& production = cfg_.pixel_dtype == ImagePixelDType::kF16
      ? out.pixel_values_f16 : out.pixel_values_bf16;
  production.resize(values);

  // Patchify with the exact transformers permute ordering
  // (_preprocess :196-217). Row index and column index (temporal duplicates):
  //   r = ((gh*Gw + gw)*merge + mh)*merge + mw
  //   k = ((c*tp + t)*patch + ph)*patch + pw
  //   src pixel  H = (gh*merge + mh)*patch + ph ; W = (gw*merge + mw)*patch + pw
  const int64_t rowstride = rw * 3;  // resized HWC uint8 source stride
  for (int64_t gh = 0; gh < Gh; ++gh) {
    for (int64_t gw = 0; gw < Gw; ++gw) {
      for (int64_t mh = 0; mh < merge; ++mh) {
        for (int64_t mw = 0; mw < merge; ++mw) {
          const int64_t r = ((gh * Gw + gw) * merge + mh) * merge + mw;
          for (int64_t c = 0; c < 3; ++c) {
            for (int64_t ph = 0; ph < patch; ++ph) {
              const int64_t H = (gh * merge + mh) * patch + ph;
              const int64_t Wbase = (gw * merge + mw) * patch;
              for (int64_t pw = 0; pw < patch; ++pw) {
                const int64_t W = Wbase + pw;
                const uint8_t raw = rgb[H * rowstride + W * 3 + c];
                const float v = (static_cast<float>(raw) - shifts[c]) / scales[c];
                const uint16_t encoded = cfg_.pixel_dtype == ImagePixelDType::kF16
                    ? vt::F32ToF16(v) : vt::F32ToBF16(v);
                for (int64_t t = 0; t < tp; ++t) {
                  const int64_t k = ((c * tp + t) * patch + ph) * patch + pw;
                  const size_t idx = static_cast<size_t>(r * feat + k);
                  if (cfg_.retain_pixel_values_f32) out.pixel_values_f32[idx] = v;
                  production[idx] = encoded;
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

std::vector<int32_t> ExpandImagePlaceholders(
    const std::vector<int32_t>& prompt_ids, int32_t image_token_id,
    int merge_size, const std::vector<std::array<int64_t, 3>>& grids,
    std::vector<std::array<int, 2>>* placeholders) {
  const int64_t merge_length =
      static_cast<int64_t>(merge_size) * merge_size;
  std::vector<int32_t> out;
  out.reserve(prompt_ids.size());
  if (placeholders) placeholders->clear();
  size_t item = 0;
  for (int32_t tok : prompt_ids) {
    if (tok == image_token_id) {
      if (item >= grids.size()) {
        throw std::runtime_error(
            "ExpandImagePlaceholders: more image placeholders than grids");
      }
      const auto& g = grids[item];
      const int64_t n = (g[0] * g[1] * g[2]) / merge_length;
      const int offset = static_cast<int>(out.size());
      out.insert(out.end(), static_cast<size_t>(n), image_token_id);
      if (placeholders)
        placeholders->push_back({offset, static_cast<int>(n)});
      ++item;
    } else {
      out.push_back(tok);
    }
  }
  if (item != grids.size()) {
    throw std::runtime_error(
        "ExpandImagePlaceholders: fewer image placeholders than grids");
  }
  return out;
}

// ---- VIDEO (M3c) ----------------------------------------------------------

std::array<int64_t, 2> VideoSmartResize(int64_t num_frames, int64_t height,
                                        int64_t width, int64_t temporal_factor,
                                        int64_t factor, int64_t min_pixels,
                                        int64_t max_pixels) {
  if (height < factor || width < factor)
    throw std::runtime_error("video smart_resize: h/w must be >= factor");
  const int64_t hi = std::max(height, width), lo = std::min(height, width);
  if (lo <= 0 || static_cast<double>(hi) / static_cast<double>(lo) > 200.0)
    throw std::runtime_error("video smart_resize: aspect ratio must be < 200");
  auto round_by = [factor](int64_t v) -> int64_t {
    return static_cast<int64_t>(std::nearbyint(static_cast<double>(v) / factor)) * factor;
  };
  int64_t h_bar = round_by(height), w_bar = round_by(width);
  const int64_t t_bar =
      static_cast<int64_t>(std::ceil(static_cast<double>(num_frames) / temporal_factor)) *
      temporal_factor;
  const double area = static_cast<double>(t_bar) * h_bar * w_bar;
  if (area > static_cast<double>(max_pixels)) {
    const double beta = std::sqrt(static_cast<double>(num_frames) *
                                  static_cast<double>(height) * width / max_pixels);
    h_bar = std::max<int64_t>(factor,
                              static_cast<int64_t>(std::floor(height / beta / factor)) * factor);
    w_bar = std::max<int64_t>(factor,
                              static_cast<int64_t>(std::floor(width / beta / factor)) * factor);
  } else if (area < static_cast<double>(min_pixels)) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) /
                                  (static_cast<double>(num_frames) * height * width));
    h_bar = static_cast<int64_t>(std::ceil(height * beta / factor)) * factor;
    w_bar = static_cast<int64_t>(std::ceil(width * beta / factor)) * factor;
  }
  return {h_bar, w_bar};
}

std::vector<double> ComputeVideoTimestamps(const std::vector<int64_t>& frame_indices,
                                           double video_fps, int merge_size) {
  std::vector<double> ts;
  ts.reserve(frame_indices.size());
  for (int64_t idx : frame_indices)
    ts.push_back(static_cast<double>(idx) / video_fps);
  std::vector<double> out;
  for (size_t i = 0; i + merge_size - 1 < ts.size(); i += static_cast<size_t>(merge_size))
    out.push_back((ts[i] + ts[i + static_cast<size_t>(merge_size) - 1]) / 2.0);
  return out;
}

VideoKwargs Qwen3VLImageProcessor::ProcessVideo(
    const uint8_t* thwc, int64_t num_frames, int64_t height, int64_t width,
    const std::vector<int64_t>& frame_indices, double video_fps) const {
  const int patch = cfg_.patch_size;
  const int merge = cfg_.merge_size;
  const int tp = cfg_.temporal_patch_size;
  const int64_t factor = static_cast<int64_t>(patch) * merge;
  // Video pixel budget from the video_preprocessor_config size (min/max already
  // fused into cfg_.min/max_pixels defaults for images; video uses its own bounds
  // but for the conformant fixture the identity holds regardless — the guard below
  // rejects any true resize).
  const auto rs = VideoSmartResize(num_frames, height, width, tp, factor,
                                   /*min_pixels=*/4096, /*max_pixels=*/25165824);
  if (rs[0] != height || rs[1] != width) {
    throw std::runtime_error(
        "Qwen3VLImageProcessor::ProcessVideo: frame requires resize (" +
        std::to_string(width) + "x" + std::to_string(height) + " -> " +
        std::to_string(rs[1]) + "x" + std::to_string(rs[0]) +
        "); bicubic resize path is deferred (M3c uses conformant frames)");
  }

  // Pad num_frames up to a multiple of temporal_patch_size by repeating the last
  // frame (transformers _preprocess:228-232); frame reads clamp to the last real.
  const int64_t padded = ((num_frames + tp - 1) / tp) * tp;
  const int64_t grid_t = padded / tp;
  const int64_t grid_h = height / patch;
  const int64_t grid_w = width / patch;
  const int64_t Gh = grid_h / merge;
  const int64_t Gw = grid_w / merge;
  const int64_t num_patches = grid_t * grid_h * grid_w;
  const int64_t feat = static_cast<int64_t>(3) * tp * patch * patch;

  const float shift = static_cast<float>(cfg_.image_mean / cfg_.rescale_factor);
  const float scale = static_cast<float>(cfg_.image_std / cfg_.rescale_factor);

  VideoKwargs out;
  out.num_patches = num_patches;
  out.patch_feature_dim = feat;
  out.video_grid_thw = {grid_t, grid_h, grid_w};
  out.timestamps = ComputeVideoTimestamps(frame_indices, video_fps, tp);
  out.pixel_values_f32.resize(static_cast<size_t>(num_patches * feat));
  out.pixel_values_bf16.resize(static_cast<size_t>(num_patches * feat));

  const int64_t framestride = height * width * 3;  // THWC uint8 source, per frame
  const int64_t rowstride = width * 3;
  // row r enumerates C-order [gt, Gh, Gw, mh, mw]; col k enumerates [c, tp, ph, pw];
  // source frame = gt*tp + t (2 real frames per temporal group), clamped to last.
  for (int64_t gt = 0; gt < grid_t; ++gt) {
    for (int64_t gh = 0; gh < Gh; ++gh) {
      for (int64_t gw = 0; gw < Gw; ++gw) {
        for (int64_t mh = 0; mh < merge; ++mh) {
          for (int64_t mw = 0; mw < merge; ++mw) {
            const int64_t r =
                (((gt * Gh + gh) * Gw + gw) * merge + mh) * merge + mw;
            for (int64_t c = 0; c < 3; ++c) {
              for (int64_t t = 0; t < tp; ++t) {
                int64_t frame = gt * tp + t;
                if (frame >= num_frames) frame = num_frames - 1;  // repeat-last pad
                const uint8_t* fp = thwc + frame * framestride;
                for (int64_t ph = 0; ph < patch; ++ph) {
                  const int64_t H = (gh * merge + mh) * patch + ph;
                  const int64_t Wbase = (gw * merge + mw) * patch;
                  for (int64_t pw = 0; pw < patch; ++pw) {
                    const int64_t W = Wbase + pw;
                    const uint8_t raw = fp[H * rowstride + W * 3 + c];
                    const float v = (static_cast<float>(raw) - shift) / scale;
                    const int64_t k = ((c * tp + t) * patch + ph) * patch + pw;
                    const size_t idx = static_cast<size_t>(r * feat + k);
                    out.pixel_values_f32[idx] = v;
                    out.pixel_values_bf16[idx] = vt::F32ToBF16(v);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

std::vector<int32_t> BuildVideoRepl(
    const std::vector<int64_t>& tokens_per_frame,
    const std::vector<std::vector<int32_t>>& timestamp_token_ids,
    int32_t vision_start_token_id, int32_t video_token_id,
    int32_t vision_end_token_id) {
  if (tokens_per_frame.size() != timestamp_token_ids.size())
    throw std::runtime_error("BuildVideoRepl: tokens_per_frame vs timestamps length");
  std::vector<int32_t> out;
  for (size_t f = 0; f < tokens_per_frame.size(); ++f) {
    for (int32_t t : timestamp_token_ids[f]) out.push_back(t);
    out.push_back(vision_start_token_id);
    out.insert(out.end(), static_cast<size_t>(tokens_per_frame[f]), video_token_id);
    out.push_back(vision_end_token_id);
  }
  return out;
}

}  // namespace vllm::multimodal

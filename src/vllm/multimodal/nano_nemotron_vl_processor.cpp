// Nemotron Nano VL / Omni image processor. See
// include/vllm/multimodal/nano_nemotron_vl_processor.h for the port map.
//
// Ported from vLLM `e126687a9a`,
// vllm/transformers_utils/processors/nano_nemotron_vl.py:61-92, :252-570,
// :1086-1098, and torch v2.11.0 aten/src/ATen/native/cpu/UpSampleKernel.cpp
// (antialiased separable bicubic) with aten/src/ATen/native/UpSample.h
// (`area_pixel_compute_scale`, `cubic_convolution1/2`).
#include "vllm/multimodal/nano_nemotron_vl_processor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/multimodal/hasher.h"
#include "vt/dtype.h"

namespace vllm::multimodal {
namespace {

// Python's `round(x)` on a float: round half to EVEN. `std::nearbyint` under
// the default FE_TONEAREST mode is exactly that; `std::round` (half away from
// zero) is not, and it disagrees at every exact .5 -- which `h / p + 0.5` hits
// whenever the side is a multiple of the patch size.
int64_t PyRound(double x) { return static_cast<int64_t>(std::nearbyint(x)); }

// HelperInterpCubic::aa_filter<float, /*use_keys_cubic=*/true>
// (UpSampleKernel.cpp:1333-1349): a = -0.5, in f32.
float AaCubic(float x) {
  constexpr float a = -0.5f;
  x = std::abs(x);
  if (x < 1.0f) return ((a + 2) * x - (a + 3)) * x * x + 1;          // UpSample.h:399-401
  if (x < 2.0f) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;    // UpSample.h:403-406
  return 0.0f;
}

struct AaAxis {
  std::vector<int64_t> xmin, xsize;
  std::vector<float> w;  // [out, max_interp]
  int64_t max_interp = 0;
};

// `_compute_index_ranges_weights` + `_compute_indices_min_size_weights_aa`
// (UpSampleKernel.cpp:746-783, :857-983) for scalar_t = float, reproducing its
// mixed float/double promotions term by term.
AaAxis ComputeAaAxis(int64_t in, int64_t out) {
  AaAxis ax;
  // area_pixel_compute_scale<float>, align_corners=False, no scale given.
  const float scale = static_cast<float>(in) / static_cast<float>(out);
  const float support =
      (scale >= 1.0) ? static_cast<float>((4 * 0.5) * scale) : static_cast<float>(4 * 0.5);
  ax.max_interp = static_cast<int64_t>(std::ceil(support)) * 2 + 1;
  ax.xmin.resize(static_cast<size_t>(out));
  ax.xsize.resize(static_cast<size_t>(out));
  ax.w.assign(static_cast<size_t>(out * ax.max_interp), 0.0f);
  const float invscale = (scale >= 1.0) ? static_cast<float>(1.0 / scale) : 1.0f;
  for (int64_t i = 0; i < out; ++i) {
    const float center = static_cast<float>(scale * (static_cast<double>(i) + 0.5));
    float total = 0.0f;
    const int64_t xmin = std::max(
        static_cast<int64_t>(static_cast<double>(center - support) + 0.5), int64_t{0});
    int64_t xsize = std::min(static_cast<int64_t>(static_cast<double>(center + support) + 0.5), in) - xmin;
    xsize = std::clamp(xsize, int64_t{0}, ax.max_interp);
    float* wp = &ax.w[static_cast<size_t>(i * ax.max_interp)];
    for (int64_t j = 0; j < xsize; ++j) {
      const float d = static_cast<float>(j + xmin) - center;
      const float wj = AaCubic(static_cast<float>((static_cast<double>(d) + 0.5) *
                                                  static_cast<double>(invscale)));
      wp[j] = wj;
      total += wj;
    }
    if (total != 0.0f)
      for (int64_t j = 0; j < xsize; ++j) wp[j] /= total;
    ax.xmin[static_cast<size_t>(i)] = xmin;
    ax.xsize[static_cast<size_t>(i)] = xsize;
  }
  return ax;
}

}  // namespace

int64_t NanoNemotronVLTokensAvailable(const NanoNemotronVLProcessorConfig& cfg,
                                      int64_t text_prompt_length) {
  return cfg.max_model_len - text_prompt_length - 4;
}

// process_media (:377-462).
std::pair<NanoNemotronVLTileParams, int64_t> NanoNemotronVLProcessMedia(
    int64_t width, int64_t height, int64_t num_tokens_available,
    const NanoNemotronVLProcessorConfig& cfg) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Nano Nemotron VL image processor: the image is " +
                                std::to_string(width) + "x" + std::to_string(height) +
                                " pixels; both sides must be positive");
  }
  const double p = static_cast<double>(cfg.patch_size);
  const int64_t current = num_tokens_available;
  const int64_t cph = PyRound(static_cast<double>(height) / p + 0.5);
  const int64_t cpw = PyRound(static_cast<double>(width) / p + 0.5);
  const int64_t patches = cph * cpw;
  const double factor = std::min(
      std::sqrt(static_cast<double>(current) / static_cast<double>(patches)), cfg.factor_max);
  int64_t tph = static_cast<int64_t>(std::floor(factor * static_cast<double>(cph)));
  int64_t tpw = static_cast<int64_t>(std::floor(factor * static_cast<double>(cpw)));

  if (current > cfg.min_num_patches && tph * tpw < cfg.min_num_patches) {
    if (tph * tpw == 0) {
      // Upstream divides by this product (:419) and raises ZeroDivisionError.
      throw std::invalid_argument(
          "Nano Nemotron VL image processor: the token budget scales this image "
          "to an empty patch grid");
    }
    const double up = std::sqrt(static_cast<double>(cfg.min_num_patches) /
                                static_cast<double>(tph * tpw));
    tph = static_cast<int64_t>(std::ceil(up * static_cast<double>(tph)));
    tpw = static_cast<int64_t>(std::ceil(up * static_cast<double>(tpw)));
  }

  // PIXEL_SHUFFLE only (CONV_MERGING = False): round the grid to a multiple of
  // 2, growing when the budget allows and shrinking otherwise (:430-460).
  const int64_t d = 2;
  const int64_t rem_h = tph % d;
  if (rem_h != 0) {
    const int64_t inc = d - rem_h;
    if ((tph + inc) * tpw <= current) {
      tph += inc;
    } else {
      tph = std::max(d, tph - rem_h);
    }
  }
  const int64_t rem_w = tpw % d;
  if (rem_w != 0) {
    const int64_t inc = d - rem_w;
    if (tph * (tpw + inc) <= current) {
      tpw += inc;
    } else {
      tpw = std::max(d, tpw - rem_w);
    }
  }

  NanoNemotronVLTileParams out;
  out.grid_w = tpw;
  out.grid_h = tph;
  // _get_num_embeddings (:286-289) at (tpw * p, tph * p) pixels.
  out.num_embeddings = (tpw * tph) / (cfg.reduction * cfg.reduction);
  return {out, tpw * tph};
}

// compute_params (:464-550).
std::vector<NanoNemotronVLTileParams> NanoNemotronVLComputeParams(
    const std::vector<std::pair<int64_t, int64_t>>& sizes_wh,
    int64_t num_tokens_available, const NanoNemotronVLProcessorConfig& cfg) {
  const int64_t n = static_cast<int64_t>(sizes_wh.size());
  if (n == 0) return {};
  // Post-shuffle tokens -> patches: x4 for the pixel shuffle.
  int64_t avail = num_tokens_available * cfg.reduction * cfg.reduction;
  avail = std::max(avail, cfg.min_num_patches * n);
  const int64_t max_np = cfg.max_num_patches > 0 ? cfg.max_num_patches
                                                 : std::numeric_limits<int64_t>::max();
  std::vector<int64_t> per(static_cast<size_t>(n),
                           std::max(std::min(avail, max_np), cfg.min_num_patches));
  for (int iter = 0; iter < 10; ++iter) {
    std::vector<NanoNemotronVLTileParams> params;
    std::vector<int64_t> counts;
    int64_t total = 0;
    for (int64_t i = 0; i < n; ++i) {
      auto [p, c] = NanoNemotronVLProcessMedia(sizes_wh[static_cast<size_t>(i)].first,
                                               sizes_wh[static_cast<size_t>(i)].second,
                                               per[static_cast<size_t>(i)], cfg);
      params.push_back(p);
      counts.push_back(c);
      total += c;
    }
    if (total <= avail) return params;
    const double sf = static_cast<double>(avail) / static_cast<double>(total);
    std::vector<int64_t> scaled(static_cast<size_t>(n));
    bool scaled_down = false;
    for (int64_t i = 0; i < n; ++i) {
      scaled[static_cast<size_t>(i)] = std::max(
          cfg.min_num_patches,
          static_cast<int64_t>(static_cast<double>(counts[static_cast<size_t>(i)]) * sf));
      if (scaled[static_cast<size_t>(i)] < per[static_cast<size_t>(i)]) scaled_down = true;
    }
    if (!scaled_down) {
      per.assign(static_cast<size_t>(n), cfg.min_num_patches);
    } else {
      per = std::move(scaled);
    }
  }
  throw std::runtime_error(
      "Nano Nemotron VL image processor: the token budgeting loop did not "
      "converge in 10 rounds (upstream raises the same 'Should be unreachable' "
      "ValueError, processors/nano_nemotron_vl.py:547-550)");
}

// The separable driver (UpSampleKernel.cpp:1590-1640): the contiguous (width)
// axis first, then height; an axis whose size is unchanged is skipped.
std::vector<float> TorchResizeBicubicAntialias(const std::vector<float>& chw,
                                               int64_t channels, int64_t in_h,
                                               int64_t in_w, int64_t out_h,
                                               int64_t out_w) {
  if (static_cast<int64_t>(chw.size()) != channels * in_h * in_w) {
    throw std::invalid_argument("TorchResizeBicubicAntialias: input is not [C, H, W]");
  }
  if (out_h <= 0 || out_w <= 0) {
    throw std::invalid_argument("TorchResizeBicubicAntialias: empty output size");
  }
  std::vector<float> cur = chw;
  int64_t cw = in_w;
  if (out_w != in_w) {
    const AaAxis ax = ComputeAaAxis(in_w, out_w);
    std::vector<float> tmp(static_cast<size_t>(channels * in_h * out_w));
    for (int64_t c = 0; c < channels; ++c)
      for (int64_t y = 0; y < in_h; ++y) {
        const float* src = &cur[static_cast<size_t>((c * in_h + y) * in_w)];
        float* dst = &tmp[static_cast<size_t>((c * in_h + y) * out_w)];
        for (int64_t x = 0; x < out_w; ++x) {
          const int64_t x0 = ax.xmin[static_cast<size_t>(x)];
          const int64_t xs = ax.xsize[static_cast<size_t>(x)];
          const float* wp = &ax.w[static_cast<size_t>(x * ax.max_interp)];
          float acc = src[x0] * wp[0];  // interpolate_aa_single_dim (:190-213)
          for (int64_t j = 1; j < xs; ++j) acc += src[x0 + j] * wp[j];
          dst[x] = acc;
        }
      }
    cur = std::move(tmp);
    cw = out_w;
  }
  if (out_h != in_h) {
    const AaAxis ax = ComputeAaAxis(in_h, out_h);
    std::vector<float> tmp(static_cast<size_t>(channels * out_h * cw));
    for (int64_t c = 0; c < channels; ++c)
      for (int64_t y = 0; y < out_h; ++y) {
        const int64_t y0 = ax.xmin[static_cast<size_t>(y)];
        const int64_t ys = ax.xsize[static_cast<size_t>(y)];
        const float* wp = &ax.w[static_cast<size_t>(y * ax.max_interp)];
        float* dst = &tmp[static_cast<size_t>((c * out_h + y) * cw)];
        for (int64_t x = 0; x < cw; ++x) {
          const float* col = &cur[static_cast<size_t>((c * in_h + y0) * cw + x)];
          float acc = col[0] * wp[0];
          for (int64_t j = 1; j < ys; ++j) acc += col[static_cast<size_t>(j * cw)] * wp[j];
          dst[x] = acc;
        }
      }
    cur = std::move(tmp);
  }
  return cur;
}

std::vector<float> NanoNemotronVLResizeNormalize(const uint8_t* rgb, int64_t height,
                                                 int64_t width,
                                                 const NanoNemotronVLTileParams& p,
                                                 const NanoNemotronVLProcessorConfig& cfg,
                                                 bool round_to_bf16) {
  if (rgb == nullptr || height <= 0 || width <= 0) {
    throw std::invalid_argument("Nano Nemotron VL image processor: empty image");
  }
  // _pil_to_nhwc_tensor + permute(0, 3, 1, 2).to(float32): HWC u8 -> CHW f32.
  std::vector<float> chw(static_cast<size_t>(3 * height * width));
  for (int64_t y = 0; y < height; ++y)
    for (int64_t x = 0; x < width; ++x)
      for (int64_t c = 0; c < 3; ++c)
        chw[static_cast<size_t>((c * height + y) * width + x)] =
            static_cast<float>(rgb[static_cast<size_t>((y * width + x) * 3 + c)]);
  const int64_t oh = p.grid_h * cfg.patch_size;
  const int64_t ow = p.grid_w * cfg.patch_size;
  std::vector<float> out = TorchResizeBicubicAntialias(chw, 3, height, width, oh, ow);
  // ((t / 255.0 - mean) / std).to(dtype) (:80-81), f32 arithmetic.
  for (int64_t c = 0; c < 3; ++c) {
    const float mean = cfg.norm_mean[static_cast<size_t>(c)];
    const float sd = cfg.norm_std[static_cast<size_t>(c)];
    float* plane = &out[static_cast<size_t>(c * oh * ow)];
    for (int64_t i = 0; i < oh * ow; ++i) {
      float v = (plane[i] / 255.0f - mean) / sd;
      if (round_to_bf16) v = vt::BF16ToF32(vt::F32ToBF16(v));
      plane[i] = v;
    }
  }
  return out;
}

std::vector<float> NanoNemotronVLPatchify(const std::vector<float>& chw,
                                          int64_t height, int64_t width,
                                          int64_t patch_size) {
  const int64_t pz = patch_size;
  if (height % pz != 0 || width % pz != 0 ||
      static_cast<int64_t>(chw.size()) != 3 * height * width) {
    throw std::invalid_argument(
        "Nano Nemotron VL patchify: the image must be [3, H, W] with H and W "
        "multiples of the patch size");
  }
  const int64_t py = height / pz;
  const int64_t px = width / pz;
  const int64_t dim = 3 * pz * pz;
  std::vector<float> out(static_cast<size_t>(py * px * dim));
  for (int64_t gy = 0; gy < py; ++gy)
    for (int64_t gx = 0; gx < px; ++gx) {
      float* row = &out[static_cast<size_t>((gy * px + gx) * dim)];
      for (int64_t c = 0; c < 3; ++c)
        for (int64_t yy = 0; yy < pz; ++yy)
          for (int64_t xx = 0; xx < pz; ++xx)
            row[(c * pz + yy) * pz + xx] =
                chw[static_cast<size_t>((c * height + gy * pz + yy) * width + gx * pz + xx)];
    }
  return out;
}

PromptReplacement NanoNemotronVLImageReplacement(int32_t img_start_id,
                                                 int32_t img_context_id,
                                                 int32_t img_end_id,
                                                 const std::vector<int>& num_embeddings) {
  PromptReplacement out;
  out.modality = "image";
  // `target=[vocab["<image>"]]` (models/nano_nemotron_vl.py:476-480): ONE token.
  out.target = {img_context_id};
  for (int n : num_embeddings) {
    if (n <= 0) {
      throw std::runtime_error(
          "Nano Nemotron VL image replacement: an image produced " + std::to_string(n) +
          " embedding rows; the tiler's grid is at least 2x2, so this is a caller defect");
    }
    PromptUpdateContent c;
    c.full.reserve(static_cast<size_t>(n) + 2);
    c.full.push_back(img_start_id);
    c.full.insert(c.full.end(), static_cast<size_t>(n), img_context_id);
    c.full.push_back(img_end_id);
    c.embed_offset = 1;
    c.embed_length = n;
    out.items.push_back(std::move(c));
  }
  return out;
}

std::string NanoNemotronVLImageMmContent(int num_images, const std::string& text) {
  std::string mm;
  if (text.find("<image>") == std::string::npos) {
    if (num_images > 1) {
      for (int i = 0; i < num_images; ++i) {
        if (i > 0) mm += ' ';
        mm += "<image " + std::to_string(i + 1) + "><image>";
      }
      mm += '\n';
    } else if (num_images == 1) {
      mm = "<image>\n";
    }
  }
  // `text_ns.val.lstrip('\n')`
  size_t first = 0;
  while (first < text.size() && text[first] == '\n') ++first;
  return mm + text.substr(first);
}

MultiModalInputs NanoNemotronVLPrepareInputs(const std::vector<int32_t>& prompt_ids,
                                             int64_t text_prompt_length,
                                             const std::vector<NanoNemotronVLImageRgb>& images,
                                             const NanoNemotronVLProcessorConfig& cfg,
                                             const NanoNemotronVLTokenIds& ids,
                                             const std::string& model_id) {
  MultiModalInputs out;
  if (images.empty()) {
    out.prompt_token_ids = prompt_ids;
    return out;
  }
  if (ids.img_start < 0 || ids.img_context < 0 || ids.img_end < 0) {
    throw std::runtime_error(
        "Nano Nemotron VL processor: the <img>, <image> and </img> token ids "
        "were not resolved from the tokenizer");
  }
  const int64_t n_targets = static_cast<int64_t>(
      std::count(prompt_ids.begin(), prompt_ids.end(), ids.img_context));
  if (n_targets != static_cast<int64_t>(images.size())) {
    throw std::invalid_argument("Nano Nemotron VL processor: expected " +
                                std::to_string(images.size()) +
                                " <image> tokens in the prompt but found " +
                                std::to_string(n_targets));
  }
  std::vector<std::pair<int64_t, int64_t>> sizes;
  for (const NanoNemotronVLImageRgb& im : images) sizes.emplace_back(im.width, im.height);
  const std::vector<NanoNemotronVLTileParams> params = NanoNemotronVLComputeParams(
      sizes, NanoNemotronVLTokensAvailable(cfg, text_prompt_length), cfg);

  std::vector<int> counts;
  std::vector<std::shared_ptr<ImageKwargs>> kwargs;
  for (size_t i = 0; i < images.size(); ++i) {
    const NanoNemotronVLTileParams& p = params[i];
    const std::vector<float> chw = NanoNemotronVLResizeNormalize(
        images[i].rgb, images[i].height, images[i].width, p, cfg, /*round_to_bf16=*/true);
    auto kw = std::make_shared<ImageKwargs>();
    kw->pixel_values_f32 = NanoNemotronVLPatchify(chw, p.grid_h * cfg.patch_size,
                                                  p.grid_w * cfg.patch_size, cfg.patch_size);
    kw->pixel_values_bf16.resize(kw->pixel_values_f32.size());
    for (size_t k = 0; k < kw->pixel_values_f32.size(); ++k)
      kw->pixel_values_bf16[k] = vt::F32ToBF16(kw->pixel_values_f32[k]);
    kw->num_patches = p.grid_h * p.grid_w;
    kw->patch_feature_dim = 3 * cfg.patch_size * cfg.patch_size;
    kw->image_grid_thw = {1, p.grid_h, p.grid_w};
    kwargs.push_back(std::move(kw));
    counts.push_back(static_cast<int>(p.num_embeddings));
  }
  const PromptReplacement rep =
      NanoNemotronVLImageReplacement(ids.img_start, ids.img_context, ids.img_end, counts);
  std::vector<AppliedPromptUpdate> applied;
  out.prompt_token_ids = ApplyPromptReplacements(prompt_ids, {rep}, &applied);
  if (applied.size() != images.size()) {
    throw std::runtime_error("Nano Nemotron VL processor: the expansion placed " +
                             std::to_string(applied.size()) + " images for " +
                             std::to_string(images.size()));
  }
  for (size_t i = 0; i < images.size(); ++i) {
    MultiModalFeatureSpec spec;
    spec.modality = "image";
    spec.offset = applied[i].offset;
    spec.length = applied[i].length;
    spec.mm_hash = MultiModalHasher::HashImageRGB(model_id, images[i].rgb, images[i].height,
                                                  images[i].width);
    spec.data = kwargs[i];
    out.mm_features.push_back(std::move(spec));
  }
  return out;
}

}  // namespace vllm::multimodal

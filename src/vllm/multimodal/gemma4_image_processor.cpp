#include "vllm/multimodal/gemma4_image_processor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vllm::multimodal {
namespace {
// ATen 70d99e998b49 UpSampleKernel.cpp: HelperInterpCubic and
// _compute_index_ranges_int16_weights. Unlike Pillow: adaptive int16 precision.
struct AxisWeights {
  struct Window { int64_t first; std::vector<double> weights; };
  std::vector<Window> windows;
  int precision = 0;
};
AxisWeights TensorWeights(int64_t in, int64_t out) {
  AxisWeights a;
  const double scale = static_cast<double>(in) / out;
  const double support = 2 * std::max(1.0, scale);
  double maximum = 0;
  for (int64_t i = 0; i < out; ++i) {
    const double center = scale * (i + 0.5);
    const int64_t first = std::max<int64_t>(0, static_cast<int64_t>(center - support + 0.5));
    const int64_t end = std::min<int64_t>(in, static_cast<int64_t>(center + support + 0.5));
    AxisWeights::Window window{first, {}};
    double total = 0;
    for (int64_t j = first; j < end; ++j) {
      const double x = std::abs((j - center + 0.5) * (scale >= 1.0 ? 1.0 / scale : 1.0));
      const double weight = x < 1 ? ((1.5 * x - 2.5) * x * x + 1) :
                            x < 2 ? (((x - 5) * x + 8) * x - 4) * -0.5 : 0;
      window.weights.push_back(weight);
      total += weight;
    }
    for (auto& weight : window.weights) {
      weight /= total;
      maximum = std::max(maximum, weight);
    }
    a.windows.push_back(std::move(window));
  }
  for (; a.precision < 22; ++a.precision)
    if (static_cast<int>(0.5 + maximum * (1 << (a.precision + 1))) >= (1 << 15)) break;
  for (auto& window : a.windows) for (auto& weight : window.weights) {
    const double v = weight * (1 << a.precision);
    weight = static_cast<int>(v < 0 ? v - 0.5 : v + 0.5);
  }
  return a;
}
uint8_t TensorClip(int64_t sum, int precision) {
  return static_cast<uint8_t>(std::clamp<int64_t>(sum >> precision, 0, 255));
}
std::vector<uint8_t> TensorResize(std::span<const uint8_t> rgb, int64_t h, int64_t w,
                                  int64_t oh, int64_t ow) {
  const auto xs = TensorWeights(w, ow), ys = TensorWeights(h, oh);
  std::vector<uint8_t> out(oh * ow * 3);
  // A single accumulator row: never allocate input-height * output-width.
  std::vector<int64_t> accum(ow * 3);
  for (int64_t y = 0; y < oh; ++y) {
    std::fill(accum.begin(), accum.end(), int64_t{1} << (ys.precision - 1));
    const auto& yw = ys.windows[y];
    for (size_t j = 0; j < yw.weights.size(); ++j) {
      if (yw.weights[j] == 0) continue;
      const int64_t sy = yw.first + j;
      for (int64_t x = 0; x < ow; ++x) {
        const auto& xw = xs.windows[x];
        for (int c = 0; c < 3; ++c) {
          int64_t sum = int64_t{1} << (xs.precision - 1);
          for (size_t k = 0; k < xw.weights.size(); ++k)
            sum += rgb[(sy * w + xw.first + k) * 3 + c] * static_cast<int64_t>(xw.weights[k]);
          accum[x * 3 + c] += TensorClip(sum, xs.precision) * static_cast<int64_t>(yw.weights[j]);
        }
      }
    }
    for (int64_t x = 0; x < ow * 3; ++x) out[y * ow * 3 + x] = TensorClip(accum[x], ys.precision);
  }
  return out;
}
}  // namespace

// HF a08ace4bbd97e721c98751deec37d87b026acadc:
// get_aspect_ratio_preserving_size and Gemma4ImageProcessor._preprocess.
Gemma4ImageProcessor::Gemma4ImageProcessor(int64_t n) : max_soft_tokens_(n) {
  if (n != 70 && n != 140 && n != 280 && n != 560 && n != 1120)
    throw std::invalid_argument("Gemma4 unsupported soft-token budget");
}
std::array<int64_t, 2> Gemma4ImageProcessor::TargetSize(int64_t h, int64_t w) const {
  if (h <= 0 || w <= 0 || h > 65536 || w > 65536 || h * w > 64 * 1024 * 1024)
    throw std::invalid_argument("Gemma4 decoded image dimensions exceed bounded API");
  const double scale = std::sqrt(static_cast<double>(max_soft_tokens_ * 2304) /
                                 static_cast<double>(h * w));
  int64_t th = static_cast<int64_t>(std::floor(scale * h / 48)) * 48;
  int64_t tw = static_cast<int64_t>(std::floor(scale * w / 48)) * 48;
  if (th == 0) {
    th = 48;
    tw = std::min((w / h) * 48, max_soft_tokens_ * 48);
  } else if (tw == 0) {
    tw = 48;
    th = std::min((h / w) * 48, max_soft_tokens_ * 48);
  }
  if (th <= 0 || tw <= 0 || th * tw > max_soft_tokens_ * 2304)
    throw std::invalid_argument("Gemma4 invalid target geometry");
  return {th, tw};
}
Gemma4ImageFeatures Gemma4ImageProcessor::ProcessImage(std::span<const uint8_t> rgb,
                                                      int64_t h, int64_t w) const {
  const auto target = TargetSize(h, w);
  if (rgb.size() != static_cast<size_t>(h * w * 3))
    throw std::invalid_argument("Gemma4 decoded RGB byte count mismatch");
  std::vector<uint8_t> resized;
  if (target != std::array<int64_t, 2>{h, w}) {
    resized = TensorResize(rgb, h, w, target[0], target[1]);
    rgb = resized;
    h = target[0];
    w = target[1];
  }
  Gemma4ImageFeatures out;
  auto& patches = out.patches;
  patches.num_patches = max_soft_tokens_ * 9;
  patches.patch_feature_dim = 768;
  patches.image_grid_thw = {1, h / 16, w / 16};
  patches.pixel_values_f32.resize(static_cast<size_t>(patches.num_patches * 768), 0);
  out.position_ids.resize(static_cast<size_t>(patches.num_patches * 2), -1);
  out.num_soft_tokens = h * w / 2304;
  for (int64_t y = 0; y < h; ++y) {
    for (int64_t x = 0; x < w; ++x) {
      const int64_t row = (y / 16) * (w / 16) + x / 16;
      out.position_ids[row * 2] = x / 16;
      out.position_ids[row * 2 + 1] = y / 16;
      for (int64_t c = 0; c < 3; ++c)
        patches.pixel_values_f32[row * 768 + (y % 16) * 48 + (x % 16) * 3 + c] =
            static_cast<float>(rgb[(y * w + x) * 3 + c]) * (1.0f / 255.0f);
    }
  }
  return out;
}
}  // namespace vllm::multimodal

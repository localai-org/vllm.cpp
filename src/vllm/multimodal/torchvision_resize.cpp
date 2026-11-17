// CPU uint8 antialiased separable bicubic resize, adapted from PyTorch
// cf30153c4c131c8164ee7798e5022d810682e2cb:
// aten/src/ATen/native/cpu/UpSampleKernel.cpp (HelperInterpCubic,
// _compute_index_ranges_int16_weights, _compute_indices_min_size_weights_aa),
// cpu/UpSampleKernelAVXAntialias.h (width-first uint8 passes), and
// aten/src/ATen/native/UpSample.h (cubic_convolution{1,2}).
// PyTorch copyright and BSD notices: third_party/pytorch_vision_resize/LICENSE.
#include "vllm/multimodal/torchvision_resize.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vllm::multimodal {
namespace {

size_t RgbSize(int64_t h, int64_t w) {
  if (h <= 0 || w <= 0 || h > 32768 || w > 32768 || h * w > 16777216)
    throw std::runtime_error("torchvision resize: image extent/pixel limit exceeded");
  return static_cast<size_t>(h * w * 3);
}

double Cubic(double x) {
  x = std::abs(x);
  if (x < 1.0) return ((1.5 * x - 2.5) * x) * x + 1.0;
  if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
  return 0.0;
}

struct Taps {
  int64_t first;
  std::vector<int16_t> weights;
};

struct Axis {
  std::vector<Taps> taps;
  unsigned precision = 0;
};

Axis Coefficients(int64_t in_size, int64_t out_size) {
  const double scale = static_cast<double>(in_size) / out_size;
  const double support = 2.0 * std::max(scale, 1.0);
  const double inverse = scale >= 1.0 ? 1.0 / scale : 1.0;
  const int64_t capacity = static_cast<int64_t>(std::ceil(support)) * 2 + 1;
  Axis result;
  result.taps.reserve(static_cast<size_t>(out_size));
  std::vector<std::vector<double>> normalized;
  normalized.reserve(static_cast<size_t>(out_size));
  double max_weight = 0.0;
  for (int64_t i = 0; i < out_size; ++i) {
    const double center = scale * (i + 0.5);
    const int64_t first = std::max<int64_t>(0, static_cast<int64_t>(center - support + 0.5));
    const int64_t end = std::min(in_size, static_cast<int64_t>(center + support + 0.5));
    std::vector<double> weights(static_cast<size_t>(std::clamp(end - first, int64_t{0}, capacity)));
    double total = 0.0;
    for (size_t j = 0; j < weights.size(); ++j) {
      weights[j] = Cubic((static_cast<int64_t>(j) + first - center + 0.5) * inverse);
      total += weights[j];
    }
    if (weights.empty() || total == 0.0)
      throw std::runtime_error("torchvision resize: empty interpolation support");
    for (double& weight : weights) {
      weight /= total;
      max_weight = std::max(max_weight, weight);
    }
    result.taps.push_back({first, std::vector<int16_t>(weights.size())});
    normalized.push_back(std::move(weights));
  }
  // One precision for the entire axis, selected using the largest normalized
  // positive coefficient. Fixed Pillow precision is not interchangeable.
  while (result.precision < 22 &&
         static_cast<int>(0.5 + max_weight * (1 << (result.precision + 1))) < (1 << 15))
    ++result.precision;
  if (result.precision == 0)
    throw std::runtime_error("torchvision resize: invalid coefficient precision");
  const double unit = static_cast<double>(1 << result.precision);
  for (size_t i = 0; i < normalized.size(); ++i) {
    for (size_t j = 0; j < normalized[i].size(); ++j) {
      const double scaled = normalized[i][j] * unit;
      const int64_t quantized = static_cast<int64_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
      if (quantized < -32768 || quantized > 32767)
        throw std::runtime_error("torchvision resize: coefficient overflows int16");
      result.taps[i].weights[j] = static_cast<int16_t>(quantized);
    }
  }
  return result;
}

uint8_t RoundByte(int64_t sum, unsigned precision) {
  // Negative division may truncate differently from arithmetic right shift,
  // but clipping to zero makes that distinction immaterial to output bytes.
  const int64_t rounded = (sum + (int64_t{1} << (precision - 1))) /
                          (int64_t{1} << precision);
  return static_cast<uint8_t>(std::clamp(rounded, int64_t{0}, int64_t{255}));
}

}  // namespace

std::vector<uint8_t> TorchvisionResizeBicubicRgb(
    const uint8_t* rgb, int64_t in_h, int64_t in_w, int64_t out_h, int64_t out_w) {
  const size_t in_size = RgbSize(in_h, in_w);
  const size_t out_size = RgbSize(out_h, out_w);
  if (!rgb) throw std::runtime_error("torchvision resize: null RGB image");
  if (in_h == out_h && in_w == out_w) return {rgb, rgb + in_size};
  // Check width-first intermediate independently: transposed aspect ratios
  // can exceed both the input and output pixel budgets.
  std::vector<uint8_t> horizontal(RgbSize(in_h, out_w));
  if (in_w == out_w) {
    std::copy(rgb, rgb + in_size, horizontal.begin());
  } else {
    const auto axis = Coefficients(in_w, out_w);
    for (int64_t y = 0; y < in_h; ++y) {
      for (int64_t x = 0; x < out_w; ++x) {
        const auto& taps = axis.taps[static_cast<size_t>(x)];
        for (int64_t c = 0; c < 3; ++c) {
          const size_t start = static_cast<size_t>((y * in_w + taps.first) * 3 + c);
          int64_t sum = static_cast<int64_t>(rgb[start]) * taps.weights[0];
          for (size_t j = 1; j < taps.weights.size(); ++j)
            sum += static_cast<int64_t>(rgb[start + j * 3]) * taps.weights[j];
          horizontal[static_cast<size_t>((y * out_w + x) * 3 + c)] = RoundByte(sum, axis.precision);
        }
      }
    }
  }
  if (in_h == out_h) return horizontal;
  std::vector<uint8_t> output(out_size);
  const auto axis = Coefficients(in_h, out_h);
  const size_t stride = static_cast<size_t>(out_w * 3);
  for (int64_t y = 0; y < out_h; ++y) {
    const auto& taps = axis.taps[static_cast<size_t>(y)];
    for (size_t x = 0; x < stride; ++x) {
      const size_t start = static_cast<size_t>(taps.first) * stride + x;
      int64_t sum = static_cast<int64_t>(horizontal[start]) * taps.weights[0];
      for (size_t j = 1; j < taps.weights.size(); ++j)
        sum += static_cast<int64_t>(horizontal[start + j * stride]) * taps.weights[j];
      output[static_cast<size_t>(y) * stride + x] = RoundByte(sum, axis.precision);
    }
  }
  return output;
}

}  // namespace vllm::multimodal

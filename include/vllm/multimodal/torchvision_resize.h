#pragma once

#include <cstdint>
#include <vector>

namespace vllm::multimodal {

// HWC RGB bytes -> HWC RGB bytes. Matches the pinned torchvision v2 CPU
// uint8 bicubic antialias=True/align_corners=False path: normalized float64
// coefficients quantized to signed int16 with per-axis adaptive precision,
// then clipped uint8 intermediate/output. Reproduces the AVX2 reference math
// without AVX2 or Torch dependencies. Pillow instead uses fixed 22-bit weights.
// Extents are bounded to 32768 and 16,777,216 pixels before allocation;
// processed-image admission (currently 4,194,304 pixels) is the caller's job.
std::vector<uint8_t> TorchvisionResizeBicubicRgb(
    const uint8_t* rgb, int64_t in_h, int64_t in_w, int64_t out_h, int64_t out_w);

}  // namespace vllm::multimodal

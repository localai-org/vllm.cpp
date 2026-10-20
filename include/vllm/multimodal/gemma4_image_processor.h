#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "vllm/multimodal/inputs.h"

namespace vllm::multimodal {

// Decoded uint8 RGB only. No template, decoding, normalization overrides or
// optional overrides. CPU tensor uint8 antialiased resize; patch=16/pool=3.
struct Gemma4ImageFeatures {
  ImageKwargs patches;  // float32 [budget*9, 768]; BF16 intentionally empty
  std::vector<int64_t> position_ids;  // [budget*9, 2], xy; (-1,-1) padding mask
  int64_t num_soft_tokens = 0;
};

class Gemma4ImageProcessor {
 public:
  explicit Gemma4ImageProcessor(int64_t max_soft_tokens = 280);
  std::array<int64_t, 2> TargetSize(int64_t height, int64_t width) const;
  Gemma4ImageFeatures ProcessImageRGB(std::span<const uint8_t> rgb,
                                     int64_t height, int64_t width) const {
    return ProcessImage(rgb, height, width);
  }
  Gemma4ImageFeatures ProcessImage(std::span<const uint8_t> rgb,
                                  int64_t height, int64_t width) const;

 private:
  int64_t max_soft_tokens_;
};

}  // namespace vllm::multimodal

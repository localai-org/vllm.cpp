#include "vllm/model_executor/models/gemma4_vision.h"

#include <cmath>
#include <stdexcept>

namespace vllm::multimodal {
int64_t ValidateGemma4VisionInput(const std::vector<float>& pixels,
                                 const std::vector<int64_t>& positions,
                                 int64_t n_total, const Gemma4VisionConfig& cfg) {
  auto reject = [] { throw std::invalid_argument("Gemma4 malformed vision patches or positions"); };
  if (cfg.patch_size <= 0 || cfg.patch_size > 256 || cfg.pooling_kernel_size <= 0 ||
      cfg.pooling_kernel_size > 256 || cfg.position_embedding_size <= 0 ||
      n_total <= 0 || n_total > 10080) reject();
  const int64_t pd = cfg.patch_dim();
  if (pixels.size() != static_cast<size_t>(n_total * pd) ||
      positions.size() != static_cast<size_t>(n_total * 2)) reject();
  int64_t valid = 0;
  while (valid < n_total && positions[valid * 2] >= 0 && positions[valid * 2 + 1] >= 0)
    ++valid;
  if (valid == 0) reject();
  int64_t width = 0;
  while (width < valid && positions[width * 2 + 1] == 0) ++width;
  if (width == 0 || width % cfg.pooling_kernel_size != 0 || valid % width != 0 ||
      (valid / width) % cfg.pooling_kernel_size != 0) reject();
  for (int64_t row = 0; row < n_total; ++row) {
    if (row < valid) {
      if (positions[row * 2] != row % width || positions[row * 2 + 1] != row / width ||
          positions[row * 2] >= cfg.position_embedding_size ||
          positions[row * 2 + 1] >= cfg.position_embedding_size) reject();
    } else if (positions[row * 2] != -1 || positions[row * 2 + 1] != -1) reject();
    for (int64_t col = 0; col < pd; ++col) {
      const float value = pixels[row * pd + col];
      if (!std::isfinite(value) || value < 0 || value > 1 || (row >= valid && value != 0)) reject();
    }
  }
  return valid;
}
}  // namespace vllm::multimodal

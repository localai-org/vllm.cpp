// Typed XPU vision attention arguments. Keep the legacy BF16 forward separate.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "vt/ops.h"

namespace vllm::multimodal {
inline vt::AttentionArgs TypedVisionAttentionArgs(int64_t head_dim) {
  if (head_dim <= 0) throw std::invalid_argument("typed vision attention requires positive head dimension");
  // The pinned Python vision constructor computes in binary64; the XPU
  // operator narrows that scalar once to FP32. Float sqrt + reciprocal has
  // different bits at D=72, even though it is mathematically equivalent.
  return {static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))), false};
}
}  // namespace vllm::multimodal

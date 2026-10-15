#pragma once
#include "xpu_common.h"
#include <sycl/ext/intel/math.hpp>

namespace vt::xpu {
// Pinned Torch XPU mean for contiguous F32 squares at D256: four adjacent
// registers per virtual lane, then its ascending-offset subgroup tree.
// At >=32 outputs the reduction has 32 lanes and two vectors per lane;
// smaller output counts have >=64 lanes and combine vectors after summing
// their registers. All wider geometries have the same active 64 vectors.
inline float ProducerQkPartial256(View src, int64_t base, int lane, bool paired_vectors) {
  float sums[4];
  for (int j = 0; j < 4; ++j) {
    const float first = Load(src, base + 4 * lane + j);
    sums[j] = first * first;
    if (paired_vectors) {
      const float second = Load(src, base + 128 + 4 * lane + j);
      sums[j] += second * second;
    }
  }
  return ((sums[0] + sums[1]) + sums[2]) + sums[3];
}

inline float ProducerQkMean256(View src, int64_t base, sycl::sub_group group,
                       int lane, int64_t outputs) {
  const bool paired = outputs >= 32;
  float first = ProducerQkPartial256(src, base, lane, paired);
  float second = ProducerQkPartial256(src, base, lane + 16, paired);
  if (!paired) {
    first += ProducerQkPartial256(src, base, lane + 32, false);
    second += ProducerQkPartial256(src, base, lane + 48, false);
  }
  for (int offset = 1; offset < 16; offset *= 2) {
    first += sycl::shift_group_left(group, first, offset);
    second += sycl::shift_group_left(group, second, offset);
  }
  return sycl::group_broadcast(group, first + second, 0) / 256.0f;
}

inline float ProducerQkMean256Scalar(View src, int64_t base, int64_t outputs) {
  const bool paired = outputs >= 32;
  float partials[32];
  for (int lane = 0; lane < 32; ++lane) {
    partials[lane] = ProducerQkPartial256(src, base, lane, paired);
    if (!paired) partials[lane] += ProducerQkPartial256(src, base, lane + 32, false);
  }
  for (int offset = 1; offset < 32; offset *= 2)
    for (int lane = 0; lane + offset < 32; ++lane)
      partials[lane] += partials[lane + offset];
  return partials[0] / 256.0f;
}

inline float ProducerFloatProduct(float first, float second) {
  const float product = sycl::ext::intel::math::fmul_rn(first, second);
  // The pinned XPU multiply returns +0 for a negative value times the
  // Gemma factor +0. Preserve IEEE multiplication's operand-sign XOR,
  // including signed-zero inputs and products that underflow to zero.
  if (product == 0.0f) {
    const uint32_t sign = (sycl::bit_cast<uint32_t>(first) ^
                           sycl::bit_cast<uint32_t>(second)) & 0x80000000u;
    return sycl::bit_cast<float>(sign);
  }
  return product;
}

inline float ProducerQkNormValue(float value, float inverse, float weight, bool gemma) {
  // Original eager IR materializes each F32 multiply before narrowing to
  // F16. Preserve each operation, including an incoming signed zero.
  const float effective_weight = gemma
      ? sycl::ext::intel::math::fadd_rn(1.0f, weight) : weight;
  const float normalized = ProducerFloatProduct(value, inverse);
  return ProducerFloatProduct(normalized, effective_weight);
}
}  // namespace vt::xpu

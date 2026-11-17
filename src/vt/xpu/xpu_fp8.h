#pragma once
#include "xpu_common.h"

namespace vt::xpu {
// E4M3FN, saturating and round-to-nearest-even. Integer rounding avoids FP64
// and does not depend on the device's floating-point rounding mode.
inline uint8_t EncodeE4M3(float value) {
  const auto bits = sycl::bit_cast<uint32_t>(value);
  const auto magnitude = bits & 0x7fffffffu;
  const uint8_t sign = (bits >> 24) & 0x80;
  if (magnitude > 0x7f800000u) return 0x7f;
  if (magnitude >= 0x43e00000u) return sign | 0x7e;  // +/-448, including infinity
  if (magnitude < 0x3c800000u) {
    return sign | uint8_t(sycl::rint(sycl::bit_cast<float>(magnitude) * 512.0f));
  }
  const auto rounded = magnitude + 0x7ffffu + ((magnitude >> 20) & 1u);
  return sign | uint8_t((rounded >> 20) - 120 * 8);
}
inline float DecodeE4M3(uint8_t byte) {
  const uint32_t magnitude = byte & 0x7f, sign = uint32_t(byte & 0x80) << 24;
  if (magnitude == 0x7f) return sycl::bit_cast<float>(sign | 0x7fc00000u);
  if (magnitude < 8) {
    const float value = float(magnitude) * (1.0f / 512.0f);
    return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(value) | sign);
  }
  return sycl::bit_cast<float>(sign | ((magnitude + 120 * 8) << 20));
}
inline float LoadKV(View cache, int64_t offset, float scale) {
  return cache.dtype == DType::kI8
      ? DecodeE4M3(static_cast<const uint8_t*>(cache.data)[offset]) * scale
      : Load(cache, offset);
}
}  // namespace vt::xpu

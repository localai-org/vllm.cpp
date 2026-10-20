#pragma once
#include "vt/dtype.h"
#include <cstdlib>
#include <string_view>
namespace vt {
inline constexpr const char* kXpuSigmoidGateTypedDefault = "0";
inline bool XpuSigmoidGateTyped(int64_t rows, int64_t width) {
  if (rows < 512 || rows > 4096 || width != 6144) return false;
  const char* raw = std::getenv("VT_XPU_SIGMOID_GATE_TYPED");
  const std::string_view value = raw ? raw : kXpuSigmoidGateTypedDefault;
  VT_CHECK(value == "0" || value == "1", "VT_XPU_SIGMOID_GATE_TYPED must be0/1");
  return value == "1";
}
}  // namespace vt

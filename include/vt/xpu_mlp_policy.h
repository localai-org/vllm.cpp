#pragma once
#include "vt/dtype.h"
#include <cstdlib>
#include <string_view>

namespace vt {
inline constexpr const char* kXpuMlpSiluPrepareDefault = "1";
inline bool XpuMlpSiluPrepare(int64_t rows, int64_t width, int64_t output) {
  if (rows < 512 || rows > 4096 || width != 17408 || output != 5120) return false;
  const char* setting = std::getenv("VT_XPU_MLP_SILU_PREPARE");
  const std::string_view value = setting ? setting : kXpuMlpSiluPrepareDefault;
  VT_CHECK(value == "0" || value == "1", "VT_XPU_MLP_SILU_PREPARE must be0/1");
  return value == "1";
}
}  // namespace vt

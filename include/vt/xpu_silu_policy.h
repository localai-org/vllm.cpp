#pragma once
#include "vt/dtype.h"
#include <cstdlib>
#include <string_view>

namespace vt {
// Qualified on real M1/M4 operands, graph replay and paired C1 serving.
// Setting0 retains the original small-row control for diagnostics.
inline constexpr const char* kXpuSmallSiluDefault = "1";
inline bool XpuSmallSiluDecode(int64_t rows, int64_t width) {
  if ((rows != 1 && rows != 4) || width != 17408) return false;
  const char* setting = std::getenv("VT_XPU_SILU_SMALL_DECODE");
  const std::string_view value = setting ? setting : kXpuSmallSiluDefault;
  VT_CHECK(value == "0" || value == "1", "VT_XPU_SILU_SMALL_DECODE must be0/1");
  return value == "1";
}
}  // namespace vt

#pragma once
#include "vt/dtype.h"
#include <cstdlib>
#include <string_view>

namespace vt {
// Qualified on real M1/M4 operands, graph replay and paired C1 serving.
// Mode0 disables small rows; mode1 retains M1/M4; mode2 experiments with
// measured concurrency shapes. Reuse the same immutable FP16 table.
inline constexpr const char* kXpuSmallSiluDefault = "1";
inline bool XpuSmallSiluDecode(int64_t rows, int64_t width) {
  if ((rows != 1 && rows != 2 && rows != 3 && rows != 4 && rows != 8 &&
       rows != 12 && rows != 16) || width != 17408) return false;
  const char* setting = std::getenv("VT_XPU_SILU_SMALL_DECODE");
  const std::string_view value = setting ? setting : kXpuSmallSiluDefault;
  VT_CHECK(value == "0" || value == "1" || value == "2",
           "VT_XPU_SILU_SMALL_DECODE must be 0/1/2");
  return value == "2" || (value == "1" && (rows == 1 || rows == 4));
}
}  // namespace vt

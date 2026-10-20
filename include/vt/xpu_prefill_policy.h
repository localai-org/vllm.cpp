#pragma once
#include "vt/dtype.h"
#include <cstdlib>
#include <string_view>

namespace vt {
// Experimental: keep the reviewed auto/2 baseline unless explicitly enabled.
inline constexpr const char* kXpuOneDnnMixedPrefillDefault = "0";
inline bool XpuOneDnnMixedPrefillEnabled() {
  const char* setting = std::getenv("VT_XPU_ONEDNN_MIXED_PREFILL");
  const std::string_view value = setting ? setting : kXpuOneDnnMixedPrefillDefault;
  VT_CHECK(value == "0" || value == "1", "VT_XPU_ONEDNN_MIXED_PREFILL must be 0/1");
  return value == "1";
}
}  // namespace vt

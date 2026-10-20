#pragma once
#include "vt/dtype.h"
#include <cstdlib>
#include <cstring>

namespace vt {
inline constexpr const char* kXpuGemmaLargeTypedDefault = "1";
inline bool XpuGemmaLargeTyped() {
  const char* value = std::getenv("VT_XPU_GEMMA_LARGE_TYPED");
  if (!value) value = kXpuGemmaLargeTypedDefault;
  if (std::strcmp(value, "0") == 0) return false;
  if (std::strcmp(value, "1") == 0) return true;
  VT_CHECK(false, "VT_XPU_GEMMA_LARGE_TYPED must be 0/1");
  return false;
}
}  // namespace vt

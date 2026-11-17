#pragma once
#include "vt/dtype.h"
#include <cstdlib>
#include <cstring>

namespace vt {
inline constexpr const char* kExl3W8A8SkipFullInitializationDefault = "0";
inline bool Exl3W8A8SkipFullInitialization(int64_t rows, int64_t padded_rows) {
  if (rows <= 0 || rows != padded_rows) return false;
  const char* setting = std::getenv("VT_XPU_W8A8_SKIP_FULL_INIT");
  if (!setting) setting = kExl3W8A8SkipFullInitializationDefault;
  VT_CHECK(std::strcmp(setting,"0") == 0 || std::strcmp(setting,"1") == 0,
           "VT_XPU_W8A8_SKIP_FULL_INIT must be 0/1");
  return std::strcmp(setting,"1") == 0;
}
}  // namespace vt

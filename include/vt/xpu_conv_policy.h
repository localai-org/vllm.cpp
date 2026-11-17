#pragma once
#include "vt/dtype.h"
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace vt {
inline constexpr const char* kXpuConvPrefillTypedDefault = "4";
inline int XpuConvPrefillTypedTile() {
  const char* value = std::getenv("VT_XPU_CONV_PREFILL_TYPED_TILE");
  if (!value) value = kXpuConvPrefillTypedDefault;
  for (int tile : {0,1,2,4}) {
    const char* text = tile == 0 ? "0" : tile == 1 ? "1" : tile == 2 ? "2" : "4";
    if (std::strcmp(value,text) == 0) return tile;
  }
  VT_CHECK(false,"VT_XPU_CONV_PREFILL_TYPED_TILE must be 0/1/2/4");
  return 0;
}
}  // namespace vt

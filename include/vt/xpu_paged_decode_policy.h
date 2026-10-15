#pragma once
#include "vt/paged_attn_route.h"
#include "vt/dtype.h"
#include <cstdlib>
#include <string_view>

namespace vt {
inline constexpr const char* kXpuLongDecodeDefault = "1";
inline constexpr const char* kXpuLongDecodeSplitsDefault = "32";
inline XpuPagedDecodePlan XpuPagedDecodePlanFromEnvironment(int64_t length, int64_t page) {
  const auto* enabled = std::getenv("VT_XPU_XE2_LONG_DECODE");
  const auto* parts = std::getenv("VT_XPU_XE2_DECODE_SPLITS");
  const std::string_view flag = enabled ? enabled : kXpuLongDecodeDefault;
  const std::string_view splits = parts ? parts : kXpuLongDecodeSplitsDefault;
  VT_CHECK(flag == "0" || flag == "1", "VT_XPU_XE2_LONG_DECODE must be0/1");
  VT_CHECK(splits == "16" || splits == "32", "VT_XPU_XE2_DECODE_SPLITS must be16/32");
  return PagedAttnXpuDecodePlan(length, page, flag == "1", splits == "16" ? 16 : 32);
}
}  // namespace vt
